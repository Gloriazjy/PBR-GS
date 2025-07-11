#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation, get_minimum_axis, flip_align_view
from scipy.spatial import KDTree

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._normal=torch.empty(0)
        self._normal2=torch.empty(0)
        self._specular = torch.empty(0)
        self._roughness = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.tmp_radii = torch.empty(0)
        self.setup_functions()
        self.albedo = None
        self._metallic = torch.empty(0)
        self.setup_metallic_parameters()

        self.diffuse_activation = torch.sigmoid
        self.specular_activation = torch.sigmoid
        self.default_roughness = 0.6
        self.roughness_activation = torch.sigmoid
        self.roughness_bias = 0.
        self.current_dir_pp_normalized = None

    def setup_metallic_parameters(self):
        if self._metallic.dim() ==1:
            self._metallic=self._metallic.unsqueeze(-1)

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_diffuse(self):
        return self._features_dc
    
    @property
    def get_specular(self):
        return self.specular_activation(self._specular)
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure
    
    @property
    def get_minimum_axis(self):
        return get_minimum_axis(self.get_scaling, self.get_rotation)

    @property
    def get_roughness(self):
        roughness = self.roughness_activation(self._roughness + self.roughness_bias)
        return roughness

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        
        normals = np.zeros_like(np.asarray(pcd.points, dtype=np.float32))
        normals2 = np.copy(normals)

        self._normal = nn.Parameter(torch.from_numpy(normals).to(self._xyz.device).requires_grad_(True))
        self._normal2 = nn.Parameter(torch.from_numpy(normals2).to(self._xyz.device).requires_grad_(True))
        specular_len = 3 
        self._specular = nn.Parameter(torch.zeros((fused_point_cloud.shape[0], specular_len), device="cuda").requires_grad_(True))
        self._roughness = nn.Parameter(self.default_roughness*torch.ones((fused_point_cloud.shape[0], 1), device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        metallic_init = 0.05 * torch.ones((fused_point_cloud.shape[0], 1), device="cuda")
        self._metallic = nn.Parameter(metallic_init.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self._normal.requires_grad_(requires_grad=False)
        self._normal2.requires_grad_(requires_grad=False)

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._normal], 'lr': training_args.normal_lr, "name": "normal"},
            {'params': [self._normal2], 'lr': training_args.normal_lr, "name": "normal2"},
            {'params': [self._roughness], 'lr': training_args.roughness_lr, "name": "roughness"},
            {'params': [self._specular], 'lr': training_args.specular_lr, "name": "specular"},
            {'params': [self._metallic], 'lr': training_args.metallic_lr, "name": "metallic"}
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # 添加 normals2 的字段名
        l.extend(['nx2', 'ny2', 'nz2'])
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        l.append('roughness')
        # 添加 specular 的字段名
        for i in range(self._specular.shape[1]):
            l.append('specular_{}'.format(i))
        # 添加 metallic 的字段名
        l.append('metallic')
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = self._normal.detach().cpu().numpy()
        normals2 = self._normal2.detach().cpu().numpy()
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        roughness = self._roughness.detach().cpu().numpy()
        specular = self._specular.detach().cpu().numpy()
        metallic = self._metallic.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, normals2, f_dc, f_rest, opacities, scale, rotation, roughness, specular, metallic), axis=1)
        elements[:] = list(map(tuple, attributes))

        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        normals = np.stack((np.asarray(plydata.elements[0]["nx"]),
                            np.asarray(plydata.elements[0]["ny"]),
                            np.asarray(plydata.elements[0]["nz"])),  axis=1)
        roughness = np.asarray(plydata.elements[0]["roughness"])[..., np.newaxis]
        specular_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("specular")]
        specular = np.zeros((xyz.shape[0], len(specular_names)))
        for idx, attr_name in enumerate(specular_names):
            specular[:, idx] = np.asarray(plydata.elements[0][attr_name])

        normal2 = np.stack((
            np.asarray(plydata.elements[0]["nx2"]),
            np.asarray(plydata.elements[0]["ny2"]),
            np.asarray(plydata.elements[0]["nz2"])
        ), axis=1)
        metallic = np.asarray(plydata.elements[0]["metallic"])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._normal = nn.Parameter(torch.tensor(normals, dtype=torch.float, device="cuda").requires_grad_(True))
        self._normal2 = nn.Parameter(torch.tensor(normal2, dtype=torch.float, device="cuda").requires_grad_(True))
        self._roughness = nn.Parameter(torch.tensor(roughness, dtype=torch.float, device="cuda").requires_grad_(True))
        self._specular = nn.Parameter(torch.tensor(specular, dtype=torch.float, device="cuda").requires_grad_(True))
        self._metallic = nn.Parameter(torch.tensor(metallic, dtype=torch.float, device="cuda").requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            param = group["params"][0]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            # assert mask.shape[0] == param.shape[0], \
            f"参数{group['name']}的维度{param.shape[0]}与掩码{mask.shape[0]}不匹配"
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        # assert mask.shape[0] == self.get_xyz.shape[0], f"掩码长度{mask.shape[0]}与高斯点数{self.get_xyz.shape[0]}不匹配"
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._specular = optimizable_tensors["specular"]
        self._normal = optimizable_tensors["normal"]
        self._normal2 = optimizable_tensors["normal2"]
        self._roughness = optimizable_tensors["roughness"]
        self._metallic = optimizable_tensors["metallic"]
        # assert self._metallic.shape[0] == mask.shape[0], "剪枝掩码与Metallic参数维度不匹配"

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            group_name = group["name"]
            if group_name not in tensors_dict:
                continue
            extension_tensor = tensors_dict[group_name]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group_name] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group_name] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii, new_normal, new_normal2, new_specular, new_roughness, new_metallic):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "normal": new_normal,
        "normal2": new_normal2,
        "roughness": new_roughness,
        "specular":new_specular,
        "metallic": new_metallic}
    
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._normal = optimizable_tensors["normal"]
        self._normal2 = optimizable_tensors["normal2"]
        self._roughness = optimizable_tensors["roughness"]
        self._specular = optimizable_tensors["specular"]
        self._metallic = optimizable_tensors["metallic"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)
        new_normal = self._normal[selected_pts_mask].repeat(N, 1)
        new_normal2 = self._normal2[selected_pts_mask].repeat(N, 1)
        new_specular = self._specular[selected_pts_mask].repeat(N,1)
        new_roughness = self._roughness[selected_pts_mask].repeat(N, 1)
        new_metallic = self._metallic[selected_pts_mask].repeat(N, 1)

        # assert new_metallic.shape[0] == N * selected_pts_mask.sum(), "Metallic参数分裂错误"
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii, new_normal, new_normal2, new_specular, new_roughness, new_metallic)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

        # 在 densify_and_split 方法的末尾添加
        # assert self._features_dc.shape[0] == self._specular.shape[0], "Size mismatch between _features_dc and _specular"
        # assert self._features_rest.shape[0] == self._specular.shape[0], "Size mismatch between _features_rest and _specular"

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]
        new_normal = self._normal[selected_pts_mask]
        new_normal2 = self._normal2[selected_pts_mask]
        new_specular = self._specular[selected_pts_mask]
        new_roughness = self._roughness[selected_pts_mask]
        new_metallic = self._metallic[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii, new_normal, new_normal2, new_specular, new_roughness, new_metallic)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
    
    # 计算反照率
    def compute_albedo(self):
        features = self.get_features
        if features is None:
            print("Error: self.get_features() returned None!")
            return None
        dc_features = features[:, :3, 0]
        albedo = dc_features.clamp(min=0, max=1)
        return albedo

    def compute_ambient_occlusion(self, points, normals, num_rays=100, max_distance=1.0):
        """
        计算所有点的环境光遮蔽值
        :param points: 物体表面的点，torch.Tensor，形状为 (N, 3)
        :param normals: 这些点的法线，torch.Tensor，形状为 (N, 3)
        :param num_rays: 发射的射线数量
        :param max_distance: 射线的最大传播距离
        :return: 环境光遮蔽值，范围在 [0, 1] 之间，0 表示完全遮蔽，1 表示无遮蔽，形状为 (N, 1)
        """
        N = points.shape[0]
        ao_values = torch.zeros(N, device=points.device)
        for i in range(N):
            point = points[i]
            normal = normals[i]
            hit_count = 0
            for _ in range(num_rays):
                # 在半球面内随机生成一个方向
                random_direction = torch.randn(3, device=point.device)
                # 确保方向在法线所在的半球面内
                if torch.dot(random_direction, normal) < 0:
                    random_direction = -random_direction
                random_direction = random_direction / torch.norm(random_direction)

                # 发射射线，检查是否与周围物体相交
                # 这里简单假设已经有一个函数可以进行射线追踪
                if self.ray_trace(point, random_direction, max_distance):
                    hit_count += 1

            # 计算环境光遮蔽值
            ao = 1 - (hit_count / num_rays)
            ao_values[i] = ao
        return ao_values.unsqueeze(-1)

    def compute_total_light(self, points, normals, light_sources, ambient_light=0.1):
        """
        计算所有点的总光照
        :param points: 物体表面的点，torch.Tensor，形状为 (N, 3)
        :param normals: 这些点的法线，torch.Tensor，形状为 (N, 3)
        :param light_sources: 光源列表，每个光源是一个字典，包含 'position' 和 'intensity'
        :param ambient_light: 环境光强度
        :return: 总光照强度，torch.Tensor，形状为 (N, 3)
        """
        N = points.shape[0]
        total_lights = torch.zeros((N, 3), device=points.device)
        for i in range(N):
            point = points[i]
            normal = normals[i]
            direct_light = torch.zeros(3, device=point.device)
            # 计算直接光照
            for light in light_sources:
                light_direction = (light['position'] - point).normalized()
                light_intensity = light['intensity']
                # 兰伯特漫反射模型
                diffuse = torch.dot(normal, light_direction).clamp(min=0) * light_intensity
                direct_light += diffuse

            # 计算环境光遮蔽
            ao = self.compute_ambient_occlusion(point.unsqueeze(0), normal.unsqueeze(0)).squeeze()

            # 计算间接光照
            indirect_light = ao * ambient_light

            # 总光照
            total_light = direct_light + indirect_light
            total_lights[i] = total_light
        return total_lights
    
    def get_normals(self, dir_pp_normalized, return_delta=False):
        normal_axis = self.get_minimum_axis
        normal_axis = normal_axis
        normal_axis, positive = flip_align_view(normal_axis, dir_pp_normalized)
        delta_normal1 = self._normal  # (N, 3) 
        delta_normal2 = self._normal2 # (N, 3) 
        delta_normal = torch.stack([delta_normal1, delta_normal2], dim=-1) # (N, 3, 2)
        idx = torch.where(positive, 0, 1).long()[:,None,:].repeat(1, 3, 1) # (N, 3, 1)
        delta_normal = torch.gather(delta_normal, index=idx, dim=-1).squeeze(-1) # (N, 3)
        normal = delta_normal + normal_axis 
        normal = normal/normal.norm(dim=1, keepdim=True) # (N, 3)
        if return_delta:
            return normal, delta_normal
        else:
            return normal

    def get_albedo(self):
        albedo = self.compute_albedo()
        if albedo is None:
            print("Error: compute_albedo returned None!")
        return albedo
    
    # def compute_metallic(self):
    #     specular_reflectance = self._specular[:, 0]
    #     metallic_threshold = 0.5
    #     self._metallic = (specular_reflectance > metallic_threshold).float()
    
    def get_metallic(self):
        return torch.sigmoid(self._metallic)

