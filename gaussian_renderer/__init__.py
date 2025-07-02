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
import random
import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.general_utils import flip_align_view
from torch.cuda.amp import autocast

def fresnel_schlick(H, V, metallic):
    F0 = torch.lerp(torch.tensor([0.04, 0.04, 0.04], dtype=torch.float32).to(H.device), metallic, metallic)
    cos_theta = torch.dot(H, V).clamp(min=0, max=1)
    return F0 + (1 - F0) * (1 - cos_theta) ** 5

def normal_distribution_function(N, H, roughness):
    alpha = roughness ** 2
    N_dot_H = torch.dot(N, H).clamp(min=1e-8)
    alpha_sq = alpha ** 2
    denom = (N_dot_H ** 2) * (alpha_sq - 1) + 1
    return alpha_sq / (torch.pi * denom ** 2)

def geometry_function(N, V, L, roughness):
    def geometry_schlick_ggx(N_dot_V, k):
        return N_dot_V / (N_dot_V * (1 - k) + k)
    
    alpha = roughness ** 2
    k = (alpha + 1) ** 2 / 8
    N_dot_V = torch.dot(N, V).clamp(min=1e-8)
    N_dot_L = torch.dot(N, L).clamp(min=1e-8)
    ggx1 = geometry_schlick_ggx(N_dot_V, k)
    ggx2 = geometry_schlick_ggx(N_dot_L, k)
    return ggx1 * ggx2

@autocast()
def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = True, override_color = None, use_trained_exp=False, debug=False, speed=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_opacity.shape[0], 1))
    dir_norm = dir_pp.norm(dim=1, keepdim=True)
    if (dir_norm == 0).any():
        print("Warning: dir_norm contains zero values!")
    dir_pp_normalized = dir_pp/dir_norm
    pc.current_dir_pp_normalized = dir_pp_normalized
    if not speed:
        if debug:
            normal_axis = pc.get_minimum_axis
            normal_axis, _ = flip_align_view(normal_axis, dir_pp_normalized)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            sh_colors = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color

    # 获取所有高斯点属性（向量化计算）
    xyz = pc.get_xyz
    normals = pc.get_normals(dir_pp_normalized=dir_pp_normalized)   # (N,3)
    albedo = pc.get_albedo()    # (N,3)
    if albedo is None:
        print("Error: albedo is None!")
        # 可以选择返回默认值或者进行其他处理
        return None
    roughness = pc.get_roughness  # (N,1)
    metallic = pc.get_metallic()  # (N,1)
    if metallic.dim() == 1:
        metallic = metallic.unsqueeze(1)

    n_points = xyz.shape[0]

    # 计算视线方向（View Direction）
    camera_center = viewpoint_camera.camera_center
    dir_to_cam = camera_center - xyz
    V = dir_to_cam / dir_to_cam.norm(dim=1, keepdim=True)  # (N,3) 单位向量

    # 定义光源方向
    light_direction = torch.tensor([0.0, 0.0, 1.0], device=xyz.device).repeat(xyz.shape[0], 1)  # (N,3)
    theta = random.uniform(0, 2*math.pi)
    light_direction = torch.tensor(
        [math.cos(theta), math.sin(theta), 1.0], 
        device=xyz.device
        ).repeat(xyz.shape[0], 1)
    L = light_direction / light_direction.norm(dim=1, keepdim=True)

    # 计算半角向量（H）
    H = (V + L) / torch.norm(V + L, dim=1, keepdim=True)  # (N,3)
    assert H.shape == (n_points, 3), f"半角向量维度错误: {H.shape} != ({n_points},3)"

    # 法线分布函数（GGX）
    alpha = roughness**2  # (N,1)
    alpha_sq = alpha**2
    # N_dot_H = torch.sum(normals * H, dim=1).unsqueeze(-1).clamp(min=1e-8)  # (N,1)
    N_dot_H = torch.sum(normals * H, dim=1, keepdim=True).clamp(1e-6,1.0)  # (N,1)
    # N_dot_H = (torch.sum(normals * H, dim=1, keepdim=True) + 1e-6).clamp(0, 1)
    D = alpha_sq / (torch.pi * (N_dot_H**2 * (alpha_sq - 1) + 1)**2)  # (N,1)
    # D_denominator = (N_dot_H**2 * (alpha_sq - 1) + 1)
    # D = alpha_sq / (torch.pi * D_denominator**2 + 1e-6)

    # 几何函数（Smith-Schlick GGX）
    k = (alpha + 1)**2 / 8
    N_dot_V = torch.sum(normals * V, dim=1, keepdim=True)
    N_dot_L = torch.sum(normals * L, dim=1, keepdim=True)

    G = (N_dot_V / (N_dot_V * (1 - k) + k + 1e-6)) * (N_dot_L / (N_dot_L * (1 - k) + k + 1e-6))

    # 菲涅尔项（修正后的向量化实现）
    F0 = torch.lerp(torch.tensor([0.04], device=metallic.device), albedo, metallic)
    cos_theta = torch.sum(H * V, dim=1, keepdim=True)  # (N,1)
    F = F0 + (1 - F0) * (1 - cos_theta)**5  # (N,3)

    specular = (D * G * F) / (4 * N_dot_V * N_dot_L + 1e-6)

    diffuse = (1 - metallic) * albedo / torch.pi  # (N,3)

    pbr_colors = ((diffuse + specular) * N_dot_L).clamp(0, 1)

    use_pbr_model = True

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    if use_pbr_model:  # 定义一个明确的标志
        # 使用微平面模型 + SH混合颜色
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            colors_precomp = colors_precomp,  # 混合后的颜色
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    else:
        # 使用原始3DGS的SH着色
        if separate_sh:
            rendered_image, radii, depth_image = rasterizer(
                means3D = means3D,
                means2D = means2D,
                dc = dc,
                shs = shs,  # 注意：不要同时提供shs和colors_precomp
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = cov3D_precomp)
        else:
            rendered_image, radii, depth_image = rasterizer(
                means3D = means3D,
                means2D = means2D,
                shs = shs,
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = cov3D_precomp)
        
    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        "depth" : depth_image
        }
    
    return out
