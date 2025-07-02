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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
except:
    pass

C1 = 0.01 ** 2
C2 = 0.03 ** 2

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        ssim_map = fusedssim(C1, C2, img1, img2)
        ctx.save_for_backward(img1.detach(), img2)
        ctx.C1 = C1
        ctx.C2 = C2
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        return None, None, grad, None

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def fast_ssim(img1, img2):
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()

# def calculate_geometric_consistency_loss(render_pkg, viewpoint_cam):
#     viewspace_point_tensor = render_pkg["viewspace_points"]
#     K = viewpoint_cam.K
#     R = torch.tensor(viewpoint_cam.R, device=K.device, dtype=torch.float32)
#     T = torch.tensor(viewpoint_cam.T, device=K.device, dtype=torch.float32)
#     viewspace_point_tensor = viewspace_point_tensor.to(torch.float32)

#     def project_points(points, K, R, T):
#         ones = torch.ones(points.shape[0], 1, device=points.device, dtype=torch.float32)
#         points_homogeneous = torch.cat([points, ones], dim=1)
#         points_camera = torch.matmul(R, points_homogeneous[:, :3].T).T + T
#         points_image = torch.matmul(K, points_camera.T).T
#         points_image = points_image[:, :2] / points_image[:, 2:3]
#         del ones, points_homogeneous, points_camera  # 释放不再使用的张量
#         return points_image

#     projected_points = project_points(viewspace_point_tensor, K, R, T)

#     def unproject_points(points_image, K, R, T):
#         ones = torch.ones(points_image.shape[0], 1, device=points_image.device, dtype=torch.float32)
#         points_camera = torch.matmul(torch.inverse(K), torch.cat([points_image, ones], dim=1).T).T
#         points_world = torch.matmul(torch.inverse(R), (points_camera - T).T).T
#         del ones, points_camera  # 释放不再使用的张量
#         return points_world

#     unprojected_points = unproject_points(projected_points, K, R, T)

#     geometric_loss = torch.mean(torch.norm(viewspace_point_tensor - unprojected_points, dim=1))

#     return geometric_loss

def calculate_geometric_consistency_loss(depth_ref, depth_nbr, R_rn, T_rn, K_ref, K_nbr, valid_mask_ref):
    """
    计算几何一致性损失
    Args:
        depth_ref: 参考视图的深度图 (H, W)
        depth_nbr: 邻接视图的深度图 (H, W)
        R_rn: 参考视图到邻接视图的旋转矩阵 (3, 3)
        T_rn: 参考视图到邻接视图的平移向量 (3, 1)
        K_ref: 参考视图内参矩阵 (3, 3)
        K_nbr: 邻接视图内参矩阵 (3, 3)
        valid_mask_ref: 参考视图的有效像素掩码 (H, W)
    Returns:
        L_geo: 几何一致性损失 (标量)
    """
    # 确保输入为二维张量
    depth_ref = depth_ref.squeeze(0)  # (1, H, W) -> (H, W)
    depth_nbr = depth_nbr.squeeze(0)
    valid_mask_ref = valid_mask_ref.squeeze(0) if valid_mask_ref.dim() == 3 else valid_mask_ref

    H, W = depth_ref.shape
    device = depth_ref.device

    # 生成像素网格 (参考视图)
    y_coords, x_coords = torch.meshgrid(torch.arange(H, device=device), 
                                        torch.arange(W, device=device), 
                                        indexing='ij')
    ones = torch.ones_like(x_coords)
    p_r = torch.stack([x_coords, y_coords, ones], dim=-1).float()  # (H, W, 3)

    # 将像素坐标转换为参考视图的3D坐标
    K_ref_inv = torch.inverse(K_ref)
    P_r = depth_ref.unsqueeze(-1) * (p_r @ K_ref_inv.T)  # (H, W, 3)

    # 将3D点转换到邻接视图坐标系
    P_n = (torch.einsum('ij,hwj->hwi', R_rn, P_r) + T_rn)  # (H, W, 3)

    # 投影到邻接视图像素坐标
    p_n_homo = (P_n @ K_nbr.T)  # (H, W, 3)
    p_n_uv = p_n_homo[..., :2] / (p_n_homo[..., 2:] + 1e-6)  # (H, W, 2)

    # 双线性插值获取邻接视图深度
    u_norm = (p_n_uv[..., 0] / (W - 1)) * 2 - 1  # 归一化到[-1, 1]
    v_norm = (p_n_uv[..., 1] / (H - 1)) * 2 - 1
    grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0)  # (1, H, W, 2)
    depth_nbr_sampled = F.grid_sample(
        depth_nbr.unsqueeze(0).unsqueeze(0),  # 输入需要是4D (B, C, H, W)
        grid, 
        mode='bilinear', 
        align_corners=True
    ).squeeze()  # (H, W)

    # 检查投影是否在图像范围内
    valid_uv = (p_n_uv[..., 0] >= 0) & (p_n_uv[..., 0] < W) & \
               (p_n_uv[..., 1] >= 0) & (p_n_uv[..., 1] < H)
    # valid_mask_total = valid_mask_ref & valid_uv
    # 确保 valid_mask_total 是二维布尔张量 depth_nbr_sampled>0 确保临接视图的采样深度是有效的
    valid_mask_total = valid_mask_ref & valid_uv & (depth_nbr_sampled > 0)

    # 反向投影回参考视图
    p_n_sampled = torch.stack([p_n_uv[..., 0], p_n_uv[..., 1], ones], dim=-1)  # (H, W, 3)
    K_nbr_inv = torch.inverse(K_nbr)
    P_n_reconstructed = depth_nbr_sampled.unsqueeze(-1) * (p_n_sampled @ K_nbr_inv.T)  # (H, W, 3)

    # 转换回参考视图坐标系
    R_nr = R_rn.T
    T_nr = -R_nr @ T_rn
    P_r_prime = (torch.einsum('ij,hwj->hwi', R_nr, P_n_reconstructed) + T_nr)  # (H, W, 3)

    # 投影回参考视图的像素坐标
    p_r_prime_homo = (P_r_prime @ K_ref.T)  # (H, W, 3)
    p_r_prime_uv = p_r_prime_homo[..., :2] / (p_r_prime_homo[..., 2:] + 1e-6)

    # 计算欧氏距离
    diff = torch.norm(p_r[..., :2] - p_r_prime_uv, dim=-1)
    valid_diff = diff[valid_mask_total]

    if valid_diff.numel() == 0:
        return torch.tensor(0.0, device=device)
    
    return valid_diff.mean()

# 将图像转换为灰度图像
def rgb_to_gray(image):
    if image.dim() == 3:
        image = image.unsqueeze(0)
    gray_image = 0.299 * image[:, 0:1, :, :] + 0.587 * image[:, 1:2, :, :] + 0.114 * image[:, 2:3, :, :]
    return gray_image

def calculate_photometric_consistency_loss(rendered_image, gt_image):
    """
    计算光度一致性损失
    :param rendered_image: 渲染得到的图像
    :param gt_image: 真实图像
    :return: 光度一致性损失
    """

    rendered_gray = rgb_to_gray(rendered_image)
    gt_gray = rgb_to_gray(gt_image)

    # 计算归一化互相关（NCC）
    def ncc(x, y):
        x_mean = torch.mean(x, dim=(2, 3), keepdim=True)
        y_mean = torch.mean(y, dim=(2, 3), keepdim=True)
        x_std = torch.std(x, dim=(2, 3), keepdim=True)
        y_std = torch.std(y, dim=(2, 3), keepdim=True)
        ncc_value = torch.mean((x - x_mean) * (y - y_mean) / (x_std * y_std + 1e-8))
        del x_mean, y_mean, x_std, y_std  # 释放不再使用的张量
        return ncc_value

    ncc_value = ncc(rendered_gray, gt_gray)

    # 光度一致性损失，使用 1 - NCC
    photometric_loss = 1 - ncc_value

    return photometric_loss
