#!/usr/bin/env python3
"""
DeepIM 손실 함수
- Position Loss + Rotation Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def rotation_6d_to_matrix(rot_6d):
    """6D 표현 → 3x3 회전 행렬 (Gram-Schmidt 정규화)
    
    Args:
        rot_6d: (N, 6) 6D rotation representation
    
    Returns:
        R: (N, 3, 3) 회전 행렬
    """
    if rot_6d.dim() == 1:
        rot_6d = rot_6d.unsqueeze(0)
    
    a1 = rot_6d[:, :3]
    a2 = rot_6d[:, 3:6]
    
    b1 = F.normalize(a1, dim=1)
    b2 = a2 - (b1 * a2).sum(dim=1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    
    R = torch.stack([b1, b2, b3], dim=2)  # (N, 3, 3)
    return R


def compose_rotation_6d(base_6d, delta_6d):
    """두 6D 회전 표현을 합성
    
    Args:
        base_6d: (N, 6) 기본 회전
        delta_6d: (N, 6) 추가 회전 (residual)
    
    Returns:
        composed_6d: (N, 6) 합성된 회전
    """
    # 6D → 회전 행렬
    R_base = rotation_6d_to_matrix(base_6d)
    R_delta = rotation_6d_to_matrix(delta_6d)
    
    # 회전 합성: R_new = R_delta @ R_base
    R_composed = torch.bmm(R_delta, R_base)
    
    # 회전 행렬 → 6D
    composed_6d = torch.cat([R_composed[:, :, 0], R_composed[:, :, 1]], dim=1)
    
    return composed_6d


class DeepIMLoss(nn.Module):
    """DeepIM 학습용 손실 함수"""
    
    def __init__(self, lambda_pos=1.0, lambda_rot=0.5):
        """
        Args:
            lambda_pos: 위치 손실 가중치
            lambda_rot: 회전 손실 가중치
        """
        super().__init__()
        self.lambda_pos = lambda_pos
        self.lambda_rot = lambda_rot
    
    def forward(self, pred_pos, pred_rot, gt_pos, gt_rot, 
                initial_pos=None, initial_rot=None):
        """
        Args:
            pred_pos: (N, 3) 예측된 최종 위치
            pred_rot: (N, 6) 예측된 최종 회전 (6D)
            gt_pos: (N, 3) GT 위치
            gt_rot: (N, 6) GT 회전 (6D)
            initial_pos: (N, 3) 초기 위치 (옵션)
            initial_rot: (N, 6) 초기 회전 (옵션)
        
        Returns:
            total_loss: 총 손실
            loss_dict: 개별 손실 딕셔너리
        """
        # Position Loss (SmoothL1)
        pos_loss = F.smooth_l1_loss(pred_pos, gt_pos)
        
        # Rotation Loss (6D representation에 대한 L1)
        rot_loss = F.smooth_l1_loss(pred_rot, gt_rot)
        
        # Total Loss
        total_loss = self.lambda_pos * pos_loss + self.lambda_rot * rot_loss
        
        loss_dict = {
            'total': total_loss.item(),
            'position': pos_loss.item(),
            'rotation': rot_loss.item(),
        }
        
        return total_loss, loss_dict


class GeodesicLoss(nn.Module):
    """회전 행렬 간의 Geodesic 거리 기반 손실 함수"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred_6d, gt_6d):
        """
        Args:
            pred_6d: (N, 6) 예측 6D rotation
            gt_6d: (N, 6) GT 6D rotation
        
        Returns:
            loss: 평균 geodesic 거리
        """
        # 6D → 회전 행렬
        R_pred = rotation_6d_to_matrix(pred_6d)
        R_gt = rotation_6d_to_matrix(gt_6d)
        
        # R_pred^T @ R_gt
        R_diff = torch.bmm(R_pred.transpose(1, 2), R_gt)
        
        # trace 계산
        trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
        
        # arccos((trace - 1) / 2) - 수치 안정화
        cos_theta = torch.clamp((trace - 1) / 2, -0.999, 0.999)
        theta = torch.acos(cos_theta)
        
        return theta.mean()


class DeepIMLossWithGeodesic(nn.Module):
    """Geodesic Loss를 사용하는 DeepIM 손실 함수"""
    
    def __init__(self, lambda_pos=1.0, lambda_rot=0.5, use_geodesic=True):
        super().__init__()
        self.lambda_pos = lambda_pos
        self.lambda_rot = lambda_rot
        self.use_geodesic = use_geodesic
        
        if use_geodesic:
            self.geodesic_loss = GeodesicLoss()
    
    def forward(self, pred_pos, pred_rot, gt_pos, gt_rot):
        # Position Loss
        pos_loss = F.smooth_l1_loss(pred_pos, gt_pos)
        
        # Rotation Loss
        if self.use_geodesic:
            rot_loss = self.geodesic_loss(pred_rot, gt_rot)
        else:
            rot_loss = F.smooth_l1_loss(pred_rot, gt_rot)
        
        total_loss = self.lambda_pos * pos_loss + self.lambda_rot * rot_loss
        
        loss_dict = {
            'total': total_loss.item(),
            'position': pos_loss.item(),
            'rotation': rot_loss.item(),
        }
        
        return total_loss, loss_dict


# 유틸리티 함수
def compute_pose_error(pred_pos, pred_rot, gt_pos, gt_rot):
    """포즈 오차 계산
    
    Args:
        pred_pos: (N, 3) 예측 위치 (meters)
        pred_rot: (N, 6) 예측 회전 (6D)
        gt_pos: (N, 3) GT 위치
        gt_rot: (N, 6) GT 회전
    
    Returns:
        pos_error_mm: (N,) 위치 오차 (mm)
        rot_error_deg: (N,) 회전 오차 (degrees)
    """
    # Position error (mm)
    pos_error = torch.sqrt(((pred_pos - gt_pos) ** 2).sum(dim=1)) * 1000
    
    # Rotation error (degrees)
    R_pred = rotation_6d_to_matrix(pred_rot)
    R_gt = rotation_6d_to_matrix(gt_rot)
    
    R_diff = torch.bmm(R_pred.transpose(1, 2), R_gt)
    trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
    cos_theta = torch.clamp((trace - 1) / 2, -1, 1)
    theta = torch.acos(cos_theta)
    rot_error = torch.rad2deg(theta)
    
    return pos_error, rot_error

