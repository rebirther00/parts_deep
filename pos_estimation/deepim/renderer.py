#!/usr/bin/env python3
"""
3D 모델 렌더러
- OBJ 파일 로드 및 렌더링
- RGB + Depth 이미지 생성
- trimesh + pyrender 기반 (PyTorch3D 대체)
"""

import os
import numpy as np
import torch

# pyrender는 off-screen 렌더링을 위해 EGL 백엔드 사용
os.environ['PYOPENGL_PLATFORM'] = 'egl'

try:
    import trimesh
    import pyrender
    RENDERER_AVAILABLE = True
except ImportError:
    RENDERER_AVAILABLE = False
    print("⚠️ trimesh/pyrender가 설치되어 있지 않습니다.")

# PyTorch3D 호환성을 위한 플래그 (레거시 코드 지원)
PYTORCH3D_AVAILABLE = RENDERER_AVAILABLE


class MeshRenderer:
    """trimesh + pyrender 기반 3D 메쉬 렌더러"""
    
    def __init__(self, assets_dir, class_names, image_size=224, device='cuda'):
        """
        Args:
            assets_dir: OBJ 파일이 있는 디렉토리
            class_names: 클래스 이름 리스트 (OBJ 파일명과 매칭)
            image_size: 출력 이미지 크기
            device: 'cuda' 또는 'cpu' (pyrender는 CPU 렌더링)
        """
        if not RENDERER_AVAILABLE:
            raise ImportError("trimesh/pyrender가 설치되어 있지 않습니다.")
        
        self.device = torch.device(device)
        self.image_size = image_size
        self.assets_dir = assets_dir
        self.class_names = class_names
        
        # 메쉬 로드
        self.meshes = {}
        self._load_meshes()
        
        # 카메라 설정 (데이터셋과 동일한 intrinsics)
        self.focal_length = 768.0
        self.principal_point = (512.0, 512.0)
        self.original_size = 1024
        
        # 스케일 조정
        self.scale = image_size / self.original_size
        self.fx = self.focal_length * self.scale
        self.fy = self.focal_length * self.scale
        self.cx = self.principal_point[0] * self.scale
        self.cy = self.principal_point[1] * self.scale
        
        # Off-screen 렌더러 생성
        self.renderer = pyrender.OffscreenRenderer(image_size, image_size)
    
    def _load_meshes(self):
        """OBJ 파일들 로드"""
        print(f"📦 3D 메쉬 로드 중...")
        
        for class_name in self.class_names:
            obj_path = os.path.join(self.assets_dir, f"{class_name}.obj")
            
            if not os.path.exists(obj_path):
                print(f"  ⚠️ {class_name}.obj 파일을 찾을 수 없습니다.")
                continue
            
            try:
                # trimesh로 OBJ 로드
                mesh = trimesh.load(obj_path, force='mesh')
                
                # 중심을 원점으로 이동
                mesh.vertices -= mesh.centroid
                
                # pyrender 메쉬로 변환
                material = pyrender.MetallicRoughnessMaterial(
                    baseColorFactor=[0.5, 0.5, 0.5, 1.0],
                    metallicFactor=0.2,
                    roughnessFactor=0.8
                )
                pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
                
                self.meshes[class_name] = {
                    'trimesh': mesh,
                    'pyrender': pyrender_mesh,
                    'centroid': mesh.centroid.copy(),
                }
                
                print(f"  ✅ {class_name}: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
                
            except Exception as e:
                print(f"  ❌ {class_name} 로드 실패: {e}")
        
        print(f"📦 총 {len(self.meshes)}개 메쉬 로드 완료")
    
    def _create_camera_matrix(self):
        """카메라 intrinsic 행렬 생성"""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])
    
    def render(self, class_name, position, rotation_matrix):
        """주어진 포즈로 3D 모델 렌더링
        
        Args:
            class_name: 렌더링할 클래스 이름
            position: (3,) 객체 위치 [x, y, z] (meters)
            rotation_matrix: (3, 3) 회전 행렬
        
        Returns:
            rgb: (3, H, W) 렌더링된 RGB 이미지 (torch.Tensor)
            depth: (1, H, W) 렌더링된 Depth 이미지 (torch.Tensor)
        """
        if class_name not in self.meshes:
            # 메쉬가 없으면 검은 이미지 반환
            rgb = torch.zeros(3, self.image_size, self.image_size, device=self.device)
            depth = torch.zeros(1, self.image_size, self.image_size, device=self.device)
            return rgb, depth
        
        # 텐서를 numpy로 변환
        if isinstance(position, torch.Tensor):
            position = position.detach().cpu().numpy()
        if isinstance(rotation_matrix, torch.Tensor):
            rotation_matrix = rotation_matrix.detach().cpu().numpy()
        
        # pyrender 씬 생성
        scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])
        
        # 메쉬 추가 (객체 포즈 적용)
        mesh_data = self.meshes[class_name]
        
        # 변환 행렬 생성 (4x4)
        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = rotation_matrix
        pose_matrix[:3, 3] = position
        
        scene.add(mesh_data['pyrender'], pose=pose_matrix)
        
        # 조명 추가
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, 0, -3]
        scene.add(light, pose=light_pose)
        
        # 카메라 추가 (원점에 위치, -Z 방향 바라봄)
        camera = pyrender.IntrinsicsCamera(
            fx=self.fx, fy=self.fy,
            cx=self.cx, cy=self.cy,
            znear=0.01, zfar=100.0
        )
        
        # 카메라 포즈: OpenGL 좌표계 → OpenCV 좌표계 변환
        # OpenGL: Y up, -Z forward
        # OpenCV: Y down, +Z forward
        camera_pose = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        scene.add(camera, pose=camera_pose)
        
        # 렌더링
        try:
            color, depth_img = self.renderer.render(scene)
        except Exception as e:
            print(f"렌더링 실패: {e}")
            rgb = torch.zeros(3, self.image_size, self.image_size, device=self.device)
            depth = torch.zeros(1, self.image_size, self.image_size, device=self.device)
            return rgb, depth
        
        # numpy → torch 변환
        # RGB: (H, W, 3) → (3, H, W), 정규화 0-1
        rgb = torch.from_numpy(color.astype(np.float32) / 255.0).permute(2, 0, 1)
        
        # Depth: (H, W) → (1, H, W)
        depth = torch.from_numpy(depth_img.astype(np.float32)).unsqueeze(0)
        
        # GPU로 이동
        rgb = rgb.to(self.device)
        depth = depth.to(self.device)
        
        return rgb, depth
    
    def render_batch(self, class_names_batch, positions_batch, rotation_matrices_batch):
        """배치 렌더링
        
        Args:
            class_names_batch: 클래스 이름 리스트 (batch_size,)
            positions_batch: (batch_size, 3) 위치들
            rotation_matrices_batch: (batch_size, 3, 3) 회전 행렬들
        
        Returns:
            rgbs: (batch_size, 3, H, W)
            depths: (batch_size, 1, H, W)
        """
        batch_size = len(class_names_batch)
        rgbs = []
        depths = []
        
        for i in range(batch_size):
            rgb, depth = self.render(
                class_names_batch[i],
                positions_batch[i],
                rotation_matrices_batch[i]
            )
            rgbs.append(rgb)
            depths.append(depth)
        
        return torch.stack(rgbs), torch.stack(depths)
    
    def __del__(self):
        """렌더러 정리"""
        if hasattr(self, 'renderer'):
            self.renderer.delete()


# 테스트 코드
if __name__ == "__main__":
    if RENDERER_AVAILABLE:
        print("✅ Renderer 사용 가능")
        
        # 테스트
        assets_dir = "/home/kotest/issac-minsu/assets"
        class_names = ["arm_link_25_nomat", "arm_link_30_notmat", 
                       "boom_link_25_nomat", "boom_link_30_notmat"]
        
        try:
            renderer = MeshRenderer(assets_dir, class_names, device='cuda')
            
            # 테스트 렌더링
            position = np.array([0.0, 2.0, 4.0])
            rotation = np.eye(3)
            
            rgb, depth = renderer.render("arm_link_25_nomat", position, rotation)
            print(f"RGB shape: {rgb.shape}, Depth shape: {depth.shape}")
            print(f"RGB range: [{rgb.min():.3f}, {rgb.max():.3f}]")
            print(f"Depth range: [{depth.min():.3f}, {depth.max():.3f}]")
            
            # 이미지 저장 테스트
            from PIL import Image
            rgb_np = (rgb.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            Image.fromarray(rgb_np).save("/tmp/test_render.png")
            print("✅ 테스트 렌더링 저장: /tmp/test_render.png")
            
        except Exception as e:
            print(f"❌ 렌더러 테스트 실패: {e}")
    else:
        print("❌ trimesh/pyrender가 설치되어 있지 않습니다.")
