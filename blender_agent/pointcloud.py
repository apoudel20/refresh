"""
Zero-shot image-to-point-cloud using monocular depth estimation.
Supports DepthPro (Apple / Replicate), ZoeDepth (Replicate), and a local
open3d-based fallback that uses any depth PNG already on disk.
"""

import base64
import json
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import httpx
import numpy as np
from . import secrets as _secrets


DepthBackend = Literal["depthpro_replicate", "zoedepth_replicate", "local_depth_png"]


@dataclass
class PointCloudConfig:
    backend: DepthBackend = "depthpro_replicate"
    replicate_key: str = field(default_factory=lambda: __import__("os").environ.get("REPLICATE_API_TOKEN", ""))
    # Filters applied before writing PLY
    max_depth: float = 10.0          # clip far points
    depth_scale: float = 1.0         # multiply raw depth values
    point_density: int = 131072      # target point count (downsample if exceeded)


class ImageToPointCloud:
    """
    Converts a single RGB image to a PLY point cloud via zero-shot depth estimation.

    Usage:
        itp = ImageToPointCloud()
        ply_path = itp.convert("photo.jpg", "output/mesh.ply")
    """

    def __init__(self, config: PointCloudConfig | None = None):
        _secrets.load()
        self.cfg = config or PointCloudConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def convert(self, image_path: str, output_ply_path: str) -> str:
        """
        Full pipeline: image → depth map → point cloud PLY.
        Returns the path to the saved PLY file.
        """
        image_path = Path(image_path)
        output_ply_path = Path(output_ply_path)
        output_ply_path.parent.mkdir(parents=True, exist_ok=True)

        image_bytes = image_path.read_bytes()
        rgb = self._load_rgb(image_bytes)

        if self.cfg.backend == "depthpro_replicate":
            depth = self._depthpro_replicate(image_bytes)
        elif self.cfg.backend == "zoedepth_replicate":
            depth = self._zoedepth_replicate(image_bytes)
        elif self.cfg.backend == "local_depth_png":
            depth_path = image_path.with_suffix(".depth.png")
            depth = self._load_depth_png(depth_path)
        else:
            raise ValueError(f"Unknown depth backend: {self.cfg.backend}")

        points, colors = self._depth_to_pointcloud(depth, rgb)
        self._write_ply(output_ply_path, points, colors)
        return str(output_ply_path)

    def convert_depth_png(self, depth_png_path: str, rgb_image_path: str, output_ply_path: str) -> str:
        """Build a point cloud from a pre-computed depth PNG + matching RGB image."""
        depth = self._load_depth_png(Path(depth_png_path))
        rgb = self._load_rgb(Path(rgb_image_path).read_bytes())
        points, colors = self._depth_to_pointcloud(depth, rgb)
        output_ply_path = Path(output_ply_path)
        output_ply_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_ply(output_ply_path, points, colors)
        return str(output_ply_path)

    # ------------------------------------------------------------------
    # Depth estimation backends
    # ------------------------------------------------------------------

    def _depthpro_replicate(self, image_bytes: bytes) -> np.ndarray:
        b64 = f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode()}"
        return self._replicate_run(
            "apple/depth-pro:a2220d6e",
            {"image": b64},
        )

    def _zoedepth_replicate(self, image_bytes: bytes) -> np.ndarray:
        b64 = f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode()}"
        return self._replicate_run(
            "cjwbw/zoedepth:edb5c0ca",
            {"image": b64},
        )

    def _replicate_run(self, model_version: str, input_payload: dict) -> np.ndarray:
        headers = {
            "Authorization": f"Bearer {self.cfg.replicate_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=180) as client:
            resp = client.post(
                "https://api.replicate.com/v1/predictions",
                headers=headers,
                json={"version": model_version.split(":")[1] if ":" in model_version else model_version,
                      "input": input_payload},
            )
            resp.raise_for_status()
            poll_url = resp.json()["urls"]["get"]
            for _ in range(90):
                time.sleep(2)
                result = client.get(poll_url, headers=headers).json()
                if result["status"] == "succeeded":
                    depth_url = result["output"] if isinstance(result["output"], str) else result["output"]["depth"]
                    depth_bytes = client.get(depth_url).content
                    return self._decode_depth_image(depth_bytes)
                if result["status"] == "failed":
                    raise RuntimeError(f"Replicate depth prediction failed: {result.get('error')}")
        raise TimeoutError("Replicate depth prediction timed out")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_rgb(image_bytes: bytes) -> np.ndarray:
        from PIL import Image
        img = Image.open(__import__("io").BytesIO(image_bytes)).convert("RGB")
        return np.array(img, dtype=np.float32) / 255.0

    @staticmethod
    def _load_depth_png(path: Path) -> np.ndarray:
        from PIL import Image
        img = Image.open(path).convert("I")   # 32-bit grayscale
        return np.array(img, dtype=np.float32) / 65535.0

    @staticmethod
    def _decode_depth_image(depth_bytes: bytes) -> np.ndarray:
        from PIL import Image
        img = Image.open(__import__("io").BytesIO(depth_bytes))
        arr = np.array(img, dtype=np.float32)
        if arr.max() > 1.0:
            arr = arr / arr.max()
        return arr

    def _depth_to_pointcloud(
        self, depth: np.ndarray, rgb: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        h, w = depth.shape[:2]

        # Resize rgb to match depth if needed
        if rgb.shape[:2] != (h, w):
            from PIL import Image
            rgb_img = Image.fromarray((rgb * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
            rgb = np.array(rgb_img, dtype=np.float32) / 255.0

        # Simple pinhole back-projection (unit focal length, centred principal point)
        xs = (np.arange(w) - w / 2) / w
        ys = (np.arange(h) - h / 2) / h
        gx, gy = np.meshgrid(xs, ys)

        d = depth * self.cfg.depth_scale
        mask = d < self.cfg.max_depth

        z = d[mask]
        x = gx[mask] * z
        y = -gy[mask] * z   # flip Y for right-handed coords

        points = np.stack([x, y, z], axis=1).astype(np.float32)
        colors = rgb.reshape(-1, 3)[mask.ravel()].astype(np.float32)

        # Downsample
        if len(points) > self.cfg.point_density:
            idx = np.random.choice(len(points), self.cfg.point_density, replace=False)
            points, colors = points[idx], colors[idx]

        return points, colors

    @staticmethod
    def _write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
        n = len(points)
        header = (
            f"ply\nformat binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            f"property float x\nproperty float y\nproperty float z\n"
            f"property uchar red\nproperty uchar green\nproperty uchar blue\n"
            f"end_header\n"
        ).encode()

        rgb_uint8 = (colors * 255).clip(0, 255).astype(np.uint8)
        # Pack each vertex as: float32 x3 + uint8 x3
        vertex_bytes = bytearray()
        for i in range(n):
            vertex_bytes += struct.pack("<fff", *points[i])
            vertex_bytes += struct.pack("BBB", *rgb_uint8[i])

        path.write_bytes(header + bytes(vertex_bytes))
