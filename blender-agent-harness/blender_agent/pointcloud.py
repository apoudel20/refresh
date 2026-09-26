"""
Zero-shot image-to-point-cloud.

Backends:
  depthpro_replicate  — Apple DepthPro via Replicate API
  zoedepth_replicate  — ZoeDepth via Replicate API
  local_depth_png     — pre-computed depth PNG already on disk
  multi_view          — visual hull from N silhouette images at known Y-rotation angles
                        (no depth map needed — carves a voxel grid using projections only)
"""

import base64
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import httpx
import numpy as np
from . import secrets as _secrets


DepthBackend = Literal["depthpro_replicate", "zoedepth_replicate", "multi_view"]


@dataclass
class PointCloudConfig:
    backend: DepthBackend = "depthpro_replicate"
    replicate_key: str = field(default_factory=lambda: __import__("os").environ.get("REPLICATE_API_TOKEN", ""))
    # Filters applied before writing PLY
    max_depth: float = 10.0
    depth_scale: float = 1.0
    point_density: int = 131072

    # multi_view backend — image paths and their Y-rotation angles in degrees
    multi_view_image_paths: list[str] = field(default_factory=list)
    multi_view_angles_deg: list[float] = field(default_factory=list)
    multi_view_resolution: int = 128        # voxel grid side length
    multi_view_bg_tolerance: int = 30       # pixel distance from corner colour → background


class ImageToPointCloud:
    def __init__(self, config: PointCloudConfig | None = None):
        _secrets.load()
        self.cfg = config or PointCloudConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def convert(self, image_path: str, output_ply_path: str) -> str:
        """image → PLY point cloud. Returns the saved PLY path."""
        output_ply_path = Path(output_ply_path)
        output_ply_path.parent.mkdir(parents=True, exist_ok=True)

        if self.cfg.backend == "multi_view":
            points, colors = self._multi_view_visual_hull(
                self.cfg.multi_view_image_paths,
                self.cfg.multi_view_angles_deg,
                self.cfg.multi_view_resolution,
            )
        else:
            image_path = Path(image_path)
            image_bytes = image_path.read_bytes()
            rgb = self._load_rgb(image_bytes)

            if self.cfg.backend == "depthpro_replicate":
                depth = self._depthpro_replicate(image_bytes)
            elif self.cfg.backend == "zoedepth_replicate":
                depth = self._zoedepth_replicate(image_bytes)
            else:
                raise ValueError(f"Unknown depth backend: {self.cfg.backend}")

            points, colors = self._depth_to_pointcloud(depth, rgb)

        self._write_ply(output_ply_path, points, colors)
        return str(output_ply_path)

    # ------------------------------------------------------------------
    # Multi-view visual hull
    # ------------------------------------------------------------------

    def _multi_view_visual_hull(
        self,
        image_paths: list[str],
        angles_deg: list[float],
        resolution: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Visual hull reconstruction from orthographic silhouettes.

        For each view at Y-rotation angle θ:
          u = x·cos(θ) + z·sin(θ)    (horizontal image axis)
          v = -y                       (vertical, flipped)

        Voxels whose projected pixel falls outside any silhouette are carved away.
        Surface voxels (occupied with at least one unoccupied neighbour) become points.
        Colors are sampled from the nearest front-facing view.
        """
        from PIL import Image
        from scipy.ndimage import binary_erosion

        masks: list[np.ndarray] = []
        color_imgs: list[np.ndarray] = []
        angles_rad = [np.radians(a) for a in angles_deg]

        for path in image_paths:
            img = Image.open(path).convert("RGB").resize((resolution * 2, resolution * 2))
            arr = np.array(img, dtype=np.float32)
            # Estimate background from corner pixels
            corners = np.stack([arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]])
            bg = np.median(corners, axis=0)
            diff = np.abs(arr - bg).max(axis=2)
            masks.append(diff > self.cfg.multi_view_bg_tolerance)
            color_imgs.append(arr.astype(np.uint8))

        H, W = masks[0].shape
        N = resolution

        # Voxel grid in world coords [-1, 1]³  (X=right, Y=up, Z=forward)
        lin = np.linspace(-1.0, 1.0, N, dtype=np.float32)
        # gx[i,j,k] = X coord of voxel (i,j,k), etc.
        gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")

        occupied = np.ones((N, N, N), dtype=bool)

        for mask, theta in zip(masks, angles_rad):
            # Orthographic project onto this camera's image plane
            u = gx * np.cos(theta) + gz * np.sin(theta)  # [-1, 1]
            v = -gy                                         # [-1, 1]

            px = np.round((u + 1.0) / 2.0 * (W - 1)).astype(np.int32).clip(0, W - 1)
            py = np.round((v + 1.0) / 2.0 * (H - 1)).astype(np.int32).clip(0, H - 1)

            # Carve voxels whose projection lands outside the silhouette
            occupied &= mask[py, px]

        # Keep only surface voxels
        surface = occupied & ~binary_erosion(occupied)
        if not surface.any():
            surface = occupied  # fallback: keep everything

        ix, iy, iz = np.where(surface)
        pts_x = lin[ix]
        pts_y = lin[iy]
        pts_z = lin[iz]
        points = np.stack([pts_x, pts_y, pts_z], axis=1)

        # Color: blend contributions from each view, weighted by |cos(θ)| (face-on views count more)
        color_acc = np.zeros((len(points), 3), dtype=np.float32)
        weight_acc = np.zeros(len(points), dtype=np.float32)

        for img_arr, theta in zip(color_imgs, angles_rad):
            u = pts_x * np.cos(theta) + pts_z * np.sin(theta)
            v = -pts_y
            px = np.round((u + 1.0) / 2.0 * (W - 1)).astype(np.int32).clip(0, W - 1)
            py = np.round((v + 1.0) / 2.0 * (H - 1)).astype(np.int32).clip(0, H - 1)
            w = abs(np.cos(theta))
            color_acc += img_arr[py, px].astype(np.float32) * w
            weight_acc += w

        colors = (color_acc / weight_acc[:, None]).clip(0, 255).astype(np.float32) / 255.0

        # Downsample
        if len(points) > self.cfg.point_density:
            idx = np.random.default_rng().choice(len(points), self.cfg.point_density, replace=False)
            points, colors = points[idx], colors[idx]

        return points.astype(np.float32), colors

    # ------------------------------------------------------------------
    # Depth estimation backends
    # ------------------------------------------------------------------

    def _depthpro_replicate(self, image_bytes: bytes) -> np.ndarray:
        b64 = f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode()}"
        return self._replicate_run("apple/depth-pro:a2220d6e", {"image": b64})

    def _zoedepth_replicate(self, image_bytes: bytes) -> np.ndarray:
        b64 = f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode()}"
        return self._replicate_run("cjwbw/zoedepth:edb5c0ca", {"image": b64})

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
                    return self._decode_depth_image(client.get(depth_url).content)
                if result["status"] == "failed":
                    raise RuntimeError(f"Replicate depth prediction failed: {result.get('error')}")
        raise TimeoutError("Replicate depth prediction timed out")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_rgb(image_bytes: bytes) -> np.ndarray:
        import io
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return np.array(img, dtype=np.float32) / 255.0

    @staticmethod
    def _decode_depth_image(depth_bytes: bytes) -> np.ndarray:
        import io
        from PIL import Image
        img = Image.open(io.BytesIO(depth_bytes))
        arr = np.array(img, dtype=np.float32)
        if arr.max() > 1.0:
            arr = arr / arr.max()
        return arr

    def _depth_to_pointcloud(self, depth: np.ndarray, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from PIL import Image
        h, w = depth.shape[:2]
        if rgb.shape[:2] != (h, w):
            rgb_img = Image.fromarray((rgb * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
            rgb = np.array(rgb_img, dtype=np.float32) / 255.0

        xs = (np.arange(w) - w / 2) / w
        ys = (np.arange(h) - h / 2) / h
        gx, gy = np.meshgrid(xs, ys)

        d = depth * self.cfg.depth_scale
        mask = d < self.cfg.max_depth
        z = d[mask]
        x = gx[mask] * z
        y = -gy[mask] * z

        points = np.stack([x, y, z], axis=1).astype(np.float32)
        colors = rgb.reshape(-1, 3)[mask.ravel()].astype(np.float32)

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
        vertex_bytes = bytearray()
        for i in range(n):
            vertex_bytes += struct.pack("<fff", *points[i])
            vertex_bytes += struct.pack("BBB", *rgb_uint8[i])
        path.write_bytes(header + bytes(vertex_bytes))
