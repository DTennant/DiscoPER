"""
Visual feature extractor: converts images to numeric features for the discovery pipeline.

Three extraction levels:
1. Classical CV features — color histograms, edge density, texture metrics, spatial statistics
2. Semantic features — detected object counts, spatial layout metrics
3. Deep features — CLIP/ResNet embeddings (optional, requires torch)

The classical features are designed to approximately (but noisily) recover the
causal variables encoded into the scene, making discovery non-trivial.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image


def _rgb_to_hsv_array(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB array (H, W, 3) uint8 to HSV array (H, W, 3) float32.

    H: [0, 360), S: [0, 1], V: [0, 1]
    """
    rgb_f = rgb.astype(np.float32) / 255.0
    r, g, b = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    # Hue
    h = np.zeros_like(delta)
    mask_r = (cmax == r) & (delta > 0)
    mask_g = (cmax == g) & (delta > 0)
    mask_b = (cmax == b) & (delta > 0)
    h[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / delta[mask_r]) + 360) % 360
    h[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 120) % 360
    h[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 240) % 360

    # Saturation
    s = np.where(cmax > 0, delta / cmax, 0)

    # Value
    v = cmax

    return np.stack([h, s, v], axis=-1)


def _sobel_edges(gray: np.ndarray) -> np.ndarray:
    """Apply simple Sobel edge detection. Returns edge magnitude array."""
    # Horizontal and vertical kernels via convolution
    # Pad to avoid boundary issues
    padded = np.pad(gray, 1, mode="edge")
    gx = (
        -padded[:-2, :-2]
        + padded[:-2, 2:]
        - 2 * padded[1:-1, :-2]
        + 2 * padded[1:-1, 2:]
        - padded[2:, :-2]
        + padded[2:, 2:]
    )
    gy = (
        -padded[:-2, :-2]
        - 2 * padded[:-2, 1:-1]
        - padded[:-2, 2:]
        + padded[2:, :-2]
        + 2 * padded[2:, 1:-1]
        + padded[2:, 2:]
    )
    return np.sqrt(gx**2 + gy**2)


def _label_connected_components(binary: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Label connected components in a binary image.

    Uses scipy.ndimage.label if available, otherwise a simple flood-fill.

    Returns:
        (labeled_array, num_components)
    """
    try:
        from scipy.ndimage import label as scipy_label

        return scipy_label(binary)
    except ImportError:
        pass

    # Fallback: simple row-by-row union-find style labeling
    labels = np.zeros_like(binary, dtype=np.int32)
    current_label = 0
    h, w = binary.shape

    for y in range(h):
        for x in range(w):
            if binary[y, x] and labels[y, x] == 0:
                current_label += 1
                # BFS flood fill
                stack = [(y, x)]
                while stack:
                    cy, cx = stack.pop()
                    if cy < 0 or cy >= h or cx < 0 or cx >= w:
                        continue
                    if labels[cy, cx] != 0 or not binary[cy, cx]:
                        continue
                    labels[cy, cx] = current_label
                    stack.extend(
                        [(cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)]
                    )

    return labels, current_label


class VisualFeatureExtractor:
    """Extracts numeric features from ecology scene images."""

    def __init__(
        self,
        feature_set: str = "classical",
        include_spatial: bool = True,
        include_color: bool = True,
        include_texture: bool = True,
        include_object_detection: bool = True,
    ):
        """
        Initialize feature extractor.

        Args:
            feature_set: "classical", "deep", or "both"
            include_spatial: Include spatial layout features
            include_color: Include color-based features
            include_texture: Include texture features
            include_object_detection: Include blob-based object detection
        """
        self.feature_set = feature_set
        self.include_spatial = include_spatial
        self.include_color = include_color
        self.include_texture = include_texture
        self.include_object_detection = include_object_detection

    def extract(self, image: Image.Image) -> dict[str, float]:
        """Extract all configured features from a single image."""
        img_rgb = np.array(image.convert("RGB"))
        h, w = img_rgb.shape[:2]

        features: dict[str, float] = {}

        if self.feature_set in ("classical", "both"):
            if self.include_color:
                features.update(self._extract_color_features(img_rgb))
            if self.include_spatial:
                features.update(self._extract_spatial_features(img_rgb))
            if self.include_texture:
                features.update(self._extract_texture_features(img_rgb))
            if self.include_object_detection:
                features.update(self._extract_object_features(img_rgb))

        if self.feature_set in ("deep", "both"):
            deep = self._extract_deep_features(image)
            if deep is not None:
                features.update(deep)

        return features

    def extract_batch(self, images: list[Image.Image]) -> pd.DataFrame:
        """Extract features from multiple images, return as DataFrame."""
        records = []
        for img in images:
            features = self.extract(img)
            records.append(features)
        return pd.DataFrame(records)

    def get_feature_names(self) -> list[str]:
        """Return list of feature names that will be extracted."""
        names = []
        if self.feature_set in ("classical", "both"):
            if self.include_color:
                names.extend(
                    [
                        "color_warmth",
                        "green_fraction",
                        "blue_fraction",
                        "white_fraction",
                        "sky_grey_fraction",
                        "avg_brightness",
                        "color_diversity",
                    ]
                )
            if self.include_spatial:
                names.extend(
                    [
                        "horizon_height",
                        "terrain_roughness",
                        "upper_complexity",
                    ]
                )
            if self.include_texture:
                names.extend(["texture_energy", "texture_uniformity"])
            if self.include_object_detection:
                names.extend(
                    [
                        "dark_blob_count",
                        "large_blob_count",
                        "small_blob_count",
                        "sky_dot_count",
                        "tree_count",
                        "vegetation_coverage",
                        "water_body_area",
                    ]
                )
        return names

    # =========================================================================
    # COLOR FEATURES
    # =========================================================================

    def _extract_color_features(self, img_rgb: np.ndarray) -> dict[str, float]:
        """Extract color-based features that approximate causal variables."""
        h, w = img_rgb.shape[:2]
        hsv = _rgb_to_hsv_array(img_rgb)
        hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]

        features: dict[str, float] = {}

        # Color warmth: fraction of warm hues (yellow 30-70, orange 10-30, red 0-10 & 340-360)
        warm_mask = (((hue >= 0) & (hue <= 70)) | ((hue >= 340) & (hue <= 360))) & (
            sat > 0.15
        )
        cool_mask = ((hue >= 180) & (hue <= 260)) & (sat > 0.15)
        n_warm = warm_mask.sum()
        n_cool = cool_mask.sum()
        total_chromatic = max(n_warm + n_cool, 1)
        features["color_warmth"] = float(n_warm / total_chromatic)

        # Green fraction: vegetation proxy
        green_mask = ((hue >= 70) & (hue <= 170)) & (sat > 0.2) & (val > 0.15)
        features["green_fraction"] = float(green_mask.sum() / (h * w))

        # Blue fraction: water proxy
        blue_mask = ((hue >= 190) & (hue <= 250)) & (sat > 0.2) & (val > 0.2)
        features["blue_fraction"] = float(blue_mask.sum() / (h * w))

        # White fraction: snow proxy (high value, low saturation)
        white_mask = (val > 0.85) & (sat < 0.15)
        features["white_fraction"] = float(white_mask.sum() / (h * w))

        # Sky grey fraction: cloud proxy (upper half, grey-ish)
        upper_half = img_rgb[: h // 2]
        upper_gray = np.abs(
            upper_half.astype(np.float32).std(axis=-1)
        )  # Low std = grey
        grey_mask = (upper_gray < 15) & (upper_half.mean(axis=-1) > 100)
        features["sky_grey_fraction"] = float(
            grey_mask.sum() / max(upper_half.shape[0] * upper_half.shape[1], 1)
        )

        # Average brightness
        features["avg_brightness"] = float(val.mean())

        # Color diversity: Shannon entropy of hue histogram
        hue_hist, _ = np.histogram(hue[sat > 0.1], bins=36, range=(0, 360))
        hue_probs = hue_hist / max(hue_hist.sum(), 1)
        hue_probs = hue_probs[hue_probs > 0]
        features["color_diversity"] = float(
            -np.sum(hue_probs * np.log2(hue_probs + 1e-10))
        )

        return features

    # =========================================================================
    # SPATIAL FEATURES
    # =========================================================================

    def _extract_spatial_features(self, img_rgb: np.ndarray) -> dict[str, float]:
        """Extract spatial layout features approximating elevation/terrain."""
        h, w = img_rgb.shape[:2]
        gray = img_rgb.astype(np.float32).mean(axis=-1)

        features: dict[str, float] = {}

        # Horizon height: find transition from terrain to sky
        # Use vertical gradient — big change in green channel signals horizon
        green_col = img_rgb[:, :, 1].astype(np.float32)
        row_means = green_col.mean(axis=1)

        # Find largest gradient change
        diffs = np.abs(np.diff(row_means))
        if len(diffs) > 10:
            # Smooth diffs and find peak in middle portion
            kernel = np.ones(5) / 5
            smooth_diffs = np.convolve(diffs, kernel, mode="same")
            # Search in middle 80% of image
            search_start = int(h * 0.1)
            search_end = int(h * 0.9)
            search_region = smooth_diffs[search_start:search_end]
            if len(search_region) > 0:
                horizon_row = search_start + np.argmax(search_region)
                features["horizon_height"] = float(1.0 - horizon_row / h)
            else:
                features["horizon_height"] = 0.5
        else:
            features["horizon_height"] = 0.5

        # Terrain roughness: edge density in lower half
        lower_half = gray[h // 2 :]
        edges = _sobel_edges(lower_half)
        features["terrain_roughness"] = float(edges.mean() / 255.0)

        # Upper complexity: edge density in upper quarter
        upper_quarter = gray[: h // 4]
        upper_edges = _sobel_edges(upper_quarter)
        features["upper_complexity"] = float(upper_edges.mean() / 255.0)

        return features

    # =========================================================================
    # TEXTURE FEATURES
    # =========================================================================

    def _extract_texture_features(self, img_rgb: np.ndarray) -> dict[str, float]:
        """Extract texture features approximating precipitation/insects."""
        gray = img_rgb.astype(np.float32).mean(axis=-1)

        features: dict[str, float] = {}

        # Texture energy: average local contrast (Laplacian approximation)
        padded = np.pad(gray, 1, mode="edge")
        laplacian = (
            padded[:-2, 1:-1]
            + padded[2:, 1:-1]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
            - 4 * padded[1:-1, 1:-1]
        )
        features["texture_energy"] = float(np.abs(laplacian).mean() / 255.0)

        # Texture uniformity: inverse variance of local contrast
        local_var = np.abs(laplacian).var()
        features["texture_uniformity"] = float(1.0 / (1.0 + local_var / 1000.0))

        return features

    # =========================================================================
    # OBJECT DETECTION FEATURES
    # =========================================================================

    def _extract_object_features(self, img_rgb: np.ndarray) -> dict[str, float]:
        """Extract object count features approximating animal/vegetation counts."""
        h, w = img_rgb.shape[:2]
        hsv = _rgb_to_hsv_array(img_rgb)
        hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]

        features: dict[str, float] = {}

        # Dark blob detection (animals) — dark, low-saturation blobs in terrain zone
        terrain_zone = img_rgb[h // 3 :, :]
        terrain_gray = terrain_zone.astype(np.float32).mean(axis=-1)
        # Animals are dark, smallish blobs
        dark_mask = terrain_gray < 80
        labeled, n_blobs = _label_connected_components(dark_mask)

        # Count by size
        small_count = 0
        large_count = 0
        min_area = 4  # Minimum pixels to be a blob
        large_threshold = 40  # Pixels threshold for "large"

        if n_blobs > 0:
            for label_id in range(1, n_blobs + 1):
                area = (labeled == label_id).sum()
                if area >= min_area:
                    if area >= large_threshold:
                        large_count += 1
                    else:
                        small_count += 1

        features["dark_blob_count"] = float(small_count + large_count)
        features["small_blob_count"] = float(small_count)
        features["large_blob_count"] = float(large_count)

        # Sky dot count (birds) — small dark blobs in upper quarter
        sky_zone = img_rgb[: h // 4, :]
        sky_gray = sky_zone.astype(np.float32).mean(axis=-1)
        sky_dark = sky_gray < 100
        sky_labeled, sky_n = _label_connected_components(sky_dark)
        bird_count = 0
        if sky_n > 0:
            for label_id in range(1, sky_n + 1):
                area = (sky_labeled == label_id).sum()
                if 2 <= area <= 30:  # Bird-sized
                    bird_count += 1
        features["sky_dot_count"] = float(bird_count)

        # Tree count: green blobs in middle terrain zone
        mid_zone_hsv = hsv[h // 4 : h * 3 // 4, :]
        green_blob_mask = (
            (mid_zone_hsv[..., 0] >= 70)
            & (mid_zone_hsv[..., 0] <= 170)
            & (mid_zone_hsv[..., 1] > 0.25)
            & (mid_zone_hsv[..., 2] > 0.1)
        )
        tree_labeled, tree_n = _label_connected_components(green_blob_mask)
        tree_count = 0
        if tree_n > 0:
            for label_id in range(1, min(tree_n + 1, 200)):
                area = (tree_labeled == label_id).sum()
                if area >= 10:  # Minimum tree size
                    tree_count += 1
        features["tree_count"] = float(tree_count)

        # Vegetation coverage: green fraction in terrain zone only
        terrain_hsv = hsv[h // 3 :, :]
        veg_mask = (
            (terrain_hsv[..., 0] >= 60)
            & (terrain_hsv[..., 0] <= 175)
            & (terrain_hsv[..., 1] > 0.15)
            & (terrain_hsv[..., 2] > 0.1)
        )
        terrain_pixels = terrain_hsv.shape[0] * terrain_hsv.shape[1]
        features["vegetation_coverage"] = float(veg_mask.sum() / max(terrain_pixels, 1))

        # Water body area: blue region in lower half
        lower_hsv = hsv[h // 2 :, :]
        water_mask = (
            (lower_hsv[..., 0] >= 180)
            & (lower_hsv[..., 0] <= 250)
            & (lower_hsv[..., 1] > 0.2)
            & (lower_hsv[..., 2] > 0.2)
        )
        lower_pixels = lower_hsv.shape[0] * lower_hsv.shape[1]
        features["water_body_area"] = float(water_mask.sum() / max(lower_pixels, 1))

        return features

    # =========================================================================
    # DEEP FEATURES (OPTIONAL)
    # =========================================================================

    def _extract_deep_features(self, image: Image.Image) -> Optional[dict[str, float]]:
        """
        Extract deep features using CLIP or ResNet.

        Returns None if torch/torchvision not available.
        """
        try:
            import torch
            from torchvision import transforms, models
        except ImportError:
            warnings.warn(
                "torch/torchvision not available — skipping deep features. "
                "Install with: pip install torch torchvision",
                stacklevel=2,
            )
            return None

        features: dict[str, float] = {}

        # ResNet50 features
        try:
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            model.eval()

            # Remove final FC layer to get embeddings
            modules = list(model.children())[:-1]
            feature_model = torch.nn.Sequential(*modules)

            preprocess = transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )

            img_tensor = preprocess(image.convert("RGB")).unsqueeze(0)

            with torch.no_grad():
                embedding = feature_model(img_tensor).squeeze().numpy()

            # Store first 64 components as named features for the pipeline
            for i in range(min(64, len(embedding))):
                features[f"resnet_feat_{i:03d}"] = float(embedding[i])

        except Exception as e:
            warnings.warn(f"ResNet feature extraction failed: {e}", stacklevel=2)

        return features if features else None

    def extract_embeddings(
        self, images: list[Image.Image], embedding_type: str = "resnet"
    ) -> Optional[np.ndarray]:
        """
        Extract dense embedding vectors for cluster_and_enrich.

        Args:
            images: List of PIL images.
            embedding_type: "resnet" or "clip"

        Returns:
            Array of shape (n_images, embedding_dim) or None if unavailable.
        """
        try:
            import torch
            from torchvision import transforms, models
        except ImportError:
            warnings.warn(
                "torch/torchvision not available for embeddings",
                stacklevel=2,
            )
            return None

        if embedding_type == "resnet":
            return self._extract_resnet_embeddings(images)

        warnings.warn(f"Unknown embedding type: {embedding_type}", stacklevel=2)
        return None

    def _extract_resnet_embeddings(
        self, images: list[Image.Image]
    ) -> Optional[np.ndarray]:
        """Extract ResNet50 pool5 embeddings for all images."""
        try:
            import torch
            from torchvision import transforms, models

            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            model.eval()
            modules = list(model.children())[:-1]
            feature_model = torch.nn.Sequential(*modules)

            preprocess = transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )

            embeddings = []
            for img in images:
                tensor = preprocess(img.convert("RGB")).unsqueeze(0)
                with torch.no_grad():
                    emb = feature_model(tensor).squeeze().numpy()
                embeddings.append(emb)

            return np.array(embeddings)

        except Exception as e:
            warnings.warn(f"ResNet embedding extraction failed: {e}", stacklevel=2)
            return None
