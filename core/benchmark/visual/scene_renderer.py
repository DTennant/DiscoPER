"""
Scene renderer: generates ecology-themed images from visual property values.

Architecture:
- Scenes are composed of layers: sky, terrain, water, vegetation, snow, animals
- Each layer is controlled by visual properties from the mapping
- Pseudo-Perlin noise (summed sinusoids) adds natural variation
- All rendering is deterministic given a seed

The renderer produces stylized but recognizable landscape scenes suitable
for benchmarking visual discovery systems.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# ---------------------------------------------------------------------------
# Polygon sprite data – normalized to a 0-100 coordinate grid.
# Each animal maps to a list of named parts; every part is a list of (x, y).
# ---------------------------------------------------------------------------
_ANIMAL_SPRITES: dict[str, list[list[tuple[float, float]]]] = {
    # --- Herbivores ---
    "deer": [
        # body – elongated oval
        [
            (25, 45),
            (35, 35),
            (55, 32),
            (75, 35),
            (80, 45),
            (75, 55),
            (55, 58),
            (35, 55),
        ],
        # head
        [(80, 35), (90, 28), (98, 32), (95, 40), (85, 42)],
        # leg front-left
        [(35, 55), (33, 80), (37, 80), (39, 55)],
        # leg front-right
        [(45, 55), (43, 82), (47, 82), (49, 55)],
        # leg back-left
        [(60, 58), (58, 82), (62, 82), (64, 58)],
        # leg back-right
        [(70, 55), (68, 80), (72, 80), (74, 55)],
    ],
    "rabbit": [
        # body – compact round
        [
            (30, 50),
            (40, 38),
            (60, 35),
            (72, 40),
            (75, 52),
            (70, 62),
            (50, 65),
            (35, 60),
        ],
        # head
        [(72, 38), (82, 30), (88, 35), (85, 45), (75, 45)],
        # ear left
        [(78, 30), (76, 8), (80, 6), (83, 28)],
        # ear right
        [(84, 28), (82, 10), (86, 8), (88, 26)],
        # tail
        [(28, 52), (22, 48), (20, 54), (25, 58)],
        # hind leg
        [(40, 62), (35, 78), (42, 78), (48, 62)],
    ],
    # --- Predators ---
    "wolf": [
        # body – elongated angular
        [
            (15, 45),
            (25, 34),
            (50, 30),
            (75, 33),
            (82, 42),
            (78, 52),
            (50, 55),
            (25, 52),
        ],
        # head – pointed snout
        [(82, 34), (95, 28), (100, 35), (96, 42), (85, 44)],
        # tail – bushy
        [(15, 42), (5, 30), (2, 35), (8, 45), (15, 48)],
        # leg front-left
        [(55, 55), (53, 80), (57, 80), (59, 55)],
        # leg front-right
        [(65, 52), (63, 78), (67, 78), (69, 52)],
        # leg back-left
        [(30, 52), (28, 78), (32, 78), (34, 52)],
        # leg back-right
        [(40, 55), (38, 80), (42, 80), (44, 55)],
    ],
    "bear": [
        # body – broad stocky
        [
            (18, 42),
            (28, 30),
            (55, 26),
            (78, 30),
            (85, 42),
            (82, 58),
            (55, 62),
            (28, 58),
        ],
        # head – round with snout bump
        [(82, 30), (92, 24), (98, 30), (96, 40), (88, 42), (84, 38)],
        # leg front-left
        [(55, 60), (52, 85), (58, 85), (61, 60)],
        # leg front-right
        [(68, 58), (65, 84), (71, 84), (74, 58)],
        # leg back-left
        [(30, 58), (27, 84), (33, 84), (36, 58)],
        # leg back-right
        [(42, 60), (39, 85), (45, 85), (48, 60)],
    ],
    # --- Birds (top-down, flying) ---
    "bird": [
        # body – small elongated
        [(42, 40), (48, 35), (58, 35), (62, 40), (58, 45), (48, 45)],
        # wing left – swept back
        [(45, 40), (20, 20), (15, 30), (40, 42)],
        # wing right – swept back
        [(45, 40), (20, 60), (15, 50), (40, 38)],
        # tail – small V
        [(60, 38), (75, 32), (70, 40), (75, 48), (60, 42)],
    ],
}


def _scale_and_place(
    points: list[tuple[float, float]],
    cx: int,
    cy: int,
    size: int,
    flip_h: bool = False,
) -> list[tuple[int, int]]:
    """Scale 0-100 normalized points to pixel coords centered at (cx, cy)."""
    scale = size / 100.0
    result = []
    for x, y in points:
        px = x if not flip_h else (100 - x)
        result.append(
            (
                int(px * scale + cx - size // 2),
                int(y * scale + cy - size // 2),
            )
        )
    return result


def _lerp_color(
    c1: tuple[int, int, int], c2: tuple[int, int, int], t: float
) -> tuple[int, int, int]:
    """Linearly interpolate between two RGB colors."""
    t = max(0.0, min(1.0, t))
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


def _pseudo_perlin_1d(x: np.ndarray, seed: int, octaves: int = 4) -> np.ndarray:
    """
    Generate pseudo-Perlin 1D noise using summed sinusoids.

    Returns values roughly in [-1, 1].
    """
    rng = np.random.default_rng(seed)
    result = np.zeros_like(x, dtype=np.float64)
    for i in range(octaves):
        freq = 2.0**i
        phase = rng.uniform(0, 2 * np.pi)
        amplitude = 1.0 / (1.5**i)
        result += amplitude * np.sin(freq * x * 0.02 + phase)
    # Normalize to roughly [-1, 1]
    result /= max(np.abs(result).max(), 1e-6)
    return result


class EcologySceneRenderer:
    """Renders ecology scenes from visual property dictionaries."""

    # Color palettes
    SKY_COLD = (140, 170, 210)
    SKY_WARM = (255, 190, 100)
    SKY_MID = (170, 200, 235)
    CLOUD_COLOR = (200, 200, 205)
    CLOUD_DARK = (150, 155, 165)

    TERRAIN_GREEN = (80, 130, 60)
    TERRAIN_BROWN = (140, 110, 70)
    TERRAIN_DRY = (180, 160, 110)
    TERRAIN_DARK = (90, 80, 55)

    WATER_SHALLOW = (90, 150, 190)
    WATER_DEEP = (40, 90, 140)

    SNOW_WHITE = (245, 248, 252)
    SNOW_SHADOW = (210, 220, 235)

    TREE_GREEN_DARK = (35, 90, 30)
    TREE_GREEN_MID = (55, 120, 45)
    TREE_GREEN_LIGHT = (80, 150, 60)
    TREE_TRUNK = (90, 65, 40)

    ANIMAL_BROWN = (110, 80, 50)
    ANIMAL_DARK = (60, 50, 40)
    PREDATOR_COLOR = (80, 60, 45)
    BIRD_COLOR = (50, 45, 40)

    def __init__(
        self,
        image_size: tuple[int, int] = (1024, 1024),
        style: str = "naturalistic",
    ):
        self.image_size = image_size
        self.w, self.h = image_size
        self.style = style
        self._scale = self.w / 256.0

    def render(self, visual_props: dict[str, float], seed: int = 0) -> Image.Image:
        """
        Render a complete ecology scene from visual properties.

        Args:
            visual_props: Dict of visual property name → value (0-1 normalized).
                Expected keys: color_warmth, green_density, sky_coverage,
                object_count_small, object_count_large, water_area,
                terrain_height, snow_coverage, texture_roughness,
                sky_dot_count, lighting_offset, viewpoint_jitter
            seed: Random seed for deterministic rendering.

        Returns:
            PIL Image of the rendered scene.
        """
        rng = np.random.default_rng(seed)

        # Extract properties with defaults
        p = {
            "color_warmth": 0.5,
            "green_density": 0.4,
            "sky_coverage": 0.3,
            "object_count_small": 0.3,
            "object_count_large": 0.1,
            "water_area": 0.0,
            "terrain_height": 0.3,
            "snow_coverage": 0.0,
            "texture_roughness": 0.3,
            "sky_dot_count": 0.1,
            "lighting_offset": 0.5,
            "viewpoint_jitter": 0.0,
        }
        for k, v in visual_props.items():
            if k in p:
                p[k] = v

        img = Image.new("RGB", self.image_size, (0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Compute terrain profile (shared across layers)
        terrain_profile = self._compute_terrain_profile(p, rng)

        # Compose layers in order
        self._draw_sky(img, draw, p, rng)
        self._draw_terrain(img, draw, p, rng, terrain_profile)
        self._draw_water(img, draw, p, rng, terrain_profile)
        self._draw_vegetation(img, draw, p, rng, terrain_profile)
        self._draw_snow(img, draw, p, rng, terrain_profile)
        self._draw_animals(img, draw, p, rng, terrain_profile)
        self._draw_sky_birds(img, draw, p, rng)

        # Apply global lighting adjustment
        img = self._apply_lighting(img, p)

        # Optional slight blur for naturalistic look
        if self.style == "naturalistic":
            img = img.filter(ImageFilter.SMOOTH_MORE)

        return img

    def _compute_terrain_profile(
        self, p: dict[str, float], rng: np.random.Generator
    ) -> np.ndarray:
        """
        Compute terrain height profile as array of y-coordinates per x-pixel.

        terrain_height controls the overall horizon line.
        """
        x = np.arange(self.w, dtype=np.float64)
        noise = _pseudo_perlin_1d(x, seed=int(rng.integers(0, 100000)), octaves=4)

        # Base horizon: terrain_height pushes horizon up (mountains = high horizon)
        base_horizon = self.h * (1.0 - 0.25 - 0.4 * p["terrain_height"])

        # Amplitude of terrain variation
        amplitude = self.h * 0.08 * (0.5 + p["terrain_height"])

        profile = base_horizon + noise * amplitude

        # Add viewpoint jitter
        jitter = p.get("viewpoint_jitter", 0.0) * self.h * 0.05
        profile = profile + jitter

        return profile.astype(np.int32)

    def _draw_sky(
        self,
        img: Image.Image,
        draw: ImageDraw.ImageDraw,
        p: dict[str, float],
        rng: np.random.Generator,
    ) -> None:
        """Draw sky gradient with clouds."""
        warmth = p["color_warmth"]
        sky_top = _lerp_color(self.SKY_COLD, self.SKY_WARM, warmth * 0.7)
        sky_bottom = _lerp_color(self.SKY_MID, self.SKY_WARM, warmth * 0.5)

        # Draw gradient
        for y in range(self.h):
            t = y / max(self.h - 1, 1)
            row_color = _lerp_color(sky_top, sky_bottom, t)
            draw.line([(0, y), (self.w - 1, y)], fill=row_color)

        # Draw clouds
        cloud_coverage = p["sky_coverage"]
        n_clouds = int(cloud_coverage * 12 + rng.integers(0, 3))
        for _ in range(n_clouds):
            cx = int(rng.uniform(0, self.w))
            cy = int(rng.uniform(0, self.h * 0.5))
            cw = int(rng.uniform(self.w * 0.08, self.w * 0.25))
            ch = int(rng.uniform(self.h * 0.03, self.h * 0.08))

            cloud_col = _lerp_color(
                self.CLOUD_COLOR, self.CLOUD_DARK, cloud_coverage * 0.6
            )
            # Cloud puffs
            n_puffs = rng.integers(3, 7)
            for _ in range(n_puffs):
                px = cx + int(rng.normal(0, cw * 0.3))
                py = cy + int(rng.normal(0, ch * 0.3))
                pr = int(rng.uniform(ch * 0.5, ch * 1.2))
                bbox = [px - pr, py - pr, px + pr, py + pr]
                draw.ellipse(bbox, fill=cloud_col)

    def _draw_terrain(
        self,
        img: Image.Image,
        draw: ImageDraw.ImageDraw,
        p: dict[str, float],
        rng: np.random.Generator,
        terrain_profile: np.ndarray,
    ) -> None:
        """Draw terrain below the horizon profile."""
        warmth = p["color_warmth"]
        green_d = p["green_density"]

        # Base terrain color depends on warmth and greenness
        if green_d > 0.4:
            base_color = _lerp_color(self.TERRAIN_BROWN, self.TERRAIN_GREEN, green_d)
        else:
            base_color = _lerp_color(
                self.TERRAIN_DRY, self.TERRAIN_BROWN, warmth * 0.5 + green_d
            )

        # Draw column by column
        pixels = np.array(img)
        for x in range(self.w):
            y_top = max(0, min(int(terrain_profile[x]), self.h - 1))
            for y in range(y_top, self.h):
                # Depth gradient: darker further from horizon
                depth = (y - y_top) / max(self.h - y_top, 1)
                shade = 1.0 - depth * 0.3

                # Add texture noise
                noise_val = rng.uniform(-0.05, 0.05) * p["texture_roughness"]
                shade = max(0.3, min(1.0, shade + noise_val))

                c = (
                    int(base_color[0] * shade),
                    int(base_color[1] * shade),
                    int(base_color[2] * shade),
                )
                pixels[y, x] = c

        # Write pixels back
        img.paste(Image.fromarray(pixels), (0, 0))

    def _draw_water(
        self,
        img: Image.Image,
        draw: ImageDraw.ImageDraw,
        p: dict[str, float],
        rng: np.random.Generator,
        terrain_profile: np.ndarray,
    ) -> None:
        """Draw water body in lower portion if water_area > threshold."""
        water_area = p["water_area"]
        if water_area < 0.08:
            return

        # Water body positioned in the lower-center area
        water_w = int(self.w * (0.2 + 0.6 * water_area))
        water_h = int(self.h * (0.05 + 0.15 * water_area))
        water_cx = self.w // 2 + int(rng.integers(-self.w // 8, self.w // 8))
        water_top = int(self.h * 0.7 - water_h * 0.3)

        # Elliptical water body
        bbox = [
            water_cx - water_w // 2,
            water_top,
            water_cx + water_w // 2,
            water_top + water_h,
        ]

        # Water color with slight warmth tint
        water_col = _lerp_color(
            self.WATER_DEEP, self.WATER_SHALLOW, 0.3 + p["color_warmth"] * 0.2
        )
        draw.ellipse(bbox, fill=water_col)

        # Water edge highlight
        highlight = _lerp_color(water_col, (255, 255, 255), 0.2)
        pad = max(1, int(3 * self._scale))
        inner_bbox = [
            bbox[0] + pad,
            bbox[1] + pad,
            bbox[2] - pad,
            bbox[1] + water_h // 3,
        ]
        draw.ellipse(inner_bbox, fill=highlight)

    def _draw_vegetation(
        self,
        img: Image.Image,
        draw: ImageDraw.ImageDraw,
        p: dict[str, float],
        rng: np.random.Generator,
        terrain_profile: np.ndarray,
    ) -> None:
        """Draw trees and bushes based on green_density."""
        green_d = p["green_density"]
        if green_d < 0.05:
            return

        n_trees = int(green_d * 30 + rng.integers(0, 5))

        for _ in range(n_trees):
            # Position tree along terrain
            x = int(rng.uniform(0, self.w))
            terrain_y = int(terrain_profile[min(x, self.w - 1)])

            y = terrain_y + int(rng.uniform(-3 * self._scale, self.h * 0.15))
            if y < 0 or y >= self.h:
                continue

            # Tree size varies — farther from viewer (higher y) = smaller
            depth_factor = (y - terrain_y) / max(self.h - terrain_y, 1)
            base_size = int(self.w * 0.02 * (1.0 + green_d * 0.5))
            size = max(3, int(base_size * (1.0 - depth_factor * 0.5)))

            # Randomly choose tree type
            tree_type = rng.choice(["conifer", "deciduous", "bush"])

            # Green shade with variation
            greens = [self.TREE_GREEN_DARK, self.TREE_GREEN_MID, self.TREE_GREEN_LIGHT]
            green_shade = greens[int(rng.integers(0, 3))]
            # Adjust for warmth
            if p["color_warmth"] < 0.3:
                green_shade = _lerp_color(green_shade, (60, 80, 70), 0.3)

            if tree_type == "conifer":
                self._draw_conifer(draw, x, y, size, green_shade, rng)
            elif tree_type == "deciduous":
                self._draw_deciduous(draw, x, y, size, green_shade, rng)
            else:
                self._draw_bush(draw, x, y, size, green_shade, rng)

    def _draw_conifer(
        self,
        draw: ImageDraw.ImageDraw,
        x: int,
        y: int,
        size: int,
        color: tuple[int, int, int],
        rng: np.random.Generator,
    ) -> None:
        """Draw a conifer tree (triangle shape)."""
        trunk_h = size // 2
        tw = max(1, int(self._scale))
        draw.rectangle(
            [x - tw, y, x + tw, y + trunk_h],
            fill=self.TREE_TRUNK,
        )
        for i in range(3):
            top = y - size + i * (size // 4)
            base = y - size // 4 + i * (size // 4)
            width = size - i * (size // 5)
            shade = _lerp_color(color, self.TREE_GREEN_DARK, i * 0.15)
            draw.polygon(
                [(x, top), (x - width // 2, base), (x + width // 2, base)],
                fill=shade,
            )

    def _draw_deciduous(
        self,
        draw: ImageDraw.ImageDraw,
        x: int,
        y: int,
        size: int,
        color: tuple[int, int, int],
        rng: np.random.Generator,
    ) -> None:
        """Draw a deciduous tree (circle crown)."""
        trunk_h = size // 2
        tw = max(1, int(self._scale))
        draw.rectangle(
            [x - tw, y, x + tw, y + trunk_h],
            fill=self.TREE_TRUNK,
        )
        crown_r = max(2, size // 2 + int(rng.integers(-2, 3)))
        cy_base = y - crown_r // 2
        jitter = max(1, int(2 * self._scale))
        for k in range(3):
            ox = int(rng.integers(-jitter, jitter + 1))
            oy = int(rng.integers(-jitter, jitter + 1))
            r = max(2, crown_r - k * max(1, crown_r // 6))
            shade = _lerp_color(color, self.TREE_GREEN_LIGHT, k * 0.12)
            draw.ellipse(
                [x + ox - r, cy_base + oy - r, x + ox + r, cy_base + oy + r],
                fill=shade,
            )

    def _draw_bush(
        self,
        draw: ImageDraw.ImageDraw,
        x: int,
        y: int,
        size: int,
        color: tuple[int, int, int],
        rng: np.random.Generator,
    ) -> None:
        """Draw a small bush (short wide ellipse)."""
        jitter = max(1, int(2 * self._scale))
        for k in range(3):
            ox = int(rng.integers(-jitter, jitter + 1))
            oy = int(rng.integers(-jitter // 2, jitter // 2 + 1))
            bw = max(2, size + int(rng.integers(-2, 3)) - k * max(1, size // 5))
            bh = max(2, size // 2 - k * max(1, size // 8))
            shade = _lerp_color(color, self.TREE_GREEN_LIGHT, k * 0.1)
            draw.ellipse(
                [x + ox - bw, y + oy - bh, x + ox + bw, y + oy + bh // 2],
                fill=shade,
            )

    def _draw_snow(
        self,
        img: Image.Image,
        draw: ImageDraw.ImageDraw,
        p: dict[str, float],
        rng: np.random.Generator,
        terrain_profile: np.ndarray,
    ) -> None:
        """Add snow patches on terrain and mountaintops."""
        snow = p["snow_coverage"]
        if snow < 0.15:
            return

        pixels = np.array(img)

        # Snow on mountaintops: higher terrain areas get more snow
        for x in range(self.w):
            ty = int(terrain_profile[min(x, self.w - 1)])
            # More snow at higher terrain (lower ty values)
            snow_depth = int(snow * 20 * self._scale * (1.0 - ty / self.h))
            snow_depth = max(0, min(snow_depth, self.h - ty))

            for y in range(ty, min(ty + snow_depth, self.h)):
                # Blend existing pixel with snow color
                blend = min(1.0, snow * (1.0 - (y - ty) / max(snow_depth, 1)) * 1.5)
                if rng.random() < blend:
                    snow_col = _lerp_color(
                        self.SNOW_SHADOW, self.SNOW_WHITE, rng.uniform(0.5, 1.0)
                    )
                    pixels[y, x] = snow_col

        img.paste(Image.fromarray(pixels), (0, 0))

    def _draw_animals(
        self,
        img: Image.Image,
        draw: ImageDraw.ImageDraw,
        p: dict[str, float],
        rng: np.random.Generator,
        terrain_profile: np.ndarray,
    ) -> None:
        """Draw animal shapes on terrain."""
        s = self._scale
        herbivore_types = ["deer", "rabbit"]
        predator_types = ["wolf", "bear"]

        n_small = int(p["object_count_small"] * 12)
        for _ in range(n_small):
            x = int(rng.uniform(self.w * 0.05, self.w * 0.95))
            ty = int(terrain_profile[min(x, self.w - 1)])
            y = ty + int(rng.uniform(5 * s, self.h * 0.25))
            if y >= self.h:
                continue

            size = max(2, int(rng.uniform(3 * s, 6 * s)))
            color = _lerp_color(
                self.ANIMAL_BROWN, self.ANIMAL_DARK, rng.uniform(0, 0.5)
            )
            flip = bool(rng.integers(0, 2))
            sprite_name = herbivore_types[int(rng.integers(0, len(herbivore_types)))]
            for part in _ANIMAL_SPRITES[sprite_name]:
                pts = _scale_and_place(part, x, y, size, flip_h=flip)
                draw.polygon(pts, fill=color)

        n_large = int(p["object_count_large"] * 4)
        for _ in range(n_large):
            x = int(rng.uniform(self.w * 0.1, self.w * 0.9))
            ty = int(terrain_profile[min(x, self.w - 1)])
            y = ty + int(rng.uniform(8 * s, self.h * 0.2))
            if y >= self.h:
                continue

            size = max(4, int(rng.uniform(6 * s, 12 * s)))
            color = self.PREDATOR_COLOR
            flip = bool(rng.integers(0, 2))
            sprite_name = predator_types[int(rng.integers(0, len(predator_types)))]
            for part in _ANIMAL_SPRITES[sprite_name]:
                pts = _scale_and_place(part, x, y, size, flip_h=flip)
                draw.polygon(pts, fill=color)

    def _draw_sky_birds(
        self,
        img: Image.Image,
        draw: ImageDraw.ImageDraw,
        p: dict[str, float],
        rng: np.random.Generator,
    ) -> None:
        """Draw small bird silhouettes in the sky."""
        n_birds = int(p.get("sky_dot_count", 0) * 15)
        if n_birds < 1:
            return

        s = self._scale
        bird_parts = _ANIMAL_SPRITES["bird"]

        for _ in range(n_birds):
            bx = int(rng.uniform(0, self.w))
            by = int(rng.uniform(self.h * 0.02, self.h * 0.4))
            bird_size = max(3, int(rng.uniform(2 * s, 5 * s)))
            flip = bool(rng.integers(0, 2))
            for part in bird_parts:
                pts = _scale_and_place(part, bx, by, bird_size, flip_h=flip)
                draw.polygon(pts, fill=self.BIRD_COLOR)

    def _apply_lighting(self, img: Image.Image, p: dict[str, float]) -> Image.Image:
        """Apply global lighting/brightness adjustment."""
        lighting = p.get("lighting_offset", 0.5)
        # lighting_offset of 0.5 = neutral, <0.5 = darker, >0.5 = brighter
        factor = 0.7 + lighting * 0.6  # Range: 0.7 to 1.3

        pixels = np.array(img, dtype=np.float32)
        pixels = pixels * factor
        pixels = np.clip(pixels, 0, 255).astype(np.uint8)
        return Image.fromarray(pixels)
