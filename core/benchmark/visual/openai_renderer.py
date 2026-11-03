"""
OpenAI image generation renderer.

Uses the OpenAI Images API (gpt-image-1.5) to generate photorealistic
ecology scenes from causal variable values. Each set of visual properties
is translated into a detailed text prompt that describes the scene, then
sent to the API for generation.

Requires: openai package + OPENAI_API_KEY env var

This is the "hard mode" renderer — images are photorealistic with natural
variation, making feature extraction and discovery significantly more
challenging than the programmatic renderer.
"""

from __future__ import annotations

import base64
import io
import os
import warnings
from typing import Optional

from PIL import Image


def _build_ecology_prompt(visual_props: dict[str, float]) -> str:
    """
    Build a detailed text prompt describing an ecology scene from visual
    property values. The prompt is designed so DALL-E renders each visual
    property faithfully.
    """
    parts: list[str] = []

    # Overall style anchor
    parts.append(
        "A photorealistic landscape photograph of a natural ecology scene, "
        "taken with a DSLR camera. No text, no watermarks, no people."
    )

    # Temperature / color warmth
    warmth = visual_props.get("color_warmth", 0.5)
    if warmth < 0.25:
        parts.append(
            "The scene has a cold, blueish-grey atmosphere, overcast winter light."
        )
    elif warmth < 0.45:
        parts.append("The scene has cool, muted tones with soft daylight.")
    elif warmth < 0.65:
        parts.append("The scene is lit by warm daylight with neutral colors.")
    else:
        parts.append(
            "The scene is bathed in warm golden-hour light with orange and amber tones."
        )

    # Terrain height / elevation
    terrain = visual_props.get("terrain_height", 0.3)
    if terrain > 0.7:
        parts.append(
            "Tall mountain peaks and ridges dominate the background with rugged terrain."
        )
    elif terrain > 0.4:
        parts.append("Rolling hills and moderate elevation in the background.")
    else:
        parts.append(
            "Flat, open terrain stretching to the horizon, like a grassland or plain."
        )

    # Sky coverage / clouds
    sky = visual_props.get("sky_coverage", 0.3)
    if sky > 0.7:
        parts.append("The sky is heavily overcast with thick grey clouds.")
    elif sky > 0.4:
        parts.append("Partly cloudy sky with scattered cumulus clouds.")
    else:
        parts.append("Clear blue sky with very few clouds.")

    # Vegetation / green density
    green = visual_props.get("green_density", 0.4)
    if green > 0.7:
        parts.append(
            "Dense, lush vegetation everywhere — thick forest canopy, abundant "
            "green grass, and many trees."
        )
    elif green > 0.35:
        parts.append(
            "Moderate vegetation with scattered trees, bushes, and green grass patches."
        )
    elif green > 0.1:
        parts.append(
            "Sparse vegetation — a few shrubs and tufts of dry grass on bare ground."
        )
    else:
        parts.append(
            "Barren, arid landscape with almost no vegetation, rocky or sandy ground."
        )

    # Snow
    snow = visual_props.get("snow_coverage", 0.0)
    if snow > 0.6:
        parts.append(
            "Thick snow covers the ground and mountaintops, a winter landscape."
        )
    elif snow > 0.25:
        parts.append("Patches of snow on higher ground and in shaded areas.")

    # Water
    water = visual_props.get("water_area", 0.0)
    if water > 0.5:
        parts.append(
            "A large lake or wide river occupies the center-foreground of the scene."
        )
    elif water > 0.2:
        parts.append("A small pond or narrow stream is visible in the landscape.")
    elif water > 0.08:
        parts.append("A tiny puddle or creek is barely visible.")

    # Small animals (herbivores)
    n_small = visual_props.get("object_count_small", 0.0)
    count_small = int(n_small * 12)
    if count_small >= 8:
        parts.append(
            f"A herd of about {count_small} small grazing animals (like deer or rabbits) scattered across the terrain."
        )
    elif count_small >= 3:
        parts.append(
            f"A few ({count_small}) small grazing animals visible in the mid-ground."
        )
    elif count_small >= 1:
        parts.append("One or two small animals barely visible in the distance.")

    # Large animals (predators)
    n_large = visual_props.get("object_count_large", 0.0)
    count_large = int(n_large * 4)
    if count_large >= 2:
        parts.append(
            f"{count_large} large predators (like wolves or bears) are present in the scene."
        )
    elif count_large >= 1:
        parts.append("A single large predator is visible in the scene.")

    # Birds
    birds = visual_props.get("sky_dot_count", 0.0)
    count_birds = int(birds * 15)
    if count_birds >= 8:
        parts.append(f"A flock of about {count_birds} birds flying across the sky.")
    elif count_birds >= 3:
        parts.append(f"A few birds ({count_birds}) soaring in the sky.")
    elif count_birds >= 1:
        parts.append("One or two birds in the sky.")

    # Texture / roughness
    texture = visual_props.get("texture_roughness", 0.3)
    if texture > 0.6:
        parts.append(
            "The ground has rough, uneven texture — wet mud, loose rocks, tangled roots."
        )
    elif texture < 0.15:
        parts.append("The ground is smooth and uniform.")

    return " ".join(parts)


class OpenAIImageRenderer:
    SUPPORTED_SIZES = {"1024x1024", "1024x1536", "1536x1024"}

    def __init__(
        self,
        model: str = "gpt-image-1.5",
        size: str = "1024x1024",
        quality: str = "low",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model = model
        if size not in self.SUPPORTED_SIZES:
            size = "1024x1024"
        self.size = size
        self.quality = quality
        self._api_key = api_key
        self._base_url = base_url
        self._client: Optional[object] = None

    @staticmethod
    def is_available() -> bool:
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LITELLM_API_KEY")
        return key is not None

    def _ensure_client(self) -> bool:
        if self._client is not None:
            return True

        try:
            from openai import OpenAI

            key = (
                self._api_key
                or os.environ.get("OPENAI_API_KEY")
                or os.environ.get("LITELLM_API_KEY")
            )
            if not key:
                warnings.warn(
                    "No API key found. Set OPENAI_API_KEY.",
                    stacklevel=2,
                )
                return False

            kwargs: dict = {"api_key": key}
            if self._base_url:
                kwargs["base_url"] = self._base_url

            self._client = OpenAI(**kwargs)
            return True

        except ImportError:
            warnings.warn(
                "openai package not installed. pip install openai",
                stacklevel=2,
            )
            return False

    def render(
        self,
        visual_props: dict[str, float],
        seed: int = 0,
        prompt_override: Optional[str] = None,
    ) -> Optional[Image.Image]:
        if not self._ensure_client():
            return None

        prompt = prompt_override or _build_ecology_prompt(visual_props)

        try:
            response = self._client.images.generate(  # type: ignore[union-attr]
                model=self.model,
                prompt=prompt,
                size=self.size,
                quality=self.quality,
            )

            image_data = response.data[0]

            if hasattr(image_data, "b64_json") and image_data.b64_json:
                img_bytes = base64.b64decode(image_data.b64_json)
                return Image.open(io.BytesIO(img_bytes)).convert("RGB")

            warnings.warn("OpenAI returned no image data", stacklevel=2)
            return None

        except Exception as e:
            warnings.warn(f"OpenAI image generation failed: {e}", stacklevel=2)
            return None

    def render_batch(
        self,
        visual_props_list: list[dict[str, float]],
        seeds: Optional[list[int]] = None,
    ) -> list[Optional[Image.Image]]:
        results = []
        for i, vp in enumerate(visual_props_list):
            s = seeds[i] if seeds else i
            img = self.render(vp, seed=s)
            results.append(img)
        return results

    def get_prompt(self, visual_props: dict[str, float]) -> str:
        return _build_ecology_prompt(visual_props)
