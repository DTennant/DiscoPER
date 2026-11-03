"""
Optional diffusion model stylizer.

Applies Stable Diffusion img2img to transform programmatically rendered
ecology scenes into photorealistic images while preserving the visual
properties that encode causal variables.

Requires: torch, diffusers, transformers (optional dependencies)
Falls back gracefully (returns original images) if not installed.
"""

from __future__ import annotations

import warnings
from typing import Optional

from PIL import Image


class DiffusionStylizer:
    """Applies diffusion model style transfer to rendered scenes."""

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        strength: float = 0.4,
        device: str = "auto",
        seed: int = 42,
    ):
        """
        Initialize stylizer. Loads model lazily on first use.

        Args:
            model_id: HuggingFace model ID for the diffusion model.
            strength: How much to transform (0=no change, 1=full regeneration).
                     Use 0.3-0.5 to preserve structure while adding realism.
            device: Device for inference ("auto", "cuda", "cpu").
            seed: Random seed for reproducibility.
        """
        self.model_id = model_id
        self.strength = strength
        self.device = device
        self.seed = seed
        self._pipeline = None
        self._available = None

    @staticmethod
    def is_available() -> bool:
        """Check if torch and diffusers are installed."""
        try:
            import torch  # noqa: F401
            import diffusers  # noqa: F401

            return True
        except ImportError:
            return False

    def _ensure_pipeline(self) -> bool:
        """Lazily load the diffusion pipeline. Returns True if available."""
        if self._available is False:
            return False
        if self._pipeline is not None:
            return True

        try:
            import torch
            from diffusers import AutoPipelineForImage2Image

            if self.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device

            dtype = torch.float16 if device == "cuda" else torch.float32

            self._pipeline = AutoPipelineForImage2Image.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                variant="fp16" if device == "cuda" else None,
            ).to(device)

            # Enable memory optimizations
            if device == "cuda":
                try:
                    self._pipeline.enable_model_cpu_offload()
                except Exception:
                    pass

            self._available = True
            return True

        except ImportError:
            warnings.warn(
                "diffusers/torch not available. Install with: "
                "pip install torch diffusers transformers accelerate",
                stacklevel=2,
            )
            self._available = False
            return False

        except Exception as e:
            warnings.warn(f"Failed to load diffusion model: {e}", stacklevel=2)
            self._available = False
            return False

    def stylize(
        self,
        image: Image.Image,
        prompt: str = "photorealistic ecology landscape, nature photography, "
        "detailed natural scenery, wildlife habitat",
        negative_prompt: str = "cartoon, drawing, sketch, low quality, blurry, "
        "text, watermark, logo",
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Apply style transfer to a single image.

        Args:
            image: Input PIL image (programmatically rendered scene).
            prompt: Text prompt guiding the style transfer.
            negative_prompt: What to avoid in the output.
            seed: Override random seed for this image.

        Returns:
            Stylized PIL image, or original if model unavailable.
        """
        if not self._ensure_pipeline():
            return image

        import torch

        gen_seed = seed if seed is not None else self.seed
        generator = torch.Generator(device=self._pipeline.device).manual_seed(gen_seed)

        # Resize to model-friendly dimensions
        target_size = (768, 768) if "xl" in self.model_id else (512, 512)
        original_size = image.size
        resized = image.resize(target_size, Image.Resampling.LANCZOS)

        try:
            result = self._pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=resized,
                strength=self.strength,
                guidance_scale=7.5,
                num_inference_steps=30,
                generator=generator,
            ).images[0]

            # Resize back to original dimensions
            result = result.resize(original_size, Image.Resampling.LANCZOS)
            return result

        except Exception as e:
            warnings.warn(f"Diffusion stylization failed: {e}", stacklevel=2)
            return image

    def stylize_batch(
        self,
        images: list[Image.Image],
        prompts: Optional[list[str]] = None,
        batch_size: int = 4,
    ) -> list[Image.Image]:
        """
        Apply style transfer to a batch of images.

        Args:
            images: List of input PIL images.
            prompts: Optional per-image prompts (uses default if None).
            batch_size: Number of images to process at once.

        Returns:
            List of stylized images.
        """
        results = []
        for i, img in enumerate(images):
            prompt = prompts[i] if prompts else None
            seed = self.seed + i
            if prompt:
                result = self.stylize(img, prompt=prompt, seed=seed)
            else:
                result = self.stylize(img, seed=seed)
            results.append(result)
        return results
