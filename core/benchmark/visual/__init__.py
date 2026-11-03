"""
Visual ecology benchmark generation system.

Generates ecology scene images from causal DAG simulations, where causal
variables manifest as visual properties (color, spatial density, object counts,
texture, layout). The discovery system operates on extracted features from
these images, testing whether it can recover the underlying causal patterns.

Components:
- visual_mapping: Maps causal variables to visual scene properties
- scene_renderer: Renders ecology scenes using PIL from visual properties
- feature_extractor: Extracts numeric features from images for the pipeline
- visual_dataset_adapter: DatasetAdapter serving extracted visual features
- diffusion_stylizer: Optional diffusion model style transfer for realism
"""

from core.benchmark.visual.visual_mapping import (
    VariableVisualMapping,
    VisualMappingSpec,
    map_values_to_visual,
    DEFAULT_ECOLOGY_MAPPING,
)
from core.benchmark.visual.scene_renderer import EcologySceneRenderer
from core.benchmark.visual.feature_extractor import VisualFeatureExtractor
from core.benchmark.visual.openai_renderer import OpenAIImageRenderer

__all__ = [
    "VariableVisualMapping",
    "VisualMappingSpec",
    "map_values_to_visual",
    "DEFAULT_ECOLOGY_MAPPING",
    "EcologySceneRenderer",
    "VisualFeatureExtractor",
    "OpenAIImageRenderer",
]
