"""
Visual mapping layer: translates causal DAG variable values to visual scene properties.

Each VariableVisualMapping defines how a numeric causal variable maps to a visual
attribute in the rendered scene. Mappings are monotonic with optional noise to
prevent trivial pixel-counting recovery of the causal variables.

Visual channels:
- color_warmth: Scene color temperature (0=cold blue, 1=warm orange)
- green_density: Vegetation coverage fraction (0=barren, 1=dense forest)
- sky_coverage: Cloud/overcast coverage (0=clear, 1=fully overcast)
- object_count_small: Number of small animal shapes (herbivores, birds)
- object_count_large: Number of large animal shapes (predators)
- water_area: Water body size fraction (0=none, 1=large lake)
- terrain_height: Horizon line / mountain prominence (0=flat, 1=mountainous)
- snow_coverage: Snow/frost coverage (0=none, 1=fully snow-covered)
- texture_roughness: Surface texture complexity (0=smooth, 1=rough/busy)
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np


@dataclasses.dataclass
class VariableVisualMapping:
    """Defines how a single causal variable maps to a visual property."""

    variable_name: str
    visual_channel: str
    value_range: tuple[float, float]
    visual_range: tuple[float, float] = (0.0, 1.0)
    noise_std: float = 0.05
    description: str = ""

    def map_value(
        self, raw_value: float, rng: Optional[np.random.Generator] = None
    ) -> float:
        """
        Map a raw causal variable value to a normalized visual property value.

        Args:
            raw_value: The raw causal variable value.
            rng: Optional random generator for adding noise.

        Returns:
            Normalized visual property value in visual_range, with optional noise.
        """
        lo, hi = self.value_range
        v_lo, v_hi = self.visual_range

        if hi == lo:
            t = 0.5
        else:
            t = (raw_value - lo) / (hi - lo)
        t = float(np.clip(t, 0.0, 1.0))

        visual_val = v_lo + t * (v_hi - v_lo)

        if rng is not None and self.noise_std > 0:
            visual_val += rng.normal(0, self.noise_std)

        return float(np.clip(visual_val, 0.0, 1.0))


@dataclasses.dataclass
class VisualMappingSpec:
    """
    Complete specification for mapping causal variables to visual scene properties.

    Contains a list of variable-to-visual mappings plus global rendering parameters.
    """

    mappings: list[VariableVisualMapping]
    image_size: tuple[int, int] = (256, 256)
    style: str = "naturalistic"
    nuisance_lighting_std: float = 0.03
    nuisance_viewpoint_jitter: float = 0.02

    def get_mapping(self, variable_name: str) -> Optional[VariableVisualMapping]:
        """Get mapping for a specific causal variable."""
        for m in self.mappings:
            if m.variable_name == variable_name:
                return m
        return None

    def get_visual_channels(self) -> list[str]:
        """Get list of all visual channels used."""
        return list(set(m.visual_channel for m in self.mappings))


def map_values_to_visual(
    mapping_spec: VisualMappingSpec,
    variable_values: dict[str, float],
    rng: Optional[np.random.Generator] = None,
) -> dict[str, float]:
    """
    Map a set of causal variable values to visual property values.

    Multiple causal variables can map to the same visual channel — their
    effects are combined additively (clipped to [0, 1]).

    Args:
        mapping_spec: The visual mapping specification.
        variable_values: Dict of causal variable name → raw numeric value.
        rng: Optional random generator for noise and nuisance factors.

    Returns:
        Dict of visual channel name → normalized value (0-1).
    """
    visual_props: dict[str, float] = {}
    channel_counts: dict[str, int] = {}

    for mapping in mapping_spec.mappings:
        if mapping.variable_name not in variable_values:
            continue

        raw_val = variable_values[mapping.variable_name]
        vis_val = mapping.map_value(raw_val, rng)
        channel = mapping.visual_channel

        if channel in visual_props:
            # Additive combination: average contributions
            channel_counts[channel] += 1
            n = channel_counts[channel]
            visual_props[channel] = visual_props[channel] * ((n - 1) / n) + vis_val / n
        else:
            visual_props[channel] = vis_val
            channel_counts[channel] = 1

    # Add nuisance factors (independent of causal variables)
    if rng is not None:
        if "lighting_offset" not in visual_props:
            visual_props["lighting_offset"] = float(
                np.clip(rng.normal(0.5, mapping_spec.nuisance_lighting_std), 0.0, 1.0)
            )
        if "viewpoint_jitter" not in visual_props:
            visual_props["viewpoint_jitter"] = float(
                np.clip(
                    rng.normal(0.0, mapping_spec.nuisance_viewpoint_jitter), -0.1, 0.1
                )
            )

    # Clip all values to [0, 1]
    for k in visual_props:
        if k != "viewpoint_jitter":
            visual_props[k] = float(np.clip(visual_props[k], 0.0, 1.0))

    return visual_props


# =============================================================================
# Default ecology mapping for the GOL benchmark presets
# =============================================================================

DEFAULT_ECOLOGY_MAPPING = VisualMappingSpec(
    mappings=[
        # Climate → color/snow
        VariableVisualMapping(
            variable_name="temperature",
            visual_channel="color_warmth",
            value_range=(-30, 50),
            visual_range=(0.0, 1.0),
            noise_std=0.04,
            description="Temperature maps to scene color temperature (cold=blue, warm=orange)",
        ),
        VariableVisualMapping(
            variable_name="temperature",
            visual_channel="snow_coverage",
            value_range=(-30, 50),
            visual_range=(1.0, 0.0),  # Inverted: cold → more snow
            noise_std=0.05,
            description="Cold temperatures increase snow coverage",
        ),
        # Elevation → terrain
        VariableVisualMapping(
            variable_name="elevation",
            visual_channel="terrain_height",
            value_range=(0, 5000),
            visual_range=(0.1, 0.9),
            noise_std=0.03,
            description="Elevation controls mountain prominence and horizon height",
        ),
        # Vegetation → green density
        VariableVisualMapping(
            variable_name="vegetation_density",
            visual_channel="green_density",
            value_range=(0, 100),
            visual_range=(0.0, 1.0),
            noise_std=0.06,
            description="Vegetation density controls tree/grass coverage",
        ),
        # Canopy cover also contributes to green density
        VariableVisualMapping(
            variable_name="canopy_cover",
            visual_channel="green_density",
            value_range=(0, 100),
            visual_range=(0.0, 0.8),
            noise_std=0.05,
            description="Canopy cover adds to overall greenery",
        ),
        # Precipitation → sky coverage and texture
        VariableVisualMapping(
            variable_name="precipitation",
            visual_channel="sky_coverage",
            value_range=(50, 3000),
            visual_range=(0.05, 0.9),
            noise_std=0.06,
            description="Precipitation increases cloud/overcast coverage",
        ),
        VariableVisualMapping(
            variable_name="precipitation",
            visual_channel="texture_roughness",
            value_range=(50, 3000),
            visual_range=(0.1, 0.7),
            noise_std=0.05,
            description="Precipitation adds wet/rough surface texture",
        ),
        # Herbivores → small animal count
        VariableVisualMapping(
            variable_name="herbivore_abundance",
            visual_channel="object_count_small",
            value_range=(0, 80),
            visual_range=(0.0, 1.0),
            noise_std=0.08,
            description="Herbivore abundance controls number of small animal shapes",
        ),
        # Handle herbivore_a / herbivore_b for medium/hard presets
        VariableVisualMapping(
            variable_name="herbivore_a",
            visual_channel="object_count_small",
            value_range=(0, 80),
            visual_range=(0.0, 0.7),
            noise_std=0.08,
            description="Herbivore A population as small shapes",
        ),
        VariableVisualMapping(
            variable_name="herbivore_b",
            visual_channel="object_count_small",
            value_range=(0, 80),
            visual_range=(0.0, 0.5),
            noise_std=0.08,
            description="Herbivore B population as additional small shapes",
        ),
        # Predators → large animal shapes
        VariableVisualMapping(
            variable_name="predator_presence",
            visual_channel="object_count_large",
            value_range=(0, 1),
            visual_range=(0.0, 1.0),
            noise_std=0.06,
            description="Predator presence controls number of large predator shapes",
        ),
        VariableVisualMapping(
            variable_name="keystone_predator",
            visual_channel="object_count_large",
            value_range=(0, 1),
            visual_range=(0.0, 0.6),
            noise_std=0.06,
            description="Keystone predator adds additional large shapes",
        ),
        # Water proximity → water body area
        VariableVisualMapping(
            variable_name="water_proximity",
            visual_channel="water_area",
            value_range=(0, 50),
            visual_range=(0.8, 0.0),  # Inverted: closer = more water visible
            noise_std=0.05,
            description="Water proximity (inverse): closer → larger water body",
        ),
        # Insects → texture roughness
        VariableVisualMapping(
            variable_name="insect_abundance",
            visual_channel="texture_roughness",
            value_range=(0, 300),
            visual_range=(0.0, 0.5),
            noise_std=0.06,
            description="Insect abundance adds to surface texture busyness",
        ),
        # Birds → sky dots
        VariableVisualMapping(
            variable_name="bird_richness",
            visual_channel="sky_dot_count",
            value_range=(0, 40),
            visual_range=(0.0, 1.0),
            noise_std=0.07,
            description="Bird species richness shown as small dots in sky",
        ),
    ],
    image_size=(256, 256),
    style="naturalistic",
    nuisance_lighting_std=0.03,
    nuisance_viewpoint_jitter=0.02,
)


def get_feature_variable_mapping() -> dict[str, str]:
    """
    Return mapping from extracted feature names to the causal variables
    they approximate. Used for evaluation only — NOT exposed to discovery.
    """
    return {
        # Color features → causal variables
        "color_warmth": "temperature",
        "white_fraction": "temperature",  # snow proxy
        "green_fraction": "vegetation_density",
        "blue_fraction": "water_proximity",
        "sky_grey_fraction": "precipitation",
        # Spatial features
        "horizon_height": "elevation",
        "terrain_roughness": "elevation",
        # Object count features
        "dark_blob_count": "herbivore_abundance",
        "small_blob_count": "herbivore_abundance",
        "large_blob_count": "predator_presence",
        "sky_dot_count": "bird_richness",
        # Vegetation features
        "vegetation_coverage": "vegetation_density",
        "tree_count": "vegetation_density",
        # Water features
        "water_body_area": "water_proximity",
        # Texture features
        "texture_energy": "precipitation",
    }
