"""
Ecology-themed preset configurations for causal DAG benchmarks.

Three difficulty levels with increasing numbers of variables, rules,
and emergent structural complexity.
"""

from __future__ import annotations

from core.benchmark.rules import (
    CausalDAG,
    CompetitionRule,
    InteractionRule,
    LinearCausalRule,
    ModulationRule,
    RuleSet,
    SaturationRule,
    ThresholdCausalRule,
    VariableSpec,
)


# =============================================================================
# EASY — Grassland Water Chain
# =============================================================================

def make_easy_preset() -> RuleSet:
    """
    Easy preset: Grassland Water Chain.

    ~9 variables, 8 rules, 3000 samples.

    DAG:
        elevation --(-0.7)--> temperature --(0.3)--> vegetation_density --(0.55)--> herbivore_abundance --(0.5)--> predator_presence
        latitude ---(-0.5)--> temperature                 ^                               ^
                                               precipitation --(0.6)--+         water_proximity --(-0.4)--+
                                                            +--(0.5)--> insect_abundance
    """
    dag = CausalDAG()

    # Exogenous variables
    dag.add_variable(VariableSpec(
        name="elevation", var_type="exogenous",
        distribution="uniform", distribution_params={"low": 0, "high": 4000},
        description="Elevation in meters above sea level",
    ))
    dag.add_variable(VariableSpec(
        name="latitude", var_type="exogenous",
        distribution="uniform", distribution_params={"low": 25, "high": 65},
        description="Latitude in degrees",
    ))
    dag.add_variable(VariableSpec(
        name="precipitation", var_type="exogenous",
        distribution="normal", distribution_params={"mean": 800, "std": 300},
        clip_min=50, clip_max=3000,
        description="Annual precipitation in mm",
    ))
    dag.add_variable(VariableSpec(
        name="water_proximity", var_type="exogenous",
        distribution="exponential", distribution_params={"scale": 5.0},
        clip_min=0, clip_max=50,
        description="Distance to nearest water source in km",
    ))

    # Derived variables
    dag.add_variable(VariableSpec(
        name="temperature", var_type="derived",
        base_value=20.0, observation_noise_std=0.15,
        clip_min=-30, clip_max=50,
        description="Mean annual temperature in Celsius",
    ))
    dag.add_variable(VariableSpec(
        name="vegetation_density", var_type="derived",
        base_value=50.0, observation_noise_std=0.15,
        clip_min=0, clip_max=100,
        description="Vegetation density index (0-100)",
    ))
    dag.add_variable(VariableSpec(
        name="insect_abundance", var_type="derived",
        base_value=100.0, observation_noise_std=0.15,
        clip_min=0,
        description="Insect abundance count",
    ))
    dag.add_variable(VariableSpec(
        name="herbivore_abundance", var_type="derived",
        base_value=30.0, observation_noise_std=0.15,
        clip_min=0,
        description="Herbivore population abundance",
    ))
    dag.add_variable(VariableSpec(
        name="predator_presence", var_type="derived",
        base_value=0.5, observation_noise_std=0.1,
        clip_min=0, clip_max=1,
        description="Predator presence probability",
    ))

    # Causal rules
    dag.add_rule(LinearCausalRule(
        id="elev_temp", sources=["elevation"], target="temperature",
        strength=-0.7, noise_std=0.05,
        description="Temperature decreases with elevation (lapse rate ~6.5°C/km)",
        ecological_principle="Adiabatic lapse rate",
    ))
    dag.add_rule(LinearCausalRule(
        id="lat_temp", sources=["latitude"], target="temperature",
        strength=-0.5, noise_std=0.05,
        description="Higher latitudes are colder",
        ecological_principle="Solar radiation gradient",
    ))
    dag.add_rule(LinearCausalRule(
        id="temp_veg", sources=["temperature"], target="vegetation_density",
        strength=0.3, noise_std=0.08,
        description="Warmer temperatures increase vegetation density",
        ecological_principle="Growing season length",
    ))
    dag.add_rule(LinearCausalRule(
        id="precip_veg", sources=["precipitation"], target="vegetation_density",
        strength=0.6, noise_std=0.08,
        description="More precipitation increases vegetation density",
        ecological_principle="Water availability for plant growth",
    ))
    dag.add_rule(LinearCausalRule(
        id="precip_insect", sources=["precipitation"], target="insect_abundance",
        strength=0.5, noise_std=0.1,
        description="More precipitation supports higher insect populations",
        ecological_principle="Habitat moisture and food availability",
    ))
    dag.add_rule(LinearCausalRule(
        id="veg_herb", sources=["vegetation_density"], target="herbivore_abundance",
        strength=0.55, noise_std=0.1,
        description="Denser vegetation supports more herbivores",
        ecological_principle="Bottom-up trophic cascade",
    ))
    dag.add_rule(LinearCausalRule(
        id="herb_pred", sources=["herbivore_abundance"], target="predator_presence",
        strength=0.5, noise_std=0.08,
        description="More herbivores attract predators",
        ecological_principle="Predator-prey dynamics",
    ))
    dag.add_rule(LinearCausalRule(
        id="water_herb", sources=["water_proximity"], target="herbivore_abundance",
        strength=-0.4, noise_std=0.08,
        description="Greater distance from water reduces herbivore abundance",
        ecological_principle="Water dependence of large herbivores",
    ))

    return RuleSet(
        name="gol_easy",
        description="Grassland Water Chain — simple trophic cascade with linear effects",
        dag=dag,
        n_samples=3000,
        seed=42,
        global_noise_scale=1.0,
        correlation_threshold=0.15,
        prior_rule_ids=["elev_temp", "lat_temp"],
    )


# =============================================================================
# MEDIUM — Forest Trophic Cascade
# =============================================================================

def make_medium_preset() -> RuleSet:
    """
    Medium preset: Forest Trophic Cascade.

    ~14 variables, ~15 rules, 5000 samples.
    Adds: seasonal modulation, interaction effects, competition,
    threshold (treeline), Bergmann's rule, observer confound.
    """
    dag = CausalDAG()

    # Exogenous variables
    dag.add_variable(VariableSpec(
        name="elevation", var_type="exogenous",
        distribution="uniform", distribution_params={"low": 0, "high": 4000},
        description="Elevation in meters",
    ))
    dag.add_variable(VariableSpec(
        name="latitude", var_type="exogenous",
        distribution="uniform", distribution_params={"low": 25, "high": 65},
        description="Latitude in degrees",
    ))
    dag.add_variable(VariableSpec(
        name="precipitation", var_type="exogenous",
        distribution="normal", distribution_params={"mean": 900, "std": 350},
        clip_min=50, clip_max=3000,
        description="Annual precipitation in mm",
    ))
    dag.add_variable(VariableSpec(
        name="water_proximity", var_type="exogenous",
        distribution="exponential", distribution_params={"scale": 5.0},
        clip_min=0, clip_max=50,
        description="Distance to water in km",
    ))
    dag.add_variable(VariableSpec(
        name="season_indicator", var_type="exogenous",
        distribution="uniform", distribution_params={"low": 0, "high": 1},
        description="Seasonal warmth indicator (0=winter, 1=summer)",
    ))

    # Derived variables
    dag.add_variable(VariableSpec(
        name="temperature", var_type="derived",
        base_value=18.0, observation_noise_std=0.2,
        clip_min=-30, clip_max=50,
        description="Mean annual temperature in Celsius",
    ))
    dag.add_variable(VariableSpec(
        name="vegetation_density", var_type="derived",
        base_value=50.0, observation_noise_std=0.2,
        clip_min=0, clip_max=100,
        description="Vegetation density index",
    ))
    dag.add_variable(VariableSpec(
        name="canopy_cover", var_type="derived",
        base_value=40.0, observation_noise_std=0.2,
        clip_min=0, clip_max=100,
        description="Canopy cover percentage",
    ))
    dag.add_variable(VariableSpec(
        name="insect_abundance", var_type="derived",
        base_value=100.0, observation_noise_std=0.2,
        clip_min=0,
        description="Insect abundance count",
    ))
    dag.add_variable(VariableSpec(
        name="herbivore_a", var_type="derived",
        base_value=30.0, observation_noise_std=0.2,
        clip_min=0,
        description="Herbivore species A abundance",
    ))
    dag.add_variable(VariableSpec(
        name="herbivore_b", var_type="derived",
        base_value=25.0, observation_noise_std=0.2,
        clip_min=0,
        description="Herbivore species B abundance",
    ))
    dag.add_variable(VariableSpec(
        name="predator_presence", var_type="derived",
        base_value=0.4, observation_noise_std=0.1,
        clip_min=0, clip_max=1,
        description="Predator presence probability",
    ))
    dag.add_variable(VariableSpec(
        name="bird_richness", var_type="derived",
        base_value=15.0, observation_noise_std=0.2,
        clip_min=0,
        description="Bird species richness",
    ))
    dag.add_variable(VariableSpec(
        name="body_size", var_type="derived",
        base_value=50.0, observation_noise_std=0.2,
        clip_min=5,
        description="Average body size in grams (Bergmann's rule)",
    ))

    # --- Core climate rules (prior knowledge) ---
    dag.add_rule(LinearCausalRule(
        id="elev_temp", sources=["elevation"], target="temperature",
        strength=-0.7, noise_std=0.05,
        description="Temperature decreases with elevation (lapse rate)",
        ecological_principle="Adiabatic lapse rate",
    ))
    dag.add_rule(LinearCausalRule(
        id="lat_temp", sources=["latitude"], target="temperature",
        strength=-0.5, noise_std=0.05,
        description="Higher latitudes are colder",
        ecological_principle="Solar radiation gradient",
    ))
    dag.add_rule(LinearCausalRule(
        id="precip_veg", sources=["precipitation"], target="vegetation_density",
        strength=0.55, noise_std=0.08,
        description="More precipitation increases vegetation density",
        ecological_principle="Water availability",
    ))

    # --- Vegetation and canopy ---
    dag.add_rule(LinearCausalRule(
        id="temp_veg", sources=["temperature"], target="vegetation_density",
        strength=0.3, noise_std=0.08,
        description="Warmer temperatures support more vegetation",
        ecological_principle="Growing season",
    ))
    dag.add_rule(InteractionRule(
        id="precip_temp_veg", sources=["precipitation", "temperature"],
        target="vegetation_density",
        strength=0.35, noise_std=0.05,
        description="Precipitation effect on vegetation is amplified by warmth",
        ecological_principle="Water-energy dynamics",
    ))
    dag.add_rule(ModulationRule(
        id="season_temp_veg", sources=["temperature", "season_indicator"],
        target="vegetation_density",
        strength=0.25, noise_std=0.05,
        modulator_center=0.5, modulator_scale=0.2,
        description="Seasonal warmth modulates temperature effect on vegetation",
        ecological_principle="Phenological timing",
    ))
    dag.add_rule(ThresholdCausalRule(
        id="elev_treeline", sources=["elevation"], target="canopy_cover",
        strength=-0.8, noise_std=0.05,
        threshold=2500, above=True,
        description="Above treeline (~2500m), canopy cover drops sharply",
        ecological_principle="Alpine treeline",
    ))
    dag.add_rule(LinearCausalRule(
        id="veg_canopy", sources=["vegetation_density"], target="canopy_cover",
        strength=0.5, noise_std=0.08,
        description="Higher vegetation density correlates with canopy cover",
        ecological_principle="Forest structure",
    ))

    # --- Trophic cascade ---
    dag.add_rule(LinearCausalRule(
        id="precip_insect", sources=["precipitation"], target="insect_abundance",
        strength=0.45, noise_std=0.1,
        description="More precipitation supports more insects",
        ecological_principle="Moisture-dependent insect life cycles",
    ))
    dag.add_rule(LinearCausalRule(
        id="veg_herb_a", sources=["vegetation_density"], target="herbivore_a",
        strength=0.5, noise_std=0.1,
        description="Denser vegetation supports herbivore species A",
        ecological_principle="Bottom-up trophic cascade",
    ))
    dag.add_rule(LinearCausalRule(
        id="veg_herb_b", sources=["vegetation_density"], target="herbivore_b",
        strength=0.45, noise_std=0.1,
        description="Denser vegetation supports herbivore species B",
        ecological_principle="Bottom-up trophic cascade",
    ))
    dag.add_rule(CompetitionRule(
        id="herb_competition", sources=["herbivore_a", "water_proximity"],
        target="herbivore_b",
        strength=0.4, noise_std=0.08,
        modifier_weakening=0.6,
        description="Herbivore A competes with B; water proximity weakens competition",
        ecological_principle="Interspecific competition + resource partitioning",
    ))
    dag.add_rule(LinearCausalRule(
        id="herb_pred", sources=["herbivore_a"], target="predator_presence",
        strength=0.45, noise_std=0.08,
        description="More herbivore A attracts predators",
        ecological_principle="Predator-prey dynamics",
    ))
    dag.add_rule(LinearCausalRule(
        id="canopy_bird", sources=["canopy_cover"], target="bird_richness",
        strength=0.5, noise_std=0.08,
        description="Greater canopy cover supports more bird species",
        ecological_principle="Habitat structural complexity",
    ))

    # --- Bergmann's rule ---
    dag.add_rule(LinearCausalRule(
        id="temp_bodysize", sources=["temperature"], target="body_size",
        strength=-0.45, noise_std=0.1,
        description="Colder temperatures favor larger body size (Bergmann's rule)",
        ecological_principle="Thermoregulation surface-area-to-volume ratio",
    ))

    return RuleSet(
        name="gol_medium",
        description="Forest Trophic Cascade — interactions, competition, thresholds",
        dag=dag,
        n_samples=5000,
        seed=42,
        global_noise_scale=1.0,
        correlation_threshold=0.12,
        prior_rule_ids=["elev_temp", "lat_temp", "precip_veg"],
    )


# =============================================================================
# HARD — Multi-Biome Complex
# =============================================================================

def make_hard_preset() -> RuleSet:
    """
    Hard preset: Multi-Biome Complex.

    ~20 variables, ~25 rules, 10000 samples.
    Adds: saturation (carrying capacity), keystone species, multiple thresholds,
    weak effects in high noise, masked correlations, Simpson's paradox structures.
    """
    dag = CausalDAG()

    # --- Exogenous variables ---
    dag.add_variable(VariableSpec(
        name="elevation", var_type="exogenous",
        distribution="uniform", distribution_params={"low": 0, "high": 5000},
        description="Elevation in meters",
    ))
    dag.add_variable(VariableSpec(
        name="latitude", var_type="exogenous",
        distribution="uniform", distribution_params={"low": 15, "high": 70},
        description="Latitude in degrees",
    ))
    dag.add_variable(VariableSpec(
        name="precipitation", var_type="exogenous",
        distribution="normal", distribution_params={"mean": 1000, "std": 450},
        clip_min=50, clip_max=4000,
        description="Annual precipitation in mm",
    ))
    dag.add_variable(VariableSpec(
        name="water_proximity", var_type="exogenous",
        distribution="exponential", distribution_params={"scale": 8.0},
        clip_min=0, clip_max=80,
        description="Distance to water in km",
    ))
    dag.add_variable(VariableSpec(
        name="season_indicator", var_type="exogenous",
        distribution="uniform", distribution_params={"low": 0, "high": 1},
        description="Seasonal warmth indicator",
    ))
    dag.add_variable(VariableSpec(
        name="soil_nitrogen", var_type="exogenous",
        distribution="lognormal", distribution_params={"mean": 1.5, "sigma": 0.8},
        clip_min=0, clip_max=100,
        description="Soil nitrogen content (mg/kg)",
    ))
    dag.add_variable(VariableSpec(
        name="disturbance_history", var_type="exogenous",
        distribution="bernoulli", distribution_params={"p": 0.3},
        dtype="bool",
        description="Recent fire or logging disturbance (True/False)",
    ))

    # --- Derived: climate ---
    dag.add_variable(VariableSpec(
        name="temperature", var_type="derived",
        base_value=18.0, observation_noise_std=0.3,
        clip_min=-40, clip_max=55,
        description="Mean annual temperature",
    ))

    # --- Derived: vegetation ---
    dag.add_variable(VariableSpec(
        name="vegetation_density", var_type="derived",
        base_value=50.0, observation_noise_std=0.3,
        clip_min=0, clip_max=100,
        description="Vegetation density index",
    ))
    dag.add_variable(VariableSpec(
        name="canopy_cover", var_type="derived",
        base_value=40.0, observation_noise_std=0.3,
        clip_min=0, clip_max=100,
        description="Canopy cover percentage",
    ))
    dag.add_variable(VariableSpec(
        name="root_biomass", var_type="derived",
        base_value=200.0, observation_noise_std=0.3,
        clip_min=0,
        description="Root biomass (g/m^2)",
    ))

    # --- Derived: fauna ---
    dag.add_variable(VariableSpec(
        name="insect_abundance", var_type="derived",
        base_value=120.0, observation_noise_std=0.3,
        clip_min=0,
        description="Insect abundance",
    ))
    dag.add_variable(VariableSpec(
        name="herbivore_a", var_type="derived",
        base_value=30.0, observation_noise_std=0.3,
        clip_min=0,
        description="Herbivore species A",
    ))
    dag.add_variable(VariableSpec(
        name="herbivore_b", var_type="derived",
        base_value=25.0, observation_noise_std=0.3,
        clip_min=0,
        description="Herbivore species B",
    ))
    dag.add_variable(VariableSpec(
        name="keystone_predator", var_type="derived",
        base_value=0.3, observation_noise_std=0.15,
        clip_min=0, clip_max=1,
        description="Keystone predator presence probability",
    ))
    dag.add_variable(VariableSpec(
        name="predator_presence", var_type="derived",
        base_value=0.4, observation_noise_std=0.15,
        clip_min=0, clip_max=1,
        description="General predator presence probability",
    ))
    dag.add_variable(VariableSpec(
        name="bird_richness", var_type="derived",
        base_value=15.0, observation_noise_std=0.3,
        clip_min=0,
        description="Bird species richness",
    ))
    dag.add_variable(VariableSpec(
        name="body_size", var_type="derived",
        base_value=50.0, observation_noise_std=0.3,
        clip_min=5,
        description="Average body size",
    ))
    dag.add_variable(VariableSpec(
        name="species_diversity", var_type="derived",
        base_value=3.0, observation_noise_std=0.3,
        clip_min=0,
        description="Shannon diversity index",
    ))
    dag.add_variable(VariableSpec(
        name="decomposition_rate", var_type="derived",
        base_value=5.0, observation_noise_std=0.3,
        clip_min=0,
        description="Litter decomposition rate (g/month)",
    ))

    # ===== RULES =====

    # --- Climate (prior knowledge) ---
    dag.add_rule(LinearCausalRule(
        id="elev_temp", sources=["elevation"], target="temperature",
        strength=-0.7, noise_std=0.05,
        description="Temperature decreases with elevation",
        ecological_principle="Adiabatic lapse rate",
    ))
    dag.add_rule(LinearCausalRule(
        id="lat_temp", sources=["latitude"], target="temperature",
        strength=-0.5, noise_std=0.05,
        description="Higher latitudes are colder",
        ecological_principle="Solar radiation gradient",
    ))

    # --- Vegetation: multiple drivers ---
    dag.add_rule(LinearCausalRule(
        id="precip_veg", sources=["precipitation"], target="vegetation_density",
        strength=0.5, noise_std=0.08,
        description="Precipitation drives vegetation",
        ecological_principle="Water availability",
    ))
    dag.add_rule(LinearCausalRule(
        id="temp_veg", sources=["temperature"], target="vegetation_density",
        strength=0.25, noise_std=0.08,
        description="Warmer temps support more vegetation",
        ecological_principle="Growing season",
    ))
    dag.add_rule(InteractionRule(
        id="precip_temp_veg", sources=["precipitation", "temperature"],
        target="vegetation_density",
        strength=0.3, noise_std=0.05,
        description="Precipitation × temperature interaction on vegetation",
        ecological_principle="Water-energy dynamics",
    ))
    dag.add_rule(LinearCausalRule(
        id="nitrogen_veg", sources=["soil_nitrogen"], target="vegetation_density",
        strength=0.35, noise_std=0.08,
        description="Soil nitrogen promotes vegetation growth",
        ecological_principle="Nutrient limitation",
    ))
    dag.add_rule(ModulationRule(
        id="season_veg", sources=["temperature", "season_indicator"],
        target="vegetation_density",
        strength=0.2, noise_std=0.05,
        modulator_center=0.5, modulator_scale=0.2,
        description="Seasonal warmth modulates temperature effect on vegetation",
        ecological_principle="Phenological timing",
    ))

    # --- Canopy and forest structure ---
    dag.add_rule(ThresholdCausalRule(
        id="elev_treeline", sources=["elevation"], target="canopy_cover",
        strength=-0.8, noise_std=0.05,
        threshold=2500, above=True,
        description="Canopy cover drops above treeline",
        ecological_principle="Alpine treeline",
    ))
    dag.add_rule(LinearCausalRule(
        id="veg_canopy", sources=["vegetation_density"], target="canopy_cover",
        strength=0.5, noise_std=0.08,
        description="Vegetation density supports canopy",
        ecological_principle="Forest structure",
    ))

    # --- Root biomass (saturation) ---
    dag.add_rule(SaturationRule(
        id="veg_root_sat", sources=["vegetation_density"], target="root_biomass",
        strength=0.6, noise_std=0.08,
        capacity=80.0,
        description="Root biomass saturates as vegetation density increases",
        ecological_principle="Carrying capacity / diminishing returns",
    ))
    dag.add_rule(LinearCausalRule(
        id="nitrogen_root", sources=["soil_nitrogen"], target="root_biomass",
        strength=0.3, noise_std=0.08,
        description="Soil nitrogen enhances root growth",
        ecological_principle="Below-ground nutrient allocation",
    ))

    # --- Insects ---
    dag.add_rule(LinearCausalRule(
        id="precip_insect", sources=["precipitation"], target="insect_abundance",
        strength=0.4, noise_std=0.1,
        description="Precipitation supports insects",
        ecological_principle="Moisture",
    ))
    dag.add_rule(LinearCausalRule(
        id="veg_insect", sources=["vegetation_density"], target="insect_abundance",
        strength=0.35, noise_std=0.1,
        description="Vegetation supports insect populations",
        ecological_principle="Habitat and food",
    ))

    # --- Herbivores with carrying capacity ---
    dag.add_rule(SaturationRule(
        id="veg_herb_a_sat", sources=["vegetation_density"], target="herbivore_a",
        strength=0.55, noise_std=0.1,
        capacity=70.0,
        description="Herbivore A population saturates with vegetation",
        ecological_principle="Carrying capacity",
    ))
    dag.add_rule(LinearCausalRule(
        id="veg_herb_b", sources=["vegetation_density"], target="herbivore_b",
        strength=0.45, noise_std=0.1,
        description="Vegetation supports herbivore B",
        ecological_principle="Bottom-up control",
    ))
    dag.add_rule(LinearCausalRule(
        id="water_herb_a", sources=["water_proximity"], target="herbivore_a",
        strength=-0.3, noise_std=0.08,
        description="Far from water reduces herbivore A",
        ecological_principle="Water dependence",
    ))

    # --- Keystone predator ---
    dag.add_rule(LinearCausalRule(
        id="herb_a_keystone", sources=["herbivore_a"], target="keystone_predator",
        strength=0.5, noise_std=0.08,
        description="Herbivore A abundance attracts keystone predator",
        ecological_principle="Prey-dependent predator establishment",
    ))

    # --- Competition modified by keystone ---
    dag.add_rule(CompetitionRule(
        id="herb_competition", sources=["herbivore_a", "keystone_predator"],
        target="herbivore_b",
        strength=0.5, noise_std=0.08,
        modifier_weakening=0.7,
        description="Herbivore A competes with B; keystone predator weakens competition (trophic cascade)",
        ecological_principle="Keystone predation reduces dominant competitor",
    ))

    # --- General predator ---
    dag.add_rule(LinearCausalRule(
        id="herb_pred", sources=["herbivore_a"], target="predator_presence",
        strength=0.4, noise_std=0.08,
        description="Herbivore abundance supports predator presence",
        ecological_principle="Predator-prey dynamics",
    ))
    dag.add_rule(LinearCausalRule(
        id="keystone_pred", sources=["keystone_predator"], target="predator_presence",
        strength=0.3, noise_std=0.05,
        description="Keystone predator presence boosts general predator index",
        ecological_principle="Predator guild facilitation",
    ))

    # --- Bird richness ---
    dag.add_rule(LinearCausalRule(
        id="canopy_bird", sources=["canopy_cover"], target="bird_richness",
        strength=0.45, noise_std=0.08,
        description="Canopy cover supports bird species richness",
        ecological_principle="Habitat structural complexity",
    ))
    dag.add_rule(LinearCausalRule(
        id="insect_bird", sources=["insect_abundance"], target="bird_richness",
        strength=0.3, noise_std=0.1,
        description="Insect abundance feeds insectivorous birds",
        ecological_principle="Food web",
    ))

    # --- Body size (Bergmann's rule) ---
    dag.add_rule(LinearCausalRule(
        id="temp_bodysize", sources=["temperature"], target="body_size",
        strength=-0.4, noise_std=0.1,
        description="Colder temperatures favor larger body size",
        ecological_principle="Bergmann's rule",
    ))

    # --- Species diversity (weak, hard to detect) ---
    dag.add_rule(LinearCausalRule(
        id="veg_diversity", sources=["vegetation_density"], target="species_diversity",
        strength=0.2, noise_std=0.12,
        description="Vegetation density weakly promotes species diversity",
        ecological_principle="Habitat heterogeneity",
    ))
    dag.add_rule(ThresholdCausalRule(
        id="disturbance_diversity", sources=["disturbance_history"],
        target="species_diversity",
        strength=-0.35, noise_std=0.1,
        threshold=0.5, above=True,
        description="Recent disturbance reduces diversity",
        ecological_principle="Intermediate disturbance hypothesis (extreme end)",
    ))

    # --- Decomposition ---
    dag.add_rule(LinearCausalRule(
        id="temp_decomp", sources=["temperature"], target="decomposition_rate",
        strength=0.4, noise_std=0.08,
        description="Warmer temperatures increase decomposition",
        ecological_principle="Temperature-dependent microbial activity",
    ))
    dag.add_rule(LinearCausalRule(
        id="precip_decomp", sources=["precipitation"], target="decomposition_rate",
        strength=0.25, noise_std=0.1,
        description="Moisture aids decomposition (weak effect)",
        ecological_principle="Moisture for microbial activity",
    ))

    return RuleSet(
        name="gol_hard",
        description="Multi-Biome Complex — saturation, keystone species, weak effects, thresholds",
        dag=dag,
        n_samples=10000,
        seed=42,
        global_noise_scale=1.0,
        correlation_threshold=0.10,
        prior_rule_ids=["elev_temp", "lat_temp"],
    )


# =============================================================================
# EXTREME — Cross-Biome Trophic Web
# =============================================================================

def make_extreme_preset() -> RuleSet:
    """
    Extreme preset: Cross-Biome Trophic Web.

    ~30 variables, ~46 rules, 18000 samples.
    Three interacting ecosystems (aquatic, grassland, forest) connected
    through shared climate drivers, migratory species (herons), and
    hydrological coupling.

    Designed to stress-test discovery agents with:
    - Deep mediation chains (5+ edges)
    - Cross-biome confound forks
    - Multiple confound hubs (temperature, precipitation, nitrogen each
      drive 4-5 children)
    - All 6 rule types used extensively
    - Weak signals (low strength, high noise)
    - Decoy correlations (strong surface correlations between unlinked
      variables driven by shared confounds)
    """
    dag = CausalDAG()

    # ===== EXOGENOUS VARIABLES (8) =====

    dag.add_variable(VariableSpec(
        name="elevation", var_type="exogenous",
        distribution="uniform", distribution_params={"low": 0, "high": 5000},
        description="Elevation in meters",
    ))
    dag.add_variable(VariableSpec(
        name="latitude", var_type="exogenous",
        distribution="uniform", distribution_params={"low": 10, "high": 70},
        description="Latitude in degrees",
    ))
    dag.add_variable(VariableSpec(
        name="annual_precipitation", var_type="exogenous",
        distribution="normal", distribution_params={"mean": 1100, "std": 500},
        clip_min=50, clip_max=5000,
        description="Annual precipitation in mm",
    ))
    dag.add_variable(VariableSpec(
        name="water_proximity", var_type="exogenous",
        distribution="exponential", distribution_params={"scale": 8.0},
        clip_min=0, clip_max=80,
        description="Distance to permanent water in km",
    ))
    dag.add_variable(VariableSpec(
        name="season_indicator", var_type="exogenous",
        distribution="uniform", distribution_params={"low": 0, "high": 1},
        description="Seasonal warmth indicator (0=winter, 1=summer)",
    ))
    dag.add_variable(VariableSpec(
        name="soil_nitrogen", var_type="exogenous",
        distribution="lognormal", distribution_params={"mean": 1.5, "sigma": 0.8},
        clip_min=0, clip_max=100,
        description="Soil nitrogen content (mg/kg)",
    ))
    dag.add_variable(VariableSpec(
        name="disturbance_history", var_type="exogenous",
        distribution="bernoulli", distribution_params={"p": 0.25},
        dtype="bool",
        description="Recent fire or logging disturbance (True/False)",
    ))
    dag.add_variable(VariableSpec(
        name="lake_depth", var_type="exogenous",
        distribution="lognormal", distribution_params={"mean": 1.0, "sigma": 0.7},
        clip_min=0.5, clip_max=50,
        description="Depth of nearest lake in meters",
    ))

    # ===== DERIVED: CLIMATE & ABIOTIC (3) =====

    dag.add_variable(VariableSpec(
        name="temperature", var_type="derived",
        base_value=18.0, observation_noise_std=0.35,
        clip_min=-40, clip_max=55,
        description="Mean annual temperature in Celsius",
    ))
    dag.add_variable(VariableSpec(
        name="soil_moisture", var_type="derived",
        base_value=40.0, observation_noise_std=0.35,
        clip_min=0, clip_max=100,
        description="Soil moisture index (%)",
    ))
    dag.add_variable(VariableSpec(
        name="water_clarity", var_type="derived",
        base_value=50.0, observation_noise_std=0.35,
        clip_min=0, clip_max=100,
        description="Lake water clarity (Secchi depth proxy)",
    ))

    # ===== DERIVED: PRODUCERS / PRIMARY HABITAT (5) =====

    dag.add_variable(VariableSpec(
        name="vegetation_density", var_type="derived",
        base_value=50.0, observation_noise_std=0.35,
        clip_min=0, clip_max=100,
        description="Terrestrial vegetation density index",
    ))
    dag.add_variable(VariableSpec(
        name="canopy_cover", var_type="derived",
        base_value=40.0, observation_noise_std=0.35,
        clip_min=0, clip_max=100,
        description="Forest canopy cover (%)",
    ))
    dag.add_variable(VariableSpec(
        name="grass_biomass", var_type="derived",
        base_value=60.0, observation_noise_std=0.35,
        clip_min=0, clip_max=200,
        description="Grassland above-ground biomass (g/m2)",
    ))
    dag.add_variable(VariableSpec(
        name="aquatic_algae", var_type="derived",
        base_value=30.0, observation_noise_std=0.35,
        clip_min=0, clip_max=100,
        description="Algal density in lake (chlorophyll-a proxy)",
    ))
    dag.add_variable(VariableSpec(
        name="root_biomass", var_type="derived",
        base_value=200.0, observation_noise_std=0.35,
        clip_min=0,
        description="Below-ground root biomass (g/m2)",
    ))

    # ===== DERIVED: PRIMARY CONSUMERS / INVERTEBRATES (5) =====

    dag.add_variable(VariableSpec(
        name="insect_abundance", var_type="derived",
        base_value=120.0, observation_noise_std=0.35,
        clip_min=0,
        description="Terrestrial insect abundance",
    ))
    dag.add_variable(VariableSpec(
        name="zooplankton_density", var_type="derived",
        base_value=50.0, observation_noise_std=0.35,
        clip_min=0,
        description="Lake zooplankton density",
    ))
    dag.add_variable(VariableSpec(
        name="herbivore_grazer", var_type="derived",
        base_value=30.0, observation_noise_std=0.35,
        clip_min=0,
        description="Grassland grazer abundance (e.g. ungulates)",
    ))
    dag.add_variable(VariableSpec(
        name="herbivore_browser", var_type="derived",
        base_value=25.0, observation_noise_std=0.35,
        clip_min=0,
        description="Forest browser abundance (e.g. deer)",
    ))
    dag.add_variable(VariableSpec(
        name="aquatic_invertebrate", var_type="derived",
        base_value=80.0, observation_noise_std=0.35,
        clip_min=0,
        description="Benthic macroinvertebrate abundance",
    ))

    # ===== DERIVED: SECONDARY CONSUMERS / HIGHER TROPHIC (5) =====

    dag.add_variable(VariableSpec(
        name="fish_abundance", var_type="derived",
        base_value=40.0, observation_noise_std=0.3,
        clip_min=0,
        description="Lake fish abundance",
    ))
    dag.add_variable(VariableSpec(
        name="bird_richness", var_type="derived",
        base_value=15.0, observation_noise_std=0.35,
        clip_min=0,
        description="Terrestrial bird species richness",
    ))
    dag.add_variable(VariableSpec(
        name="raptor_presence", var_type="derived",
        base_value=0.3, observation_noise_std=0.15,
        clip_min=0, clip_max=1,
        description="Raptor/apex predator presence (probability)",
    ))
    dag.add_variable(VariableSpec(
        name="keystone_predator", var_type="derived",
        base_value=0.3, observation_noise_std=0.15,
        clip_min=0, clip_max=1,
        description="Keystone terrestrial predator presence (probability)",
    ))
    dag.add_variable(VariableSpec(
        name="heron_abundance", var_type="derived",
        base_value=10.0, observation_noise_std=0.3,
        clip_min=0,
        description="Wading bird (heron) abundance — cross-biome species",
    ))

    # ===== DERIVED: ECOSYSTEM-LEVEL / EMERGENT (4) =====

    dag.add_variable(VariableSpec(
        name="species_diversity", var_type="derived",
        base_value=3.0, observation_noise_std=0.35,
        clip_min=0,
        description="Shannon diversity index",
    ))
    dag.add_variable(VariableSpec(
        name="decomposition_rate", var_type="derived",
        base_value=5.0, observation_noise_std=0.35,
        clip_min=0,
        description="Litter decomposition rate (g/month)",
    ))
    dag.add_variable(VariableSpec(
        name="body_size_index", var_type="derived",
        base_value=50.0, observation_noise_std=0.35,
        clip_min=5,
        description="Average vertebrate body size (g)",
    ))
    dag.add_variable(VariableSpec(
        name="nutrient_cycling_rate", var_type="derived",
        base_value=10.0, observation_noise_std=0.35,
        clip_min=0,
        description="Nitrogen cycling rate (mg/kg/month)",
    ))

    # ===== RULES (46) =====

    # --- Block A: Climate (3 rules — 2 are prior knowledge) ---
    dag.add_rule(LinearCausalRule(
        id="elev_temp", sources=["elevation"], target="temperature",
        strength=-0.7, noise_std=0.05,
        description="Temperature decreases with elevation (lapse rate ~6.5°C/km)",
        ecological_principle="Adiabatic lapse rate",
    ))
    dag.add_rule(LinearCausalRule(
        id="lat_temp", sources=["latitude"], target="temperature",
        strength=-0.5, noise_std=0.05,
        description="Higher latitudes are colder",
        ecological_principle="Solar radiation gradient",
    ))
    dag.add_rule(LinearCausalRule(
        id="precip_soilmoist", sources=["annual_precipitation"], target="soil_moisture",
        strength=0.55, noise_std=0.08,
        description="More rainfall increases soil moisture",
        ecological_principle="Hydrological cycle",
    ))

    # --- Block B: Soil & Water Abiotic (3 rules) ---
    dag.add_rule(LinearCausalRule(
        id="temp_soilmoist", sources=["temperature"], target="soil_moisture",
        strength=-0.25, noise_std=0.08,
        description="Warmer temperatures increase evaporation, reducing soil moisture",
        ecological_principle="Evapotranspiration",
    ))
    dag.add_rule(LinearCausalRule(
        id="lake_clarity", sources=["lake_depth"], target="water_clarity",
        strength=0.4, noise_std=0.1,
        description="Deeper lakes tend to have clearer water",
        ecological_principle="Limnology — depth stratification",
    ))
    dag.add_rule(LinearCausalRule(
        id="precip_clarity", sources=["annual_precipitation"], target="water_clarity",
        strength=-0.3, noise_std=0.1,
        description="High precipitation increases runoff, reducing water clarity",
        ecological_principle="Sediment transport",
    ))

    # --- Block C: Terrestrial Vegetation (8 rules) ---
    dag.add_rule(LinearCausalRule(
        id="precip_veg", sources=["annual_precipitation"], target="vegetation_density",
        strength=0.45, noise_std=0.08,
        description="Precipitation drives vegetation density",
        ecological_principle="Water availability for plant growth",
    ))
    dag.add_rule(LinearCausalRule(
        id="temp_veg", sources=["temperature"], target="vegetation_density",
        strength=0.25, noise_std=0.08,
        description="Warmer temperatures support more vegetation",
        ecological_principle="Growing season length",
    ))
    dag.add_rule(InteractionRule(
        id="precip_temp_veg", sources=["annual_precipitation", "temperature"],
        target="vegetation_density",
        strength=0.3, noise_std=0.05,
        description="Precipitation × temperature synergy on vegetation",
        ecological_principle="Water-energy dynamics",
    ))
    dag.add_rule(ModulationRule(
        id="season_temp_veg", sources=["temperature", "season_indicator"],
        target="vegetation_density",
        strength=0.2, noise_std=0.05,
        modulator_center=0.5, modulator_scale=0.2,
        description="Seasonal warmth modulates temperature effect on vegetation",
        ecological_principle="Phenological timing",
    ))
    dag.add_rule(LinearCausalRule(
        id="nitrogen_veg", sources=["soil_nitrogen"], target="vegetation_density",
        strength=0.3, noise_std=0.08,
        description="Soil nitrogen promotes vegetation growth",
        ecological_principle="Nutrient limitation",
    ))
    dag.add_rule(ThresholdCausalRule(
        id="elev_treeline", sources=["elevation"], target="canopy_cover",
        strength=-0.8, noise_std=0.05,
        threshold=2500, above=True,
        description="Above treeline (~2500m), canopy cover drops sharply",
        ecological_principle="Alpine treeline",
    ))
    dag.add_rule(LinearCausalRule(
        id="veg_canopy", sources=["vegetation_density"], target="canopy_cover",
        strength=0.5, noise_std=0.08,
        description="Higher vegetation density correlates with canopy cover",
        ecological_principle="Forest structure",
    ))
    dag.add_rule(LinearCausalRule(
        id="soilmoist_grass", sources=["soil_moisture"], target="grass_biomass",
        strength=0.5, noise_std=0.08,
        description="Soil moisture drives grassland biomass",
        ecological_principle="Grassland hydrology",
    ))

    # --- Block D: Grassland Subsystem (2 rules) ---
    dag.add_rule(LinearCausalRule(
        id="nitrogen_grass", sources=["soil_nitrogen"], target="grass_biomass",
        strength=0.35, noise_std=0.08,
        description="Nitrogen fertilizes grassland",
        ecological_principle="Nutrient enrichment",
    ))
    dag.add_rule(ThresholdCausalRule(
        id="disturbance_grass", sources=["disturbance_history"], target="grass_biomass",
        strength=-0.4, noise_std=0.1,
        threshold=0.5, above=True,
        description="Recent disturbance reduces grass biomass",
        ecological_principle="Fire/grazing disturbance history",
    ))

    # --- Block E: Aquatic Subsystem (4 rules) ---
    dag.add_rule(LinearCausalRule(
        id="clarity_algae", sources=["water_clarity"], target="aquatic_algae",
        strength=-0.35, noise_std=0.1,
        description="Clearer water indicates less algae (inverse eutrophication signal)",
        ecological_principle="Trophic state index",
    ))
    dag.add_rule(LinearCausalRule(
        id="temp_algae", sources=["temperature"], target="aquatic_algae",
        strength=0.4, noise_std=0.08,
        description="Warmer water promotes algal blooms",
        ecological_principle="Temperature-dependent algal growth",
    ))
    dag.add_rule(SaturationRule(
        id="nitrogen_algae", sources=["soil_nitrogen"], target="aquatic_algae",
        strength=0.5, noise_std=0.08,
        capacity=60.0,
        description="Nitrogen runoff fuels algae with diminishing returns",
        ecological_principle="Nutrient loading and eutrophication",
    ))
    dag.add_rule(InteractionRule(
        id="precip_nitrogen_algae", sources=["annual_precipitation", "soil_nitrogen"],
        target="aquatic_algae",
        strength=0.25, noise_std=0.08,
        description="Precipitation washes nitrogen into lake (precip × nitrogen interaction)",
        ecological_principle="Nutrient transport via runoff",
    ))

    # --- Block F: Root Biomass (2 rules — saturation) ---
    dag.add_rule(SaturationRule(
        id="veg_root_sat", sources=["vegetation_density"], target="root_biomass",
        strength=0.6, noise_std=0.08,
        capacity=80.0,
        description="Root biomass saturates as vegetation density increases",
        ecological_principle="Carrying capacity / diminishing returns",
    ))
    dag.add_rule(LinearCausalRule(
        id="nitrogen_root", sources=["soil_nitrogen"], target="root_biomass",
        strength=0.25, noise_std=0.08,
        description="Soil nitrogen enhances root growth",
        ecological_principle="Below-ground nutrient allocation",
    ))

    # --- Block G: Invertebrates (4 rules) ---
    dag.add_rule(LinearCausalRule(
        id="precip_insect", sources=["annual_precipitation"], target="insect_abundance",
        strength=0.35, noise_std=0.1,
        description="Precipitation supports insect populations",
        ecological_principle="Moisture-dependent insect life cycles",
    ))
    dag.add_rule(LinearCausalRule(
        id="veg_insect", sources=["vegetation_density"], target="insect_abundance",
        strength=0.35, noise_std=0.1,
        description="Vegetation supports insect populations",
        ecological_principle="Habitat and food availability",
    ))
    dag.add_rule(SaturationRule(
        id="algae_zooplankton", sources=["aquatic_algae"], target="zooplankton_density",
        strength=0.55, noise_std=0.08,
        capacity=70.0,
        description="Algae feed zooplankton with diminishing returns",
        ecological_principle="Aquatic food web — grazer saturation",
    ))
    dag.add_rule(LinearCausalRule(
        id="clarity_invertebrate", sources=["water_clarity"],
        target="aquatic_invertebrate",
        strength=0.4, noise_std=0.1,
        description="Clear water supports benthic invertebrates",
        ecological_principle="Substrate visibility and dissolved oxygen",
    ))

    # --- Block H: Herbivores with Competition (5 rules) ---
    dag.add_rule(SaturationRule(
        id="grass_grazer_sat", sources=["grass_biomass"], target="herbivore_grazer",
        strength=0.55, noise_std=0.1,
        capacity=65.0,
        description="Grazer population saturates with grass biomass",
        ecological_principle="Carrying capacity",
    ))
    dag.add_rule(LinearCausalRule(
        id="water_grazer", sources=["water_proximity"], target="herbivore_grazer",
        strength=-0.3, noise_std=0.08,
        description="Greater distance from water reduces grazer abundance",
        ecological_principle="Water dependence of large herbivores",
    ))
    dag.add_rule(LinearCausalRule(
        id="veg_browser", sources=["vegetation_density"], target="herbivore_browser",
        strength=0.45, noise_std=0.1,
        description="Vegetation density supports forest browsers",
        ecological_principle="Bottom-up trophic control",
    ))
    dag.add_rule(CompetitionRule(
        id="grazer_browser_competition",
        sources=["herbivore_grazer", "water_proximity"],
        target="herbivore_browser",
        strength=0.4, noise_std=0.08,
        modifier_weakening=0.6,
        description="Grazers compete with browsers; water proximity weakens competition via resource partitioning",
        ecological_principle="Interspecific competition + niche partitioning",
    ))
    dag.add_rule(ThresholdCausalRule(
        id="disturbance_browser", sources=["disturbance_history"],
        target="herbivore_browser",
        strength=-0.35, noise_std=0.1,
        threshold=0.5, above=True,
        description="Disturbance drives browsers away from affected habitat",
        ecological_principle="Habitat sensitivity",
    ))

    # --- Block I: Keystone Predator and Trophic Cascade (3 rules) ---
    dag.add_rule(LinearCausalRule(
        id="grazer_keystone", sources=["herbivore_grazer"], target="keystone_predator",
        strength=0.5, noise_std=0.08,
        description="Grazer abundance attracts keystone predator",
        ecological_principle="Prey-dependent predator establishment",
    ))
    dag.add_rule(CompetitionRule(
        id="keystone_cascade",
        sources=["herbivore_grazer", "keystone_predator"],
        target="herbivore_browser",
        strength=0.3, noise_std=0.08,
        modifier_weakening=0.7,
        description="Keystone predator suppresses grazers, indirectly benefiting browsers (trophic cascade)",
        ecological_principle="Keystone predation reduces dominant competitor",
    ))
    dag.add_rule(LinearCausalRule(
        id="grazer_raptor", sources=["herbivore_grazer"], target="raptor_presence",
        strength=0.4, noise_std=0.08,
        description="Prey abundance supports raptor presence",
        ecological_principle="Predator-prey dynamics",
    ))

    # --- Block J: Aquatic Predators and Cross-Biome Link (3 rules) ---
    dag.add_rule(LinearCausalRule(
        id="zooplankton_fish", sources=["zooplankton_density"], target="fish_abundance",
        strength=0.5, noise_std=0.08,
        description="Zooplankton feed fish",
        ecological_principle="Aquatic food chain",
    ))
    dag.add_rule(InteractionRule(
        id="invertebrate_clarity_fish",
        sources=["aquatic_invertebrate", "water_clarity"],
        target="fish_abundance",
        strength=0.3, noise_std=0.1,
        description="Fish catch benthic invertebrates better in clear water (invertebrate × clarity)",
        ecological_principle="Visual predation efficiency",
    ))
    dag.add_rule(CompetitionRule(
        id="fish_heron_depth",
        sources=["fish_abundance", "lake_depth"],
        target="heron_abundance",
        strength=-0.5, noise_std=0.08,
        modifier_weakening=0.5,
        description="Fish attract herons, but deep lakes weaken the link (wading limitation)",
        ecological_principle="Piscivore foraging constraints",
    ))

    # --- Block K: Birds and Community (3 rules) ---
    dag.add_rule(LinearCausalRule(
        id="canopy_bird", sources=["canopy_cover"], target="bird_richness",
        strength=0.45, noise_std=0.08,
        description="Canopy cover supports bird species richness",
        ecological_principle="Habitat structural complexity",
    ))
    dag.add_rule(ModulationRule(
        id="insect_season_bird",
        sources=["insect_abundance", "season_indicator"],
        target="bird_richness",
        strength=0.3, noise_std=0.1,
        modulator_center=0.5, modulator_scale=0.2,
        description="Insect-dependent bird richness modulated by season (migratory birds in summer)",
        ecological_principle="Seasonal migration and food availability",
    ))
    dag.add_rule(ModulationRule(
        id="heron_canopy_bird",
        sources=["canopy_cover", "heron_abundance"],
        target="bird_richness",
        strength=0.2, noise_std=0.08,
        modulator_center=10.0, modulator_scale=5.0,
        description="Heron presence modulates canopy benefit on birds (wetland-forest ecotone effect)",
        ecological_principle="Ecotone species diversity boost",
    ))

    # --- Block L: Ecosystem-Level Quantities (6 rules) ---
    dag.add_rule(LinearCausalRule(
        id="temp_bodysize", sources=["temperature"], target="body_size_index",
        strength=-0.4, noise_std=0.1,
        description="Colder temperatures favor larger body size",
        ecological_principle="Bergmann's rule — thermoregulation",
    ))
    dag.add_rule(LinearCausalRule(
        id="veg_diversity", sources=["vegetation_density"], target="species_diversity",
        strength=0.18, noise_std=0.14,
        description="Vegetation density weakly promotes species diversity (weak, noisy signal)",
        ecological_principle="Habitat heterogeneity",
    ))
    dag.add_rule(ThresholdCausalRule(
        id="disturbance_diversity", sources=["disturbance_history"],
        target="species_diversity",
        strength=-0.3, noise_std=0.12,
        threshold=0.5, above=True,
        description="Recent disturbance reduces species diversity",
        ecological_principle="Intermediate disturbance hypothesis (extreme end)",
    ))
    dag.add_rule(ModulationRule(
        id="temp_moisture_decomp",
        sources=["temperature", "soil_moisture"],
        target="decomposition_rate",
        strength=0.4, noise_std=0.08,
        modulator_center=40.0, modulator_scale=15.0,
        description="Temperature drives decomposition, modulated by soil moisture",
        ecological_principle="Moisture-temperature synergy on microbial activity",
    ))
    dag.add_rule(LinearCausalRule(
        id="root_nutrient", sources=["root_biomass"], target="nutrient_cycling_rate",
        strength=0.4, noise_std=0.1,
        description="Root biomass drives nutrient cycling via exudates and mycorrhizae",
        ecological_principle="Rhizosphere nutrient cycling",
    ))
    dag.add_rule(InteractionRule(
        id="decomp_root_nutrient",
        sources=["decomposition_rate", "root_biomass"],
        target="nutrient_cycling_rate",
        strength=0.35, noise_std=0.1,
        description="Decomposition and root biomass synergistically accelerate nutrient cycling",
        ecological_principle="Above-below ground nutrient feedback",
    ))

    return RuleSet(
        name="gol_extreme",
        description=(
            "Cross-Biome Trophic Web — deep chains, cross-biome coupling, "
            "Simpson's structures, decoy correlations"
        ),
        dag=dag,
        n_samples=18000,
        seed=42,
        global_noise_scale=1.0,
        correlation_threshold=0.08,
        prior_rule_ids=["elev_temp", "lat_temp", "precip_veg"],
    )


# =============================================================================
# Preset registry
# =============================================================================

PRESET_BUILDERS = {
    "easy": make_easy_preset,
    "medium": make_medium_preset,
    "hard": make_hard_preset,
    "extreme": make_extreme_preset,
}


def get_preset(name: str) -> RuleSet:
    """Get a preset RuleSet by name."""
    if name not in PRESET_BUILDERS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(PRESET_BUILDERS.keys())}")
    return PRESET_BUILDERS[name]()
