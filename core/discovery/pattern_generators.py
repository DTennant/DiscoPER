# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Pattern generators for synthetic benchmark datasets.

Functions to inject various statistical patterns into synthetic data
with controllable effect sizes for benchmarking discovery agents.
"""

from typing import Any, Optional
import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _standardize(x: np.ndarray) -> np.ndarray:
    """Standardize array to mean 0, std 1."""
    std = np.std(x)
    if std == 0:
        return x - np.mean(x)
    return (x - np.mean(x)) / std


def _create_correlated_var(
    base: np.ndarray,
    r: float,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Create a variable with specified correlation to a base variable.
    
    Args:
        base: Base variable to correlate with
        r: Target correlation coefficient (-1 to 1)
        rng: Random number generator
        
    Returns:
        New variable with approximately correlation r to base
    """
    n = len(base)
    base_std = _standardize(base)
    noise = rng.standard_normal(n)
    
    # Create correlated variable: r * base + sqrt(1-r^2) * noise
    r_clipped = np.clip(r, -0.999, 0.999)
    result = r_clipped * base_std + np.sqrt(1 - r_clipped**2) * noise
    
    return result


# =============================================================================
# BASE FEATURE GENERATORS
# =============================================================================

def generate_base_environmental_features(
    n_samples: int,
    rng: np.random.Generator
) -> pd.DataFrame:
    """
    Generate base environmental features for ecology dataset.
    
    Features: latitude, longitude, elevation, temperature, precipitation, canopy_cover
    """
    df = pd.DataFrame()
    
    # Geographic coordinates
    df["latitude"] = rng.uniform(25, 65, n_samples)  # Temperate to subarctic
    df["longitude"] = rng.uniform(-130, -60, n_samples)  # North America
    
    # Elevation (correlated with latitude - higher latitudes tend to have mountains)
    df["elevation"] = np.clip(
        500 + 2000 * ((df["latitude"] - 25) / 40) + rng.normal(0, 500, n_samples),
        0, 4000
    )
    
    # Temperature (strongly correlated with latitude and elevation - this is prior knowledge)
    # Lapse rate: ~6.5°C per 1000m
    df["temperature"] = (
        30  # Base temp at equator level
        - 0.7 * (df["latitude"] - 25)  # Latitude effect
        - 0.0065 * df["elevation"]  # Elevation effect (lapse rate)
        + rng.normal(0, 3, n_samples)  # Noise
    )
    
    # Precipitation (varies with latitude and longitude)
    df["precipitation"] = np.clip(
        100 + 50 * np.sin((df["latitude"] - 35) * np.pi / 30)  # Latitude bands
        + 30 * np.cos((df["longitude"] + 95) * np.pi / 35)  # West-east gradient
        + rng.normal(0, 30, n_samples),
        5, 300
    )
    
    # Canopy cover (correlated with precipitation and temperature)
    df["canopy_cover"] = np.clip(
        0.3 + 0.003 * df["precipitation"] + 0.01 * np.clip(df["temperature"], 0, 25)
        + rng.normal(0, 0.15, n_samples),
        0, 1
    )
    
    # Vegetation density (correlated with precipitation - prior knowledge)
    df["vegetation_density"] = np.clip(
        0.2 + 0.002 * df["precipitation"] + 0.005 * np.clip(df["temperature"], 5, 30)
        + rng.normal(0, 0.1, n_samples),
        0, 1
    )
    
    return df


def generate_species_features(
    n_samples: int,
    n_species: int,
    env_df: pd.DataFrame,
    rng: np.random.Generator
) -> pd.DataFrame:
    """
    Generate species-related features.
    
    Features: species_id, body_size, wing_length, diet_type, is_nocturnal, 
              taxonomic_class, species_family, body_mass
    """
    df = pd.DataFrame()
    
    # Species assignment
    species_names = [f"species_{i}" for i in range(n_species)]
    df["species_id"] = rng.choice(species_names, n_samples)
    
    # Taxonomic classification (deterministic based on species)
    classes = ["Mammalia", "Aves", "Reptilia"]
    families = [f"family_{i}" for i in range(n_species // 3 + 1)]
    
    species_to_class = {sp: classes[i % len(classes)] for i, sp in enumerate(species_names)}
    species_to_family = {sp: families[i // 3] for i, sp in enumerate(species_names)}
    
    df["taxonomic_class"] = df["species_id"].map(species_to_class)
    df["species_family"] = df["species_id"].map(species_to_family)
    
    # Body size - will be modified by patterns (e.g., Bergmann's rule)
    df["body_size"] = rng.lognormal(mean=3, sigma=0.5, size=n_samples)
    
    # Body mass (correlated with body size)
    df["body_mass"] = df["body_size"] ** 2.5 * rng.uniform(0.8, 1.2, n_samples)
    
    # Wing length (for birds, correlated with body size)
    df["wing_length"] = np.where(
        df["taxonomic_class"] == "Aves",
        df["body_size"] * 2 + rng.normal(0, 2, n_samples),
        0  # Non-birds don't have wings
    )
    
    # Diet type
    diet_types = ["herbivore", "carnivore", "omnivore", "insectivore"]
    df["diet_type"] = rng.choice(diet_types, n_samples)
    
    # Diet diversity (number of food types)
    df["diet_diversity"] = rng.integers(1, 8, n_samples)
    
    # Activity pattern
    df["is_nocturnal"] = rng.choice([0, 1], n_samples, p=[0.7, 0.3])
    
    # Range size (correlated with body size - larger animals have larger ranges)
    df["range_size"] = df["body_size"] * 10 + rng.exponential(50, n_samples)
    
    # Foraging height
    df["foraging_height"] = np.clip(rng.exponential(5, n_samples), 0, 50)
    
    # Prey size (correlated with body size)
    df["prey_size"] = df["body_size"] * 0.1 + rng.exponential(1, n_samples)
    
    # Migration distance (for birds)
    df["migration_distance"] = np.where(
        df["taxonomic_class"] == "Aves",
        rng.exponential(500, n_samples),
        0
    )
    
    return df


def generate_observation_features(
    n_samples: int,
    n_regions: int,
    rng: np.random.Generator
) -> pd.DataFrame:
    """
    Generate observation/sampling-related features.
    
    Features: observer_id, observer_experience, timestamp, season, biogeographic_region
    """
    df = pd.DataFrame()
    
    # Observers
    n_observers = 50
    observer_ids = [f"observer_{i}" for i in range(n_observers)]
    df["observer_id"] = rng.choice(observer_ids, n_samples)
    
    # Observer experience (years, deterministic per observer)
    observer_exp = {obs: rng.integers(1, 30) for obs in observer_ids}
    df["observer_experience"] = df["observer_id"].map(observer_exp)
    
    # Timestamp and season
    start_date = pd.Timestamp("2015-01-01")
    days_offset = rng.integers(0, 365 * 8, n_samples)  # 8 years of data
    df["timestamp"] = start_date + pd.to_timedelta(days_offset, unit="D")
    
    season_map = {1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
                  6: "summer", 7: "summer", 8: "summer", 9: "fall", 10: "fall", 
                  11: "fall", 12: "winter"}
    df["season"] = df["timestamp"].dt.month.map(season_map)
    
    # Biogeographic region
    regions = [f"region_{i}" for i in range(n_regions)]
    df["biogeographic_region"] = rng.choice(regions, n_samples)
    
    # Sampling in biodiversity hotspot (confound variable)
    df["sampling_in_hotspot"] = rng.choice([0, 1], n_samples, p=[0.7, 0.3])
    
    # Climate zone
    climate_zones = ["tropical", "subtropical", "temperate", "boreal", "arctic"]
    df["climate_zone"] = rng.choice(climate_zones, n_samples, p=[0.1, 0.2, 0.4, 0.2, 0.1])
    
    return df


def generate_ecological_outcomes(
    n_samples: int,
    rng: np.random.Generator
) -> pd.DataFrame:
    """
    Generate ecological outcome variables.
    
    Features: abundance, species_richness, species_count, elevation_preference, etc.
    """
    df = pd.DataFrame()
    
    # Abundance (count data)
    df["abundance"] = rng.poisson(lam=20, size=n_samples)
    
    # Species richness / count
    df["species_richness"] = rng.poisson(lam=15, size=n_samples)
    df["species_count"] = df["species_richness"] + rng.integers(-2, 3, n_samples)
    df["species_count"] = np.clip(df["species_count"], 1, None)
    
    # Elevation preference (continuous)
    df["elevation_preference"] = rng.normal(1500, 500, n_samples)
    
    # Elevation use (where actually found)
    df["elevation_use"] = rng.normal(1200, 600, n_samples)
    
    # Bird richness (for mediation chain)
    df["bird_richness"] = rng.poisson(lam=8, size=n_samples)
    
    # Insect abundance
    df["insect_abundance"] = rng.exponential(100, n_samples)
    
    # Prey density
    df["prey_density"] = rng.exponential(50, n_samples)
    
    # Water proximity
    df["water_proximity"] = rng.exponential(5, n_samples)
    
    # Tree presence (binary)
    df["tree_presence"] = rng.choice([0, 1], n_samples, p=[0.3, 0.7])
    
    # Habitat type
    habitat_types = ["forest", "grassland", "wetland", "desert", "tundra"]
    df["habitat_type"] = rng.choice(habitat_types, n_samples, p=[0.3, 0.3, 0.15, 0.15, 0.1])
    
    # Species presence columns (for spatial clustering)
    df["species_A_presence"] = rng.choice([0, 1], n_samples, p=[0.6, 0.4])
    df["species_B_presence"] = rng.choice([0, 1], n_samples, p=[0.5, 0.5])
    df["species_C_presence"] = rng.choice([0, 1], n_samples, p=[0.55, 0.45])
    
    # Keystone species presence
    df["keystone_species_present"] = rng.choice([0, 1], n_samples, p=[0.7, 0.3])
    
    # Species X and Y abundance (for competition)
    df["species_X_abundance"] = rng.poisson(lam=10, size=n_samples)
    df["species_Y_abundance"] = rng.poisson(lam=10, size=n_samples)
    
    # Species diversity
    df["species_diversity"] = rng.uniform(0.3, 0.9, n_samples)
    
    return df


# =============================================================================
# SIMPLE PATTERN INJECTORS
# =============================================================================

def inject_correlation(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    target_r: float,
    rng: np.random.Generator,
    subset_mask: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Inject a correlation between two existing columns.
    
    Args:
        df: DataFrame to modify
        x_col: Independent variable column
        y_col: Dependent variable column (will be modified)
        target_r: Target correlation coefficient
        rng: Random number generator
        subset_mask: Optional boolean mask to apply only to subset
        
    Returns:
        Modified DataFrame
    """
    df = df.copy()
    
    # Convert to float to allow fractional modifications
    df[y_col] = df[y_col].astype(float)
    
    if subset_mask is None:
        subset_mask = np.ones(len(df), dtype=bool)
    
    x = df.loc[subset_mask, x_col].values.astype(float)
    n = len(x)
    
    # Create correlated values
    new_y = _create_correlated_var(x, target_r, rng)
    
    # Scale to match original distribution
    orig_y = df.loc[subset_mask, y_col].values.astype(float)
    new_y = new_y * np.std(orig_y) + np.mean(orig_y)
    
    df.loc[subset_mask, y_col] = new_y
    
    return df


def inject_group_difference(
    df: pd.DataFrame,
    var_col: str,
    group_col: str,
    effect_size: float,
    rng: np.random.Generator,
    group_effects: Optional[dict[str, float]] = None
) -> pd.DataFrame:
    """
    Inject mean differences between groups.
    
    Args:
        df: DataFrame to modify
        var_col: Variable to modify
        group_col: Grouping column
        effect_size: Cohen's d effect size
        rng: Random number generator
        group_effects: Optional dict of group -> effect multiplier
        
    Returns:
        Modified DataFrame
    """
    df = df.copy()
    
    # Convert to float to allow fractional modifications
    original_dtype = df[var_col].dtype
    df[var_col] = df[var_col].astype(float)
    
    var_std = df[var_col].std()
    groups = df[group_col].unique()
    
    if group_effects is None:
        # Create default effects: first half positive, second half negative
        n_groups = len(groups)
        group_effects = {}
        for i, g in enumerate(groups):
            if i < n_groups // 2:
                group_effects[g] = effect_size
            else:
                group_effects[g] = -effect_size
    
    for group, effect in group_effects.items():
        mask = df[group_col] == group
        df.loc[mask, var_col] = df.loc[mask, var_col] + effect * var_std
    
    return df


def inject_threshold(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    threshold: float,
    effect_below: float,
    effect_above: float,
    rng: np.random.Generator,
    condition_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Inject a threshold effect where relationship changes at a cutoff.
    
    Args:
        df: DataFrame to modify
        x_col: Predictor column
        y_col: Outcome column (will be modified)
        threshold: Threshold value
        effect_below: Correlation below threshold
        effect_above: Correlation above threshold
        rng: Random number generator
        condition_col: Optional column to use for threshold (if different from x_col)
        
    Returns:
        Modified DataFrame
    """
    df = df.copy()
    
    # Convert to float to allow fractional modifications
    df[y_col] = df[y_col].astype(float)
    
    threshold_col = condition_col if condition_col else x_col
    
    below_mask = df[threshold_col] < threshold
    above_mask = ~below_mask
    
    # Apply different correlations below and above threshold
    if below_mask.sum() > 10:
        df = inject_correlation(df, x_col, y_col, effect_below, rng, below_mask.values)
    if above_mask.sum() > 10:
        df = inject_correlation(df, x_col, y_col, effect_above, rng, above_mask.values)
    
    return df


def inject_simpson_paradox(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    overall_effect: float,
    within_effect: float,
    rng: np.random.Generator
) -> pd.DataFrame:
    """
    Inject Simpson's paradox: overall correlation is opposite to within-group correlation.
    
    Args:
        df: DataFrame to modify
        x_col: X variable
        y_col: Y variable (will be modified)
        group_col: Grouping column
        overall_effect: Target marginal correlation (e.g., 0.3)
        within_effect: Target within-group correlation (e.g., -0.4, opposite sign)
        rng: Random number generator
        
    Returns:
        Modified DataFrame
    """
    df = df.copy()
    
    # Convert to float to allow fractional modifications
    df[y_col] = df[y_col].astype(float)
    
    groups = df[group_col].unique()
    y_std = df[y_col].std()
    
    # First, set up group means to create the overall positive correlation
    # Groups with high mean X should have high mean Y
    group_x_means = df.groupby(group_col)[x_col].mean().sort_values()
    
    for i, (group, x_mean) in enumerate(group_x_means.items()):
        mask = df[group_col] == group
        # Shift Y based on group X mean to create overall positive correlation
        y_shift = overall_effect * 2 * y_std * (i / len(groups) - 0.5)
        df.loc[mask, y_col] = df.loc[mask, y_col] + y_shift
    
    # Then, within each group, create negative correlation
    for group in groups:
        mask = df[group_col] == group
        if mask.sum() > 10:
            df = inject_correlation(df, x_col, y_col, within_effect, rng, mask.values)
    
    return df


# =============================================================================
# MODERATE PATTERN INJECTORS
# =============================================================================

def inject_two_way_interaction(
    df: pd.DataFrame,
    outcome_col: str,
    x1_col: str,
    x2_col: str,
    main_effect_x1: float,
    main_effect_x2: float,
    interaction_effect: float,
    rng: np.random.Generator,
    subset_filter: Optional[dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Inject a two-way interaction effect: Y = b1*X1 + b2*X2 + b12*X1*X2.
    
    Args:
        df: DataFrame to modify
        outcome_col: Outcome variable
        x1_col, x2_col: Predictor columns
        main_effect_x1, main_effect_x2: Main effects
        interaction_effect: Interaction effect
        rng: Random number generator
        subset_filter: Optional dict of column->value to filter subset
        
    Returns:
        Modified DataFrame
    """
    df = df.copy()
    
    # Convert to float to allow fractional modifications
    df[outcome_col] = df[outcome_col].astype(float)
    
    # Create subset mask
    mask = np.ones(len(df), dtype=bool)
    if subset_filter:
        for col, val in subset_filter.items():
            if col in df.columns:
                mask &= (df[col] == val).values
    
    # Standardize predictors
    x1 = _standardize(df.loc[mask, x1_col].values.astype(float))
    x2 = _standardize(df.loc[mask, x2_col].values.astype(float))
    
    # Compute interaction term
    y_std = df[outcome_col].std()
    interaction_contribution = (
        main_effect_x1 * x1 + 
        main_effect_x2 * x2 + 
        interaction_effect * x1 * x2
    ) * y_std
    
    df.loc[mask, outcome_col] = df.loc[mask, outcome_col].values + interaction_contribution
    
    return df


def inject_conditional_confound(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    confound_col: str,
    apparent_effect: float,
    true_effect: float,
    rng: np.random.Generator
) -> pd.DataFrame:
    """
    Inject a spurious correlation via confounding.
    
    X and Y appear correlated, but it's because confound affects both.
    After controlling for confound, true correlation is much weaker.
    
    Args:
        df: DataFrame to modify
        x_col: First variable
        y_col: Second variable
        confound_col: Confounding variable
        apparent_effect: Observed marginal correlation
        true_effect: True partial correlation after controlling
        rng: Random number generator
        
    Returns:
        Modified DataFrame
    """
    df = df.copy()
    
    # Convert to float to allow fractional modifications
    df[x_col] = df[x_col].astype(float)
    df[y_col] = df[y_col].astype(float)
    
    # Make confound affect both X and Y
    confound = _standardize(df[confound_col].values.astype(float))
    confound_strength = np.sqrt(apparent_effect**2 - true_effect**2) if apparent_effect**2 > true_effect**2 else 0.3
    
    x_std = df[x_col].std()
    y_std = df[y_col].std()
    
    # Add confound effect to both variables
    df[x_col] = df[x_col].values + confound_strength * confound * x_std
    df[y_col] = df[y_col].values + confound_strength * confound * y_std
    
    # Add small true correlation
    if abs(true_effect) > 0.01:
        df = inject_correlation(df, x_col, y_col, true_effect, rng)
    
    return df


def inject_masked_correlation(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    confounder_col: str,
    marginal_effect: float,
    partial_effect: float,
    rng: np.random.Generator
) -> pd.DataFrame:
    """
    Inject a suppression effect: true correlation is hidden by confounder.
    
    X and Y have weak marginal correlation, but strong partial correlation
    after controlling for confounder (suppression).
    
    Args:
        df: DataFrame to modify
        x_col: First variable
        y_col: Second variable
        confounder_col: Suppressor variable
        marginal_effect: Weak observed correlation
        partial_effect: Strong true correlation after controlling
        rng: Random number generator
        
    Returns:
        Modified DataFrame
    """
    df = df.copy()
    
    # Convert to float to allow fractional modifications
    df[x_col] = df[x_col].astype(float)
    df[y_col] = df[y_col].astype(float)
    
    # First, create the true strong correlation between X and Y
    df = inject_correlation(df, x_col, y_col, partial_effect, rng)
    
    # Then, make confounder correlate with both in opposite directions
    # This will suppress the marginal correlation
    confounder = df[confounder_col].values.astype(float)
    x = df[x_col].values
    y = df[y_col].values
    
    # Calculate suppression strength needed
    suppression = partial_effect - marginal_effect
    
    # Adjust confounder's effect
    confounder_std = _standardize(confounder)
    x_std = np.std(x)
    y_std = np.std(y)
    
    # Confounder positively affects X, negatively affects Y (or vice versa)
    df[x_col] = x + suppression * 0.5 * confounder_std * x_std
    df[y_col] = y - suppression * 0.5 * confounder_std * y_std
    
    return df


# =============================================================================
# COMPLEX PATTERN INJECTORS
# =============================================================================

def inject_three_way_interaction(
    df: pd.DataFrame,
    outcome_col: str,
    x1_col: str,
    x2_col: str,
    x3_col: str,
    main_effects: tuple[float, float, float],
    two_way_effects: Optional[dict[str, float]],
    three_way_effect: float,
    rng: np.random.Generator
) -> pd.DataFrame:
    """
    Inject a three-way interaction effect.
    
    Y = b1*X1 + b2*X2 + b3*X3 + b12*X1*X2 + b13*X1*X3 + b23*X2*X3 + b123*X1*X2*X3
    
    Args:
        df: DataFrame to modify
        outcome_col: Outcome variable
        x1_col, x2_col, x3_col: Predictor columns
        main_effects: Tuple of main effects (b1, b2, b3)
        two_way_effects: Dict of two-way effects {"x1_x2": b12, "x1_x3": b13, "x2_x3": b23}
        three_way_effect: Three-way interaction effect (b123)
        rng: Random number generator
        
    Returns:
        Modified DataFrame
    """
    df = df.copy()
    
    # Convert to float to allow fractional modifications
    df[outcome_col] = df[outcome_col].astype(float)
    
    x1 = _standardize(df[x1_col].values.astype(float))
    x2 = _standardize(df[x2_col].values.astype(float))
    x3 = _standardize(df[x3_col].values.astype(float))
    
    y_std = df[outcome_col].std()
    
    # Main effects
    contribution = main_effects[0] * x1 + main_effects[1] * x2 + main_effects[2] * x3
    
    # Two-way effects
    if two_way_effects:
        contribution += two_way_effects.get("x1_x2", 0) * x1 * x2
        contribution += two_way_effects.get("x1_x3", 0) * x1 * x3
        contribution += two_way_effects.get("x2_x3", 0) * x2 * x3
    
    # Three-way effect
    contribution += three_way_effect * x1 * x2 * x3
    
    df[outcome_col] = df[outcome_col].values + contribution * y_std
    
    return df


def inject_mediation_chain(
    df: pd.DataFrame,
    chain_cols: list[str],
    direct_effects: list[float],
    rng: np.random.Generator
) -> pd.DataFrame:
    """
    Inject a mediation chain: A -> B -> C -> D.
    
    Each variable in the chain affects the next.
    
    Args:
        df: DataFrame to modify
        chain_cols: List of column names in causal order [A, B, C, D]
        direct_effects: List of effects between consecutive variables [a->b, b->c, c->d]
        rng: Random number generator
        
    Returns:
        Modified DataFrame
    """
    df = df.copy()
    
    for i in range(len(chain_cols) - 1):
        from_col = chain_cols[i]
        to_col = chain_cols[i + 1]
        effect = direct_effects[i]
        
        # Create correlation from cause to effect
        df = inject_correlation(df, from_col, to_col, effect, rng)
    
    return df


def inject_hierarchical_effect(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hierarchy_col: str,
    effects_by_level: dict[str, float],
    rng: np.random.Generator
) -> pd.DataFrame:
    """
    Inject different effects for different hierarchical groups.
    
    E.g., Bergmann's rule (negative temp-body size correlation) for mammals,
    but positive for reptiles.
    
    Args:
        df: DataFrame to modify
        x_col: Predictor column
        y_col: Outcome column
        hierarchy_col: Grouping column (e.g., taxonomic_class)
        effects_by_level: Dict of group -> effect size
        rng: Random number generator
        
    Returns:
        Modified DataFrame
    """
    df = df.copy()
    
    # Convert to float to allow fractional modifications
    df[y_col] = df[y_col].astype(float)
    
    for level, effect in effects_by_level.items():
        mask = df[hierarchy_col] == level
        if mask.sum() > 10:
            df = inject_correlation(df, x_col, y_col, effect, rng, mask.values)
    
    return df


def inject_latent_variable(
    df: pd.DataFrame,
    observed_cols: list[str],
    induced_correlations: dict[str, float],
    rng: np.random.Generator
) -> pd.DataFrame:
    """
    Inject correlations among observed variables due to a latent common cause.
    
    Args:
        df: DataFrame to modify
        observed_cols: List of observed variable columns
        induced_correlations: Dict of "var1-var2" -> correlation
        rng: Random number generator
        
    Returns:
        Modified DataFrame
    """
    df = df.copy()
    
    # Convert to float to allow fractional modifications
    for col in observed_cols:
        df[col] = df[col].astype(float)
    
    # Create latent variable
    n = len(df)
    latent = rng.standard_normal(n)
    
    # Calculate loadings from desired correlations
    # For simplicity, make each observed var correlated with latent
    avg_corr = np.mean(list(induced_correlations.values())) if induced_correlations else 0.5
    loading = np.sqrt(max(0.01, min(0.99, avg_corr)))
    
    # Modify each observed variable to include latent effect
    for col in observed_cols:
        orig = df[col].values.astype(float)
        orig_std = np.std(orig)
        orig_mean = np.mean(orig)
        
        # New value = loading * latent + sqrt(1-loading^2) * original_standardized
        new_val = loading * latent + np.sqrt(1 - loading**2) * _standardize(orig)
        df[col] = new_val * orig_std + orig_mean
    
    return df


def inject_nonlinear_piecewise(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    breakpoints: list[float],
    slopes: list[float],
    rng: np.random.Generator
) -> pd.DataFrame:
    """
    Inject a piecewise linear (potentially U-shaped) relationship.
    
    Args:
        df: DataFrame to modify
        x_col: Predictor column
        y_col: Outcome column
        breakpoints: List of x values where slope changes
        slopes: List of slopes for each segment (len = len(breakpoints) + 1)
        rng: Random number generator
        
    Returns:
        Modified DataFrame
    """
    df = df.copy()
    
    # Convert to float to allow fractional modifications
    df[y_col] = df[y_col].astype(float)
    
    x = df[x_col].values.astype(float)
    y_std = df[y_col].std()
    
    # Calculate piecewise contribution
    contribution = np.zeros(len(x))
    
    # First segment
    mask = x < breakpoints[0]
    contribution[mask] = slopes[0] * (x[mask] - breakpoints[0])
    
    # Middle segments
    for i in range(len(breakpoints) - 1):
        mask = (x >= breakpoints[i]) & (x < breakpoints[i + 1])
        contribution[mask] = slopes[i + 1] * (x[mask] - breakpoints[i])
    
    # Last segment
    mask = x >= breakpoints[-1]
    contribution[mask] = slopes[-1] * (x[mask] - breakpoints[-1])
    
    # Normalize and apply
    contribution = _standardize(contribution)
    df[y_col] = df[y_col].values + contribution * y_std * 0.5
    
    return df


def inject_temporal_interaction(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    time_col: str,
    effects_by_time: dict[str, float],
    rng: np.random.Generator
) -> pd.DataFrame:
    """
    Inject an effect that varies by time period (e.g., season).
    
    Args:
        df: DataFrame to modify
        x_col: Predictor column
        y_col: Outcome column
        time_col: Time grouping column (e.g., season)
        effects_by_time: Dict of time_value -> effect
        rng: Random number generator
        
    Returns:
        Modified DataFrame
    """
    df = df.copy()
    
    # Convert to float to allow fractional modifications
    df[y_col] = df[y_col].astype(float)
    
    for time_val, effect in effects_by_time.items():
        mask = df[time_col] == time_val
        if mask.sum() > 10:
            df = inject_correlation(df, x_col, y_col, effect, rng, mask.values)
    
    return df


def inject_spatial_cluster(
    df: pd.DataFrame,
    presence_cols: list[str],
    region_col: str,
    cluster_pattern: dict[str, list[int]],
    rng: np.random.Generator
) -> pd.DataFrame:
    """
    Inject region-specific co-occurrence patterns.
    
    Args:
        df: DataFrame to modify
        presence_cols: List of species presence columns
        region_col: Region column
        cluster_pattern: Dict of region -> list of presence values [0,1,1] etc.
        rng: Random number generator
        
    Returns:
        Modified DataFrame
    """
    df = df.copy()
    
    for region, pattern in cluster_pattern.items():
        mask = df[region_col] == region
        n_in_region = mask.sum()
        
        if n_in_region == 0:
            continue
            
        for i, col in enumerate(presence_cols):
            if i < len(pattern) and col in df.columns:
                # Set presence probability based on pattern
                prob = 0.8 if pattern[i] == 1 else 0.2
                df.loc[mask, col] = rng.choice([0, 1], size=n_in_region, p=[1 - prob, prob])
    
    return df


def inject_competition_masking(
    df: pd.DataFrame,
    sp1_col: str,
    sp2_col: str,
    keystone_col: str,
    effect_without_keystone: float,
    effect_with_keystone: float,
    rng: np.random.Generator
) -> pd.DataFrame:
    """
    Inject competition that disappears when keystone species is present.
    
    Args:
        df: DataFrame to modify
        sp1_col: First species column
        sp2_col: Second species column
        keystone_col: Keystone presence column
        effect_without_keystone: Correlation without keystone (negative = competition)
        effect_with_keystone: Correlation with keystone present
        rng: Random number generator
        
    Returns:
        Modified DataFrame
    """
    df = df.copy()
    
    # Convert to float to allow fractional modifications
    df[sp1_col] = df[sp1_col].astype(float)
    df[sp2_col] = df[sp2_col].astype(float)
    
    # Without keystone: competition (negative correlation)
    no_keystone = df[keystone_col] == 0
    if no_keystone.sum() > 10:
        df = inject_correlation(df, sp1_col, sp2_col, effect_without_keystone, rng, no_keystone.values)
    
    # With keystone: neutral or positive
    with_keystone = df[keystone_col] == 1
    if with_keystone.sum() > 10:
        df = inject_correlation(df, sp1_col, sp2_col, effect_with_keystone, rng, with_keystone.values)
    
    return df


# =============================================================================
# MAIN GENERATOR FUNCTION
# =============================================================================

def generate_benchmark_data(
    n_samples: int,
    n_species: int,
    n_regions: int,
    seed: int
) -> pd.DataFrame:
    """
    Generate base benchmark dataset with all features.
    
    Patterns should be injected separately using the inject_* functions.
    
    Args:
        n_samples: Number of samples
        n_species: Number of species
        n_regions: Number of biogeographic regions
        seed: Random seed
        
    Returns:
        DataFrame with all base features
    """
    rng = np.random.default_rng(seed)
    
    # Generate feature groups
    env_df = generate_base_environmental_features(n_samples, rng)
    species_df = generate_species_features(n_samples, n_species, env_df, rng)
    obs_df = generate_observation_features(n_samples, n_regions, rng)
    outcome_df = generate_ecological_outcomes(n_samples, rng)
    
    # Combine all features
    df = pd.concat([env_df, species_df, obs_df, outcome_df], axis=1)
    
    # Add sample ID
    df.insert(0, "sample_id", range(n_samples))
    
    return df
