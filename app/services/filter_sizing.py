import numpy as np
from typing import Optional, Tuple

from app.models.schemas import FilterDimensions, OperatingConditions


DEFAULT_VELOCITY = 0.3  # m/s
DEFAULT_EBCT = 5.0  # seconds
MIN_VELOCITY = 0.1  # m/s
MAX_VELOCITY = 0.5  # m/s
MIN_EBCT = 2.0  # seconds
MAX_EBCT = 10.0  # seconds


def calculate_dimensions(
    flow_rate: float,
    bulk_density: float,
    bed_voidage: float,
    target_velocity: Optional[float] = None,
    target_ebct: Optional[float] = None,
) -> Tuple[FilterDimensions, OperatingConditions]:
    """
    Calculate filter dimensions based on flow rate and design constraints.
    
    Args:
        flow_rate: Volumetric flow rate (m³/h)
        bulk_density: Carbon bulk density (kg/m³)
        bed_voidage: Bed void fraction (0-1)
        target_velocity: Target superficial velocity (m/s)
        target_ebct: Target EBCT (s)
    
    Returns:
        Tuple of (FilterDimensions, OperatingConditions)
    """
    flow_rate_m3s = flow_rate / 3600  # Convert to m³/s
    
    # Determine velocity and EBCT
    if target_velocity is not None:
        velocity = np.clip(target_velocity, MIN_VELOCITY, MAX_VELOCITY)
    elif target_ebct is not None:
        velocity = None  # Will be calculated from EBCT
    else:
        velocity = DEFAULT_VELOCITY
    
    if target_ebct is not None:
        ebct = np.clip(target_ebct, MIN_EBCT, MAX_EBCT)
    else:
        ebct = DEFAULT_EBCT
    
    # Calculate cross-section from velocity (if velocity is known)
    if velocity is not None:
        cross_section = flow_rate_m3s / velocity
        bed_volume = flow_rate_m3s * ebct
        bed_height = bed_volume / cross_section
    else:
        # Calculate from EBCT first
        bed_volume = flow_rate_m3s * ebct
        # Use aspect ratio H/D ~ 2-3 (typical for cylindrical beds)
        aspect_ratio = 2.5
        diameter = (4 * bed_volume / (np.pi * aspect_ratio)) ** (1/3)
        cross_section = np.pi * (diameter / 2) ** 2
        bed_height = bed_volume / cross_section
        velocity = flow_rate_m3s / cross_section
    
    # Calculate diameter from cross-section
    diameter = np.sqrt(4 * cross_section / np.pi)
    
    # Carbon mass
    bed_mass = bed_volume * bulk_density
    
    # Actual residence time (accounting for voidage)
    residence_time = ebct * bed_voidage
    
    # Pressure drop (Ergun equation simplified for gas)
    # ΔP = K × H × v^1.8 where K depends on particle size and voidage
    # Simplified: ΔP ≈ 150 × (1-ε)² / ε³ × μ × v / d_p² × H
    # For gas at low velocity, use empirical correlation
    particle_diameter = 0.003  # Default 3mm if not specified
    k_ergun = 150 * ((1 - bed_voidage) ** 2) / (bed_voidage ** 3)
    mu_air = 1.8e-5  # Dynamic viscosity of air (Pa·s)
    pressure_drop = k_ergun * mu_air * velocity / (particle_diameter ** 2) * bed_height
    # Add turbulent term for higher velocities
    rho_air = 1.2  # kg/m³
    pressure_drop += 1.75 * (1 - bed_voidage) / bed_voidage ** 3 * rho_air * velocity ** 2 / particle_diameter * bed_height
    
    dimensions = FilterDimensions(
        bed_volume=round(bed_volume, 4),
        bed_mass=round(bed_mass, 1),
        bed_diameter=round(diameter, 3),
        bed_height=round(bed_height, 3),
        cross_section=round(cross_section, 4),
    )
    
    operating = OperatingConditions(
        superficial_velocity=round(velocity, 3),
        ebct=round(ebct, 2),
        residence_time=round(residence_time, 2),
        pressure_drop=round(pressure_drop, 1),
    )
    
    return dimensions, operating
