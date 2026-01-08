"""Utility functions for CADET-Process integration."""
import numpy as np
from typing import Dict, Tuple, Optional


def estimate_film_diffusion(
    velocity: float,
    particle_diameter: float,
    molecular_diffusivity: float = 1e-5,
    kinematic_viscosity: float = 1.5e-5,
) -> float:
    """
    Estimate film mass transfer coefficient using Wakao-Funazkri correlation.

    Sh = 2.0 + 1.1 * Re^0.6 * Sc^0.33
    k_f = Sh * D_m / d_p

    Args:
        velocity: Superficial velocity (m/s)
        particle_diameter: Particle diameter (m)
        molecular_diffusivity: Molecular diffusivity in gas (m²/s)
        kinematic_viscosity: Kinematic viscosity of gas (m²/s)

    Returns:
        Film mass transfer coefficient k_f (m/s)
    """
    Re = velocity * particle_diameter / kinematic_viscosity
    Sc = kinematic_viscosity / molecular_diffusivity

    Sh = 2.0 + 1.1 * (Re ** 0.6) * (Sc ** 0.33)
    k_f = Sh * molecular_diffusivity / particle_diameter

    return k_f


def estimate_pore_diffusion(
    molecular_diffusivity: float = 1e-5,
    particle_porosity: float = 0.7,
    tortuosity: float = 3.0,
) -> float:
    """
    Estimate effective pore diffusion coefficient.

    D_p = (eps_p / tau) * D_m

    Args:
        molecular_diffusivity: Molecular diffusivity (m²/s)
        particle_porosity: Particle porosity
        tortuosity: Tortuosity factor (typically 2-4)

    Returns:
        Effective pore diffusion coefficient D_p (m²/s)
    """
    return (particle_porosity / tortuosity) * molecular_diffusivity


def estimate_axial_dispersion(
    velocity: float,
    particle_diameter: float,
    molecular_diffusivity: float = 1e-5,
    bed_porosity: float = 0.4,
) -> float:
    """
    Estimate axial dispersion coefficient using Chung-Wen correlation.

    D_ax = D_m / eps_b + 0.5 * v * d_p

    Args:
        velocity: Superficial velocity (m/s)
        particle_diameter: Particle diameter (m)
        molecular_diffusivity: Molecular diffusivity (m²/s)
        bed_porosity: Bed porosity

    Returns:
        Axial dispersion coefficient D_ax (m²/s)
    """
    return molecular_diffusivity / bed_porosity + 0.5 * velocity * particle_diameter


def convert_langmuir_to_cadet(
    K_eq: float,
    q_max: float,
    reference_desorption_rate: float = 1.0,
) -> Tuple[float, float, float]:
    """
    Convert Langmuir isotherm parameters to CADET kinetic parameters.

    Langmuir: q = (q_max * K * c) / (1 + K * c)
    CADET kinetic: dq/dt = k_a * c * (q_max - q) - k_d * q

    At equilibrium: K_eq = k_a / k_d

    Args:
        K_eq: Langmuir equilibrium constant (m³/mol)
        q_max: Maximum adsorption capacity (mol/m³)
        reference_desorption_rate: Reference k_d value (1/s)

    Returns:
        Tuple of (k_a, k_d, q_max) for CADET
    """
    k_d = reference_desorption_rate
    k_a = K_eq * k_d
    return k_a, k_d, q_max


def convert_freundlich_to_cadet(
    K_f: float,
    n: float,
    k_kin: float = 0.1,
) -> Tuple[float, float, float]:
    """
    Convert Freundlich isotherm parameters to CADET LDF parameters.

    Freundlich: q = K_f * c^(1/n)
    CADET LDF: dq/dt = k_kin * (q* - q) where q* = K_f * c^(1/n)

    Args:
        K_f: Freundlich constant
        n: Freundlich exponent (typically 0.7-1.0)
        k_kin: LDF kinetic coefficient (1/s)

    Returns:
        Tuple of (k_kin, K_f, n) for CADET Freundlich LDF
    """
    return k_kin, K_f, n


def mg_m3_to_mol_m3(
    concentration_mg_m3: float,
    molecular_weight: float,
) -> float:
    """Convert concentration from mg/m³ to mol/m³."""
    return concentration_mg_m3 / (molecular_weight * 1000)


def mol_m3_to_mg_m3(
    concentration_mol_m3: float,
    molecular_weight: float,
) -> float:
    """Convert concentration from mol/m³ to mg/m³."""
    return concentration_mol_m3 * molecular_weight * 1000


def calculate_residence_time(
    bed_length: float,
    bed_diameter: float,
    flow_rate_m3_h: float,
    bed_porosity: float,
) -> float:
    """
    Calculate residence time in the column.

    Args:
        bed_length: Bed length (m)
        bed_diameter: Bed diameter (m)
        flow_rate_m3_h: Volumetric flow rate (m³/h)
        bed_porosity: Bed porosity

    Returns:
        Residence time (seconds)
    """
    cross_section = np.pi * (bed_diameter / 2) ** 2
    bed_volume = cross_section * bed_length
    void_volume = bed_volume * bed_porosity
    flow_rate_m3_s = flow_rate_m3_h / 3600

    return void_volume / flow_rate_m3_s


def estimate_simulation_time(
    bed_mass_kg: float,
    q_max_mol_m3: float,
    inlet_concentration_mol_m3: float,
    flow_rate_m3_h: float,
    safety_factor: float = 2.0,
) -> float:
    """
    Estimate required simulation time based on stoichiometric breakthrough.

    Args:
        bed_mass_kg: Carbon bed mass (kg)
        q_max_mol_m3: Maximum adsorption capacity (mol/m³)
        inlet_concentration_mol_m3: Inlet concentration (mol/m³)
        flow_rate_m3_h: Flow rate (m³/h)
        safety_factor: Multiplier for safety margin

    Returns:
        Estimated simulation time (hours)
    """
    moles_capacity = bed_mass_kg * q_max_mol_m3 / 1000
    moles_per_hour = inlet_concentration_mol_m3 * flow_rate_m3_h

    if moles_per_hour > 0:
        stoichiometric_time = moles_capacity / moles_per_hour
        return stoichiometric_time * safety_factor
    return 100.0
