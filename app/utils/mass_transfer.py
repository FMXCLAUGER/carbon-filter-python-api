"""Mass transfer coefficient calculations"""
import math
from typing import Optional
from .constants import R, AIR_VISCOSITY_25C, AIR_DENSITY_25C, D_AIR_TYPICAL


def calculate_air_properties(temperature: float, pressure: float = 101325) -> dict:
    """
    Calculate air properties at given temperature and pressure.

    Args:
        temperature: Temperature in °C
        pressure: Pressure in Pa

    Returns:
        Dictionary with density, viscosity, and kinematic viscosity
    """
    T_K = temperature + 273.15

    # Ideal gas law for density
    M_air = 0.029  # kg/mol
    density = (pressure * M_air) / (R * T_K)

    # Sutherland's formula for viscosity
    T_ref = 298.15  # K
    mu_ref = AIR_VISCOSITY_25C
    S = 110.4  # Sutherland constant for air

    viscosity = mu_ref * (T_K / T_ref) ** 1.5 * (T_ref + S) / (T_K + S)

    return {
        "density": density,  # kg/m³
        "viscosity": viscosity,  # Pa·s
        "kinematic_viscosity": viscosity / density,  # m²/s
    }


def calculate_diffusivity(
    molecular_weight: float,
    temperature: float = 25,
    pressure: float = 101325,
) -> float:
    """
    Estimate molecular diffusivity in air using Fuller-Schettler-Giddings.

    Args:
        molecular_weight: Molecular weight of adsorbate (g/mol)
        temperature: Temperature (°C)
        pressure: Pressure (Pa)

    Returns:
        Molecular diffusivity in m²/s
    """
    T_K = temperature + 273.15
    P_atm = pressure / 101325

    # Fuller correlation parameters
    M_air = 29  # g/mol

    # Diffusion volumes (simplified)
    # Air diffusion volume ≈ 20.1
    # Typical organic ≈ 15 + 16.5*n_C (approximate)
    V_air = 20.1
    V_organic = 15 + 0.5 * molecular_weight  # Rough estimate

    # Fuller-Schettler-Giddings equation
    D = (
        1.013e-7
        * T_K**1.75
        * math.sqrt(1 / M_air + 1 / molecular_weight)
        / (P_atm * (V_air ** (1 / 3) + V_organic ** (1 / 3)) ** 2)
    )

    return D


def calculate_reynolds(
    velocity: float,
    particle_diameter: float,
    density: float,
    viscosity: float,
) -> float:
    """
    Calculate particle Reynolds number.

    Args:
        velocity: Superficial velocity (m/s)
        particle_diameter: Particle diameter (m)
        density: Fluid density (kg/m³)
        viscosity: Dynamic viscosity (Pa·s)

    Returns:
        Reynolds number (dimensionless)
    """
    return (density * velocity * particle_diameter) / viscosity


def calculate_schmidt(viscosity: float, density: float, diffusivity: float) -> float:
    """
    Calculate Schmidt number.

    Args:
        viscosity: Dynamic viscosity (Pa·s)
        density: Fluid density (kg/m³)
        diffusivity: Molecular diffusivity (m²/s)

    Returns:
        Schmidt number (dimensionless)
    """
    return viscosity / (density * diffusivity)


def calculate_sherwood_wakao_funazkri(
    Re: float,
    Sc: float,
    bed_voidage: float,
) -> float:
    """
    Calculate Sherwood number using Wakao-Funazkri correlation.

    Valid for packed beds with 3 < Re < 10000.

    Sh = 2 + 1.1 * Sc^(1/3) * Re^0.6

    Args:
        Re: Reynolds number
        Sc: Schmidt number
        bed_voidage: Bed void fraction

    Returns:
        Sherwood number
    """
    return 2 + 1.1 * (Sc ** (1 / 3)) * (Re**0.6)


def calculate_film_transfer_coefficient(
    velocity: float,
    particle_diameter: float,
    temperature: float = 25,
    pressure: float = 101325,
    molecular_weight: float = 100,
    bed_voidage: float = 0.4,
) -> dict:
    """
    Calculate external film mass transfer coefficient.

    Uses Wakao-Funazkri correlation for packed beds.

    Args:
        velocity: Superficial velocity (m/s)
        particle_diameter: Particle diameter (m)
        temperature: Temperature (°C)
        pressure: Pressure (Pa)
        molecular_weight: Molecular weight of adsorbate (g/mol)
        bed_voidage: Bed void fraction

    Returns:
        Dictionary with kf, Re, Sc, Sh
    """
    # Air properties
    air = calculate_air_properties(temperature, pressure)

    # Diffusivity
    D = calculate_diffusivity(molecular_weight, temperature, pressure)

    # Dimensionless numbers
    Re = calculate_reynolds(
        velocity, particle_diameter, air["density"], air["viscosity"]
    )
    Sc = calculate_schmidt(air["viscosity"], air["density"], D)
    Sh = calculate_sherwood_wakao_funazkri(Re, Sc, bed_voidage)

    # Film transfer coefficient: kf = Sh * D / dp
    kf = (Sh * D) / particle_diameter

    return {
        "kf": kf,  # m/s
        "Re": Re,
        "Sc": Sc,
        "Sh": Sh,
        "diffusivity": D,  # m²/s
    }


def calculate_ldf_coefficient(
    kf: float,
    particle_diameter: float,
    particle_density: float,
    pore_diffusivity: Optional[float] = None,
    isotherm_slope: float = 1.0,
) -> float:
    """
    Calculate Linear Driving Force (LDF) coefficient.

    The LDF model simplifies intraparticle diffusion:
    dq/dt = k_LDF * (q* - q)

    Uses Glueckauf approximation: k_LDF ≈ 15 * D_eff / R_p²

    Args:
        kf: External film mass transfer coefficient (m/s)
        particle_diameter: Particle diameter (m)
        particle_density: Particle density (kg/m³)
        pore_diffusivity: Effective pore diffusivity (m²/s), optional
        isotherm_slope: dq*/dC at operating point

    Returns:
        LDF coefficient (1/s)
    """
    R_p = particle_diameter / 2  # Particle radius

    # If pore diffusivity not provided, estimate from typical values
    if pore_diffusivity is None:
        # Typical effective diffusivity in activated carbon
        pore_diffusivity = 1e-10  # m²/s

    # Internal LDF coefficient (Glueckauf)
    k_internal = 15 * pore_diffusivity / (R_p**2)

    # External contribution
    # k_external = 3 * kf / (R_p * particle_density * isotherm_slope)
    # Simplified: often internal diffusion is rate-limiting
    k_external = 3 * kf / R_p

    # Overall LDF (resistances in series)
    # 1/k_LDF = 1/k_internal + 1/k_external
    if k_internal > 0 and k_external > 0:
        k_LDF = 1 / (1 / k_internal + 1 / k_external)
    else:
        k_LDF = k_internal if k_internal > 0 else k_external

    return k_LDF
