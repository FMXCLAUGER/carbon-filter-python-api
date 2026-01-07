"""
Regeneration Energy Calculations for Activated Carbon

This module provides energy calculations for different regeneration methods:
- TSA (Temperature Swing Adsorption)
- PSA (Pressure Swing Adsorption)
- VSA (Vacuum Swing Adsorption)
- Steam regeneration

TSA is the most common method for activated carbon regeneration, using
elevated temperatures (150-300°C) to desorb contaminants.

Energy requirements include:
1. Sensible heat to raise bed temperature
2. Desorption enthalpy
3. Heat losses
4. Purge gas heating (if applicable)
"""

import math
from enum import Enum
from typing import Dict, Optional, List
from dataclasses import dataclass


class RegenerationType(Enum):
    """Types of regeneration methods."""
    TSA = "Temperature Swing Adsorption"
    PSA = "Pressure Swing Adsorption"
    VSA = "Vacuum Swing Adsorption"
    STEAM = "Steam Regeneration"
    EXTERNAL = "External Regeneration"
    NONE = "No Regeneration (Disposal)"


@dataclass
class RegenerationEnergy:
    """Container for regeneration energy calculations."""
    total_energy_kWh: float
    sensible_heat_kWh: float
    desorption_heat_kWh: float
    heat_losses_kWh: float
    specific_energy_kWh_kg: float
    regeneration_time_h: float
    power_required_kW: float
    method: RegenerationType
    efficiency: float = 0.85
    details: Dict = None


def calculate_tsa_energy(
    bed_mass: float,
    q_loading: float,
    delta_H_ads: float,
    T_ads: float,
    T_regen: float,
    C_p_carbon: float = 800.0,
    C_p_gas: float = 1005.0,
    heat_loss_factor: float = 0.15,
    heating_efficiency: float = 0.85,
    regen_time_h: float = 4.0,
    purge_gas_flow: Optional[float] = None,
    molecular_weight: float = 92.0
) -> RegenerationEnergy:
    """
    Calculate energy requirements for TSA regeneration.

    TSA involves heating the bed to desorb contaminants, then cooling
    back to adsorption temperature.

    Args:
        bed_mass: Mass of carbon bed (kg)
        q_loading: Adsorbate loading at saturation (kg/kg_carbon)
        delta_H_ads: Heat of adsorption (J/mol), use positive value
        T_ads: Adsorption temperature (°C)
        T_regen: Regeneration temperature (°C)
        C_p_carbon: Heat capacity of carbon (J/(kg·K))
        C_p_gas: Heat capacity of purge gas (J/(kg·K))
        heat_loss_factor: Fraction of heat lost to surroundings (0-1)
        heating_efficiency: Heater efficiency (0-1)
        regen_time_h: Regeneration time (hours)
        purge_gas_flow: Purge gas flow rate (m³/h), optional
        molecular_weight: Molecular weight of adsorbate (g/mol)

    Returns:
        RegenerationEnergy with complete energy analysis
    """
    # Temperature difference
    delta_T = T_regen - T_ads

    # 1. Sensible heat to raise bed temperature
    Q_sensible = bed_mass * C_p_carbon * delta_T  # Joules
    Q_sensible_kWh = Q_sensible / 3.6e6  # Convert to kWh

    # 2. Desorption heat
    # Convert loading to moles
    mass_adsorbate = bed_mass * q_loading  # kg adsorbate
    moles_adsorbate = mass_adsorbate * 1000 / molecular_weight  # moles

    Q_desorption = abs(delta_H_ads) * moles_adsorbate  # Joules
    Q_desorption_kWh = Q_desorption / 3.6e6  # kWh

    # 3. Purge gas heating (if applicable)
    Q_purge_kWh = 0.0
    if purge_gas_flow and purge_gas_flow > 0:
        rho_gas = 1.2  # kg/m³
        mass_purge = purge_gas_flow * rho_gas * regen_time_h  # kg
        Q_purge = mass_purge * C_p_gas * delta_T  # Joules
        Q_purge_kWh = Q_purge / 3.6e6

    # 4. Heat losses
    Q_gross = Q_sensible_kWh + Q_desorption_kWh + Q_purge_kWh
    Q_losses_kWh = Q_gross * heat_loss_factor

    # Total energy required (accounting for heater efficiency)
    Q_total_kWh = (Q_gross + Q_losses_kWh) / heating_efficiency

    # Power requirement
    power_kW = Q_total_kWh / regen_time_h

    # Specific energy (per kg carbon)
    specific_energy = Q_total_kWh / bed_mass if bed_mass > 0 else 0

    return RegenerationEnergy(
        total_energy_kWh=Q_total_kWh,
        sensible_heat_kWh=Q_sensible_kWh,
        desorption_heat_kWh=Q_desorption_kWh,
        heat_losses_kWh=Q_losses_kWh,
        specific_energy_kWh_kg=specific_energy,
        regeneration_time_h=regen_time_h,
        power_required_kW=power_kW,
        method=RegenerationType.TSA,
        efficiency=heating_efficiency,
        details={
            "T_ads_C": T_ads,
            "T_regen_C": T_regen,
            "delta_T_C": delta_T,
            "adsorbate_mass_kg": mass_adsorbate,
            "moles_desorbed": moles_adsorbate,
            "purge_heat_kWh": Q_purge_kWh,
            "gross_heat_kWh": Q_gross
        }
    )


def calculate_psa_energy(
    bed_mass: float,
    q_loading: float,
    P_ads: float,
    P_regen: float,
    T_operation: float = 25.0,
    bed_void_fraction: float = 0.4,
    molecular_weight: float = 92.0,
    compressor_efficiency: float = 0.75,
    cycle_time_min: float = 10.0
) -> RegenerationEnergy:
    """
    Calculate energy requirements for PSA regeneration.

    PSA uses pressure reduction to desorb contaminants. Less energy-intensive
    than TSA but requires robust pressure vessels.

    Args:
        bed_mass: Mass of carbon bed (kg)
        q_loading: Adsorbate loading (kg/kg_carbon)
        P_ads: Adsorption pressure (bar abs)
        P_regen: Regeneration pressure (bar abs)
        T_operation: Operating temperature (°C)
        bed_void_fraction: Void fraction of bed
        molecular_weight: Molecular weight of adsorbate (g/mol)
        compressor_efficiency: Compressor efficiency
        cycle_time_min: PSA cycle time (minutes)

    Returns:
        RegenerationEnergy with PSA analysis
    """
    R = 8.314  # J/(mol·K)
    T_K = T_operation + 273.15

    # Volume of void space
    rho_bulk = 480  # kg/m³ (approximate)
    bed_volume = bed_mass / rho_bulk
    void_volume = bed_volume * bed_void_fraction  # m³

    # Compression/expansion work
    # W = n × R × T × ln(P2/P1)
    pressure_ratio = P_ads / P_regen

    # Moles of gas in void space at adsorption pressure
    n_gas = (P_ads * 1e5 * void_volume) / (R * T_K)  # moles

    # Work for pressure swing (isothermal, ideal)
    W_isothermal = n_gas * R * T_K * math.log(pressure_ratio)  # Joules

    # Account for non-ideal compression
    W_actual = W_isothermal / compressor_efficiency

    # Number of cycles per hour
    cycles_per_hour = 60 / cycle_time_min

    # Energy per cycle
    W_cycle_kWh = W_actual / 3.6e6

    # Power requirement
    power_kW = W_cycle_kWh * cycles_per_hour

    # Energy per kg carbon per cycle
    specific_energy = W_cycle_kWh / bed_mass if bed_mass > 0 else 0

    return RegenerationEnergy(
        total_energy_kWh=W_cycle_kWh,
        sensible_heat_kWh=0,  # No heating in pure PSA
        desorption_heat_kWh=0,
        heat_losses_kWh=0,
        specific_energy_kWh_kg=specific_energy,
        regeneration_time_h=cycle_time_min / 60,
        power_required_kW=power_kW,
        method=RegenerationType.PSA,
        efficiency=compressor_efficiency,
        details={
            "P_ads_bar": P_ads,
            "P_regen_bar": P_regen,
            "pressure_ratio": pressure_ratio,
            "void_volume_m3": void_volume,
            "moles_gas": n_gas,
            "cycles_per_hour": cycles_per_hour,
            "cycle_time_min": cycle_time_min
        }
    )


def calculate_steam_regeneration(
    bed_mass: float,
    q_loading: float,
    delta_H_ads: float,
    T_steam: float = 150.0,
    P_steam_bar: float = 5.0,
    steam_ratio: float = 2.5,
    molecular_weight: float = 92.0,
    regen_time_h: float = 2.0
) -> RegenerationEnergy:
    """
    Calculate energy requirements for steam regeneration.

    Steam regeneration is effective for VOCs and provides both heat
    and displacement of adsorbates. The steam can be condensed and
    adsorbates recovered.

    Args:
        bed_mass: Mass of carbon bed (kg)
        q_loading: Adsorbate loading (kg/kg_carbon)
        delta_H_ads: Heat of adsorption (J/mol)
        T_steam: Steam temperature (°C)
        P_steam_bar: Steam pressure (bar)
        steam_ratio: kg steam per kg adsorbate
        molecular_weight: Molecular weight of adsorbate (g/mol)
        regen_time_h: Regeneration time (hours)

    Returns:
        RegenerationEnergy with steam regeneration analysis
    """
    # Mass of adsorbate
    mass_adsorbate = bed_mass * q_loading  # kg

    # Steam required
    steam_required = mass_adsorbate * steam_ratio  # kg

    # Steam enthalpy (approximate for saturated steam)
    # h_fg ≈ 2200 kJ/kg at 5 bar
    h_fg = 2200000  # J/kg (latent heat)

    # Sensible heat to raise bed temperature
    C_p_carbon = 800  # J/(kg·K)
    T_ambient = 25
    delta_T = T_steam - T_ambient
    Q_sensible = bed_mass * C_p_carbon * delta_T  # J
    Q_sensible_kWh = Q_sensible / 3.6e6

    # Energy in steam
    Q_steam = steam_required * h_fg  # J
    Q_steam_kWh = Q_steam / 3.6e6

    # Desorption heat
    moles_adsorbate = mass_adsorbate * 1000 / molecular_weight
    Q_desorption = abs(delta_H_ads) * moles_adsorbate
    Q_desorption_kWh = Q_desorption / 3.6e6

    # Total energy (steam provides most of the heat)
    Q_total_kWh = Q_steam_kWh

    # Power requirement
    power_kW = Q_total_kWh / regen_time_h

    # Specific energy
    specific_energy = Q_total_kWh / bed_mass if bed_mass > 0 else 0

    return RegenerationEnergy(
        total_energy_kWh=Q_total_kWh,
        sensible_heat_kWh=Q_sensible_kWh,
        desorption_heat_kWh=Q_desorption_kWh,
        heat_losses_kWh=0,  # Steam carries its own heat
        specific_energy_kWh_kg=specific_energy,
        regeneration_time_h=regen_time_h,
        power_required_kW=power_kW,
        method=RegenerationType.STEAM,
        efficiency=0.9,
        details={
            "T_steam_C": T_steam,
            "P_steam_bar": P_steam_bar,
            "steam_required_kg": steam_required,
            "steam_ratio": steam_ratio,
            "adsorbate_mass_kg": mass_adsorbate,
            "steam_energy_kWh": Q_steam_kWh
        }
    )


def calculate_vsa_energy(
    bed_mass: float,
    q_loading: float,
    P_ads: float,
    P_vacuum: float,
    T_operation: float = 25.0,
    bed_void_fraction: float = 0.4,
    vacuum_pump_efficiency: float = 0.65,
    cycle_time_min: float = 15.0
) -> RegenerationEnergy:
    """
    Calculate energy requirements for VSA regeneration.

    VSA uses vacuum to lower the pressure for desorption. More energy
    efficient than PSA for some applications.

    Args:
        bed_mass: Mass of carbon bed (kg)
        q_loading: Adsorbate loading (kg/kg_carbon)
        P_ads: Adsorption pressure (bar abs)
        P_vacuum: Vacuum pressure (bar abs, typically 0.1-0.5)
        T_operation: Operating temperature (°C)
        bed_void_fraction: Void fraction of bed
        vacuum_pump_efficiency: Vacuum pump efficiency
        cycle_time_min: Cycle time (minutes)

    Returns:
        RegenerationEnergy with VSA analysis
    """
    R = 8.314  # J/(mol·K)
    T_K = T_operation + 273.15

    # Volume calculations
    rho_bulk = 480  # kg/m³
    bed_volume = bed_mass / rho_bulk
    void_volume = bed_volume * bed_void_fraction

    # Pressure ratio (for vacuum pump work)
    pressure_ratio = P_ads / P_vacuum

    # Volume of gas to evacuate (at vacuum pressure)
    V_evacuate = void_volume * pressure_ratio  # Equivalent volume at vacuum

    # Vacuum pump work (polytropic)
    # Approximate: W = P × V × ln(P1/P2)
    n_gas = (P_ads * 1e5 * void_volume) / (R * T_K)
    W_ideal = n_gas * R * T_K * math.log(pressure_ratio)

    # Account for pump efficiency
    W_actual = W_ideal / vacuum_pump_efficiency

    # Cycles per hour
    cycles_per_hour = 60 / cycle_time_min

    # Energy per cycle
    W_cycle_kWh = W_actual / 3.6e6

    # Power
    power_kW = W_cycle_kWh * cycles_per_hour

    # Specific energy
    specific_energy = W_cycle_kWh / bed_mass if bed_mass > 0 else 0

    return RegenerationEnergy(
        total_energy_kWh=W_cycle_kWh,
        sensible_heat_kWh=0,
        desorption_heat_kWh=0,
        heat_losses_kWh=0,
        specific_energy_kWh_kg=specific_energy,
        regeneration_time_h=cycle_time_min / 60,
        power_required_kW=power_kW,
        method=RegenerationType.VSA,
        efficiency=vacuum_pump_efficiency,
        details={
            "P_ads_bar": P_ads,
            "P_vacuum_bar": P_vacuum,
            "pressure_ratio": pressure_ratio,
            "void_volume_m3": void_volume,
            "volume_evacuated_m3": V_evacuate,
            "cycles_per_hour": cycles_per_hour
        }
    )


def compare_regeneration_methods(
    bed_mass: float,
    q_loading: float,
    delta_H_ads: float = 40000,
    molecular_weight: float = 92.0
) -> Dict:
    """
    Compare different regeneration methods for a given application.

    Args:
        bed_mass: Mass of carbon bed (kg)
        q_loading: Adsorbate loading (kg/kg_carbon)
        delta_H_ads: Heat of adsorption (J/mol)
        molecular_weight: Molecular weight of adsorbate (g/mol)

    Returns:
        Dictionary with comparison of all methods
    """
    # Calculate for each method
    tsa = calculate_tsa_energy(
        bed_mass=bed_mass,
        q_loading=q_loading,
        delta_H_ads=delta_H_ads,
        T_ads=25,
        T_regen=200,
        molecular_weight=molecular_weight
    )

    psa = calculate_psa_energy(
        bed_mass=bed_mass,
        q_loading=q_loading,
        P_ads=5,
        P_regen=1,
        molecular_weight=molecular_weight
    )

    vsa = calculate_vsa_energy(
        bed_mass=bed_mass,
        q_loading=q_loading,
        P_ads=1,
        P_vacuum=0.2
    )

    steam = calculate_steam_regeneration(
        bed_mass=bed_mass,
        q_loading=q_loading,
        delta_H_ads=delta_H_ads,
        molecular_weight=molecular_weight
    )

    return {
        "TSA": {
            "energy_kWh": tsa.total_energy_kWh,
            "specific_energy_kWh_kg": tsa.specific_energy_kWh_kg,
            "power_kW": tsa.power_required_kW,
            "time_h": tsa.regeneration_time_h,
            "advantages": [
                "High regeneration efficiency",
                "Well-established technology",
                "Works for most adsorbates"
            ],
            "disadvantages": [
                "High energy consumption",
                "Thermal stress on carbon",
                "Slower cycles"
            ]
        },
        "PSA": {
            "energy_kWh": psa.total_energy_kWh,
            "specific_energy_kWh_kg": psa.specific_energy_kWh_kg,
            "power_kW": psa.power_required_kW,
            "time_h": psa.regeneration_time_h,
            "advantages": [
                "Fast cycles",
                "No thermal degradation",
                "Continuous operation possible"
            ],
            "disadvantages": [
                "Requires pressure-rated vessels",
                "Lower working capacity",
                "Not suitable for all adsorbates"
            ]
        },
        "VSA": {
            "energy_kWh": vsa.total_energy_kWh,
            "specific_energy_kWh_kg": vsa.specific_energy_kWh_kg,
            "power_kW": vsa.power_required_kW,
            "time_h": vsa.regeneration_time_h,
            "advantages": [
                "Lower energy than PSA",
                "No compression needed",
                "Good for weakly adsorbed species"
            ],
            "disadvantages": [
                "Vacuum pump maintenance",
                "Air ingress risk",
                "Limited pressure ratio"
            ]
        },
        "STEAM": {
            "energy_kWh": steam.total_energy_kWh,
            "specific_energy_kWh_kg": steam.specific_energy_kWh_kg,
            "power_kW": steam.power_required_kW,
            "time_h": steam.regeneration_time_h,
            "advantages": [
                "Effective for VOCs",
                "Enables solvent recovery",
                "Fast regeneration"
            ],
            "disadvantages": [
                "Requires steam supply",
                "Condensate handling",
                "May wet the carbon"
            ]
        },
        "recommendation": _recommend_method(q_loading, delta_H_ads, bed_mass)
    }


def _recommend_method(q_loading: float, delta_H_ads: float, bed_mass: float) -> Dict:
    """Internal function to recommend the best regeneration method."""
    if q_loading > 0.15:
        # High loading - TSA or Steam preferred
        if delta_H_ads > 40000:
            return {
                "method": "TSA",
                "reason": "High loading with strongly adsorbed species"
            }
        else:
            return {
                "method": "STEAM",
                "reason": "High loading with moderate adsorption strength"
            }
    elif bed_mass > 500:
        # Large system - economics favor TSA
        return {
            "method": "TSA",
            "reason": "Large bed size makes TSA economics favorable"
        }
    else:
        # Small system, low loading - PSA or VSA
        return {
            "method": "PSA",
            "reason": "Rapid cycling for small systems with low loading"
        }
