"""
Thermal Analysis for Activated Carbon Adsorption Beds

This module provides temperature profile calculations and safety assessments
for activated carbon adsorption systems. Key concerns include:
- Temperature rise due to heat of adsorption
- Risk of hot spots and thermal runaway
- Auto-ignition risk assessment

Activated carbon beds can experience significant temperature rises during
adsorption, especially with high concentration streams or exothermic reactions.
Safety limits:
- Normal operation: ΔT < 20°C
- Warning zone: ΔT 20-40°C
- Dangerous: ΔT > 40°C
- Auto-ignition risk: T_bed > 150°C (varies by carbon type)
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


# Physical constants
R = 8.314  # J/(mol·K)

# Auto-ignition temperatures for different carbon types (°C)
AUTO_IGNITION_TEMPS = {
    "COAL": 350,
    "COCONUT": 380,
    "WOOD": 320,
    "PEAT": 300,
    "LIGNITE": 280,
    "default": 300
}

# Safety margin (°C below auto-ignition)
SAFETY_MARGIN = 150


@dataclass
class ThermalAnalysis:
    """Container for thermal analysis results."""
    inlet_temp_C: float
    max_temp_C: float
    temp_rise_C: float
    temp_profile: List[Dict]
    safety_status: str
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    auto_ignition_risk: bool = False
    desorption_risk: bool = False


def calculate_adiabatic_temperature_rise(
    delta_H_ads: float,
    q_ads: float,
    C_p_carbon: float = 800.0,
    C_p_gas: float = 1005.0,
    rho_bulk: float = 480.0,
    gas_mass_flow_kg_h: float = 0.0
) -> float:
    """
    Calculate adiabatic temperature rise due to adsorption.

    The heat released during adsorption raises the bed temperature.
    For adiabatic conditions (no heat loss):
    ΔT = |ΔH_ads| × q_ads / (m_carbon × C_p_carbon)

    Args:
        delta_H_ads: Heat of adsorption (J/mol), negative for exothermic
        q_ads: Amount adsorbed (mol/kg_carbon)
        C_p_carbon: Heat capacity of carbon (J/(kg·K)), default 800
        C_p_gas: Heat capacity of gas (J/(kg·K)), default 1005 (air)
        rho_bulk: Bulk density (kg/m³)
        gas_mass_flow_kg_h: Gas mass flow rate (kg/h) for heat removal

    Returns:
        Adiabatic temperature rise (°C or K)
    """
    # Heat released per kg of carbon
    Q_released = abs(delta_H_ads) * q_ads  # J/kg_carbon

    # Temperature rise (adiabatic)
    delta_T = Q_released / C_p_carbon

    return delta_T


def calculate_temperature_rise(
    C0: float,
    Q_gas: float,
    delta_H_ads: float,
    rho_bulk: float = 480.0,
    bed_mass: float = 100.0,
    C_p_carbon: float = 800.0,
    C_p_gas: float = 1005.0,
    molecular_weight: float = 92.0,
    operation_time_h: float = 1.0,
    heat_transfer_coeff: float = 10.0,
    ambient_temp: float = 25.0
) -> Dict:
    """
    Calculate temperature rise during adsorption operation.

    Accounts for:
    - Heat generation from adsorption
    - Heat removal by gas flow
    - Heat loss to surroundings

    Args:
        C0: Inlet concentration (mg/m³)
        Q_gas: Gas flow rate (m³/h)
        delta_H_ads: Heat of adsorption (J/mol), typically -20000 to -50000
        rho_bulk: Bulk density (kg/m³)
        bed_mass: Mass of carbon bed (kg)
        C_p_carbon: Heat capacity of carbon (J/(kg·K))
        C_p_gas: Heat capacity of gas (J/(kg·K))
        molecular_weight: Molecular weight of adsorbate (g/mol)
        operation_time_h: Operation time (hours)
        heat_transfer_coeff: Overall heat transfer coefficient (W/(m²·K))
        ambient_temp: Ambient temperature (°C)

    Returns:
        Dictionary with temperature rise analysis
    """
    # Convert concentration to molar flow
    C0_kg_m3 = C0 / 1e6  # mg/m³ to kg/m³
    molar_flow = (C0_kg_m3 * Q_gas) / (molecular_weight / 1000)  # mol/h

    # Total moles adsorbed over operation time
    moles_adsorbed = molar_flow * operation_time_h  # mol

    # Heat generated (J)
    Q_generated = abs(delta_H_ads) * moles_adsorbed

    # Heat capacity of bed (J/K)
    bed_heat_capacity = bed_mass * C_p_carbon

    # Adiabatic temperature rise
    delta_T_adiabatic = Q_generated / bed_heat_capacity

    # Gas cooling effect
    # Heat removed by gas = ρ_gas × Q × C_p × ΔT
    rho_gas = 1.2  # kg/m³ (approximate for air)
    gas_cooling_capacity = rho_gas * Q_gas * C_p_gas * operation_time_h * 3600  # J/K

    # Actual temperature rise (steady state approximation)
    # Assumes gas reaches bed temperature
    effective_capacity = bed_heat_capacity + gas_cooling_capacity * 0.5

    delta_T_actual = Q_generated / effective_capacity
    delta_T_actual = min(delta_T_actual, delta_T_adiabatic)  # Can't exceed adiabatic

    # Heat generation rate
    Q_rate = Q_generated / (operation_time_h * 3600)  # W

    return {
        "delta_T_adiabatic_C": float(delta_T_adiabatic),
        "delta_T_actual_C": float(delta_T_actual),
        "heat_generated_J": float(Q_generated),
        "heat_generation_rate_W": float(Q_rate),
        "moles_adsorbed": float(moles_adsorbed),
        "gas_cooling_capacity_J_K": float(gas_cooling_capacity),
        "bed_heat_capacity_J_K": float(bed_heat_capacity)
    }


def calculate_bed_temperature_profile(
    inlet_temp: float,
    C0: float,
    Q_gas: float,
    bed_height: float,
    bed_area: float,
    delta_H_ads: float,
    rho_bulk: float = 480.0,
    porosity: float = 0.4,
    C_p_carbon: float = 800.0,
    C_p_gas: float = 1005.0,
    molecular_weight: float = 92.0,
    n_segments: int = 10,
    carbon_type: str = "COAL"
) -> ThermalAnalysis:
    """
    Calculate temperature profile along the adsorption bed.

    The temperature rises progressively as adsorption occurs, with
    maximum temperature near the mass transfer zone.

    Args:
        inlet_temp: Inlet gas temperature (°C)
        C0: Inlet concentration (mg/m³)
        Q_gas: Gas flow rate (m³/h)
        bed_height: Height of bed (m)
        bed_area: Cross-sectional area (m²)
        delta_H_ads: Heat of adsorption (J/mol)
        rho_bulk: Bulk density (kg/m³)
        porosity: Bed porosity
        C_p_carbon: Heat capacity of carbon (J/(kg·K))
        C_p_gas: Heat capacity of gas (J/(kg·K))
        molecular_weight: Molecular weight of adsorbate (g/mol)
        n_segments: Number of segments for profile
        carbon_type: Type of carbon for safety assessment

    Returns:
        ThermalAnalysis object with complete thermal assessment
    """
    warnings = []
    recommendations = []

    # Bed properties
    bed_volume = bed_height * bed_area  # m³
    bed_mass = bed_volume * rho_bulk  # kg

    # Gas properties
    rho_gas = 1.2  # kg/m³
    v_superficial = Q_gas / (bed_area * 3600)  # m/s
    mass_flow_gas = rho_gas * Q_gas  # kg/h

    # Concentration profile assumption: exponential decay
    # C(z) = C0 × exp(-k × z) where k is related to mass transfer
    # For simplicity, assume 95% removal over bed height

    segment_height = bed_height / n_segments
    segment_mass = bed_mass / n_segments
    segment_heat_capacity = segment_mass * C_p_carbon

    # Heat carried by gas per segment per hour
    gas_heat_capacity_rate = mass_flow_gas * C_p_gas / 3600  # W/K

    profile = []
    current_temp = inlet_temp
    max_temp = inlet_temp

    # Adsorption rate decreases along bed (front-loaded)
    for i in range(n_segments):
        z = (i + 0.5) * segment_height  # Midpoint of segment
        z_norm = z / bed_height

        # Concentration at this point (exponential profile)
        C_local = C0 * math.exp(-3 * z_norm)  # 95% removal by end

        # Local adsorption rate (mg/h in this segment)
        dC = C0 * (math.exp(-3 * i * segment_height / bed_height) -
                   math.exp(-3 * (i + 1) * segment_height / bed_height))
        ads_rate_mg_h = dC * Q_gas  # mg/h adsorbed in this segment

        # Convert to molar rate
        ads_rate_mol_h = ads_rate_mg_h / (molecular_weight * 1000)  # mol/h

        # Heat generation rate in segment (W)
        Q_gen_segment = abs(delta_H_ads) * ads_rate_mol_h / 3600  # W

        # Temperature rise in segment
        # Steady state: Q_gen = gas_heat × (T_out - T_in)
        if gas_heat_capacity_rate > 0:
            delta_T_segment = Q_gen_segment / gas_heat_capacity_rate
        else:
            delta_T_segment = 0

        # Limit temperature rise (heat loss to surroundings)
        delta_T_segment = min(delta_T_segment, 20)  # Max 20°C per segment

        new_temp = current_temp + delta_T_segment
        max_temp = max(max_temp, new_temp)

        profile.append({
            "position_m": float(z),
            "position_normalized": float(z_norm),
            "temperature_C": float(new_temp),
            "temp_rise_C": float(new_temp - inlet_temp),
            "concentration_mg_m3": float(C_local),
            "heat_generation_W": float(Q_gen_segment),
            "adsorption_rate_mol_h": float(ads_rate_mol_h)
        })

        current_temp = new_temp

    # Overall temperature rise
    temp_rise = max_temp - inlet_temp

    # Safety assessment
    auto_ignition_temp = AUTO_IGNITION_TEMPS.get(
        carbon_type, AUTO_IGNITION_TEMPS["default"]
    )
    safe_operating_temp = auto_ignition_temp - SAFETY_MARGIN

    auto_ignition_risk = max_temp > safe_operating_temp
    desorption_risk = max_temp > 80

    # Determine safety status
    if max_temp > auto_ignition_temp:
        safety_status = "CRITICAL"
        warnings.append(f"Temperature exceeds auto-ignition point ({auto_ignition_temp}°C)!")
        recommendations.append("IMMEDIATE ACTION REQUIRED: Stop operation and investigate")
    elif max_temp > safe_operating_temp:
        safety_status = "DANGEROUS"
        warnings.append(f"Temperature within {SAFETY_MARGIN}°C of auto-ignition")
        recommendations.append("Install additional temperature monitoring")
        recommendations.append("Consider reducing inlet concentration or flow rate")
    elif max_temp > 80:
        safety_status = "WARNING"
        warnings.append("High bed temperature may cause desorption")
        recommendations.append("Monitor outlet concentration for breakthrough")
    elif temp_rise > 40:
        safety_status = "CAUTION"
        warnings.append("Significant temperature rise detected")
        recommendations.append("Increase monitoring frequency")
    elif temp_rise > 20:
        safety_status = "MODERATE"
    else:
        safety_status = "NORMAL"

    # Additional recommendations based on conditions
    if temp_rise > 10:
        recommendations.append("Ensure adequate ventilation around the bed")

    if C0 > 1000:  # High concentration
        recommendations.append("Consider dilution of inlet stream")

    return ThermalAnalysis(
        inlet_temp_C=inlet_temp,
        max_temp_C=max_temp,
        temp_rise_C=temp_rise,
        temp_profile=profile,
        safety_status=safety_status,
        warnings=warnings,
        recommendations=recommendations,
        auto_ignition_risk=auto_ignition_risk,
        desorption_risk=desorption_risk
    )


def check_thermal_safety(
    operating_temp: float,
    temp_rise: float,
    carbon_type: str = "COAL",
    include_recommendations: bool = True
) -> Dict:
    """
    Quick thermal safety check for operating conditions.

    Args:
        operating_temp: Current bed temperature (°C)
        temp_rise: Temperature rise from inlet (°C)
        carbon_type: Type of carbon
        include_recommendations: Whether to include recommendations

    Returns:
        Dictionary with safety assessment
    """
    auto_ignition_temp = AUTO_IGNITION_TEMPS.get(
        carbon_type, AUTO_IGNITION_TEMPS["default"]
    )

    alerts = []
    recommendations = []

    # Check absolute temperature
    if operating_temp > auto_ignition_temp:
        alerts.append({
            "level": "CRITICAL",
            "message": f"Temperature exceeds auto-ignition ({auto_ignition_temp}°C)",
            "action": "STOP OPERATION IMMEDIATELY"
        })
    elif operating_temp > auto_ignition_temp - SAFETY_MARGIN:
        alerts.append({
            "level": "DANGER",
            "message": "Temperature approaching auto-ignition zone",
            "action": "Reduce load or increase cooling"
        })
    elif operating_temp > 80:
        alerts.append({
            "level": "WARNING",
            "message": "High temperature may cause desorption",
            "action": "Monitor outlet concentration"
        })
    elif operating_temp > 60:
        alerts.append({
            "level": "CAUTION",
            "message": "Elevated temperature detected",
            "action": "Increase monitoring"
        })

    # Check temperature rise
    if temp_rise > 50:
        alerts.append({
            "level": "WARNING",
            "message": f"Large temperature rise ({temp_rise:.1f}°C)",
            "action": "Check for hot spots"
        })

    # Determine overall status
    if any(a["level"] == "CRITICAL" for a in alerts):
        status = "CRITICAL"
    elif any(a["level"] == "DANGER" for a in alerts):
        status = "DANGER"
    elif any(a["level"] == "WARNING" for a in alerts):
        status = "WARNING"
    elif any(a["level"] == "CAUTION" for a in alerts):
        status = "CAUTION"
    else:
        status = "SAFE"

    # Generate recommendations
    if include_recommendations:
        if temp_rise > 20:
            recommendations.append("Install temperature sensors at multiple bed heights")
        if temp_rise > 40:
            recommendations.append("Consider inter-stage cooling")
            recommendations.append("Review inlet concentration limits")
        if operating_temp > 60:
            recommendations.append("Ensure adequate bed ventilation")
            recommendations.append("Check for accumulation of reactive species")

    return {
        "status": status,
        "operating_temp_C": operating_temp,
        "temp_rise_C": temp_rise,
        "auto_ignition_temp_C": auto_ignition_temp,
        "safety_margin_C": auto_ignition_temp - operating_temp,
        "alerts": alerts,
        "recommendations": recommendations if include_recommendations else []
    }


def estimate_cooling_requirements(
    heat_generation_rate: float,
    max_temp_rise: float = 20.0,
    inlet_temp: float = 25.0,
    max_bed_temp: float = 60.0
) -> Dict:
    """
    Estimate cooling requirements to maintain safe operation.

    Args:
        heat_generation_rate: Heat generation rate (W)
        max_temp_rise: Maximum acceptable temperature rise (°C)
        inlet_temp: Inlet gas temperature (°C)
        max_bed_temp: Maximum acceptable bed temperature (°C)

    Returns:
        Dictionary with cooling requirements
    """
    # Allowable temperature rise
    allowable_rise = min(max_temp_rise, max_bed_temp - inlet_temp)

    # Required heat removal rate to achieve target temperature rise
    # For gas cooling: Q_removed = m_dot × C_p × ΔT
    # Solving for minimum gas flow
    C_p_gas = 1005  # J/(kg·K)
    rho_gas = 1.2  # kg/m³

    # Heat to remove
    Q_remove = heat_generation_rate * 0.5  # Assume 50% needs active removal

    if allowable_rise > 0:
        # Minimum mass flow for cooling
        min_mass_flow = Q_remove / (C_p_gas * allowable_rise)  # kg/s
        min_volume_flow = min_mass_flow / rho_gas * 3600  # m³/h
    else:
        min_mass_flow = float('inf')
        min_volume_flow = float('inf')

    # Alternative: water cooling
    C_p_water = 4186  # J/(kg·K)
    water_temp_rise = 10  # Assume 10°C rise in cooling water
    water_flow = Q_remove / (C_p_water * water_temp_rise) * 3600  # kg/h = L/h

    return {
        "heat_to_remove_W": float(Q_remove),
        "allowable_temp_rise_C": float(allowable_rise),
        "air_cooling": {
            "min_flow_m3_h": float(min_volume_flow),
            "type": "Forced air circulation"
        },
        "water_cooling": {
            "min_flow_L_h": float(water_flow),
            "temp_rise_C": water_temp_rise,
            "type": "Jacket cooling or internal coils"
        },
        "recommendations": [
            "For heat loads > 1kW, consider active cooling",
            "For critical applications, implement redundant cooling",
            "Install high-temperature shutdown interlocks"
        ]
    }
