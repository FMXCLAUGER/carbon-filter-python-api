"""
Economic Analysis for Activated Carbon Regeneration

This module provides economic comparison between:
- On-site regeneration (TSA, PSA, VSA, Steam)
- Off-site regeneration (external service)
- Carbon replacement (disposal and new carbon)

Key metrics:
- Total Cost of Ownership (TCO)
- Return on Investment (ROI) for regeneration systems
- Break-even analysis
- Environmental impact (CO2 savings)
"""

import math
from typing import Dict, Optional, List
from dataclasses import dataclass
from enum import Enum

from .tsa_energy import (
    RegenerationType,
    RegenerationEnergy,
    calculate_tsa_energy,
    calculate_steam_regeneration
)


@dataclass
class EconomicAnalysis:
    """Container for economic analysis results."""
    annual_cost_eur: float
    npv_10y_eur: float
    payback_years: float
    roi_percent: float
    co2_kg_per_year: float
    method: str
    details: Dict


def compare_regeneration_vs_replacement(
    bed_mass: float,
    q_loading: float,
    carbon_price_eur_kg: float,
    energy_price_eur_kwh: float,
    disposal_cost_eur_kg: float,
    cycles_per_year: int,
    delta_H_ads: float = 40000,
    molecular_weight: float = 92.0,
    regen_type: RegenerationType = RegenerationType.TSA,
    regen_equipment_cost: float = 0,
    labor_cost_eur_h: float = 50.0,
    regen_labor_h: float = 4.0,
    carbon_lifetime_cycles: int = 100,
    discount_rate: float = 0.08,
    analysis_years: int = 10
) -> Dict:
    """
    Compare economics of regeneration vs. carbon replacement.

    Args:
        bed_mass: Mass of carbon bed (kg)
        q_loading: Adsorbate loading (kg/kg_carbon)
        carbon_price_eur_kg: Price of new activated carbon (€/kg)
        energy_price_eur_kwh: Electricity price (€/kWh)
        disposal_cost_eur_kg: Disposal/incineration cost (€/kg)
        cycles_per_year: Number of adsorption cycles per year
        delta_H_ads: Heat of adsorption (J/mol)
        molecular_weight: Molecular weight of adsorbate (g/mol)
        regen_type: Type of regeneration
        regen_equipment_cost: Capital cost for regeneration equipment (€)
        labor_cost_eur_h: Labor cost per hour (€/h)
        regen_labor_h: Hours of labor per regeneration cycle
        carbon_lifetime_cycles: Number of regen cycles before replacement
        discount_rate: Annual discount rate for NPV
        analysis_years: Number of years for analysis

    Returns:
        Dictionary with complete economic comparison
    """
    # === REPLACEMENT SCENARIO ===
    # Carbon needs replacement after saturation
    # Assume replacement happens when capacity drops significantly

    # Cost per replacement
    new_carbon_cost = bed_mass * carbon_price_eur_kg
    disposal_cost = bed_mass * disposal_cost_eur_kg
    replacement_cost_per_cycle = new_carbon_cost + disposal_cost

    # How often replacement is needed (assume 1 cycle per replacement for no regen)
    replacements_per_year = cycles_per_year
    annual_replacement_cost = replacements_per_year * replacement_cost_per_cycle

    # === REGENERATION SCENARIO ===
    # Calculate energy cost per regeneration

    if regen_type == RegenerationType.TSA:
        regen_energy = calculate_tsa_energy(
            bed_mass=bed_mass,
            q_loading=q_loading,
            delta_H_ads=delta_H_ads,
            T_ads=25,
            T_regen=200,
            molecular_weight=molecular_weight
        )
    elif regen_type == RegenerationType.STEAM:
        regen_energy = calculate_steam_regeneration(
            bed_mass=bed_mass,
            q_loading=q_loading,
            delta_H_ads=delta_H_ads,
            molecular_weight=molecular_weight
        )
    else:
        # Default TSA estimate
        regen_energy = calculate_tsa_energy(
            bed_mass=bed_mass,
            q_loading=q_loading,
            delta_H_ads=delta_H_ads,
            T_ads=25,
            T_regen=200,
            molecular_weight=molecular_weight
        )

    energy_cost_per_regen = regen_energy.total_energy_kWh * energy_price_eur_kwh
    labor_cost_per_regen = regen_labor_h * labor_cost_eur_h
    cost_per_regen = energy_cost_per_regen + labor_cost_per_regen

    # Carbon replacement (after lifetime cycles)
    replacements_per_year_regen = cycles_per_year / carbon_lifetime_cycles
    annual_carbon_replacement = replacements_per_year_regen * (new_carbon_cost + disposal_cost)

    # Total annual regeneration cost
    annual_regen_cost = (cycles_per_year * cost_per_regen) + annual_carbon_replacement

    # === NPV CALCULATION ===
    def calculate_npv(annual_cost: float, capex: float, years: int, rate: float) -> float:
        """Calculate Net Present Value."""
        npv = capex  # Initial investment
        for year in range(1, years + 1):
            npv += annual_cost / ((1 + rate) ** year)
        return npv

    npv_replacement = calculate_npv(annual_replacement_cost, 0, analysis_years, discount_rate)
    npv_regeneration = calculate_npv(annual_regen_cost, regen_equipment_cost, analysis_years, discount_rate)

    # === SAVINGS ===
    annual_savings = annual_replacement_cost - annual_regen_cost
    total_savings_10y = npv_replacement - npv_regeneration

    # Payback period
    if annual_savings > 0 and regen_equipment_cost > 0:
        payback_years = regen_equipment_cost / annual_savings
    elif annual_savings > 0:
        payback_years = 0
    else:
        payback_years = float('inf')

    # ROI
    if regen_equipment_cost > 0:
        roi_percent = (total_savings_10y / regen_equipment_cost) * 100
    else:
        roi_percent = float('inf') if annual_savings > 0 else 0

    # === CO2 ANALYSIS ===
    # CO2 from carbon production: ~3-5 kg CO2 per kg carbon
    co2_per_kg_carbon = 4.0
    # CO2 from electricity: ~0.4 kg CO2/kWh (EU average)
    co2_per_kwh = 0.4

    co2_replacement = replacements_per_year * bed_mass * co2_per_kg_carbon
    co2_regeneration = (
        cycles_per_year * regen_energy.total_energy_kWh * co2_per_kwh +
        replacements_per_year_regen * bed_mass * co2_per_kg_carbon
    )
    co2_savings = co2_replacement - co2_regeneration

    return {
        "replacement": {
            "annual_cost_eur": round(annual_replacement_cost, 2),
            "npv_10y_eur": round(npv_replacement, 2),
            "cost_per_cycle_eur": round(replacement_cost_per_cycle, 2),
            "replacements_per_year": replacements_per_year,
            "co2_kg_per_year": round(co2_replacement, 1)
        },
        "regeneration": {
            "annual_cost_eur": round(annual_regen_cost, 2),
            "npv_10y_eur": round(npv_regeneration, 2),
            "capex_eur": regen_equipment_cost,
            "cost_per_cycle_eur": round(cost_per_regen, 2),
            "energy_cost_per_cycle_eur": round(energy_cost_per_regen, 2),
            "labor_cost_per_cycle_eur": round(labor_cost_per_regen, 2),
            "cycles_per_year": cycles_per_year,
            "carbon_replacements_per_year": round(replacements_per_year_regen, 2),
            "co2_kg_per_year": round(co2_regeneration, 1),
            "method": regen_type.value
        },
        "comparison": {
            "annual_savings_eur": round(annual_savings, 2),
            "total_savings_10y_eur": round(total_savings_10y, 2),
            "payback_years": round(payback_years, 1) if payback_years < 100 else "N/A",
            "roi_percent": round(roi_percent, 1) if abs(roi_percent) < 10000 else "Very High",
            "co2_savings_kg_year": round(co2_savings, 1),
            "recommendation": "Regeneration" if annual_savings > 0 else "Replacement"
        },
        "parameters": {
            "bed_mass_kg": bed_mass,
            "carbon_price_eur_kg": carbon_price_eur_kg,
            "energy_price_eur_kwh": energy_price_eur_kwh,
            "cycles_per_year": cycles_per_year,
            "analysis_years": analysis_years,
            "discount_rate": discount_rate
        }
    }


def calculate_regeneration_roi(
    equipment_cost: float,
    annual_operating_cost: float,
    annual_savings: float,
    useful_life_years: int = 15,
    discount_rate: float = 0.08
) -> Dict:
    """
    Calculate ROI metrics for regeneration equipment.

    Args:
        equipment_cost: Capital cost (€)
        annual_operating_cost: Annual O&M cost (€/year)
        annual_savings: Annual savings vs. replacement (€/year)
        useful_life_years: Equipment lifetime (years)
        discount_rate: Discount rate for NPV

    Returns:
        Dictionary with ROI metrics
    """
    # Net annual benefit
    net_annual_benefit = annual_savings - annual_operating_cost

    # Simple payback
    if net_annual_benefit > 0:
        simple_payback = equipment_cost / net_annual_benefit
    else:
        simple_payback = float('inf')

    # NPV calculation
    npv = -equipment_cost
    irr_cash_flows = [-equipment_cost]

    for year in range(1, useful_life_years + 1):
        discounted_benefit = net_annual_benefit / ((1 + discount_rate) ** year)
        npv += discounted_benefit
        irr_cash_flows.append(net_annual_benefit)

    # Simple ROI
    total_benefit = net_annual_benefit * useful_life_years
    roi = ((total_benefit - equipment_cost) / equipment_cost) * 100 if equipment_cost > 0 else 0

    # IRR approximation (Newton-Raphson method)
    irr = _calculate_irr(irr_cash_flows)

    return {
        "npv_eur": round(npv, 2),
        "simple_payback_years": round(simple_payback, 1) if simple_payback < 50 else "N/A",
        "roi_percent": round(roi, 1),
        "irr_percent": round(irr * 100, 1) if irr else "N/A",
        "net_annual_benefit_eur": round(net_annual_benefit, 2),
        "total_benefit_eur": round(total_benefit, 2),
        "profitable": npv > 0,
        "analysis_period_years": useful_life_years
    }


def _calculate_irr(cash_flows: List[float], max_iterations: int = 100, tolerance: float = 0.0001) -> Optional[float]:
    """Calculate Internal Rate of Return using Newton-Raphson method."""
    if len(cash_flows) < 2:
        return None

    # Initial guess
    rate = 0.1

    for _ in range(max_iterations):
        npv = sum(cf / ((1 + rate) ** i) for i, cf in enumerate(cash_flows))
        dnpv = sum(-i * cf / ((1 + rate) ** (i + 1)) for i, cf in enumerate(cash_flows))

        if abs(dnpv) < 1e-10:
            break

        new_rate = rate - npv / dnpv

        if abs(new_rate - rate) < tolerance:
            return new_rate

        rate = new_rate

        # Prevent extreme values
        if rate < -0.99 or rate > 10:
            return None

    return rate if abs(npv) < tolerance else None


def calculate_co2_savings(
    bed_mass: float,
    cycles_per_year: int,
    regen_energy_kwh: float,
    carbon_lifetime_cycles: int = 100,
    co2_per_kg_carbon: float = 4.0,
    co2_per_kwh: float = 0.4,
    include_transport: bool = True
) -> Dict:
    """
    Calculate CO2 savings from regeneration vs. replacement.

    Args:
        bed_mass: Mass of carbon bed (kg)
        cycles_per_year: Number of cycles per year
        regen_energy_kwh: Energy per regeneration (kWh)
        carbon_lifetime_cycles: Cycles before carbon replacement
        co2_per_kg_carbon: CO2 emissions per kg new carbon (kg CO2/kg)
        co2_per_kwh: CO2 emissions per kWh electricity (kg CO2/kWh)
        include_transport: Include transport emissions

    Returns:
        Dictionary with CO2 analysis
    """
    # Transport emissions (approximate)
    transport_co2 = 0.1 if include_transport else 0  # kg CO2 per kg carbon

    # Replacement scenario
    replacements_per_year_no_regen = cycles_per_year
    co2_new_carbon = bed_mass * co2_per_kg_carbon
    co2_transport = bed_mass * transport_co2
    co2_replacement = replacements_per_year_no_regen * (co2_new_carbon + co2_transport)

    # Regeneration scenario
    replacements_per_year_with_regen = cycles_per_year / carbon_lifetime_cycles
    co2_energy = cycles_per_year * regen_energy_kwh * co2_per_kwh
    co2_with_regen = (
        replacements_per_year_with_regen * (co2_new_carbon + co2_transport) +
        co2_energy
    )

    # Savings
    co2_savings = co2_replacement - co2_with_regen

    # Equivalents for context
    # Average car emits ~4.6 tonnes CO2/year (12,000 miles at 25 mpg)
    cars_equivalent = co2_savings / 4600

    # Average tree absorbs ~22 kg CO2/year
    trees_equivalent = co2_savings / 22

    return {
        "replacement": {
            "co2_kg_year": round(co2_replacement, 1),
            "from_production": round(replacements_per_year_no_regen * co2_new_carbon, 1),
            "from_transport": round(replacements_per_year_no_regen * co2_transport, 1)
        },
        "regeneration": {
            "co2_kg_year": round(co2_with_regen, 1),
            "from_energy": round(co2_energy, 1),
            "from_replacement": round(replacements_per_year_with_regen * co2_new_carbon, 1)
        },
        "savings": {
            "co2_kg_year": round(co2_savings, 1),
            "co2_tonnes_year": round(co2_savings / 1000, 2),
            "reduction_percent": round((co2_savings / co2_replacement) * 100, 1) if co2_replacement > 0 else 0,
            "equivalent_cars": round(cars_equivalent, 2),
            "equivalent_trees": round(trees_equivalent, 0)
        },
        "assumptions": {
            "co2_per_kg_carbon": co2_per_kg_carbon,
            "co2_per_kwh": co2_per_kwh,
            "carbon_lifetime_cycles": carbon_lifetime_cycles
        }
    }


def calculate_external_regen_economics(
    bed_mass: float,
    carbon_price_eur_kg: float,
    external_regen_price_eur_kg: float,
    transport_cost_eur_kg: float,
    regen_efficiency: float = 0.95,
    turnaround_days: int = 14,
    spare_bed_factor: float = 1.0,
    cycles_per_year: int = 12
) -> Dict:
    """
    Calculate economics for external/off-site regeneration.

    Args:
        bed_mass: Mass of carbon bed (kg)
        carbon_price_eur_kg: New carbon price (€/kg)
        external_regen_price_eur_kg: External regeneration service price (€/kg)
        transport_cost_eur_kg: Two-way transport cost (€/kg)
        regen_efficiency: Capacity recovery after regeneration (0-1)
        turnaround_days: Days for external regeneration
        spare_bed_factor: Factor for spare carbon inventory (1.0 = 100% spare)
        cycles_per_year: Number of cycles per year

    Returns:
        Dictionary with external regeneration economics
    """
    # Cost per regeneration cycle
    regen_cost = bed_mass * external_regen_price_eur_kg
    transport_cost = bed_mass * transport_cost_eur_kg
    cost_per_cycle = regen_cost + transport_cost

    # Spare carbon inventory cost (one-time)
    spare_carbon_cost = bed_mass * spare_bed_factor * carbon_price_eur_kg

    # Annual cost
    annual_regen_cost = cycles_per_year * cost_per_cycle

    # Compare with replacement
    new_carbon_cost_per_cycle = bed_mass * carbon_price_eur_kg
    annual_replacement_cost = cycles_per_year * new_carbon_cost_per_cycle

    # Savings
    annual_savings = annual_replacement_cost - annual_regen_cost

    # Payback for spare carbon investment
    payback = spare_carbon_cost / annual_savings if annual_savings > 0 else float('inf')

    return {
        "external_regeneration": {
            "cost_per_cycle_eur": round(cost_per_cycle, 2),
            "regen_service_eur": round(regen_cost, 2),
            "transport_eur": round(transport_cost, 2),
            "annual_cost_eur": round(annual_regen_cost, 2),
            "spare_carbon_investment_eur": round(spare_carbon_cost, 2),
            "turnaround_days": turnaround_days,
            "capacity_recovery_percent": regen_efficiency * 100
        },
        "vs_replacement": {
            "replacement_cost_per_cycle_eur": round(new_carbon_cost_per_cycle, 2),
            "annual_replacement_cost_eur": round(annual_replacement_cost, 2),
            "annual_savings_eur": round(annual_savings, 2),
            "payback_years": round(payback, 1) if payback < 50 else "N/A",
            "recommendation": "External Regen" if annual_savings > 0 else "Replace"
        },
        "considerations": [
            "Requires spare carbon inventory for continuous operation",
            "Transport adds logistics complexity",
            f"Turnaround time of {turnaround_days} days may require planning",
            f"Capacity recovery of {regen_efficiency*100}% expected"
        ]
    }
