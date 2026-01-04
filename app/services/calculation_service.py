"""Main calculation service for filter dimensioning"""
import math
import time
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from app.models.schemas import (
    CalculationRequest,
    CalculationResult,
    PollutantResult,
    FilterDimensions,
    OperatingConditions,
    BreakthroughPoint,
)
from app.services.wheeler_jonas import (
    WheelerJonasParams,
    wheeler_jonas_breakthrough_time,
    generate_breakthrough_curve,
    estimate_kv_from_velocity,
    calculate_mass_transfer_zone,
)
from app.services.isotherm_service import estimate_capacity_from_surface_area
from app.utils.corrections import combined_correction
from app.utils.mass_transfer import (
    calculate_air_properties,
    calculate_film_transfer_coefficient,
)
from app.utils.constants import (
    VELOCITY_TYPICAL,
    EBCT_TYPICAL,
    TEMP_WARNING,
    TEMP_CRITICAL,
)


@dataclass
class DimensioningResult:
    """Intermediate result for filter dimensioning"""

    diameter: float
    height: float
    volume: float
    mass: float
    cross_section: float
    velocity: float
    ebct: float


def calculate_filter_dimensions(
    flow_rate: float,  # m³/h
    bulk_density: float,  # kg/m³
    target_velocity: Optional[float] = None,  # m/s
    target_ebct: Optional[float] = None,  # s
) -> DimensioningResult:
    """
    Calculate optimal filter dimensions.

    Either target_velocity or target_ebct should be specified.
    If both are specified, velocity takes precedence.
    If neither is specified, uses typical values.

    Args:
        flow_rate: Volumetric flow rate in m³/h
        bulk_density: Carbon bulk density in kg/m³
        target_velocity: Target superficial velocity in m/s
        target_ebct: Target Empty Bed Contact Time in seconds

    Returns:
        DimensioningResult with calculated dimensions
    """
    # Convert flow rate to m³/s
    Q = flow_rate / 3600

    # Determine velocity
    if target_velocity is not None:
        velocity = target_velocity
    elif target_ebct is not None:
        # Need to iterate to find velocity that gives target EBCT
        # Start with typical velocity and adjust
        velocity = VELOCITY_TYPICAL
    else:
        velocity = VELOCITY_TYPICAL

    # Cross-sectional area from continuity: A = Q / v
    cross_section = Q / velocity

    # Diameter from circular cross-section
    diameter = math.sqrt(4 * cross_section / math.pi)

    # Determine bed height
    if target_ebct is not None:
        # Volume = Q * EBCT, Height = Volume / Area
        volume = Q * target_ebct
        height = volume / cross_section
    else:
        # Default: height = 2 * diameter for reasonable L/D ratio
        height = max(0.5, 2 * diameter)
        volume = cross_section * height

    # Recalculate EBCT
    ebct = volume / Q

    # Mass of carbon
    mass = volume * bulk_density

    return DimensioningResult(
        diameter=diameter,
        height=height,
        volume=volume,
        mass=mass,
        cross_section=cross_section,
        velocity=velocity,
        ebct=ebct,
    )


def calculate_pressure_drop(
    velocity: float,
    bed_height: float,
    particle_diameter: float,
    bed_voidage: float,
    temperature: float,
    pressure: float = 101325,
) -> float:
    """
    Calculate pressure drop across packed bed using Ergun equation.

    ΔP/L = 150 * μ * (1-ε)² / (ε³ * dp²) * v + 1.75 * ρ * (1-ε) / (ε³ * dp) * v²

    Args:
        velocity: Superficial velocity (m/s)
        bed_height: Bed height (m)
        particle_diameter: Particle diameter (m)
        bed_voidage: Bed void fraction
        temperature: Temperature (°C)
        pressure: Pressure (Pa)

    Returns:
        Pressure drop in Pa
    """
    air = calculate_air_properties(temperature, pressure)
    mu = air["viscosity"]
    rho = air["density"]

    eps = bed_voidage
    dp = particle_diameter
    v = velocity
    L = bed_height

    # Ergun equation components
    term1 = 150 * mu * (1 - eps) ** 2 / (eps**3 * dp**2) * v
    term2 = 1.75 * rho * (1 - eps) / (eps**3 * dp) * v**2

    dP_per_L = term1 + term2
    dP = dP_per_L * L

    return dP


def calculate_thermal_effects(
    flow_rate: float,  # m³/h
    pollutants: List[dict],
    delta_H_ads: float = -40000,  # J/mol (typical)
) -> Tuple[float, bool]:
    """
    Calculate temperature rise from adsorption heat.

    Args:
        flow_rate: Flow rate in m³/h
        pollutants: List of pollutant dicts with concentration and MW
        delta_H_ads: Heat of adsorption (J/mol), negative for exothermic

    Returns:
        Tuple of (temperature rise in °C, thermal warning flag)
    """
    from app.utils.constants import AIR_DENSITY_25C, CP_AIR

    # Heat released = mass flow * concentration * heat of adsorption
    total_heat = 0  # W

    for p in pollutants:
        conc = p.get("concentration", 0)  # mg/m³
        mw = p.get("molecular_weight", 100)  # g/mol

        # Molar flow rate (mol/s)
        n_dot = (flow_rate / 3600) * (conc / 1000) / mw

        # Heat release rate (W = J/s)
        heat = n_dot * abs(delta_H_ads)
        total_heat += heat

    # Temperature rise: Q = m_dot * Cp * ΔT
    m_dot = (flow_rate / 3600) * AIR_DENSITY_25C  # kg/s
    if m_dot > 0:
        delta_T = total_heat / (m_dot * CP_AIR)
    else:
        delta_T = 0

    thermal_warning = delta_T > TEMP_WARNING

    return delta_T, thermal_warning


def perform_calculation(request: CalculationRequest) -> CalculationResult:
    """
    Perform complete filter dimensioning calculation.

    Args:
        request: CalculationRequest with all input data

    Returns:
        CalculationResult with dimensions, performance, and recommendations
    """
    start_time = time.time()

    warnings = []
    recommendations = []

    # Extract input data
    carbon = request.carbon
    pollutants = request.pollutants
    T = request.temperature
    RH = request.humidity
    P = request.pressure
    Q = request.flow_rate  # m³/h

    # Get particle diameter (default 3mm)
    dp = carbon.particle_diameter / 1000 if carbon.particle_diameter else 0.003

    # Calculate filter dimensions
    dim = calculate_filter_dimensions(
        flow_rate=Q,
        bulk_density=carbon.bulk_density,
        target_velocity=request.target_velocity,
        target_ebct=request.target_ebct,
    )

    # Pressure drop
    pressure_drop = calculate_pressure_drop(
        velocity=dim.velocity,
        bed_height=dim.height,
        particle_diameter=dp,
        bed_voidage=carbon.bed_voidage,
        temperature=T,
        pressure=P,
    )

    # Check velocity limits
    if dim.velocity < 0.1:
        warnings.append(f"Vitesse superficielle faible ({dim.velocity:.3f} m/s). Risque de mauvaise distribution.")
    elif dim.velocity > 0.5:
        warnings.append(f"Vitesse superficielle élevée ({dim.velocity:.3f} m/s). MTZ peut être importante.")

    # Pressure drop check
    if pressure_drop > 2000:  # > 20 mbar
        warnings.append(f"Perte de charge élevée ({pressure_drop:.0f} Pa). Considérer un diamètre plus grand.")

    # Calculate for each pollutant
    pollutant_results = []
    min_breakthrough_time = float("inf")
    limiting_pollutant = None

    for poll in pollutants:
        # Estimate capacity if not provided
        if poll.isotherm_params and "q_max" in poll.isotherm_params:
            q_eq = poll.isotherm_params["q_max"]
        else:
            # Estimate from surface area
            mw = poll.molecular_weight or 100
            q_eq = estimate_capacity_from_surface_area(
                carbon.surface_area, mw
            )

        # Apply temperature and humidity corrections
        q_corrected, factors = combined_correction(
            q_ref=q_eq,
            T_ref=25,
            T_new=T,
            relative_humidity=RH,
            micropore_volume=carbon.micropore_volume,
        )

        # Apply safety factor
        q_design = q_corrected / request.safety_factor

        # Wheeler-Jonas parameters
        k_v = estimate_kv_from_velocity(dim.velocity, dp, carbon.bed_voidage)

        wj_params = WheelerJonasParams(
            W_e=q_design / 1000,  # Convert mg/g to g/g
            k_v=k_v,
            rho_b=carbon.bulk_density,
            Q=Q / 60,  # m³/min
            C_in=poll.concentration,
            C_out=poll.target_outlet or poll.concentration * 0.05,
            W_bed=dim.mass,
        )

        # Breakthrough time
        t_b = wheeler_jonas_breakthrough_time(wj_params)  # minutes
        t_b_hours = t_b / 60

        # Track limiting pollutant
        if t_b_hours < min_breakthrough_time:
            min_breakthrough_time = t_b_hours
            limiting_pollutant = poll.name

        # MTZ
        mtz = calculate_mass_transfer_zone(wj_params, dim.height)

        # Removal efficiency
        efficiency = (
            (poll.concentration - (poll.target_outlet or 0)) / poll.concentration * 100
            if poll.concentration > 0
            else 0
        )

        pollutant_results.append(
            PollutantResult(
                name=poll.name,
                inlet_concentration=poll.concentration,
                outlet_concentration=poll.target_outlet or poll.concentration * 0.05,
                removal_efficiency=efficiency,
                adsorption_capacity=q_design,
                breakthrough_time=t_b_hours,
                mass_transfer_zone=mtz,
            )
        )

    # Generate breakthrough curve for limiting pollutant
    limiting_poll = next(
        (p for p in pollutants if p.name == limiting_pollutant), pollutants[0]
    )
    q_eq = estimate_capacity_from_surface_area(
        carbon.surface_area, limiting_poll.molecular_weight or 100
    )
    q_corrected, _ = combined_correction(q_eq, 25, T, RH)
    q_design = q_corrected / request.safety_factor

    wj_params = WheelerJonasParams(
        W_e=q_design / 1000,
        k_v=estimate_kv_from_velocity(dim.velocity, dp, carbon.bed_voidage),
        rho_b=carbon.bulk_density,
        Q=Q / 60,
        C_in=limiting_poll.concentration,
        C_out=limiting_poll.target_outlet or limiting_poll.concentration * 0.05,
        W_bed=dim.mass,
    )

    curve_data = generate_breakthrough_curve(wj_params, n_points=50)
    breakthrough_curve = [
        BreakthroughPoint(time=t / 60, c_c0=c) for t, c in curve_data  # min to hours
    ]

    # Service life in months
    hours_per_month = 24 * 30.5  # Average hours per month
    service_life_months = min_breakthrough_time / hours_per_month

    # Thermal effects
    poll_dicts = [
        {"concentration": p.concentration, "molecular_weight": p.molecular_weight or 100}
        for p in pollutants
    ]
    temp_rise, thermal_warning = calculate_thermal_effects(Q, poll_dicts)

    if thermal_warning:
        warnings.append(
            f"Échauffement significatif prévu ({temp_rise:.1f}°C). Surveiller température du lit."
        )
        if temp_rise > TEMP_CRITICAL - T:
            warnings.append(
                "ATTENTION: Risque de point chaud. Prévoir surveillance thermique."
            )

    # Service life check
    if service_life_months < request.design_life_months:
        warnings.append(
            f"Durée de vie calculée ({service_life_months:.1f} mois) < objectif ({request.design_life_months} mois)"
        )
        recommendations.append("Augmenter la masse de charbon ou réduire le débit")

    # Recommendations
    if len(recommendations) == 0:
        if service_life_months > request.design_life_months * 1.5:
            recommendations.append(
                "Surdimensionnement possible. Réduire volume pour optimiser coûts."
            )
        recommendations.append(
            f"Charbon recommandé: {carbon.name} ({carbon.surface_area:.0f} m²/g)"
        )

    # Overall efficiency
    overall_efficiency = sum(p.removal_efficiency for p in pollutant_results) / len(
        pollutant_results
    )

    calc_time_ms = int((time.time() - start_time) * 1000)

    return CalculationResult(
        dimensions=FilterDimensions(
            bed_volume=dim.volume,
            bed_mass=dim.mass,
            bed_diameter=dim.diameter,
            bed_height=dim.height,
            cross_section=dim.cross_section,
        ),
        operating=OperatingConditions(
            superficial_velocity=dim.velocity,
            ebct=dim.ebct,
            residence_time=dim.ebct * (1 - carbon.bed_voidage),
            pressure_drop=pressure_drop,
        ),
        pollutant_results=pollutant_results,
        overall_efficiency=overall_efficiency,
        service_life_months=service_life_months,
        breakthrough_curve=breakthrough_curve,
        breakthrough_model=request.breakthrough_model.value,
        max_temperature_rise=temp_rise,
        thermal_warning=thermal_warning,
        warnings=warnings,
        recommendations=recommendations,
        calculation_time_ms=calc_time_ms,
    )
