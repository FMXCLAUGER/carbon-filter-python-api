"""CADET-Process integration service for high-precision breakthrough simulation."""
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np

from app.services.cadet.cadet_utils import (
    estimate_film_diffusion,
    estimate_pore_diffusion,
    estimate_axial_dispersion,
    convert_langmuir_to_cadet,
    mg_m3_to_mol_m3,
    mol_m3_to_mg_m3,
)


_CADET_AVAILABLE = False
_CADET_ERROR = None

try:
    from CADETProcess.processModel import (
        ComponentSystem,
        Inlet,
        Langmuir,
        GeneralRateModel,
        Outlet,
        FlowSheet,
        Process,
    )
    from CADETProcess.simulator import Cadet

    _simulator = Cadet()
    _simulator.check_cadet()
    _CADET_AVAILABLE = True
except ImportError as e:
    _CADET_ERROR = f"CADET-Process not installed: {e}"
except Exception as e:
    _CADET_ERROR = f"CADET-Core not available: {e}"


def check_cadet_available() -> Tuple[bool, Optional[str]]:
    """Check if CADET-Process is available and working."""
    return _CADET_AVAILABLE, _CADET_ERROR


@dataclass
class CADETConfig:
    """Configuration for CADET simulation."""

    bed_length: float
    bed_diameter: float
    particle_radius: float = 1.5e-3
    bed_porosity: float = 0.4
    particle_porosity: float = 0.7

    flow_rate_m3_h: float = 1000.0
    inlet_concentration_mg_m3: float = 100.0
    molecular_weight: float = 92.0

    isotherm_model: str = "langmuir"
    isotherm_K: float = 0.001
    isotherm_qmax: float = 100.0
    isotherm_n: float = 1.0

    simulation_time_h: float = 24.0
    n_points: int = 100
    timeout_s: int = 60


@dataclass
class CADETResult:
    """Result from CADET simulation."""

    success: bool
    model_type: str = "GRM"
    breakthrough_time_h: float = 0.0
    saturation_time_h: float = 0.0
    bed_utilization: float = 0.0
    curve: List[Dict[str, float]] = field(default_factory=list)
    computation_time_s: float = 0.0
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)


class CADETBreakthroughService:
    """High-precision breakthrough simulation using CADET-Process GRM."""

    def __init__(self, timeout: int = 60):
        self.timeout = timeout
        self.available, self.error = check_cadet_available()

    def simulate(self, config: CADETConfig) -> CADETResult:
        """
        Run CADET GRM simulation for activated carbon breakthrough.

        Args:
            config: CADETConfig with all simulation parameters

        Returns:
            CADETResult with breakthrough curve and metrics
        """
        if not self.available:
            return CADETResult(
                success=False,
                error=self.error or "CADET not available",
            )

        start_time = time.time()
        warnings = []

        try:
            component_system = ComponentSystem()
            component_system.add_component("Pollutant")

            inlet = Inlet(component_system, name="inlet")
            flow_rate_m3_s = config.flow_rate_m3_h / 3600
            inlet.flow_rate = flow_rate_m3_s

            c_mol_m3 = mg_m3_to_mol_m3(
                config.inlet_concentration_mg_m3,
                config.molecular_weight,
            )
            inlet.c = [[c_mol_m3]]

            binding_model = Langmuir(component_system, name="langmuir")
            binding_model.is_kinetic = True

            k_a, k_d, qmax = convert_langmuir_to_cadet(
                K_eq=config.isotherm_K,
                q_max=config.isotherm_qmax,
            )
            binding_model.adsorption_rate = [k_a]
            binding_model.desorption_rate = [k_d]
            binding_model.capacity = [qmax]

            column = GeneralRateModel(component_system, name="column")
            column.binding_model = binding_model
            column.length = config.bed_length
            column.diameter = config.bed_diameter
            column.bed_porosity = config.bed_porosity
            column.particle_porosity = config.particle_porosity
            column.particle_radius = config.particle_radius

            cross_section = np.pi * (config.bed_diameter / 2) ** 2
            velocity = flow_rate_m3_s / cross_section

            column.axial_dispersion = estimate_axial_dispersion(
                velocity=velocity,
                particle_diameter=2 * config.particle_radius,
                bed_porosity=config.bed_porosity,
            )
            column.film_diffusion = [
                estimate_film_diffusion(
                    velocity=velocity,
                    particle_diameter=2 * config.particle_radius,
                )
            ]
            column.pore_diffusion = [
                estimate_pore_diffusion(
                    particle_porosity=config.particle_porosity,
                )
            ]
            column.surface_diffusion = [0.0]

            column.c = [0]
            column.cp = [0]
            column.q = [0]

            outlet = Outlet(component_system, name="outlet")

            flow_sheet = FlowSheet(component_system)
            flow_sheet.add_unit(inlet)
            flow_sheet.add_unit(column)
            flow_sheet.add_unit(outlet, product_outlet=True)
            flow_sheet.add_connection(inlet, column)
            flow_sheet.add_connection(column, outlet)

            process = Process(flow_sheet, "Breakthrough")
            process.cycle_time = config.simulation_time_h * 3600

            simulator = Cadet()
            simulator.timeout = self.timeout
            simulator.time_resolution = process.cycle_time / config.n_points

            results = simulator.simulate(process)

            if results.exit_flag != 0:
                warnings.append(f"Solver warning: {results.exit_message}")

            time_s = results.solution.column.outlet.time
            time_h = time_s / 3600
            c_out = results.solution.column.outlet.solution[:, 0]
            c_c0 = c_out / c_mol_m3

            curve = [
                {"time_h": float(t), "c_c0": float(c)}
                for t, c in zip(time_h, c_c0)
            ]

            breakthrough_time_h = self._find_breakthrough_time(time_h, c_c0, 0.05)
            saturation_time_h = self._find_breakthrough_time(time_h, c_c0, 0.95)

            if saturation_time_h > 0 and breakthrough_time_h > 0:
                bed_utilization = breakthrough_time_h / saturation_time_h
            else:
                bed_utilization = 0.0
                warnings.append("Could not calculate bed utilization")

            computation_time = time.time() - start_time

            return CADETResult(
                success=True,
                model_type="GRM",
                breakthrough_time_h=breakthrough_time_h,
                saturation_time_h=saturation_time_h,
                bed_utilization=bed_utilization,
                curve=curve,
                computation_time_s=computation_time,
                warnings=warnings,
                parameters={
                    "axial_dispersion": column.axial_dispersion,
                    "film_diffusion": column.film_diffusion[0],
                    "pore_diffusion": column.pore_diffusion[0],
                    "k_a": k_a,
                    "k_d": k_d,
                    "qmax": qmax,
                },
            )

        except Exception as e:
            return CADETResult(
                success=False,
                error=str(e),
                computation_time_s=time.time() - start_time,
            )

    def _find_breakthrough_time(
        self,
        time_h: np.ndarray,
        c_c0: np.ndarray,
        threshold: float,
    ) -> float:
        """Find time when C/C0 reaches threshold."""
        indices = np.where(c_c0 >= threshold)[0]
        if len(indices) > 0:
            return float(time_h[indices[0]])
        return float(time_h[-1])

    def compare_with_wheeler_jonas(
        self,
        config: CADETConfig,
        wj_breakthrough_h: float,
        wj_saturation_h: float,
    ) -> Dict[str, Any]:
        """
        Compare CADET results with Wheeler-Jonas model.

        Args:
            config: CADET configuration
            wj_breakthrough_h: Wheeler-Jonas breakthrough time (hours)
            wj_saturation_h: Wheeler-Jonas saturation time (hours)

        Returns:
            Comparison dictionary with deviations and recommendations
        """
        cadet_result = self.simulate(config)

        if not cadet_result.success:
            return {
                "comparison_available": False,
                "error": cadet_result.error,
                "wheeler_jonas": {
                    "breakthrough_time_h": wj_breakthrough_h,
                    "saturation_time_h": wj_saturation_h,
                },
            }

        bt_deviation = (
            (cadet_result.breakthrough_time_h - wj_breakthrough_h) / wj_breakthrough_h
        ) * 100 if wj_breakthrough_h > 0 else 0

        st_deviation = (
            (cadet_result.saturation_time_h - wj_saturation_h) / wj_saturation_h
        ) * 100 if wj_saturation_h > 0 else 0

        recommendations = []
        if abs(bt_deviation) > 20:
            recommendations.append(
                "Significant deviation in breakthrough time. "
                "Consider verifying mass transfer parameters."
            )
        if abs(st_deviation) > 20:
            recommendations.append(
                "Significant deviation in saturation time. "
                "Consider verifying isotherm parameters."
            )

        return {
            "comparison_available": True,
            "wheeler_jonas": {
                "breakthrough_time_h": wj_breakthrough_h,
                "saturation_time_h": wj_saturation_h,
            },
            "cadet": {
                "breakthrough_time_h": cadet_result.breakthrough_time_h,
                "saturation_time_h": cadet_result.saturation_time_h,
                "bed_utilization": cadet_result.bed_utilization,
                "curve": cadet_result.curve,
            },
            "deviations": {
                "breakthrough_percent": round(bt_deviation, 2),
                "saturation_percent": round(st_deviation, 2),
            },
            "recommendations": recommendations,
            "computation_time_s": cadet_result.computation_time_s,
        }


def get_available_models() -> List[Dict[str, Any]]:
    """Get list of available CADET binding models."""
    models = [
        {
            "id": "langmuir",
            "name": "Multi-Component Langmuir",
            "description": "Competitive adsorption with saturation capacity",
            "parameters": ["K (equilibrium constant)", "qmax (max capacity)"],
            "recommended_for": "Most VOC adsorption scenarios",
            "available": True,
        },
        {
            "id": "freundlich_ldf",
            "name": "Freundlich LDF",
            "description": "Heterogeneous surface with LDF kinetics",
            "parameters": ["K_f (Freundlich constant)", "n (exponent)", "k_kin (LDF rate)"],
            "recommended_for": "Heterogeneous activated carbons",
            "available": _CADET_AVAILABLE,
        },
        {
            "id": "linear",
            "name": "Linear",
            "description": "Linear isotherm (Henry's law)",
            "parameters": ["K (Henry constant)"],
            "recommended_for": "Low concentration, dilute systems",
            "available": _CADET_AVAILABLE,
        },
    ]
    return models


def get_cadet_status() -> Dict[str, Any]:
    """Get CADET installation status and version info."""
    available, error = check_cadet_available()

    status = {
        "available": available,
        "error": error,
        "version": None,
        "simulator": None,
    }

    if available:
        try:
            from CADETProcess import __version__ as cadet_version
            status["version"] = cadet_version
            status["simulator"] = "CADET-Core"
        except ImportError:
            status["version"] = "unknown"

    return status
