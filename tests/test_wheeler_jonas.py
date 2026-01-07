"""Tests for Wheeler-Jonas breakthrough calculations."""
import pytest
from app.services.wheeler_jonas import (
    estimate_capacity,
    estimate_kv,
    calculate_breakthrough_time,
    generate_breakthrough_curve,
    calculate_pollutant_result,
    TYPICAL_CAPACITIES,
    TYPICAL_KV,
)


class TestEstimateCapacity:
    """Tests for adsorption capacity estimation."""

    def test_known_pollutant_capacity(self):
        """Known pollutants should use lookup table values."""
        capacity = estimate_capacity(
            pollutant_name="toluene",
            concentration=100.0,
            temperature=25.0,
            humidity=50.0,
            surface_area=1000.0,
        )
        # Should be positive and reasonable
        assert capacity > 0
        assert capacity < 1.0  # Less than 100% by mass

    def test_unknown_pollutant_uses_default(self):
        """Unknown pollutants should use default capacity."""
        capacity = estimate_capacity(
            pollutant_name="unknown_compound",
            concentration=100.0,
            temperature=25.0,
            humidity=50.0,
            surface_area=1000.0,
        )
        assert capacity > 0

    def test_surface_area_affects_capacity(self):
        """Higher surface area should increase capacity."""
        cap_low = estimate_capacity("toluene", 100, 25, 50, 800)
        cap_high = estimate_capacity("toluene", 100, 25, 50, 1200)
        assert cap_high > cap_low

    def test_humidity_affects_capacity(self):
        """Higher humidity should decrease capacity."""
        cap_dry = estimate_capacity("toluene", 100, 25, 30, 1000)
        cap_wet = estimate_capacity("toluene", 100, 25, 80, 1000)
        assert cap_dry > cap_wet

    def test_temperature_affects_capacity(self):
        """Temperature should affect capacity."""
        cap_cold = estimate_capacity("toluene", 100, 10, 50, 1000)
        cap_hot = estimate_capacity("toluene", 100, 40, 50, 1000)
        # Capacity should be different at different temperatures
        assert cap_cold != cap_hot
        # Both should be positive and reasonable
        assert cap_cold > 0
        assert cap_hot > 0

    def test_minimum_capacity(self):
        """Capacity should never be below minimum threshold."""
        capacity = estimate_capacity(
            pollutant_name="unknown",
            concentration=0.1,
            temperature=80.0,
            humidity=95.0,
            surface_area=100.0,
        )
        assert capacity >= 0.01


class TestEstimateKv:
    """Tests for mass transfer coefficient estimation."""

    def test_known_pollutant_kv(self):
        """Known pollutants should have specific k_v values."""
        kv_h2s = estimate_kv("h2s", 0.3)
        kv_toluene = estimate_kv("toluene", 0.3)
        # H2S typically has higher k_v
        assert kv_h2s > kv_toluene

    def test_velocity_affects_kv(self):
        """Higher velocity should increase k_v."""
        kv_slow = estimate_kv("toluene", 0.1)
        kv_fast = estimate_kv("toluene", 0.5)
        assert kv_fast > kv_slow

    def test_particle_size_affects_kv(self):
        """Smaller particles should increase k_v."""
        kv_large = estimate_kv("toluene", 0.3, particle_diameter=5.0)
        kv_small = estimate_kv("toluene", 0.3, particle_diameter=2.0)
        assert kv_small > kv_large


class TestCalculateBreakthroughTime:
    """Tests for Wheeler-Jonas breakthrough time calculation."""

    def test_basic_breakthrough_calculation(self):
        """Basic breakthrough time should be positive."""
        t_b, mtz = calculate_breakthrough_time(
            W_e=0.3,
            carbon_mass=100.0,  # kg
            bulk_density=450.0,  # kg/m³
            C_in=100.0,  # mg/m³
            C_out=5.0,  # mg/m³
            flow_rate=1000.0,  # m³/h
            k_v=5.0,  # 1/min
            bed_height=0.5,  # m
        )
        assert t_b > 0
        assert mtz > 0
        assert mtz < 0.5  # MTZ should be less than bed height

    def test_higher_capacity_longer_breakthrough(self):
        """Higher capacity should give longer breakthrough time."""
        t_low, _ = calculate_breakthrough_time(0.2, 100, 450, 100, 5, 1000, 5, 0.5)
        t_high, _ = calculate_breakthrough_time(0.4, 100, 450, 100, 5, 1000, 5, 0.5)
        assert t_high > t_low

    def test_higher_concentration_shorter_breakthrough(self):
        """Higher inlet concentration should give shorter breakthrough."""
        t_low, _ = calculate_breakthrough_time(0.3, 100, 450, 50, 5, 1000, 5, 0.5)
        t_high, _ = calculate_breakthrough_time(0.3, 100, 450, 200, 5, 1000, 5, 0.5)
        assert t_low > t_high

    def test_higher_flow_rate_shorter_breakthrough(self):
        """Higher flow rate should give shorter breakthrough."""
        t_low, _ = calculate_breakthrough_time(0.3, 100, 450, 100, 5, 500, 5, 0.5)
        t_high, _ = calculate_breakthrough_time(0.3, 100, 450, 100, 5, 2000, 5, 0.5)
        assert t_low > t_high


class TestGenerateBreakthroughCurve:
    """Tests for breakthrough curve generation."""

    def test_curve_has_correct_points(self):
        """Curve should have specified number of points."""
        curve = generate_breakthrough_curve(100, num_points=50)
        assert len(curve) == 50

    def test_curve_starts_at_zero(self):
        """Curve should start with C/C0 near 0."""
        curve = generate_breakthrough_curve(100)
        assert curve[0].time == 0
        assert curve[0].c_c0 < 0.01  # Should be very low

    def test_curve_approaches_one(self):
        """Curve should approach C/C0 = 1.0 at end."""
        curve = generate_breakthrough_curve(100)
        assert curve[-1].c_c0 > 0.9  # Should be high

    def test_curve_is_monotonic(self):
        """Curve should be monotonically increasing."""
        curve = generate_breakthrough_curve(100)
        for i in range(1, len(curve)):
            assert curve[i].c_c0 >= curve[i - 1].c_c0


class TestCalculatePollutantResult:
    """Tests for complete pollutant result calculation."""

    def test_basic_result_structure(self):
        """Result should have all required fields."""
        result = calculate_pollutant_result(
            pollutant_name="toluene",
            concentration=100.0,
            target_outlet=5.0,
            temperature=25.0,
            humidity=50.0,
            surface_area=1050.0,
            carbon_mass=100.0,
            bulk_density=450.0,
            flow_rate=1000.0,
            velocity=0.3,
            bed_height=0.5,
            particle_diameter=3.0,
            molecular_weight=92.14,
        )

        assert result.name == "toluene"
        assert result.inlet_concentration == 100.0
        assert result.outlet_concentration == 5.0
        assert result.removal_efficiency > 0
        assert result.adsorption_capacity > 0
        assert result.breakthrough_time > 0
        assert result.mass_transfer_zone > 0

    def test_removal_efficiency_calculation(self):
        """Removal efficiency should be calculated correctly."""
        result = calculate_pollutant_result(
            pollutant_name="benzene",
            concentration=100.0,
            target_outlet=10.0,  # 10% breakthrough
            temperature=25.0,
            humidity=50.0,
            surface_area=1000.0,
            carbon_mass=50.0,
            bulk_density=450.0,
            flow_rate=500.0,
            velocity=0.3,
            bed_height=0.3,
            particle_diameter=3.0,
            molecular_weight=78.11,
        )

        # 10% breakthrough = 90% efficiency
        assert result.removal_efficiency == pytest.approx(90.0, rel=0.1)

    def test_default_breakthrough_concentration(self):
        """Without target, should default to 5% breakthrough."""
        result = calculate_pollutant_result(
            pollutant_name="toluene",
            concentration=100.0,
            target_outlet=None,
            temperature=25.0,
            humidity=50.0,
            surface_area=1000.0,
            carbon_mass=50.0,
            bulk_density=450.0,
            flow_rate=500.0,
            velocity=0.3,
            bed_height=0.3,
            particle_diameter=3.0,
            molecular_weight=92.14,
        )

        assert result.outlet_concentration == pytest.approx(5.0, rel=0.1)
        assert result.removal_efficiency == pytest.approx(95.0, rel=0.1)
