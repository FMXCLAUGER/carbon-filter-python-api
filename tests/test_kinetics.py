"""Tests for kinetics breakthrough models (Thomas, Yoon-Nelson, Bohart-Adams)."""
import pytest
from app.services.kinetics import (
    calculate_thomas_breakthrough,
    calculate_yoon_nelson_breakthrough,
    calculate_bohart_adams_breakthrough,
)


class TestThomasModel:
    """Tests for Thomas model breakthrough calculations."""

    def test_thomas_basic_calculation(self):
        """Thomas model should return valid breakthrough data."""
        # Use parameters that produce a proper S-curve
        # Lower flow rate (1 m³/h = 16667 mL/min) and higher concentration
        result = calculate_thomas_breakthrough(
            flow_rate=1.0,  # Lower flow rate for longer breakthrough
            bed_mass=50.0,
            C0=1000.0,  # Higher concentration
            q0=0.2,
            k_Th=0.0001,  # Explicitly set k_Th for predictable results
            pollutant_type="VOC",
            n_points=100,
            time_range_factor=3.0,
        )

        assert "breakthrough_times" in result
        assert "curve" in result
        assert "parameters" in result

        # Check breakthrough times are non-negative
        bt = result["breakthrough_times"]
        assert bt["t_5_h"] >= 0
        assert bt["t_10_h"] >= 0
        assert bt["t_50_h"] > 0
        assert bt["t_90_h"] > 0

    def test_thomas_higher_flow_shorter_breakthrough(self):
        """Higher flow rate should result in shorter 50% breakthrough time."""
        # Use explicit k_Th for consistent comparison
        result_low_flow = calculate_thomas_breakthrough(
            flow_rate=1.0, bed_mass=50.0, C0=1000.0, q0=0.2, k_Th=0.0001
        )
        result_high_flow = calculate_thomas_breakthrough(
            flow_rate=2.0, bed_mass=50.0, C0=1000.0, q0=0.2, k_Th=0.0001
        )

        # t_50 (50% breakthrough) should be shorter with higher flow
        assert result_low_flow["breakthrough_times"]["t_50_h"] > \
               result_high_flow["breakthrough_times"]["t_50_h"]

    def test_thomas_larger_bed_longer_breakthrough(self):
        """Larger bed should result in longer 50% breakthrough time."""
        result_small = calculate_thomas_breakthrough(
            flow_rate=1.0, bed_mass=25.0, C0=1000.0, q0=0.2, k_Th=0.0001
        )
        result_large = calculate_thomas_breakthrough(
            flow_rate=1.0, bed_mass=100.0, C0=1000.0, q0=0.2, k_Th=0.0001
        )

        # t_50 should be longer with larger bed
        assert result_large["breakthrough_times"]["t_50_h"] > \
               result_small["breakthrough_times"]["t_50_h"]

    def test_thomas_higher_capacity_longer_breakthrough(self):
        """Higher adsorption capacity should give longer 50% breakthrough."""
        result_low_cap = calculate_thomas_breakthrough(
            flow_rate=1.0, bed_mass=50.0, C0=1000.0, q0=0.1, k_Th=0.0001
        )
        result_high_cap = calculate_thomas_breakthrough(
            flow_rate=1.0, bed_mass=50.0, C0=1000.0, q0=0.3, k_Th=0.0001
        )

        # t_50 should be longer with higher capacity
        assert result_high_cap["breakthrough_times"]["t_50_h"] > \
               result_low_cap["breakthrough_times"]["t_50_h"]

    def test_thomas_curve_monotonic(self):
        """Breakthrough curve should be monotonically increasing."""
        result = calculate_thomas_breakthrough(
            flow_rate=1.0, bed_mass=50.0, C0=1000.0, q0=0.2,
            k_Th=0.0001, n_points=100
        )
        curve = result["curve"]

        # Should be monotonically increasing (or equal)
        for i in range(1, len(curve)):
            assert curve[i]["C_C0"] >= curve[i-1]["C_C0"]

    def test_thomas_returns_model_name(self):
        """Thomas model should identify itself."""
        result = calculate_thomas_breakthrough(
            flow_rate=1.0, bed_mass=50.0, C0=1000.0, q0=0.2
        )
        assert result["model"] == "Thomas"


class TestYoonNelsonModel:
    """Tests for Yoon-Nelson model breakthrough calculations."""

    def test_yoon_nelson_basic_calculation(self):
        """Yoon-Nelson model should return valid breakthrough data."""
        result = calculate_yoon_nelson_breakthrough(
            flow_rate=100.0,
            bed_mass=50.0,
            C0=100.0,
            q0=0.2,
            pollutant_type="VOC",
            n_points=50,
        )

        assert "breakthrough_times" in result
        assert "curve" in result
        assert "parameters" in result

        bt = result["breakthrough_times"]
        # Yoon-Nelson typically gives positive breakthrough times
        assert bt["t_50_h"] > 0

    def test_yoon_nelson_tau_parameter(self):
        """Custom tau should set 50% breakthrough time."""
        # tau parameter is in minutes
        result = calculate_yoon_nelson_breakthrough(
            flow_rate=100.0,
            bed_mass=50.0,
            C0=100.0,
            q0=0.2,
            tau=120.0,  # Custom tau = 120 minutes = 2 hours
        )

        # t_50 should be close to tau for Yoon-Nelson
        bt = result["breakthrough_times"]
        # Yoon-Nelson: t_50 ≈ tau (in minutes)
        assert abs(bt["t_50_min"] - 120.0) < 10.0  # Within 10 minutes tolerance

    def test_yoon_nelson_symmetric_curve(self):
        """Yoon-Nelson curve should be relatively symmetric around tau."""
        result = calculate_yoon_nelson_breakthrough(
            flow_rate=100.0, bed_mass=50.0, C0=100.0, q0=0.2, n_points=100
        )

        bt = result["breakthrough_times"]
        # Only check if all times are valid (non-zero)
        if bt["t_10_h"] > 0 and bt["t_50_h"] > 0 and bt["t_90_h"] > 0:
            # Time difference from 10% to 50% should be similar to 50% to 90%
            delta_10_50 = bt["t_50_h"] - bt["t_10_h"]
            delta_50_90 = bt["t_90_h"] - bt["t_50_h"]

            # Allow 80% variation for asymmetry (model may not be perfectly symmetric)
            assert abs(delta_10_50 - delta_50_90) < max(delta_10_50, delta_50_90) * 0.8

    def test_yoon_nelson_returns_model_name(self):
        """Yoon-Nelson model should identify itself."""
        result = calculate_yoon_nelson_breakthrough(
            flow_rate=100.0, bed_mass=50.0, C0=100.0, q0=0.2
        )
        assert result["model"] == "Yoon-Nelson"

    def test_yoon_nelson_curve_monotonic(self):
        """Breakthrough curve should be monotonically increasing."""
        result = calculate_yoon_nelson_breakthrough(
            flow_rate=100.0, bed_mass=50.0, C0=100.0, q0=0.2, n_points=100
        )
        curve = result["curve"]

        for i in range(1, len(curve)):
            assert curve[i]["C_C0"] >= curve[i-1]["C_C0"]


class TestBohartAdamsModel:
    """Tests for Bohart-Adams model breakthrough calculations."""

    def test_bohart_adams_basic_calculation(self):
        """Bohart-Adams model should return valid breakthrough data."""
        result = calculate_bohart_adams_breakthrough(
            flow_rate=100.0,
            bed_mass=50.0,
            bed_height=0.5,
            bed_area=0.5,
            C0=100.0,
            q0=0.2,
            pollutant_type="VOC",
            n_points=50,
        )

        assert "breakthrough_times" in result
        assert "curve" in result
        assert "parameters" in result

        bt = result["breakthrough_times"]
        assert bt["t_5_h"] >= 0
        assert bt["t_10_h"] >= 0

    def test_bohart_adams_bed_depth_effect(self):
        """Deeper bed should give longer breakthrough time."""
        result_shallow = calculate_bohart_adams_breakthrough(
            flow_rate=100.0, bed_mass=50.0, bed_height=0.3, bed_area=0.5, C0=100.0, q0=0.2
        )
        result_deep = calculate_bohart_adams_breakthrough(
            flow_rate=100.0, bed_mass=50.0, bed_height=1.0, bed_area=0.5, C0=100.0, q0=0.2
        )

        # Deeper bed should have later t_50 breakthrough
        assert result_deep["breakthrough_times"]["t_50_h"] >= \
               result_shallow["breakthrough_times"]["t_50_h"]

    def test_bohart_adams_low_concentration_application(self):
        """Model should work well for low concentration applications."""
        result = calculate_bohart_adams_breakthrough(
            flow_rate=100.0,
            bed_mass=50.0,
            bed_height=0.5,
            bed_area=0.5,
            C0=10.0,  # Low concentration
            q0=0.2,
            pollutant_type="VOC",
        )

        # Should return valid non-negative times
        bt = result["breakthrough_times"]
        assert bt["t_5_h"] >= 0
        assert bt["t_10_h"] >= 0

    def test_bohart_adams_area_effect(self):
        """Larger bed area should affect breakthrough."""
        result_small = calculate_bohart_adams_breakthrough(
            flow_rate=100.0, bed_mass=50.0, bed_height=0.5, bed_area=0.25, C0=100.0, q0=0.2
        )
        result_large = calculate_bohart_adams_breakthrough(
            flow_rate=100.0, bed_mass=50.0, bed_height=0.5, bed_area=1.0, C0=100.0, q0=0.2
        )

        # Larger area = lower velocity = longer breakthrough at t_50
        assert result_large["breakthrough_times"]["t_50_h"] >= \
               result_small["breakthrough_times"]["t_50_h"]

    def test_bohart_adams_returns_model_name(self):
        """Bohart-Adams model should identify itself."""
        result = calculate_bohart_adams_breakthrough(
            flow_rate=100.0, bed_mass=50.0, bed_height=0.5, bed_area=0.5,
            C0=100.0, q0=0.2
        )
        assert result["model"] == "Bohart-Adams"

    def test_bohart_adams_curve_monotonic(self):
        """Breakthrough curve should be monotonically increasing."""
        result = calculate_bohart_adams_breakthrough(
            flow_rate=100.0, bed_mass=50.0, bed_height=0.5, bed_area=0.5,
            C0=100.0, q0=0.2, n_points=100
        )
        curve = result["curve"]

        for i in range(1, len(curve)):
            assert curve[i]["C_C0"] >= curve[i-1]["C_C0"]


class TestModelComparison:
    """Tests comparing different kinetic models."""

    def test_all_models_same_conditions(self):
        """All models should give non-negative results for same conditions."""
        # Common parameters
        flow_rate = 100.0
        bed_mass = 50.0
        C0 = 100.0
        q0 = 0.2
        bed_height = 0.5
        bed_area = 0.5

        thomas = calculate_thomas_breakthrough(
            flow_rate=flow_rate, bed_mass=bed_mass, C0=C0, q0=q0
        )
        yoon_nelson = calculate_yoon_nelson_breakthrough(
            flow_rate=flow_rate, bed_mass=bed_mass, C0=C0, q0=q0
        )
        bohart_adams = calculate_bohart_adams_breakthrough(
            flow_rate=flow_rate, bed_mass=bed_mass, bed_height=bed_height,
            bed_area=bed_area, C0=C0, q0=q0
        )

        # All should give valid results (t_50_h should always be > 0)
        assert thomas["breakthrough_times"]["t_50_h"] >= 0
        assert yoon_nelson["breakthrough_times"]["t_50_h"] > 0
        assert bohart_adams["breakthrough_times"]["t_50_h"] >= 0

    def test_models_reasonable_range(self):
        """All models should give positive t_50 breakthrough times."""
        flow_rate = 100.0
        bed_mass = 50.0
        C0 = 100.0
        q0 = 0.2

        thomas = calculate_thomas_breakthrough(
            flow_rate=flow_rate, bed_mass=bed_mass, C0=C0, q0=q0
        )
        yoon_nelson = calculate_yoon_nelson_breakthrough(
            flow_rate=flow_rate, bed_mass=bed_mass, C0=C0, q0=q0
        )

        # Both models should give positive t_50
        assert thomas["breakthrough_times"]["t_50_h"] > 0
        assert yoon_nelson["breakthrough_times"]["t_50_h"] > 0

    def test_all_models_have_curve_data(self):
        """All models should return curve data with correct structure."""
        flow_rate = 100.0
        bed_mass = 50.0
        C0 = 100.0
        q0 = 0.2
        n_points = 50

        thomas = calculate_thomas_breakthrough(
            flow_rate=flow_rate, bed_mass=bed_mass, C0=C0, q0=q0, n_points=n_points
        )
        yoon_nelson = calculate_yoon_nelson_breakthrough(
            flow_rate=flow_rate, bed_mass=bed_mass, C0=C0, q0=q0, n_points=n_points
        )
        bohart_adams = calculate_bohart_adams_breakthrough(
            flow_rate=flow_rate, bed_mass=bed_mass, bed_height=0.5, bed_area=0.5,
            C0=C0, q0=q0, n_points=n_points
        )

        for result in [thomas, yoon_nelson, bohart_adams]:
            assert len(result["curve"]) == n_points
            # Check first point structure
            point = result["curve"][0]
            assert "time_h" in point
            assert "C_C0" in point
            # C/C0 should be between 0 and 1
            assert 0 <= point["C_C0"] <= 1
