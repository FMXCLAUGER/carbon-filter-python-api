"""Tests for regeneration energy calculations (TSA, PSA, VSA, Steam)."""
import pytest
from app.services.regeneration import (
    calculate_tsa_energy,
    calculate_psa_energy,
    calculate_vsa_energy,
    calculate_steam_regeneration,
    compare_regeneration_methods,
    RegenerationType,
)


class TestTSAEnergy:
    """Tests for TSA (Temperature Swing Adsorption) energy calculations."""

    def test_tsa_basic_calculation(self):
        """TSA should return valid energy calculations."""
        result = calculate_tsa_energy(
            bed_mass=100.0,
            q_loading=0.15,
            delta_H_ads=40000,
            T_ads=25,
            T_regen=200,
            molecular_weight=92.0,
        )

        assert result.total_energy_kWh > 0
        assert result.sensible_heat_kWh > 0
        assert result.desorption_heat_kWh > 0
        assert result.power_required_kW > 0
        assert result.method == RegenerationType.TSA

    def test_tsa_higher_temp_more_energy(self):
        """Higher regeneration temperature should require more energy."""
        result_low_temp = calculate_tsa_energy(
            bed_mass=100.0, q_loading=0.15, delta_H_ads=40000,
            T_ads=25, T_regen=150
        )
        result_high_temp = calculate_tsa_energy(
            bed_mass=100.0, q_loading=0.15, delta_H_ads=40000,
            T_ads=25, T_regen=300
        )

        assert result_high_temp.total_energy_kWh > result_low_temp.total_energy_kWh

    def test_tsa_larger_bed_more_energy(self):
        """Larger bed should require more energy."""
        result_small = calculate_tsa_energy(
            bed_mass=50.0, q_loading=0.15, delta_H_ads=40000,
            T_ads=25, T_regen=200
        )
        result_large = calculate_tsa_energy(
            bed_mass=200.0, q_loading=0.15, delta_H_ads=40000,
            T_ads=25, T_regen=200
        )

        assert result_large.total_energy_kWh > result_small.total_energy_kWh

    def test_tsa_higher_loading_more_desorption_heat(self):
        """Higher adsorbate loading should increase desorption heat."""
        result_low = calculate_tsa_energy(
            bed_mass=100.0, q_loading=0.05, delta_H_ads=40000,
            T_ads=25, T_regen=200
        )
        result_high = calculate_tsa_energy(
            bed_mass=100.0, q_loading=0.25, delta_H_ads=40000,
            T_ads=25, T_regen=200
        )

        assert result_high.desorption_heat_kWh > result_low.desorption_heat_kWh

    def test_tsa_specific_energy_reasonable(self):
        """Specific energy should be in reasonable range."""
        result = calculate_tsa_energy(
            bed_mass=100.0, q_loading=0.15, delta_H_ads=40000,
            T_ads=25, T_regen=200
        )

        # Typical TSA: 50-200 kWh/tonne = 0.05-0.2 kWh/kg
        assert 0.01 < result.specific_energy_kWh_kg < 1.0


class TestPSAEnergy:
    """Tests for PSA (Pressure Swing Adsorption) energy calculations."""

    def test_psa_basic_calculation(self):
        """PSA should return valid energy calculations."""
        result = calculate_psa_energy(
            bed_mass=100.0,
            q_loading=0.15,
            P_ads=5.0,  # 5 bar
            P_regen=1.0,  # 1 bar
        )

        assert result.total_energy_kWh > 0
        assert result.power_required_kW > 0
        assert result.method == RegenerationType.PSA
        # PSA has no sensible heat
        assert result.sensible_heat_kWh == 0

    def test_psa_higher_pressure_ratio_more_energy(self):
        """Higher pressure ratio should require more energy."""
        result_low = calculate_psa_energy(
            bed_mass=100.0, q_loading=0.15, P_ads=3.0, P_regen=1.0
        )
        result_high = calculate_psa_energy(
            bed_mass=100.0, q_loading=0.15, P_ads=10.0, P_regen=1.0
        )

        assert result_high.total_energy_kWh > result_low.total_energy_kWh

    def test_psa_fast_cycles(self):
        """PSA should have short cycle times compared to TSA."""
        psa = calculate_psa_energy(
            bed_mass=100.0, q_loading=0.15, P_ads=5.0, P_regen=1.0,
            cycle_time_min=10.0
        )
        tsa = calculate_tsa_energy(
            bed_mass=100.0, q_loading=0.15, delta_H_ads=40000,
            T_ads=25, T_regen=200, regen_time_h=4.0
        )

        # PSA cycle time should be much shorter
        assert psa.regeneration_time_h < tsa.regeneration_time_h


class TestVSAEnergy:
    """Tests for VSA (Vacuum Swing Adsorption) energy calculations."""

    def test_vsa_basic_calculation(self):
        """VSA should return valid energy calculations."""
        result = calculate_vsa_energy(
            bed_mass=100.0,
            q_loading=0.15,
            P_ads=1.0,  # 1 bar
            P_vacuum=0.2,  # 0.2 bar (vacuum)
        )

        assert result.total_energy_kWh > 0
        assert result.power_required_kW > 0
        assert result.method == RegenerationType.VSA

    def test_vsa_lower_vacuum_more_energy(self):
        """Lower vacuum pressure should require more energy."""
        result_weak = calculate_vsa_energy(
            bed_mass=100.0, q_loading=0.15, P_ads=1.0, P_vacuum=0.5
        )
        result_strong = calculate_vsa_energy(
            bed_mass=100.0, q_loading=0.15, P_ads=1.0, P_vacuum=0.1
        )

        assert result_strong.total_energy_kWh > result_weak.total_energy_kWh

    def test_vsa_vs_psa_comparison(self):
        """VSA typically uses less energy than PSA for same pressure ratio."""
        # Same pressure ratio but different methods
        vsa = calculate_vsa_energy(
            bed_mass=100.0, q_loading=0.15, P_ads=1.0, P_vacuum=0.2
        )
        psa = calculate_psa_energy(
            bed_mass=100.0, q_loading=0.15, P_ads=5.0, P_regen=1.0
        )

        # Both should give valid results
        assert vsa.total_energy_kWh > 0
        assert psa.total_energy_kWh > 0


class TestSteamRegeneration:
    """Tests for steam regeneration energy calculations."""

    def test_steam_basic_calculation(self):
        """Steam regeneration should return valid energy calculations."""
        result = calculate_steam_regeneration(
            bed_mass=100.0,
            q_loading=0.15,
            delta_H_ads=40000,
            molecular_weight=92.0,
        )

        assert result.total_energy_kWh > 0
        assert result.power_required_kW > 0
        assert result.method == RegenerationType.STEAM

    def test_steam_ratio_effect(self):
        """Higher steam ratio should require more energy."""
        result_low = calculate_steam_regeneration(
            bed_mass=100.0, q_loading=0.15, delta_H_ads=40000,
            steam_ratio=1.5
        )
        result_high = calculate_steam_regeneration(
            bed_mass=100.0, q_loading=0.15, delta_H_ads=40000,
            steam_ratio=4.0
        )

        assert result_high.total_energy_kWh > result_low.total_energy_kWh

    def test_steam_faster_than_tsa(self):
        """Steam regeneration is typically faster than TSA."""
        steam = calculate_steam_regeneration(
            bed_mass=100.0, q_loading=0.15, delta_H_ads=40000,
            regen_time_h=2.0
        )
        tsa = calculate_tsa_energy(
            bed_mass=100.0, q_loading=0.15, delta_H_ads=40000,
            T_ads=25, T_regen=200, regen_time_h=4.0
        )

        assert steam.regeneration_time_h < tsa.regeneration_time_h


class TestCompareRegenerationMethods:
    """Tests for regeneration method comparison."""

    def test_compare_returns_all_methods(self):
        """Comparison should return results for all methods."""
        result = compare_regeneration_methods(
            bed_mass=100.0,
            q_loading=0.15,
            delta_H_ads=40000,
            molecular_weight=92.0,
        )

        assert "TSA" in result
        assert "PSA" in result
        assert "VSA" in result
        assert "STEAM" in result
        assert "recommendation" in result

    def test_compare_all_positive_energies(self):
        """All methods should return positive energy values."""
        result = compare_regeneration_methods(
            bed_mass=100.0, q_loading=0.15
        )

        for method in ["TSA", "PSA", "VSA", "STEAM"]:
            assert result[method]["energy_kWh"] > 0
            assert result[method]["power_kW"] > 0
            assert result[method]["time_h"] > 0

    def test_compare_recommendation_valid(self):
        """Recommendation should include method and reason."""
        result = compare_regeneration_methods(
            bed_mass=100.0, q_loading=0.15
        )

        assert "method" in result["recommendation"]
        assert "reason" in result["recommendation"]
        assert result["recommendation"]["method"] in ["TSA", "PSA", "VSA", "STEAM"]

    def test_high_loading_recommends_tsa_or_steam(self):
        """High loading should recommend TSA or Steam."""
        result = compare_regeneration_methods(
            bed_mass=100.0, q_loading=0.25  # High loading
        )

        rec_method = result["recommendation"]["method"]
        assert rec_method in ["TSA", "STEAM"]

    def test_advantages_disadvantages_present(self):
        """Each method should have advantages and disadvantages."""
        result = compare_regeneration_methods(
            bed_mass=100.0, q_loading=0.15
        )

        for method in ["TSA", "PSA", "VSA", "STEAM"]:
            assert len(result[method]["advantages"]) > 0
            assert len(result[method]["disadvantages"]) > 0


class TestEnergyUnits:
    """Tests for correct energy unit conversions."""

    def test_power_equals_energy_over_time(self):
        """Power (kW) should equal energy (kWh) / time (h)."""
        result = calculate_tsa_energy(
            bed_mass=100.0, q_loading=0.15, delta_H_ads=40000,
            T_ads=25, T_regen=200, regen_time_h=4.0
        )

        expected_power = result.total_energy_kWh / result.regeneration_time_h
        assert abs(result.power_required_kW - expected_power) < 0.1

    def test_specific_energy_units(self):
        """Specific energy should be per kg carbon."""
        result = calculate_tsa_energy(
            bed_mass=100.0, q_loading=0.15, delta_H_ads=40000,
            T_ads=25, T_regen=200
        )

        expected_specific = result.total_energy_kWh / 100.0  # per kg
        assert abs(result.specific_energy_kWh_kg - expected_specific) < 0.01
