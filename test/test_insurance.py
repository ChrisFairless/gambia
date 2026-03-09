"""Tests for the InsuranceLayer and RiskTransfer classes."""

import warnings
import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI

from climada_gambia.insurance import (
    InsuranceLayer,
    RiskTransfer,
    BOUNDARY_MATCH_RTOL,
    _calc_layer_payout,
    resolve_boundary,
    calc_expected_payout,
)


# ======================================================================
# Fixtures: synthetic exceedance curve
# ======================================================================

@pytest.fixture
def simple_curve():
    """A simple 5-point exceedance curve.

    Return periods: 2, 5, 10, 50, 100
    Frequencies:    0.01, 0.02, 0.1, 0.2, 0.5 (ascending values)
    Losses:         10000, 5000, 1000, 500, 0

    Frequencies are in ascending order: index 0 = lowest frequency = rarest event.
    """
    exceedance_frequencies = np.array([0.01, 0.02, 0.1, 0.2, 0.5])
    exceedance_losses = np.array([10000, 5000, 1000, 500, 0])
    return exceedance_frequencies, exceedance_losses


# ======================================================================
# InsuranceLayer.__init__
# ======================================================================

class TestInsuranceLayerInit:
    def test_basic_creation(self):
        layer = InsuranceLayer("Test", 100, 1000)
        assert layer.name == "Test"
        assert layer.attachment_loss == 100.0
        assert layer.exhaustion_loss == 1000.0
        assert layer.rate_on_line is None
        assert layer.premium is None

    def test_with_premium(self):
        layer = InsuranceLayer("Test", 0, 500, premium=50)
        assert layer.premium == 50.0

    def test_with_rate_on_line(self):
        layer = InsuranceLayer("Test", 100, 1000, rate_on_line=0.125)
        assert layer.rate_on_line == 0.125

    def test_limit_property(self):
        layer = InsuranceLayer("Test", 200, 800)
        assert layer.limit == 600.0

    def test_negative_attachment_raises(self):
        with pytest.raises(ValueError, match="attachment_loss must be >= 0"):
            InsuranceLayer("Bad", -1, 100)

    def test_exhaustion_below_attachment_raises(self):
        with pytest.raises(ValueError, match="exhaustion_loss must be >= attachment_loss"):
            InsuranceLayer("Bad", 500, 100)

    def test_repr(self):
        layer = InsuranceLayer("Fund", 0, 1000, premium=50)
        r = repr(layer)
        assert "InsuranceLayer" in r
        assert "Fund" in r


# ======================================================================
# InsuranceLayer.from_exceedance
# ======================================================================

class TestInsuranceLayerFromExceedance:
    def test_attachment_loss_exhaustion_loss(self, simple_curve):
        freq, losses = simple_curve
        layer = InsuranceLayer.from_exceedance(
            "Direct", freq, losses,
            attachment_loss=500, exhaustion_loss=5000,
        )
        assert layer.attachment_loss == 500.0
        assert layer.exhaustion_loss == 5000.0

    def test_attachment_rp(self, simple_curve):
        freq, losses = simple_curve
        layer = InsuranceLayer.from_exceedance(
            "RP", freq, losses,
            attachment_rp=10, exhaustion_rp=100,
        )
        assert np.isclose(layer.attachment_loss, 1000.0)
        assert np.isclose(layer.exhaustion_loss, 10000.0)

    def test_attachment_frequency(self, simple_curve):
        freq, losses = simple_curve
        layer = InsuranceLayer.from_exceedance(
            "Freq", freq, losses,
            attachment_frequency=0.1, exhaustion_frequency=0.01,
        )
        assert np.isclose(layer.attachment_loss, 1000.0)
        assert np.isclose(layer.exhaustion_loss, 10000.0)

    def test_limit_kwarg(self, simple_curve):
        freq, losses = simple_curve
        layer = InsuranceLayer.from_exceedance(
            "Limit", freq, losses,
            attachment_loss=500, limit=2000,
        )
        assert layer.attachment_loss == 500.0
        assert layer.exhaustion_loss == 2500.0

    def test_rate_on_line_resolves_premium(self, simple_curve):
        freq, losses = simple_curve
        layer = InsuranceLayer.from_exceedance(
            "ROL", freq, losses,
            attachment_rp=10, exhaustion_rp=100,
            rate_on_line=0.125,
        )
        assert layer.rate_on_line == 0.125
        assert layer.premium is not None
        assert layer.premium > 0

    def test_premium_passthrough(self, simple_curve):
        freq, losses = simple_curve
        layer = InsuranceLayer.from_exceedance(
            "Fixed", freq, losses,
            attachment_loss=0, exhaustion_loss=500,
            premium=42.0,
        )
        assert layer.premium == 42.0

    def test_multiple_attachment_specs_raises(self, simple_curve):
        freq, losses = simple_curve
        with pytest.raises(ValueError, match="Exactly one"):
            InsuranceLayer.from_exceedance(
                "Bad", freq, losses,
                attachment_loss=100, attachment_rp=10,
                exhaustion_loss=5000,
            )

    def test_no_attachment_spec_raises(self, simple_curve):
        freq, losses = simple_curve
        with pytest.raises(ValueError, match="Exactly one"):
            InsuranceLayer.from_exceedance(
                "Bad", freq, losses,
                exhaustion_loss=5000,
            )

    def test_multiple_exhaustion_specs_raises(self, simple_curve):
        freq, losses = simple_curve
        with pytest.raises(ValueError, match="Exactly one"):
            InsuranceLayer.from_exceedance(
                "Bad", freq, losses,
                attachment_loss=0,
                exhaustion_loss=5000, exhaustion_rp=100,
            )


# ======================================================================
# InsuranceLayer.calc_expected_payout
# ======================================================================

class TestInsuranceLayerPayout:
    def test_full_curve_layer(self, simple_curve):
        freq, losses = simple_curve
        layer = InsuranceLayer("Full", 0, 10000)
        ep = layer.calc_expected_payout(freq, losses)
        assert ep > 0

    def test_zero_width_layer(self, simple_curve):
        freq, losses = simple_curve
        layer = InsuranceLayer("Zero", 500, 500)
        ep = layer.calc_expected_payout(freq, losses)
        assert np.isclose(ep, 0.0)

    def test_payout_matches_legacy(self, simple_curve):
        """The InsuranceLayer payout should match the legacy calc_expected_payout."""
        freq, losses = simple_curve
        att, exh = 500.0, 5000.0
        layer = InsuranceLayer("Test", att, exh)
        ep_new = layer.calc_expected_payout(freq, losses)
        ep_legacy = calc_expected_payout(
            losses, freq, att, exh, retained=False,
        )
        assert np.isclose(ep_new, ep_legacy, rtol=1e-10)


# ======================================================================
# InsuranceLayer.plot
# ======================================================================

class TestInsuranceLayerPlot:
    def test_plot_returns_axes(self, simple_curve):
        freq, losses = simple_curve
        layer = InsuranceLayer("Test", 500, 5000)
        ax = layer.plot(freq, losses, rp_max=100)
        assert ax is not None
        import matplotlib.pyplot as plt
        plt.close("all")


# ======================================================================
# RiskTransfer
# ======================================================================

class TestRiskTransfer:
    def test_basic_creation(self):
        layers = [
            InsuranceLayer("A", 0, 500, premium=0),
            InsuranceLayer("B", 500, 5000, rate_on_line=0.1),
        ]
        rt = RiskTransfer(layers)
        assert rt.n_layers == 2
        assert rt.layer_names == ["A", "B"]

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="At least one layer"):
            RiskTransfer([])

    def test_get_layer(self):
        layers = [
            InsuranceLayer("A", 0, 500),
            InsuranceLayer("B", 500, 5000),
        ]
        rt = RiskTransfer(layers)
        assert rt.get_layer("B").name == "B"
        with pytest.raises(KeyError):
            rt.get_layer("C")

    def test_gap_warning(self):
        layers = [
            InsuranceLayer("A", 0, 400),
            InsuranceLayer("B", 500, 5000),
        ]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rt = RiskTransfer(layers)
            gap_warnings = [x for x in w if "gap or overlap" in str(x.message).lower()]
            assert len(gap_warnings) == 1

    def test_no_gap_warning_when_matching(self):
        layers = [
            InsuranceLayer("A", 0, 500),
            InsuranceLayer("B", 500, 5000),
        ]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rt = RiskTransfer(layers)
            gap_warnings = [x for x in w if "gap or overlap" in str(x.message).lower()]
            assert len(gap_warnings) == 0

    def test_add_tail_risk_layer(self):
        layers = [InsuranceLayer("A", 0, 5000)]
        rt = RiskTransfer(layers)
        rt.add_tail_risk_layer(10000)
        assert rt.n_layers == 2
        assert rt.layers[-1].name == "Retained tail risk"
        assert rt.layers[-1].attachment_loss == 5000
        assert rt.layers[-1].exhaustion_loss == 10000

    def test_no_tail_risk_when_covered(self):
        layers = [InsuranceLayer("A", 0, 10000)]
        rt = RiskTransfer(layers)
        rt.add_tail_risk_layer(10000)
        assert rt.n_layers == 1  # no tail risk added


# ======================================================================
# RiskTransfer.from_layer_dicts
# ======================================================================

class TestRiskTransferFromDicts:
    def test_two_layer_example(self, simple_curve):
        freq, losses = simple_curve
        dicts = [
            {
                "name": "National Disaster Fund",
                "attachment": {"loss": 0},
                "exhaustion": {"rp": 10},
                "premium": 0,
            },
            {
                "name": "Indemnity insurance",
                "attachment": {"rp": 10},
                "exhaustion": {"rp": 100},
                "rate_on_line": 0.125,
            },
        ]
        rt = RiskTransfer.from_layer_dicts(dicts, freq, losses)
        assert rt.n_layers == 2
        ins = rt.get_layer("Indemnity insurance")
        assert np.isclose(ins.attachment_loss, 1000.0)
        assert np.isclose(ins.exhaustion_loss, 10000.0)
        assert ins.premium is not None

    def test_exceedance_frequency_spec(self, simple_curve):
        freq, losses = simple_curve
        dicts = [
            {
                "name": "Layer",
                "attachment": {"exceedance_frequency": 0.1},
                "exhaustion": {"exceedance_frequency": 0.01},
            },
        ]
        rt = RiskTransfer.from_layer_dicts(dicts, freq, losses)
        layer = rt.layers[0]
        assert np.isclose(layer.attachment_loss, 1000.0)
        assert np.isclose(layer.exhaustion_loss, 10000.0)


# ======================================================================
# RiskTransfer aggregate helpers
# ======================================================================

class TestRiskTransferAggregates:
    def test_calc_all_expected_payouts(self, simple_curve):
        freq, losses = simple_curve
        layers = [
            InsuranceLayer("A", 0, 1000),
            InsuranceLayer("B", 1000, 5000),
        ]
        rt = RiskTransfer(layers)
        payouts = rt.calc_all_expected_payouts(freq, losses)
        assert "A" in payouts
        assert "B" in payouts
        assert all(v >= 0 for v in payouts.values())

    def test_total_payout(self, simple_curve):
        freq, losses = simple_curve
        layers = [
            InsuranceLayer("A", 0, 1000),
            InsuranceLayer("B", 1000, 10000),
        ]
        rt = RiskTransfer(layers)
        total = rt.calc_total_expected_payout(freq, losses)
        # Total should equal AAL of entire curve since layers cover [0, max]
        full = InsuranceLayer("Full", 0, 10000)
        full_ep = full.calc_expected_payout(freq, losses)
        assert np.isclose(total, full_ep, rtol=1e-6)

    def test_summary_df(self, simple_curve):
        freq, losses = simple_curve
        layers = [
            InsuranceLayer("A", 0, 1000, premium=0),
            InsuranceLayer("B", 1000, 5000, rate_on_line=0.1),
        ]
        rt = RiskTransfer(layers)
        df = rt.summary_df(freq, losses)
        assert len(df) == 2
        assert "Layer" in df.columns
        assert "Expected payout" in df.columns


# ======================================================================
# RiskTransfer.plot
# ======================================================================

class TestRiskTransferPlot:
    def test_plot_returns_figure(self, simple_curve):
        freq, losses = simple_curve
        layers = [
            InsuranceLayer("A", 0, 1000),
            InsuranceLayer("B", 1000, 5000),
        ]
        rt = RiskTransfer(layers)
        fig = rt.plot(freq, losses, title="Test plot")
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close("all")


# ======================================================================
# Legacy compatibility: resolve_boundary
# ======================================================================

class TestResolveBoundary:
    def test_loss(self, simple_curve):
        freq, losses = simple_curve
        assert resolve_boundary({"loss": 42}, freq, losses) == 42.0

    def test_rp(self, simple_curve):
        freq, losses = simple_curve
        val = resolve_boundary({"rp": 10}, freq, losses)
        assert np.isclose(val, 1000.0)

    def test_exceedance_frequency(self, simple_curve):
        freq, losses = simple_curve
        val = resolve_boundary({"exceedance_frequency": 0.1}, freq, losses)
        assert np.isclose(val, 1000.0)

    def test_unknown_key_raises(self, simple_curve):
        freq, losses = simple_curve
        with pytest.raises(ValueError, match="Unknown"):
            resolve_boundary({"bad": 1}, freq, losses)


# ======================================================================
# Legacy compatibility: calc_expected_payout with retained=True
# ======================================================================

class TestCalcExpectedPayoutRetained:
    def test_retained_plus_insured_equals_total(self, simple_curve):
        freq, losses = simple_curve
        att, exh = 500.0, 5000.0
        insured = calc_expected_payout(losses, freq, att, exh, retained=False)
        retained = calc_expected_payout(losses, freq, att, exh, retained=True)
        total = InsuranceLayer("Full", 0, 10000).calc_expected_payout(freq, losses)
        assert np.isclose(insured + retained, total, rtol=1e-6)
