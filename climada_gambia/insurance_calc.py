"""Insurance policy evaluation and plotting.

This module contains the high-level routines that sweep over a parameter
space of insurance policies (varying attachment RP and rate-on-line) and
evaluate each combination against uncertainty simulations.

It uses the :class:`~climada_gambia.insurance.InsuranceLayer` and
:class:`~climada_gambia.insurance.RiskTransfer` classes defined in
:mod:`climada_gambia.insurance`.
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from functools import partial

from climada_gambia.metadata_impact import MetadataImpact
from climada_gambia.utils_total_exposed_value import get_total_exposed_value
from climada_gambia.uncertainty import gather_uncertainty_results, get_total_exposed_ratio
from climada_gambia import utils_config
from climada_gambia.scoring import calc_aal_trapezoidal
from climada_gambia.analyse_impacts import get_impf_exceedance_curves

from climada_gambia.insurance import (
    InsuranceLayer,
    RiskTransfer,
    BOUNDARY_MATCH_RTOL,
    calc_expected_payout,
)

# ======================================================================
# Default configuration
# ======================================================================

ANALYSIS_NAME = "calibration"
overwrite = True

INSURANCE_POLICIES = {
    "exhaustion_rp": 1000,
    "attachment_rp_vals": np.arange(5, 55, 5),
    "rate_on_line_vals": np.arange(0.10, 0.41, 0.025),
}

EXAMPLE_POLICY = {
    "exhaustion_rp": 1000,
    "attachment_rp": 5,
    "rate_on_line": 0.125,
}

INSURED_EXPOSURE = "economic_assets"


# ======================================================================
# Helpers
# ======================================================================

def build_impf_dict(insured_exposure, analysis_name):
    impf_dict = utils_config.gather_impact_calculation_metadata(
        filter={'exposure_type': insured_exposure}, analysis_name=analysis_name,
    )[0]
    analysis_name_full = (
        f"{analysis_name}/{impf_dict['exposure_type']}_{impf_dict['exposure_source']}"
    )
    impf_dict = MetadataImpact(impf_dict=impf_dict, analysis_name=analysis_name_full)
    return impf_dict


def _format_currency(value, unit="USD"):
    if value >= 1e6:
        return f"{value / 1e6:,.1f} mn {unit}"
    elif value >= 1e3:
        return f"{value / 1e3:,.1f} k {unit}"
    else:
        return f"{value:,.0f} {unit}"


# ======================================================================
# Main evaluation loop
# ======================================================================

def evaluate_insurance_policies(analysis_name, insured_exposure, insurance_policies):
    """Sweep over attachment-RP × rate-on-line combinations and evaluate
    each policy against uncertainty simulations.

    For each ``attachment_rp`` a :class:`RiskTransfer` is built from the
    **pricing curve** to resolve boundaries and compute expected payouts.
    Per-simulation payouts are then computed by applying each layer's
    :meth:`~InsuranceLayer.calc_expected_payout` to every row of the
    uncertainty data.
    """
    print(f"Running insurance calculations for analysis: {analysis_name}")

    assert np.all(insurance_policies["attachment_rp_vals"] < insurance_policies["exhaustion_rp"]), \
        "Attachment RPs must be less than exhaustion RP"
    assert np.all(insurance_policies["attachment_rp_vals"] >= 2), \
        "Attachment RPs must be greater than or equal to 2"
    assert np.all(insurance_policies["attachment_rp_vals"] <= 1000), \
        "Attachment RPs must be less than or equal to 1000"
    assert insurance_policies["exhaustion_rp"] >= 2, \
        "Exhaustion RP must be greater than or equal to 2"
    assert insurance_policies["exhaustion_rp"] <= 1000, \
        "Exhaustion RP must be less than or equal to 1000"

    impf_dict = build_impf_dict(insured_exposure=insured_exposure, analysis_name=analysis_name)
    impf_dict_calibrated = MetadataImpact(
        impf_dict=impf_dict,
        analysis_name=f"{impf_dict['analysis_name']}/calibrated_mid",
    )

    for scenario in impf_dict["scenarios"]:
        print(f"Running insurance calculations for scenario: {scenario}")

        # Load calibrated exceedance curves
        calibrated_curve = get_impf_exceedance_curves(
            impf_dict_calibrated,
            scenario_list=[scenario],
            impact_type_list=["economic_loss"],
            overwrite=False,
        )
        calibrated_curve["rp_level"] = "mid"
        calibrated_curve = calibrated_curve[["return_period", "impact"]]
        assert calibrated_curve.shape[0] > 0, "No calibrated curve data found"
        assert calibrated_curve.shape[0] in [9, 45], (
            "For Aqueduct flood we have 9 RPs and either 1 or 5 models "
            "(if we have multiple models, we average across them). "
            f"Expected 9 or 45 rows: found {calibrated_curve.shape[0]}"
        )
        calibrated_curve = calibrated_curve.groupby(["return_period"]).mean().reset_index()
        calibrated_curve["exceedance_frequency"] = 1 / calibrated_curve["return_period"]
        calibrated_curve = calibrated_curve.sort_values(
            "exceedance_frequency", ascending=True,
        ).reset_index(drop=True)
        exceedance_frequency = np.sort(
            np.unique(calibrated_curve["exceedance_frequency"].values)
        )

        # Load uncertainty simulation data
        uncertainty_output_paths = impf_dict.uncertainty_results_paths(
            scenario=scenario, create=False,
        )
        uncertainty_df = pd.read_csv(uncertainty_output_paths["csv"])
        uncertainty_df = uncertainty_df.drop(columns=["aai_agg"])
        assert np.all(
            [col.startswith("rp") for col in uncertainty_df.columns]
        ), "Unexpected column names in uncertainty results file"
        uncertainty_df = uncertainty_df[uncertainty_df.columns[::-1]]
        rp = pd.Series([float(s[2:]) for s in uncertainty_df.columns])
        assert np.allclose(exceedance_frequency, 1 / rp), (
            "Exceedance frequencies from RP column names do not match the "
            "calibrated curve."
        )

        exhaustion_rp = insurance_policies["exhaustion_rp"]

        # ---- Pricing curve selection (option 1) ----
        pricing_rp_losses = calibrated_curve["impact"].values

        assert np.array_equal(
            exceedance_frequency, np.sort(exceedance_frequency)
        ), "Exceedance frequencies have to be in ascending order"

        # ----- Sweep -----
        results = []
        for attachment_rp in insurance_policies["attachment_rp_vals"]:

            # Build a RiskTransfer from layer dicts using the PRICING curve.
            layer_dicts = [
                {
                    "name": "National Disaster Fund",
                    "attachment": {"loss": 0},
                    "exhaustion": {"rp": float(attachment_rp)},
                    "premium": 0,
                },
                {
                    "name": "Indemnity insurance",
                    "attachment": {"rp": float(attachment_rp)},
                    "exhaustion": {"rp": float(exhaustion_rp)},
                    # rate_on_line omitted here; premium computed per-rate below
                },
            ]
            rt = RiskTransfer.from_layer_dicts(
                layer_dicts, exceedance_frequency, pricing_rp_losses,
            )
            rt.add_tail_risk_layer(float(np.max(pricing_rp_losses)))

            insurance_layer = rt.get_layer("Indemnity insurance")
            pricing_attachment = insurance_layer.attachment_loss
            pricing_exhaustion = insurance_layer.exhaustion_loss
            pricing_expected_payout = insurance_layer.calc_expected_payout(
                exceedance_frequency, pricing_rp_losses,
            )

            if (not np.isclose(pricing_exhaustion, float(np.max(pricing_rp_losses)),
                               rtol=BOUNDARY_MATCH_RTOL)
                    and pricing_exhaustion < float(np.max(pricing_rp_losses))):
                print(
                    f"  Note: exhaustion ({pricing_exhaustion:.4g}) < max loss "
                    f"({float(np.max(pricing_rp_losses)):.4g}). Tail risk retained."
                )

            # Evaluate per-layer expected payouts across uncertainty simulations.
            # Loop through each layer and collect per-simulation payouts.
            layer_uncertainty = {}
            for layer in rt.layers:
                per_sim = uncertainty_df.apply(
                    lambda row, l=layer: l.calc_expected_payout(
                        exceedance_frequency, row.values,
                    ),
                    axis=1,
                )
                layer_uncertainty[layer.name] = per_sim

            # The insurer's layer payout and the country's retained risk
            uncertainty_payouts = layer_uncertainty["Indemnity insurance"]
            mean_payout = uncertainty_payouts.mean()

            # Retained risk = sum of all layers that are NOT the indemnity insurance layer
            retained_layer_names = [
                n for n in layer_uncertainty if n != "Indemnity insurance"
            ]
            uncertainty_retained = sum(
                layer_uncertainty[n] for n in retained_layer_names
            )
            mean_retained = uncertainty_retained.mean()

            for rate_on_line in insurance_policies["rate_on_line_vals"]:
                premium = (1 + rate_on_line) * pricing_expected_payout
                profits = premium - uncertainty_payouts
                profits_ratio = profits / premium
                mean_profit = profits.mean()
                profitable_fraction = np.sum(profits > 0) / len(profits_ratio)
                mean_cost_to_country = premium + mean_retained
                results.append({
                    "attachment_rp": attachment_rp,
                    "attachment": pricing_attachment,
                    "exhaustion_rp": exhaustion_rp,
                    "exhaustion": pricing_exhaustion,
                    "rate_on_line": rate_on_line,
                    "premium": premium,
                    "pricing_expected_payout": pricing_expected_payout,
                    "mean_payout": mean_payout,
                    "mean_profit": mean_profit,
                    "mean_profit_ratio": mean_profit / premium,
                    "profitable_fraction": profitable_fraction,
                    "mean_retained_risk": mean_retained,
                    "mean_cost_to_country": mean_cost_to_country,
                    "n_layers": rt.n_layers,
                    "layer_names": ", ".join(rt.layer_names),
                })

        output_paths = impf_dict.insurance_results_paths(scenario=scenario, create=True)
        results = pd.DataFrame(results)
        results.to_csv(output_paths["csv"], index=False)

    return results


# ======================================================================
# Plotting
# ======================================================================

def plot_example_result(policy=EXAMPLE_POLICY, insured_exposure=INSURED_EXPOSURE,
                        analysis_name=ANALYSIS_NAME):
    """Plot the exceedance curve with the example insurance policy highlighted."""
    print("Plotting example insurance policy result")
    plot_scenario = "present"
    impf_dict = build_impf_dict(insured_exposure=insured_exposure, analysis_name=analysis_name)
    output_paths = impf_dict.insurance_results_paths(scenario=plot_scenario, create=False)
    results_df = pd.read_csv(output_paths["csv"])

    uncertainty_output_paths = impf_dict.uncertainty_results_paths(
        scenario=plot_scenario, create=False,
    )
    uncertainty_df = pd.read_csv(uncertainty_output_paths["csv"])
    uncertainty_df = uncertainty_df.drop(columns=["aai_agg"])
    rps = np.array([float(col[2:]) for col in uncertainty_df.columns])
    aais = uncertainty_df.values.mean(axis=0)
    aai_agg = calc_aal_trapezoidal(aais, 1 / rps)

    example_policy_df = results_df[
        (results_df["attachment_rp"] == policy["attachment_rp"])
        & (results_df["exhaustion_rp"] == policy["exhaustion_rp"])
        & (results_df["rate_on_line"] == policy["rate_on_line"])
    ]
    assert len(example_policy_df) == 1, (
        "Example policy parameters do not uniquely identify a single policy"
    )
    example_policy_result = example_policy_df.iloc[0]

    fig, axis = plt.subplots(1, 1, figsize=(12, 6))
    axis.set_xlabel("Return period (years)")
    axis.set_ylabel("Modelled impacts (USD)")

    plt.suptitle(
        f"The Gambia flood exceedance curve with an example insurance policy ({plot_scenario})\n"
        f"Attachment RP: {example_policy_result['attachment_rp']}, "
        f"exhaustion RP: {example_policy_result['exhaustion_rp']}, "
        f"rate on line: {example_policy_result['rate_on_line']}"
    )

    rp_max = 100
    axis.set_xlim(1, rp_max)

    axis.plot(rps, aais, color="black", linewidth=1.5, label="Exceedance curve")
    axis.fill_between(
        rps,
        np.minimum(aais, example_policy_result["attachment"]),
        np.minimum(aais, example_policy_result["exhaustion"]),
        color="goldenrod",
        alpha=0.3,
        label="Insured losses",
    )

    mean_payout_str = _format_currency(example_policy_result["mean_payout"])
    mean_loss_borne = aai_agg - example_policy_result["mean_payout"]
    mean_loss_borne_str = _format_currency(mean_loss_borne)

    axis.text(
        60,
        example_policy_result["attachment"] * 0.5,
        f"Average annual losses borne by the country:\n{mean_loss_borne_str}",
        color="black",
        fontsize=10,
    )
    axis.text(
        60,
        example_policy_result["attachment"] * 1.2,
        f"Average annual payout:\n{mean_payout_str}",
        color="brown",
        fontsize=10,
    )

    axis.legend(loc="upper left")
    plt.savefig(output_paths["plot_curve"])
    plt.close(axis.figure)


def plot_policy_space(insured_exposure=INSURED_EXPOSURE, analysis_name=ANALYSIS_NAME):
    """Heatmaps of profitability across the attachment-RP × rate-on-line space."""
    print("Plotting insurance policy parameter space")
    impf_dict = build_impf_dict(insured_exposure=insured_exposure, analysis_name=analysis_name)
    scenarios = impf_dict["scenarios"]
    output_paths_dict = {
        s: impf_dict.insurance_results_paths(scenario=s, create=False)
        for s in scenarios
    }
    results_dict = {
        s: pd.read_csv(output_paths_dict[s]["csv"]) for s in scenarios
    }

    xs = np.sort(results_dict[scenarios[0]]["attachment_rp"].unique())
    ys = np.sort(results_dict[scenarios[0]]["rate_on_line"].unique())[::-1]

    pivot_fraction_dict = {
        s: df.pivot(index="rate_on_line", columns="attachment_rp",
                    values="profitable_fraction").reindex(index=ys, columns=xs)
        for s, df in results_dict.items()
    }
    arr_fraction_dict = {s: p.values for s, p in pivot_fraction_dict.items()}

    pivot_profit_dict = {
        s: df.pivot(index="rate_on_line", columns="attachment_rp",
                    values="mean_profit").reindex(index=ys, columns=xs)
        for s, df in results_dict.items()
    }
    arr_profit_dict = {s: p.values for s, p in pivot_profit_dict.items()}

    pivot_profit_ratio_dict = {
        s: df.pivot(index="rate_on_line", columns="attachment_rp",
                    values="mean_profit_ratio").reindex(index=ys, columns=xs)
        for s, df in results_dict.items()
    }
    arr_profit_ratio_dict = {s: p.values for s, p in pivot_profit_ratio_dict.items()}

    standardise_colourbars = False
    if standardise_colourbars:
        vmin_fraction = min(arr_fraction_dict[s].min() for s in scenarios)
        vmax_fraction = max(arr_fraction_dict[s].max() for s in scenarios)
        vmin_profit = min(arr_profit_dict[s].min() for s in scenarios)
        vmax_profit = max(arr_profit_dict[s].max() for s in scenarios)
        vmin_profit_ratio = min(arr_profit_ratio_dict[s].min() for s in scenarios)
        vmax_profit_ratio = max(arr_profit_ratio_dict[s].max() for s in scenarios)
    else:
        vmin_fraction = vmax_fraction = None
        vmin_profit = vmax_profit = None
        vmin_profit_ratio = vmax_profit_ratio = None

    fig, axes = plt.subplots(3, 3, figsize=(15, 18))

    for i, plot_scenario in enumerate(scenarios):
        ax1, ax2, ax3 = axes[i, 0], axes[i, 1], axes[i, 2]
        im1 = ax1.imshow(arr_fraction_dict[plot_scenario], cmap="YlGn",
                         interpolation="nearest",
                         vmin=vmin_fraction, vmax=vmax_fraction)
        im2 = ax2.imshow(arr_profit_dict[plot_scenario], cmap="YlGnBu",
                         interpolation="nearest",
                         vmin=vmin_profit, vmax=vmax_profit)
        im3 = ax3.imshow(arr_profit_ratio_dict[plot_scenario], cmap="YlGnBu",
                         interpolation="nearest",
                         vmin=vmin_profit_ratio, vmax=vmax_profit_ratio)

        unprofitable_points = pivot_profit_dict[plot_scenario][
            pivot_profit_dict[plot_scenario] < 0
        ]
        unprofitable_x = unprofitable_points.stack().index.get_level_values(1).values
        unprofitable_y = unprofitable_points.stack().index.get_level_values(0).values

        for ax in [ax1, ax2, ax3]:
            ax.scatter(
                [np.where(xs == x)[0][0] for x in unprofitable_x],
                [np.where(ys == y)[0][0] for y in unprofitable_y],
                color="chocolate", marker="x", s=10,
            )
            xr = ax.get_xlim()
            yr = ax.get_ylim()
            ax.set_xticks(np.arange(max(xr)), minor=False)
            ax.set_yticks(np.arange(max(yr)), minor=False)
            ax.grid(which="minor", snap=False, color="k", linestyle="-", linewidth=1)
            ax.tick_params(which="major", bottom=False, left=False)
            ax.tick_params(which="minor", bottom=False, left=False)
            ax.set_xticklabels([f"{x:.0f}" for x in xs])
            ax.set_yticklabels([f"{y * 100:.1f}" for y in ys])
            ax.set_xlabel("Attachment RP (years)")
            ax.set_ylabel("Rate on line (%)")

        ax1.set_title(f"Fraction of profitable policies:\nScenario: {plot_scenario}")
        ax2.set_title(f"Mean profit: {plot_scenario}")
        ax3.set_title(f"Mean profit ratio: {plot_scenario}")
        plt.colorbar(im1, ax=ax1)
        plt.colorbar(im2, ax=ax2)
        plt.colorbar(im3, ax=ax3)

    plt.suptitle("Feasibility of insurance policies\n")
    plt.savefig(output_paths_dict[scenarios[0]]["plot_policy_space"])
    plt.close(fig)


# ======================================================================
# Entry point
# ======================================================================

def main(insurance_policies=INSURANCE_POLICIES, insured_exposure=INSURED_EXPOSURE,
         analysis_name=ANALYSIS_NAME):
    _ = evaluate_insurance_policies(
        insurance_policies=insurance_policies,
        insured_exposure=insured_exposure,
        analysis_name=analysis_name,
    )
    _ = plot_example_result(insured_exposure=insured_exposure, analysis_name=analysis_name)
    _ = plot_policy_space(insured_exposure=insured_exposure, analysis_name=analysis_name)


if __name__ == "__main__":
    main()
