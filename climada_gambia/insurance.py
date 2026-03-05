import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from functools import partial
import itertools

from climada_gambia.metadata_impact import MetadataImpact
from climada_gambia.utils_total_exposed_value import get_total_exposed_value
from climada_gambia.uncertainty import gather_uncertainty_results, get_total_exposed_ratio
from climada_gambia import utils_config
from climada_gambia.scoring import calc_aal_trapezoidal
from climada_gambia.analyse_impacts import get_impf_exceedance_curves

ANALYSIS_NAME = "calibration"
overwrite = True

INSURANCE_POLICIES = {
    "exhaustion_rp": 1000, # The RP at which the insurance is exhausted, i.e. the 200-yr event
    "attachment_rp_vals": np.arange(5, 55, 5), # The RP at which the insurance starts to pay out
    "rate_on_line_vals": np.arange(0.10, 0.41, 0.025)  # Expressed as a % of the exhaustion, which is considered to be the 100-yr event
}
EXAMPLE_POLICY = {  # For plotting
    "exhaustion_rp": 1000,
    "attachment_rp": 5,  # use an aqueduct standard RP or the plot is weird
    "rate_on_line": 0.125
}
INSURED_EXPOSURE = "economic_assets"

#===============================================================================

def build_impf_dict():
    impf_dict = utils_config.gather_impact_calculation_metadata(filter={'exposure_type': INSURED_EXPOSURE}, analysis_name=ANALYSIS_NAME)[0]
    analysis_name_full = f"{ANALYSIS_NAME}/{impf_dict['exposure_type']}_{impf_dict['exposure_source']}"
    impf_dict = MetadataImpact(impf_dict=impf_dict, analysis_name=analysis_name_full)
    return impf_dict

def evaluate_insurance_policies():
    print(f"Running insurance calculations for analysis: {ANALYSIS_NAME}")

    assert np.all(INSURANCE_POLICIES["attachment_rp_vals"] < INSURANCE_POLICIES["exhaustion_rp"]), "Attachment RPs must be less than exhaustion RP"
    assert np.all(INSURANCE_POLICIES["attachment_rp_vals"] >= 2), "Attachment RPs must be greater than or equal to 2"
    assert np.all(INSURANCE_POLICIES["attachment_rp_vals"] <= 1000), "Attachment RPs must be less than or equal to 1000"
    assert INSURANCE_POLICIES["exhaustion_rp"] >= 2, "Exhaustion RP must be greater than or equal to 2"
    assert INSURANCE_POLICIES["exhaustion_rp"] <= 1000, "Exhaustion RP must be less than or equal to 1000"
    
    # In this first approach, we only use output from the uncertainty simulations for the economic assets sector.
    # We might get a better view of the uncertainty if we combined all sectors and scaled to LitPop, but we'd have to do 
    # that carefully, avoiding uncertainties cancelling each other out when they're combined (as we did in some of the
    # plotting in the uncertainty analysis).
    impf_dict = build_impf_dict()
    impf_dict_calibrated = MetadataImpact(impf_dict=impf_dict, analysis_name=f"{impf_dict['analysis_name']}/calibrated_mid")

    for scenario in impf_dict["scenarios"]:
        print(f"Running insurance calculations for scenario: {scenario}")

        #  Load calibrated exceedance curves
        calibrated_curve = get_impf_exceedance_curves(
            impf_dict_calibrated,
            scenario_list=[scenario],
            impact_type_list=["economic_loss"],
            overwrite=False
        )
        calibrated_curve["rp_level"] = "mid"
        # Average over different aqueduct models
        calibrated_curve = calibrated_curve[["return_period", "impact"]]
        assert calibrated_curve.shape[0] > 0, f"No calibrated curve data found"
        assert calibrated_curve.shape[0] in [9, 45], \
            f"For Aqueduct flood we have 9 RPs and either 1 or 5 models (if we have multiple models, we average across them). Expected 9 or 45 rows: found {calibrated_curve.shape[0]}"
        calibrated_curve = calibrated_curve.groupby(['return_period']).mean().reset_index()
        calibrated_curve["exceedance_frequency"] = 1/calibrated_curve["return_period"]
        calibrated_curve = calibrated_curve.sort_values("exceedance_frequency", ascending=True).reset_index(drop=True)
        exceedance_frequency = np.sort(np.unique(calibrated_curve["exceedance_frequency"].values))

        # Load uncertainty simulation data
        uncertainty_output_paths = impf_dict.uncertainty_results_paths(scenario=scenario, create=False)
        uncertainty_df = pd.read_csv(uncertainty_output_paths["csv"])
        uncertainty_df = uncertainty_df.drop(columns=['aai_agg'])
        assert np.all([col.startswith('rp') for col in uncertainty_df.columns]), "Unexpected column names in uncertainty results file, expected columns starting with 'rp'"
        # reverse columns so that they're in ascending order of frequency rather than RP (for easier calculations)
        uncertainty_df = uncertainty_df[uncertainty_df.columns[::-1]]
        rp = pd.Series([float(s[2:len(s)+1]) for s in uncertainty_df.columns])
        assert np.allclose(exceedance_frequency, 1/rp), "Exceedance frequencies calculated from RP column names do not match exceedance frequencies in the calibrated curve. This is unexpected"

        exhaustion_frequency = 1/INSURANCE_POLICIES["exhaustion_rp"]

        # We have to choose our 'best estimate' of AAI and other variables to set the prices under uncertainty.
        # There are a few ways we could do this, and maybe we'll compare them in later work

        # OPTION 1 (which we'll use):
        # Set prices using loss stats from the calibrated RP curve
        if True:
            pricing_rp_losses = calibrated_curve["impact"].values

        # OPTION 2:
        # Set prices using the mean losses from the uncertainty calculations
        # Means are probably a bit better than medians here, since they capture outliers a little better?
        if False:
            pricing_rp_losses = uncertainty_df.values.mean(axis=0)


        # OPTION 3:
        # Set prices using the 75th percentile of losses from the uncertainty calculations, to be a bit more conservative
        if False:
            pricing_rp_losses = uncertainty_df.apply(lambda x: np.quantile(x, 0.75), axis=0).values

        # OPTION 4:
        # Re-run the uncertainty simulations for this contract, returning payouts instead of losses
        # This would be the most accurate way to price the contract, but also the most computationally expensive, so we won't do it for now.
        # It would also give us a distribution of payouts rather than a single expected payout, which could be interesting to look at.
        if False:
            # Left as an exercise for the reader.
            pass

        assert np.array_equal(exceedance_frequency, np.sort(exceedance_frequency)), "Exceedance frequencies have to be in ascending order"
        pricing_exhaustion = np.interp(exhaustion_frequency, exceedance_frequency, pricing_rp_losses)

        # Now we test how each policy perform under different attachment RPs and rates on line 
        results = []
        for attachment_rp in INSURANCE_POLICIES["attachment_rp_vals"]:
            attachment_frequency = 1/attachment_rp
            pricing_attachment = np.interp(attachment_frequency, exceedance_frequency, pricing_rp_losses)
            
            # Using our 'best knowledge' we pick an attachment loss based on our attachment RP 
            pricing_expected_payout = calc_expected_payout(
                impacts=pricing_rp_losses,
                exceedance_frequencies=exceedance_frequency,
                attachment=pricing_attachment,
                exhaustion=pricing_exhaustion,
                retained=False
            )

            # Create a function that calculates the expected payout from a simulated RP curve sampled under uncertainty, 
            # given the attachment and exhaustion for this insurance policy. We then apply it to the uncertainty results 
            # dataframe
            partial_payout_evaluation = partial(
                calc_expected_payout,
                exceedance_frequencies=exceedance_frequency,
                attachment=pricing_attachment,
                exhaustion=pricing_exhaustion,
                retained=False
            )
            uncertainty_payouts = uncertainty_df.apply(partial_payout_evaluation, axis=1)
            mean_payout = uncertainty_payouts.mean()

            partial_retained_evaluation = partial(
                calc_expected_payout,
                exceedance_frequencies=exceedance_frequency,
                attachment=pricing_attachment,
                exhaustion=pricing_exhaustion,
                retained=True
            )
            uncertainty_retained = uncertainty_df.apply(partial_retained_evaluation, axis=1)
            mean_retained = uncertainty_retained.mean()

            for rate_on_line in INSURANCE_POLICIES["rate_on_line_vals"]:        
                premium = (1 + rate_on_line) * pricing_expected_payout
                profits = premium - uncertainty_payouts
                profits_ratio = profits / premium
                mean_profit = profits.mean()
                profitable_fraction = np.sum(profits > 0) / len(profits_ratio)
                mean_cost_to_country = premium + mean_retained
                results.append({
                    "attachment_rp": attachment_rp,
                    "attachment": pricing_attachment,
                    "exhaustion_rp": INSURANCE_POLICIES["exhaustion_rp"],
                    "exhaustion": pricing_exhaustion,
                    "rate_on_line": rate_on_line,
                    "premium": premium,
                    "pricing_expected_payout": pricing_expected_payout,
                    "mean_payout": mean_payout,
                    "mean_profit": mean_profit,
                    "mean_profit_ratio": mean_profit / premium,
                    "profitable_fraction": profitable_fraction,
                    "mean_retained_risk": mean_retained,
                    "mean_cost_to_country": mean_cost_to_country
                })

        output_paths = impf_dict.insurance_results_paths(scenario=scenario, create=True)
        results = pd.DataFrame(results)
        results.to_csv(output_paths["csv"], index=False)


def add_to_rp_series(impacts, f, rp):
    if f in impacts.index:
        assert np.isclose(rp, impacts[f]), "Calculated attachment or exhaustion does not match value on the RP curve at the same frequency, check calculations"
        return impacts
    else:
        impacts.loc[f] = rp
        impacts = impacts.sort_index(ascending=False)
        return impacts


def calc_expected_payout(impacts, exceedance_frequencies, attachment, exhaustion, retained=False):
    assert np.argmin(exceedance_frequencies) == np.argmax(impacts), "Index of min frequency does not match index of max impact"

    if isinstance(impacts, pd.Series):
        impacts = impacts.values
    if isinstance(exceedance_frequencies, pd.Series):
        exceedance_frequencies = exceedance_frequencies.values
    impacts = pd.Series(impacts, index=exceedance_frequencies)
    impacts = impacts.sort_index(ascending=False)
    assert np.array_equal(impacts.values, np.sort(impacts.values)), "Impacts have to be in ascending order for the interpolations to work"

    # Add attachment and exhaustion points to the RP curve if they don't already exist, or the trapezoid integration won't work
    attachment_frequency = np.interp(attachment, impacts.values, impacts.index.values)
    exhaustion_frequency = np.interp(exhaustion, impacts.values, impacts.index.values)

    # Deal with the case where the desired attachment/exhaustion is higher than anything modelled in this curve
    if attachment_frequency == impacts.index.min():
        attachment = impacts.values.max()
    if exhaustion_frequency == impacts.index.min():
        exhaustion = impacts.values.max()
    if attachment_frequency == impacts.index.max():
        attachment = impacts.values.min()
    if exhaustion_frequency == impacts.index.max():
        exhaustion = impacts.values.min()

    impacts = add_to_rp_series(impacts, attachment_frequency, attachment)    
    impacts = add_to_rp_series(impacts, exhaustion_frequency, exhaustion)

    if not retained:
        rp_impacts = np.maximum(0, np.minimum(impacts.values, exhaustion) - attachment)
    else:
        rp_impacts = np.minimum(attachment, impacts.values) + np.maximum(0, impacts.values - exhaustion)
    expected_impact = calc_aal_trapezoidal(rp_impacts, impacts.index.values)
    return expected_impact


def plot_example_result():
    print("Plotting example insurance policy result")
    plot_scenario = "present"
    impf_dict = build_impf_dict()
    output_paths = impf_dict.insurance_results_paths(scenario=plot_scenario, create=False)
    results_df = pd.read_csv(output_paths["csv"])

    uncertainty_output_paths = impf_dict.uncertainty_results_paths(scenario=plot_scenario, create=False)
    uncertainty_df = pd.read_csv(uncertainty_output_paths["csv"])
    uncertainty_df = uncertainty_df.drop(columns=['aai_agg'])
    rps = np.array([float(col[2:]) for col in uncertainty_df.columns])
    aais = uncertainty_df.values.mean(axis=0)
    aai_agg = calc_aal_trapezoidal(aais, 1/rps)


    example_policy_df = results_df[
        (results_df['attachment_rp'] == EXAMPLE_POLICY['attachment_rp']) &
        (results_df['exhaustion_rp'] == EXAMPLE_POLICY['exhaustion_rp']) &
        (results_df['rate_on_line'] == EXAMPLE_POLICY['rate_on_line'])
    ]
    assert len(example_policy_df) == 1, "Example policy parameters do not uniquely identify a single policy in the results"
    example_policy_result = example_policy_df.iloc[0]

    fig, axis = plt.subplots(1, 1, figsize=(12, 6))
    axis.set_xlabel("Return period (years)")
    imp_unit = "USD"
    axis.set_ylabel(f"Modelled impacts ({imp_unit})")

    plt.suptitle(
        f"The Gambia flood exceedance curve with an example insurance policy ({plot_scenario})\n" +
        f"Attachment RP: {example_policy_result['attachment_rp']}, exhaustion RP: {example_policy_result['exhaustion_rp']}, rate on line: {example_policy_result['rate_on_line']}"
    )

    rp_max = 100  # Limit x-axis to 100-yr event for better visibility of the insurance policy's effects
    axis.set_xlim(1, rp_max)

    axis.plot(rps, aais, color="black", linewidth=1.5, label="Exceedance curve")
    axis.fill_between(
        rps,
        np.minimum(aais, example_policy_result['attachment']),
        np.minimum(aais, example_policy_result['exhaustion']),
        color="goldenrod",
        alpha=0.3,
        label="Insured losses"
    )
    # Add text labels to the curve and the shaded area
    def format_currency_str(value, unit="USD"):
        if value >= 1e6:
            return f"{value/1e6:,.1f} mn {unit}"
        elif value >= 1e3:
            return f"{value/1e3:,.1f} k {unit}"
        else:
            return f"{value:,.0f} {unit}"

    aai_agg_str = format_currency_str(aai_agg)
    mean_payout_str = format_currency_str(example_policy_result['mean_payout'])
    mean_loss_borne_by_country_str = format_currency_str(aai_agg - example_policy_result['mean_payout'])

    axis.text(60, example_policy_result['attachment'] * 0.5, f"Average annual losses borne by the country:\n{mean_loss_borne_by_country_str}", color="black", fontsize=10)
    axis.text(60, example_policy_result['attachment'] * 1.2, f"Average annual payout:\n{mean_payout_str}", color="brown", fontsize=10)

    axis.legend(loc="upper left")

    plt.savefig(output_paths["plot_curve"])
    plt.close(axis.figure)


def plot_policy_space():
    print("Plotting insurance policy parameter space")
    impf_dict = build_impf_dict()
    scenarios = impf_dict["scenarios"]
    output_paths_dict = {
        plot_scenario: impf_dict.insurance_results_paths(scenario=plot_scenario, create=False)
        for plot_scenario in scenarios
    }
    results_dict = {
        plot_scenario: pd.read_csv(output_paths_dict[plot_scenario]["csv"])
        for plot_scenario in scenarios
    }

    xs = np.sort(results_dict[scenarios[0]]['attachment_rp'].unique())
    ys = np.sort(results_dict[scenarios[0]]['rate_on_line'].unique())[::-1]
    
    pivot_fraction_dict = {
        plot_scenario: df.pivot(
                index='rate_on_line',
                columns='attachment_rp',
                values='profitable_fraction'
            ).reindex(index=ys, columns=xs)
        for plot_scenario, df in results_dict.items()
    }
    arr_fraction_dict = {
        plot_scenario: pivot_fraction.values
        for plot_scenario, pivot_fraction in pivot_fraction_dict.items()
    }
    pivot_profit_dict = {
        plot_scenario: df.pivot(
                index='rate_on_line',
                columns='attachment_rp',
                values='mean_profit'
            ).reindex(index=ys, columns=xs)
        for plot_scenario, df in results_dict.items()
    }
    arr_profit_dict = {
        plot_scenario: pivot_profit.values
        for plot_scenario, pivot_profit in pivot_profit_dict.items()
    }

    pivot_profit_ratio_dict = {
        plot_scenario: df.pivot(
                index='rate_on_line',
                columns='attachment_rp',
                values='mean_profit_ratio'
            ).reindex(index=ys, columns=xs)
        for plot_scenario, df in results_dict.items()
    }
    arr_profit_ratio_dict = {
        plot_scenario: pivot_profit_ratio.values
        for plot_scenario, pivot_profit_ratio in pivot_profit_ratio_dict.items()
    }

    df_fraction_dict = {
        plot_scenario: df[df['profitable_fraction'] == 1]
        for plot_scenario, df in results_dict.items()
    }

    fig, axes = plt.subplots(3, 3, figsize=(15, 18))
    # axes = axes.flatten()

    # Calculate global min/max across all scenarios for consistent colormaps
    standardise_colourbars = False

    if standardise_colourbars:
        vmin_fraction = min(arr_fraction_dict[s].min() for s in scenarios)
        vmax_fraction = max(arr_fraction_dict[s].max() for s in scenarios)
        vmin_profit = min(arr_profit_dict[s].min() for s in scenarios)
        vmax_profit = max(arr_profit_dict[s].max() for s in scenarios)
        vmin_profit_ratio = min(arr_profit_ratio_dict[s].min() for s in scenarios)
        vmax_profit_ratio = max(arr_profit_ratio_dict[s].max() for s in scenarios)
    else:
        vmin_fraction = None
        vmax_fraction = None
        vmin_profit = None
        vmax_profit = None
        vmin_profit_ratio = None
        vmax_profit_ratio = None
    
    for i, plot_scenario in enumerate(scenarios):
        ax1 = axes[i, 0]
        ax2 = axes[i, 1]
        ax3 = axes[i, 2]
        im1 = ax1.imshow(arr_fraction_dict[plot_scenario], cmap='YlGn', interpolation='nearest', vmin=vmin_fraction, vmax=vmax_fraction)
        im2 = ax2.imshow(arr_profit_dict[plot_scenario], cmap='YlGnBu', interpolation='nearest', vmin=vmin_profit, vmax=vmax_profit)
        im3 = ax3.imshow(arr_profit_ratio_dict[plot_scenario], cmap='YlGnBu', interpolation='nearest', vmin=vmin_profit_ratio, vmax=vmax_profit_ratio)

        # Add scatter points with for all points in pivot_fraction_dict[plot_scenario] where the value is 1.
        # Plotted with x as the position of the column name (attachment RP) and y as the position of the index name (rate on line)
        profitable_points = pivot_fraction_dict[plot_scenario][pivot_fraction_dict[plot_scenario] == 1]
        profitable_x = profitable_points.stack().index.get_level_values(1).values
        profitable_y = profitable_points.stack().index.get_level_values(0).values

        unprofitable_points = pivot_profit_dict[plot_scenario][pivot_profit_dict[plot_scenario] < 0]
        unprofitable_x = unprofitable_points.stack().index.get_level_values(1).values
        unprofitable_y = unprofitable_points.stack().index.get_level_values(0).values

        ax1.scatter(
            [np.where(xs == x)[0][0] for x in unprofitable_x],
            [np.where(ys == y)[0][0] for y in unprofitable_y],
            color='chocolate',
            marker='x',
            s=10
        )
        ax2.scatter(
            [np.where(xs == x)[0][0] for x in unprofitable_x],
            [np.where(ys == y)[0][0] for y in unprofitable_y],
            color='chocolate',
            marker='x',
            s=10
        )
        ax3.scatter(
            [np.where(xs == x)[0][0] for x in unprofitable_x],
            [np.where(ys == y)[0][0] for y in unprofitable_y],
            color='chocolate',
            marker='x',
            s=10
        )

        for ax in [ax1, ax2, ax3]:
            xr = ax.get_xlim()
            yr = ax.get_ylim()
            ax.set_xticks(np.arange(max(xr)), minor=False)
            ax.set_yticks(np.arange(max(yr)), minor=False)
            ax.grid(which='minor', snap=False, color='k', linestyle='-', linewidth=1)
            ax.tick_params(which='major', bottom=False, left=False)
            ax.tick_params(which='minor', bottom=False, left=False)
            ax.set_xticklabels([f'{x:.0f}' for x in xs])
            ax.set_yticklabels([f'{y*100:.1f}' for y in ys])
            ax.set_xlabel("Attachment RP (years)")
            ax.set_ylabel("Rate on line (%)")

            
        ax1.set_title(f"Fraction of profitable policies:\nScenario: {plot_scenario}")
        ax2.set_title(f"Mean profit: {plot_scenario}")
        ax3.set_title(f"Mean profit ratio: {plot_scenario}")
        cbar1 = plt.colorbar(im1, ax=ax1)
        # cbar1.set_label('Fraction of premiums that make a profit under uncertainty')
        cbar2 = plt.colorbar(im2, ax=ax2)
        # cbar2.set_label('Mean profit')
        cbar3 = plt.colorbar(im3, ax=ax3)
        # cbar3.set_label('Mean profit ratio')

    plt.suptitle(
        f"Feasibility of insurance policies\n"
        # f"Proportion of policies that make a profit under uncertainty")
    )
    plt.savefig(output_paths_dict[scenarios[0]]["plot_policy_space"])
    plt.close(fig)


def main():
    _ = evaluate_insurance_policies()
    _ = plot_example_result()
    _ = plot_policy_space()
    

if __name__ == "__main__":
    main()
