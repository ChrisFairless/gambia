import logging
import matplotlib.pyplot as plt
import numpy as np
LOGGER = logging.getLogger(__name__)


def plot_waterfall(
    curves,
    impact_type,
    growth_factor,
    **kwargs
):
    present_year = 2025
    future_year = 2050

    df = curves[curves['impact_type'] == impact_type]
    curr_risk_df = df[df['scenario'] == 'present']
    for scenario in ['RCP4.5', 'RCP8.5']:

    imp = ImpactCalc(
        ent_future.exposures, ent_future.impact_funcs, haz_future
    ).impact(assign_centroids=hazard.centr_exp_col not in ent_future.exposures.gdf)
    fut_risk = risk_func(imp)

    if not axis:
        _, axis = plt.subplots(1, 1)
    norm_fact, norm_name = _norm_values(curr_risk)

    # current situation
    LOGGER.info("Risk at {:d}: {:.3e}".format(present_year, curr_risk))

    # changing future
    # socio-economic dev
    imp = ImpactCalc(ent_future.exposures, ent_future.impact_funcs, hazard).impact(
        assign_centroids=False
    )
    risk_dev = risk_func(imp)
    LOGGER.info(
        "Risk with development at {:d}: {:.3e}".format(future_year, risk_dev)
    )

    # socioecon + cc
    LOGGER.info(
        "Risk with development and climate change at {:d}: {:.3e}".format(
            future_year, fut_risk
        )
    )

    axis.bar(1, curr_risk / norm_fact, **kwargs)
    axis.text(
        1,
        curr_risk / norm_fact,
        str(int(round(curr_risk / norm_fact))),
        horizontalalignment="center",
        verticalalignment="bottom",
        fontsize=12,
        color="k",
    )
    axis.bar(
        2,
        height=(risk_dev - curr_risk) / norm_fact,
        bottom=curr_risk / norm_fact,
        **kwargs
    )
    axis.text(
        2,
        curr_risk / norm_fact + (risk_dev - curr_risk) / norm_fact / 2,
        str(int(round((risk_dev - curr_risk) / norm_fact))),
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=12,
        color="k",
    )
    axis.bar(
        3,
        height=(fut_risk - risk_dev) / norm_fact,
        bottom=risk_dev / norm_fact,
        **kwargs
    )
    axis.text(
        3,
        risk_dev / norm_fact + (fut_risk - risk_dev) / norm_fact / 2,
        str(int(round((fut_risk - risk_dev) / norm_fact))),
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=12,
        color="k",
    )
    axis.bar(4, height=fut_risk / norm_fact, **kwargs)
    axis.text(
        4,
        fut_risk / norm_fact,
        str(int(round(fut_risk / norm_fact))),
        horizontalalignment="center",
        verticalalignment="bottom",
        fontsize=12,
        color="k",
    )

    axis.set_xticks(np.arange(4) + 1)
    axis.set_xticklabels(
        [
            "Risk " + str(present_year),
            "Economic \ndevelopment",
            "Climate \nchange",
            "Risk " + str(future_year),
        ]
    )
    axis.set_ylabel("Impact (" + imp.unit + " " + norm_name + ")")
    axis.set_title("Risk at {:d} and {:d}".format(present_year, future_year))
    return axis