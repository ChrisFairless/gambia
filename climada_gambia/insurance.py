"""Insurance layer and risk transfer classes.

This module provides two core classes for modelling layered insurance products:

- :class:`InsuranceLayer` — a single insurance layer with attachment/exhaustion
  boundaries, expected payout calculation, and plotting.
- :class:`RiskTransfer` — an ordered collection of layers representing a complete
  risk transfer structure, with chain validation, tail-risk detection, and
  combined plotting with a summary statistics table.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from climada_gambia.scoring import calc_aal_trapezoidal


# ---------------------------------------------------------------------------
# Relative tolerance for comparing layer boundaries
# ---------------------------------------------------------------------------
BOUNDARY_MATCH_RTOL = 1e-6


# ---------------------------------------------------------------------------
# Helper: insert a point into a sorted Series
# ---------------------------------------------------------------------------
def _add_to_rp_series(impacts, f, rp):
    """Insert a (frequency, impact) point into a sorted Series if missing."""
    if f in impacts.index:
        assert np.isclose(rp, impacts[f]), (
            "Calculated attachment or exhaustion does not match value on the "
            "RP curve at the same frequency, check calculations"
        )
        return impacts
    impacts = impacts.copy()
    impacts.loc[f] = rp
    impacts = impacts.sort_index(ascending=False)
    return impacts


# ---------------------------------------------------------------------------
# Core payout helper (non-retained layer payout)
# ---------------------------------------------------------------------------
def _calc_layer_payout(impacts, exceedance_frequencies, attachment, exhaustion):
    """Expected annual payout for a layer bounded by *attachment* and *exhaustion*.

    This is the non-retained portion: losses between *attachment* and *exhaustion*.

    Parameters
    ----------
    impacts : array-like
        Loss values per return-period level.
    exceedance_frequencies : array-like
        Corresponding exceedance frequencies (ascending: low-freq first = rare).
    attachment : float
        Lower loss boundary.
    exhaustion : float
        Upper loss boundary.

    Returns
    -------
    float
    """
    if isinstance(impacts, pd.Series):
        impacts = impacts.values
    if isinstance(exceedance_frequencies, pd.Series):
        exceedance_frequencies = exceedance_frequencies.values

    impacts = np.asarray(impacts, dtype=float)
    exceedance_frequencies = np.asarray(exceedance_frequencies, dtype=float)

    assert np.argmin(exceedance_frequencies) == np.argmax(impacts), (
        "Index of min frequency does not match index of max impact"
    )

    imp_s = pd.Series(impacts, index=exceedance_frequencies).sort_index(ascending=False)
    assert np.array_equal(imp_s.values, np.sort(imp_s.values)), (
        "Impacts must be in ascending order for the interpolation to work"
    )

    att_freq = np.interp(attachment, imp_s.values, imp_s.index.values)
    exh_freq = np.interp(exhaustion, imp_s.values, imp_s.index.values)

    # Clamp when attachment/exhaustion exceeds the modelled range
    if att_freq == imp_s.index.min():
        attachment = imp_s.values.max()
    if exh_freq == imp_s.index.min():
        exhaustion = imp_s.values.max()
    if att_freq == imp_s.index.max():
        attachment = imp_s.values.min()
    if exh_freq == imp_s.index.max():
        exhaustion = imp_s.values.min()

    imp_s = _add_to_rp_series(imp_s, att_freq, attachment)
    imp_s = _add_to_rp_series(imp_s, exh_freq, exhaustion)

    rp_impacts = np.maximum(0, np.minimum(imp_s.values, exhaustion) - attachment)
    return calc_aal_trapezoidal(rp_impacts, imp_s.index.values)


# ======================================================================
# InsuranceLayer
# ======================================================================

class InsuranceLayer:
    """A single insurance / risk-retention layer.

    Parameters
    ----------
    name : str
        Human-readable layer name.
    attachment_loss : float
        Loss value at which the layer begins to pay.
    exhaustion_loss : float
        Loss value at which the layer is fully exhausted.
    rate_on_line : float or None
        If given, the premium is ``(1 + rate_on_line) * expected_payout``.
    premium : float or None
        Fixed annual premium.  Mutually exclusive with *rate_on_line* when
        the expected payout is known.
    """

    def __init__(self, name, attachment_loss, exhaustion_loss, *,
                 rate_on_line=None, premium=None):
        if attachment_loss < 0:
            raise ValueError("attachment_loss must be >= 0")
        if exhaustion_loss < attachment_loss:
            raise ValueError("exhaustion_loss must be >= attachment_loss")
        self.name = name
        self.attachment_loss = float(attachment_loss)
        self.exhaustion_loss = float(exhaustion_loss)
        self.rate_on_line = float(rate_on_line) if rate_on_line is not None else None
        self.premium = float(premium) if premium is not None else None

    # ------------------------------------------------------------------
    # Class method: build from an exceedance curve
    # ------------------------------------------------------------------
    @classmethod
    def from_exceedance(cls, name, exceedance_frequencies, exceedance_losses,
                        *,
                        attachment_loss=None, attachment_rp=None,
                        attachment_frequency=None,
                        exhaustion_loss=None, exhaustion_rp=None,
                        exhaustion_frequency=None, limit=None,
                        rate_on_line=None, premium=None):
        """Create a layer by resolving boundaries from an exceedance curve.

        Exactly one of ``attachment_loss``, ``attachment_rp``, or
        ``attachment_frequency`` must be given.  Exactly one of
        ``exhaustion_loss``, ``exhaustion_rp``, ``exhaustion_frequency``,
        or ``limit`` must be given.

        Parameters
        ----------
        name : str
        exceedance_frequencies : array-like
            Exceedance frequencies in ascending order.
        exceedance_losses : array-like
            Corresponding loss values (highest loss at lowest frequency).
        attachment_loss, attachment_rp, attachment_frequency :
            Specify the attachment point.
        exhaustion_loss, exhaustion_rp, exhaustion_frequency, limit :
            Specify the exhaustion point.
        rate_on_line, premium :
            Pricing parameters (optional).

        Returns
        -------
        InsuranceLayer
        """
        exceedance_frequencies = np.asarray(exceedance_frequencies, dtype=float)
        exceedance_losses = np.asarray(exceedance_losses, dtype=float)

        # --- Resolve attachment ---
        att_specs = {
            "attachment_loss": attachment_loss,
            "attachment_rp": attachment_rp,
            "attachment_frequency": attachment_frequency,
        }
        given_att = {k: v for k, v in att_specs.items() if v is not None}
        if len(given_att) != 1:
            raise ValueError(
                "Exactly one of attachment_loss, attachment_rp, or "
                f"attachment_frequency must be supplied; got {list(given_att)}"
            )
        if attachment_loss is not None:
            att = float(attachment_loss)
        elif attachment_rp is not None:
            att = float(np.interp(
                1.0 / float(attachment_rp),
                exceedance_frequencies, exceedance_losses,
            ))
        else:
            att = float(np.interp(
                float(attachment_frequency),
                exceedance_frequencies, exceedance_losses,
            ))

        # --- Resolve exhaustion ---
        exh_specs = {
            "exhaustion_loss": exhaustion_loss,
            "exhaustion_rp": exhaustion_rp,
            "exhaustion_frequency": exhaustion_frequency,
            "limit": limit,
        }
        given_exh = {k: v for k, v in exh_specs.items() if v is not None}
        if len(given_exh) != 1:
            raise ValueError(
                "Exactly one of exhaustion_loss, exhaustion_rp, "
                "exhaustion_frequency, or limit must be supplied; "
                f"got {list(given_exh)}"
            )
        if exhaustion_loss is not None:
            exh = float(exhaustion_loss)
        elif exhaustion_rp is not None:
            exh = float(np.interp(
                1.0 / float(exhaustion_rp),
                exceedance_frequencies, exceedance_losses,
            ))
        elif exhaustion_frequency is not None:
            exh = float(np.interp(
                float(exhaustion_frequency),
                exceedance_frequencies, exceedance_losses,
            ))
        else:  # limit
            exh = att + float(limit)

        # Resolve premium from rate_on_line if both the curve and rol are given
        resolved_premium = premium
        if rate_on_line is not None and premium is None:
            ep = _calc_layer_payout(
                exceedance_losses, exceedance_frequencies, att, exh,
            )
            resolved_premium = (1.0 + float(rate_on_line)) * ep

        return cls(
            name=name,
            attachment_loss=att,
            exhaustion_loss=exh,
            rate_on_line=rate_on_line,
            premium=resolved_premium,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def limit(self):
        """Coverage width: ``exhaustion_loss - attachment_loss``."""
        return self.exhaustion_loss - self.attachment_loss

    # ------------------------------------------------------------------
    # Payout calculation
    # ------------------------------------------------------------------
    def calc_expected_payout(self, exceedance_frequencies, exceedance_losses):
        """Expected annual payout of this layer.

        Parameters
        ----------
        exceedance_frequencies : array-like
        exceedance_losses : array-like

        Returns
        -------
        float
        """
        return _calc_layer_payout(
            exceedance_losses,
            exceedance_frequencies,
            self.attachment_loss,
            self.exhaustion_loss,
        )

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plot(self, exceedance_frequencies, exceedance_losses, *,
             ax=None, color="goldenrod", alpha=0.3, label=None,
             show_curve=True, rp_max=None):
        """Plot this layer on an exceedance curve.

        Parameters
        ----------
        exceedance_frequencies : array-like
        exceedance_losses : array-like
        ax : matplotlib Axes, optional
        color : str
        alpha : float
        label : str or None
            Defaults to ``self.name``.
        show_curve : bool
            Whether to draw the exceedance curve line.
        rp_max : float or None
            If set, limits the x-axis.

        Returns
        -------
        matplotlib.axes.Axes
        """
        rps = 1.0 / np.asarray(exceedance_frequencies, dtype=float)
        losses = np.asarray(exceedance_losses, dtype=float)
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5))
        if show_curve:
            ax.plot(rps, losses, color="black", linewidth=1.5, label="Exceedance curve")
        ax.fill_between(
            rps,
            np.minimum(losses, self.attachment_loss),
            np.minimum(losses, self.exhaustion_loss),
            color=color,
            alpha=alpha,
            label=label or self.name,
        )
        ax.set_xlabel("Return period (years)")
        ax.set_ylabel("Modelled impacts (USD)")
        if rp_max is not None:
            ax.set_xlim(1, rp_max)
        return ax

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------
    def __repr__(self):
        parts = [
            f"name={self.name!r}",
            f"attachment={self.attachment_loss:.4g}",
            f"exhaustion={self.exhaustion_loss:.4g}",
        ]
        if self.rate_on_line is not None:
            parts.append(f"rol={self.rate_on_line}")
        if self.premium is not None:
            parts.append(f"premium={self.premium:.4g}")
        return f"InsuranceLayer({', '.join(parts)})"


# ======================================================================
# RiskTransfer
# ======================================================================

class RiskTransfer:
    """Ordered collection of :class:`InsuranceLayer` objects forming a risk
    transfer structure.

    Parameters
    ----------
    layers : list[InsuranceLayer]
        Layers **in order** from lowest attachment to highest exhaustion.
    """

    def __init__(self, layers):
        if not layers:
            raise ValueError("At least one layer is required")
        self.layers = list(layers)
        self._validate_chain()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_layer_dicts(cls, layer_dicts, exceedance_frequencies, exceedance_losses):
        """Build a RiskTransfer from the dict-based layer specification format.

        Each dict should contain:

        - ``'name'`` (str)
        - ``'attachment'`` (dict): one of ``{"loss": v}``, ``{"rp": v}``,
          ``{"exceedance_frequency": v}``
        - ``'exhaustion'`` (dict): same options, or ``{"limit": v}``
        - optionally ``'rate_on_line'`` or ``'premium'``

        Parameters
        ----------
        layer_dicts : list[dict]
        exceedance_frequencies : array-like
        exceedance_losses : array-like

        Returns
        -------
        RiskTransfer
        """
        insurance_layers = []
        for d in layer_dicts:
            att_kwargs = cls._boundary_to_kwargs(d["attachment"], "attachment")
            exh_kwargs = cls._boundary_to_kwargs(d["exhaustion"], "exhaustion")
            kwargs = {**att_kwargs, **exh_kwargs}
            if "rate_on_line" in d:
                kwargs["rate_on_line"] = d["rate_on_line"]
            if "premium" in d:
                kwargs["premium"] = d["premium"]
            layer = InsuranceLayer.from_exceedance(
                name=d["name"],
                exceedance_frequencies=exceedance_frequencies,
                exceedance_losses=exceedance_losses,
                **kwargs,
            )
            insurance_layers.append(layer)
        return cls(insurance_layers)

    @staticmethod
    def _boundary_to_kwargs(spec, prefix):
        """Convert ``{"loss": v}`` / ``{"rp": v}`` / ``{"exceedance_frequency": v}``
        to keyword arguments for :meth:`InsuranceLayer.from_exceedance`."""
        if "loss" in spec:
            return {f"{prefix}_loss": spec["loss"]}
        elif "rp" in spec:
            return {f"{prefix}_rp": spec["rp"]}
        elif "exceedance_frequency" in spec:
            return {f"{prefix}_frequency": spec["exceedance_frequency"]}
        elif "limit" in spec and prefix == "exhaustion":
            return {"limit": spec["limit"]}
        else:
            raise ValueError(f"Unknown boundary spec keys: {list(spec.keys())}")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate_chain(self):
        """Warn if adjacent layers have gaps or overlaps."""
        for i in range(len(self.layers) - 1):
            cur = self.layers[i]
            nxt = self.layers[i + 1]
            if not np.isclose(
                cur.exhaustion_loss, nxt.attachment_loss,
                rtol=BOUNDARY_MATCH_RTOL,
            ):
                warnings.warn(
                    f"Layer '{cur.name}' exhaustion ({cur.exhaustion_loss:.6g}) "
                    f"does not match layer '{nxt.name}' attachment "
                    f"({nxt.attachment_loss:.6g}). "
                    "There may be a gap or overlap in coverage.",
                    UserWarning,
                    stacklevel=3,
                )

    # ------------------------------------------------------------------
    # Tail-risk layer
    # ------------------------------------------------------------------
    def add_tail_risk_layer(self, max_modelled_loss):
        """Append a retained-tail-risk layer if the top layer's exhaustion is
        below *max_modelled_loss*."""
        top = self.layers[-1]
        if (not np.isclose(top.exhaustion_loss, max_modelled_loss,
                           rtol=BOUNDARY_MATCH_RTOL)
                and top.exhaustion_loss < max_modelled_loss):
            self.layers.append(InsuranceLayer(
                name="Retained tail risk",
                attachment_loss=top.exhaustion_loss,
                exhaustion_loss=max_modelled_loss,
                premium=0,
            ))

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    @property
    def layer_names(self):
        return [l.name for l in self.layers]

    @property
    def n_layers(self):
        return len(self.layers)

    def get_layer(self, name):
        """Return the first layer matching *name*, or raise KeyError."""
        for layer in self.layers:
            if layer.name == name:
                return layer
        raise KeyError(f"No layer named {name!r}")

    # ------------------------------------------------------------------
    # Aggregate helpers
    # ------------------------------------------------------------------
    def calc_all_expected_payouts(self, exceedance_frequencies, exceedance_losses):
        """Return dict mapping layer name -> expected annual payout."""
        return {
            layer.name: layer.calc_expected_payout(
                exceedance_frequencies, exceedance_losses,
            )
            for layer in self.layers
        }

    def calc_total_expected_payout(self, exceedance_frequencies, exceedance_losses):
        """Sum of expected annual payouts across all layers."""
        return sum(self.calc_all_expected_payouts(
            exceedance_frequencies, exceedance_losses,
        ).values())

    # ------------------------------------------------------------------
    # Summary stats
    # ------------------------------------------------------------------
    def summary_df(self, exceedance_frequencies, exceedance_losses):
        """Return a :class:`~pandas.DataFrame` summarising each layer."""
        rows = []
        for layer in self.layers:
            ep = layer.calc_expected_payout(exceedance_frequencies, exceedance_losses)
            rows.append({
                "Layer": layer.name,
                "Attachment": layer.attachment_loss,
                "Exhaustion": layer.exhaustion_loss,
                "Limit": layer.limit,
                "Expected payout": ep,
                "Premium": layer.premium,
                "Rate on line": layer.rate_on_line,
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    _LAYER_COLOURS = [
        "#6baed6",   # steel blue
        "#fd8d3c",   # orange
        "#74c476",   # green
        "#9e9ac8",   # purple
        "#fb6a4a",   # red
        "#fdd0a2",   # peach
        "#bcbddc",   # lavender
    ]

    @staticmethod
    def _format_currency(value, unit="USD"):
        if value is None:
            return "—"
        if abs(value) >= 1e6:
            return f"{value / 1e6:,.1f} mn {unit}"
        if abs(value) >= 1e3:
            return f"{value / 1e3:,.1f} k {unit}"
        return f"{value:,.0f} {unit}"

    def plot(self, exceedance_frequencies, exceedance_losses, *,
             title=None, rp_max=100, figsize=(14, 6)):
        """Plot all layers on an exceedance curve with a summary stats table.

        Parameters
        ----------
        exceedance_frequencies : array-like
        exceedance_losses : array-like
        title : str or None
        rp_max : float
            Right limit of the x-axis (return period).
        figsize : tuple

        Returns
        -------
        matplotlib.figure.Figure
        """
        exceedance_frequencies = np.asarray(exceedance_frequencies, dtype=float)
        exceedance_losses = np.asarray(exceedance_losses, dtype=float)
        rps = 1.0 / exceedance_frequencies

        fig = plt.figure(figsize=figsize)
        gs = GridSpec(1, 2, width_ratios=[3, 2], figure=fig)
        ax = fig.add_subplot(gs[0])
        ax_table = fig.add_subplot(gs[1])

        # Exceedance curve
        ax.plot(rps, exceedance_losses, color="black", linewidth=1.5,
                label="Exceedance curve")

        # Fill each layer
        colours = self._LAYER_COLOURS
        for i, layer in enumerate(self.layers):
            colour = colours[i % len(colours)]
            ax.fill_between(
                rps,
                np.minimum(exceedance_losses, layer.attachment_loss),
                np.minimum(exceedance_losses, layer.exhaustion_loss),
                color=colour, alpha=0.35, label=layer.name,
            )

        ax.set_xlabel("Return period (years)")
        ax.set_ylabel("Modelled impacts (USD)")
        ax.set_xlim(1, rp_max)
        ax.legend(loc="upper left", fontsize=8)

        # Summary table
        summary = self.summary_df(exceedance_frequencies, exceedance_losses)
        table_data = []
        col_labels = ["Layer", "Attachment", "Exhaustion", "E[Payout]", "Premium"]
        for _, row in summary.iterrows():
            table_data.append([
                row["Layer"],
                self._format_currency(row["Attachment"]),
                self._format_currency(row["Exhaustion"]),
                self._format_currency(row["Expected payout"]),
                self._format_currency(row["Premium"]),
            ])

        ax_table.axis("off")
        tbl = ax_table.table(
            cellText=table_data,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1.0, 1.4)

        if title:
            fig.suptitle(title, fontsize=12)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------
    def __repr__(self):
        layer_strs = ", ".join(l.name for l in self.layers)
        return f"RiskTransfer([{layer_strs}])"


# ======================================================================
# Legacy compatibility functions
# ======================================================================

def resolve_boundary(boundary_spec, exceedance_frequencies, impacts):
    """Resolve ``{"loss": v}``, ``{"rp": v}``, or ``{"exceedance_frequency": v}``
    to a concrete loss value via interpolation.

    This is a thin wrapper kept for backward compatibility; new code should
    prefer :meth:`InsuranceLayer.from_exceedance`.
    """
    exceedance_frequencies = np.asarray(exceedance_frequencies, dtype=float)
    impacts = np.asarray(impacts, dtype=float)

    if "loss" in boundary_spec:
        return float(boundary_spec["loss"])
    elif "rp" in boundary_spec:
        freq = 1.0 / float(boundary_spec["rp"])
        return float(np.interp(freq, exceedance_frequencies, impacts))
    elif "exceedance_frequency" in boundary_spec:
        freq = float(boundary_spec["exceedance_frequency"])
        return float(np.interp(freq, exceedance_frequencies, impacts))
    else:
        raise ValueError(
            f"Unknown boundary specification keys: {list(boundary_spec.keys())}. "
            "Expected one of: 'loss', 'rp', 'exceedance_frequency'."
        )


def calc_expected_payout(impacts, exceedance_frequencies, attachment, exhaustion,
                         retained=False):
    """Calculate expected annual payout for a layer (legacy interface).

    When *retained* is ``False`` the payout is losses clipped to the
    [attachment, exhaustion] band.  When ``True`` it is losses below
    attachment plus losses above exhaustion (i.e. the country's retained
    risk).
    """
    if isinstance(impacts, pd.Series):
        impacts = impacts.values
    if isinstance(exceedance_frequencies, pd.Series):
        exceedance_frequencies = exceedance_frequencies.values

    impacts_arr = np.asarray(impacts, dtype=float)
    exceedance_frequencies_arr = np.asarray(exceedance_frequencies, dtype=float)

    assert np.argmin(exceedance_frequencies_arr) == np.argmax(impacts_arr), (
        "Index of min frequency does not match index of max impact"
    )

    imp_s = pd.Series(impacts_arr, index=exceedance_frequencies_arr)
    imp_s = imp_s.sort_index(ascending=False)
    assert np.array_equal(imp_s.values, np.sort(imp_s.values)), (
        "Impacts have to be in ascending order for the interpolations to work"
    )

    att_freq = np.interp(attachment, imp_s.values, imp_s.index.values)
    exh_freq = np.interp(exhaustion, imp_s.values, imp_s.index.values)

    if att_freq == imp_s.index.min():
        attachment = imp_s.values.max()
    if exh_freq == imp_s.index.min():
        exhaustion = imp_s.values.max()
    if att_freq == imp_s.index.max():
        attachment = imp_s.values.min()
    if exh_freq == imp_s.index.max():
        exhaustion = imp_s.values.min()

    imp_s = _add_to_rp_series(imp_s, att_freq, attachment)
    imp_s = _add_to_rp_series(imp_s, exh_freq, exhaustion)

    if not retained:
        rp_impacts = np.maximum(0, np.minimum(imp_s.values, exhaustion) - attachment)
    else:
        rp_impacts = (np.minimum(attachment, imp_s.values)
                      + np.maximum(0, imp_s.values - exhaustion))
    return calc_aal_trapezoidal(rp_impacts, imp_s.index.values)
