"""Plotly figure factory with the FactoryMind dark industrial theme.

All figures share one template (transparent panels, muted grid, professional
blue/teal colorway) so the dashboard looks like one coherent product.

Team Sodium | International AI Builders Congress (hackathon MVP).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.metrics import precision_recall_curve, roc_curve

from src import config

COLORWAY = [config.PRIMARY, config.ACCENT, config.SUCCESS, config.WARNING,
            config.DANGER, "#A78BFA", "#F472B6"]


def _build_template() -> go.layout.Template:
    t = go.layout.Template()
    t.layout = go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, 'Segoe UI', Arial, sans-serif",
                  color=config.MUTED, size=13),
        colorway=COLORWAY,
        xaxis=dict(gridcolor="rgba(148,163,184,0.12)",
                   zerolinecolor="rgba(148,163,184,0.25)", title_font_color=config.MUTED),
        yaxis=dict(gridcolor="rgba(148,163,184,0.12)",
                   zerolinecolor="rgba(148,163,184,0.25)", title_font_color=config.MUTED),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=34, b=10),
        hoverlabel=dict(bgcolor=config.PANEL, bordercolor=config.BORDER,
                        font=dict(color=config.TEXT, size=12)),
    )
    return t


pio.templates["factorymind"] = _build_template()
_RISK_ORDER = ["Low", "Medium", "High"]


def _styled(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_layout(template="factorymind", height=height,
                      autosize=True, showlegend=True)
    fig.update_xaxes(showline=True, linecolor=config.BORDER)
    fig.update_yaxes(showline=True, linecolor=config.BORDER)
    return fig


# ------------------------------------------------------------ dashboard ----
def fig_health_donut(tier_counts: dict) -> go.Figure:
    labels = [k for k in _RISK_ORDER if tier_counts.get(k, 0) > 0]
    values = [tier_counts[k] for k in labels]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.62,
        marker=dict(colors=[config.RISK_COLORS[k] for k in labels],
                    line=dict(color=config.BG, width=2)),
        textinfo="label+percent", textfont=dict(color=config.TEXT, size=12),
        hoverinfo="label+value",
    ))
    total = sum(values)
    fig.add_annotation(text=f"<b>{total}</b><br>machines", showarrow=False,
                       font=dict(color=config.TEXT, size=16))
    fig.update_layout(template="factorymind", height=320, showlegend=False,
                      margin=dict(t=10, b=10))
    return fig


def fig_risk_by_type(snapshot: pd.DataFrame) -> go.Figure:
    ct = pd.crosstab(snapshot["Type"], snapshot["risk_tier"])
    ct = ct.reindex(columns=[c for c in _RISK_ORDER if c in ct.columns], fill_value=0)
    fig = go.Figure()
    for tier in ct.columns:
        fig.add_trace(go.Bar(
            x=ct.index.astype(str), y=ct[tier], name=tier,
            marker_color=config.RISK_COLORS[tier],
            text=ct[tier], textposition="inside", insidetextfont=dict(color="#06121F"),
        ))
    fig.update_layout(barmode="stack",
                      xaxis_title="Machine type (L/M/H quality variant)",
                      yaxis_title="Machines", legend_title="Risk tier")
    return _styled(fig)


def fig_trend(df: pd.DataFrame, value_cols: list, overlay_machines: list = None) -> go.Figure:
    """Fleet-mean trend lines (hourly), with optional per-machine overlays."""
    ts = df.groupby("Timestamp")[value_cols].mean().reset_index()
    fig = go.Figure()
    names = {"Torque [Nm]": "Mean torque (Nm)",
             "Tool wear [min]": "Mean tool wear (min)",
             "temp_delta": "Mean process-air temperature delta (K)"}
    for col in value_cols:
        fig.add_trace(go.Scatter(
            x=ts["Timestamp"], y=ts[col], mode="lines", name=names.get(col, col),
            line=dict(width=2.4)))
    for mid in (overlay_machines or [])[:3]:
        sub = df[df["Machine ID"] == mid]
        if sub.empty:
            continue
        g = sub.groupby("Timestamp")[value_cols].mean().reset_index()
        for col in value_cols:
            fig.add_trace(go.Scatter(
                x=g["Timestamp"], y=g[col], mode="lines",
                name=f"{mid} - {names.get(col, col)}",
                line=dict(width=1.6, dash="dot"), opacity=0.85))
    fig.update_xaxes(title="Time (demo fleet, 48 h)")
    return _styled(fig, height=320)


def fig_scatter_ops(scored: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        scored, x="Rotational speed [rpm]", y="Torque [Nm]",
        color="risk_tier", color_discrete_map=config.RISK_COLORS,
        category_orders={"risk_tier": _RISK_ORDER},
        hover_data=["Machine ID", "Tool wear [min]"],
    )
    fig.update_traces(marker=dict(size=9, opacity=0.85,
                                  line=dict(width=1, color=config.BG)))
    fig.update_xaxes(title="Rotational speed (rpm)")
    fig.update_yaxes(title="Torque (Nm)")
    fig.update_layout(legend_title="Risk tier")
    return _styled(fig)


# ------------------------------------------------------------ prediction ----
def fig_gauge(probability: float, thresholds: dict) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=probability,
        number=dict(valueformat=".2f", font=dict(size=42, color=config.TEXT)),
        gauge=dict(
            axis=dict(range=[0, 1], tickfont=dict(color=config.MUTED, size=10),
                      tickformat=".1f"),
            bar=dict(color=config.PRIMARY, thickness=0.28),
            bgcolor="rgba(0,0,0,0)",
            steps=[
                dict(range=[0, thresholds["medium"]],
                     color="rgba(52,211,153,0.14)"),
                dict(range=[thresholds["medium"], thresholds["high"]],
                     color="rgba(251,191,36,0.16)"),
                dict(range=[thresholds["high"], 1],
                     color="rgba(248,113,113,0.18)"),
            ],
            threshold=dict(line=dict(color=config.TEXT, width=3),
                           value=thresholds["alert"]),
        ),
    ))
    fig.update_layout(template="factorymind", height=260, margin=dict(t=24, b=8))
    fig.add_annotation(
        text=f"white line = alert threshold ({thresholds['alert']:.2f})",
        xref="paper", yref="paper", x=0.5, y=-0.02, showarrow=False,
        font=dict(color=config.MUTED, size=11))
    return fig


def fig_contributions(rows: list) -> go.Figure:
    """Diverging horizontal bar chart of OCAT local contributions."""
    df = pd.DataFrame(rows)
    df = df.sort_values("delta_if_normalised")
    colors = [config.DANGER if d < 0 else config.SUCCESS for d in df["delta_if_normalised"]]
    fig = go.Figure(go.Bar(
        y=[f.replace(" [", "<br>[") for f in df["feature"]],
        x=df["delta_if_normalised"].round(4),
        orientation="h",
        marker_color=colors,
        text=[f"{d:+.3f}" for d in df["delta_if_normalised"]],
        textposition="outside", textfont=dict(color=config.MUTED, size=11),
        customdata=np.stack([df["value"].fillna(np.nan), df["healthy_typical"]], axis=-1),
        hovertemplate="%{y}<br>observed: %{customdata[0]}<br>typical healthy: %{customdata[1]}"
                      "<br>delta: %{x:+.4f}<extra></extra>",
    ))
    fig.add_annotation(
        text="red = reading pushes risk up vs typical healthy; teal = pushes risk down",
        xref="paper", yref="paper", x=0.5, y=-0.16, showarrow=False,
        font=dict(color=config.MUTED, size=11))
    fig.update_xaxes(title="Change in predicted risk if reading were normalised")
    fig.update_yaxes(title=None)
    fig.update_layout(height=max(280, 44 * len(df) + 90))
    return _styled(fig)


# ------------------------------------------------------- model performance --
def fig_confusion(cm: list) -> go.Figure:
    z = [[cm[0][0], cm[0][1]], [cm[1][0], cm[1][1]]]
    fig = go.Figure(go.Heatmap(
        z=z, x=["Predicted OK", "Predicted failure"],
        y=["Actual OK", "Actual failure"],
        text=[[str(v) for v in row] for row in z],
        texttemplate="%{text}", textfont=dict(size=18, color=config.TEXT),
        colorscale=[[0, "#13203A"], [1, config.PRIMARY]], showscale=False,
        xgap=3, ygap=3,
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=320, margin=dict(t=14, b=14))
    return _styled(fig)


def fig_roc(y_true, proba) -> go.Figure:
    fpr, tpr, _ = roc_curve(y_true, proba)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC",
                             line=dict(color=config.PRIMARY, width=3)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance",
                             line=dict(color=config.MUTED, width=1.5, dash="dash")))
    auc = float(np.trapz(tpr, fpr))
    fig.update_xaxes(title="False positive rate", range=[0, 1.02])
    fig.update_yaxes(title="True positive rate", range=[0, 1.02])
    fig.update_layout(legend=dict(x=0.98, y=0.02, xanchor="right"),
                      title=f"ROC curve (AUC = {auc:.3f})")
    return _styled(fig)


def fig_pr_curve(y_true, proba) -> go.Figure:
    prec, rec, _ = precision_recall_curve(y_true, proba)
    ap = float(np.mean([p for p in prec]))
    from sklearn.metrics import average_precision_score
    ap = float(average_precision_score(y_true, proba))
    base = float(np.mean(y_true))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name="PR curve",
                             line=dict(color=config.ACCENT, width=3)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[base, base], mode="lines",
                             name=f"Positive-rate baseline ({base:.3f})",
                             line=dict(color=config.MUTED, width=1.5, dash="dash")))
    fig.update_xaxes(title="Recall", range=[0, 1.02])
    fig.update_yaxes(title="Precision", range=[0, 1.05])
    fig.update_layout(legend=dict(x=0.98, y=0.02, xanchor="right"),
                      title=f"Precision-Recall curve (AP = {ap:.3f})")
    return _styled(fig)


def fig_importance(importance: dict, std: dict = None, top: int = 8) -> go.Figure:
    items = sorted(importance.items(), key=lambda kv: kv[1])[-top:]
    names = [k for k, _ in items]
    means = [v for _, v in items]
    errs = [std.get(k, 0.0) for k in names] if std else None
    fig = go.Figure(go.Bar(
        x=means, y=[n.replace(" [", "<br>[") for n in names],
        orientation="h", marker_color=config.ACCENT,
        error_x=dict(type="data", array=errs, color=config.MUTED) if errs else None,
        text=[f"{v:.3f}" for v in means], textposition="outside",
        textfont=dict(color=config.MUTED, size=11),
    ))
    fig.update_xaxes(title="Mean decrease in log-loss (permutation importance)")
    fig.update_layout(height=max(280, 42 * len(names) + 80))
    return _styled(fig)


# ------------------------------------------------------------- utilities ---
def style_risk_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with display-friendly columns for st.dataframe."""
    out = df.copy()
    if "alert" in out.columns:
        out["alert"] = out["alert"].map({True: "ALERT", False: "-"})
    return out
