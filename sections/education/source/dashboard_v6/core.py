"""Shared data, look & feel and chart helpers for the v6 dashboard.

Conventions
- Births are measured by the total fertility rate (TFR, SingStat M810091) only.
- Every price or income index is rebased to 2010 = 100 in notebook 04, so key numbers and charts use the same base.
- Each factor page ends with a "link" chart: one series from that page plotted against TFR, as levels,
  detrended values or yearly changes, with an optional lag.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.stats import pearsonr

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"

# ---------------------------------------------------------------- palette (entity -> colour, never by rank)
BLUE, ORANGE, AQUA, VIOLET, YELLOW, PINK, GREEN, RED = ("#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#eda100",
                                                        "#e87ba4", "#008300", "#e34948")
SERIES = [ORANGE, AQUA, VIOLET, YELLOW, PINK, GREEN]
INK, MUTED, GRID, LIGHT = "#1f1f1d", "#6b6a66", "#ececea", "#d4d3cd"
DIVERGING = [[0, BLUE], [0.5, "#efeeea"], [1, RED]]
BASE_YEAR = 2010

FACTORS = {
    "Marriage": dict(page="views/marriage.py", icon="💍", rep="F_Marriage_Rate_25_29", colour=PINK),
    "Income": dict(page="views/income.py", icon="💰", rep="Real_Income_Index", colour=AQUA),
    "Housing": dict(page="views/housing.py", icon="🏠", rep="HDB_Resale_Price_Index", colour=ORANGE),
    "Education": dict(page="views/education.py", icon="🎓", rep="Edu_CPI_Tuition", colour=VIOLET),
    "Healthcare": dict(page="views/healthcare.py", icon="🏥", rep="Health_CPI", colour=YELLOW),
}

# short names for crowded places (matrix axes, small multiples)
SHORT = {
    "TFR": "TFR", "F_Marriage_Rate_20_24": "Marriage rate 20-24", "F_Marriage_Rate_25_29": "Marriage rate 25-29",
    "F_Marriage_Rate_30_34": "Marriage rate 30-34", "F_Marriage_Rate_35_39": "Marriage rate 35-39",
    "F_Marriage_Rate_40_44": "Marriage rate 40-44", "Median_Age_First_Birth": "Age at 1st birth",
    "Brides_25_29": "Brides 25-29", "Brides_30_34": "Brides 30-34",
    "Real_Income_Index": "Real income", "Nominal_Income_Index": "Nominal income", "PM_Avg": "Avg income / member",
    "Real_Growth_%": "Real income growth", "Gini_Before_Transfers": "Gini (before)", "Gini_After_Transfers": "Gini (after)",
    "HDB_Resale_Price_Index": "HDB prices", "Private_Non_Landed_Index": "Condo prices",
    "Private_Landed_Index": "Landed prices", "HDB_YoY_%": "HDB price change",
    "Edu_CPI": "Education prices", "Edu_CPI_Tuition": "Tuition prices", "Edu_CPI_General": "General edu prices",
    "Edu_CPI_Textbooks": "Textbook prices", "Edu_Primary_per_Student": "Spend / primary pupil",
    "Edu_Secondary_per_Student": "Spend / secondary pupil", "Edu_Public_Total": "Gov edu spending",
    "Health_CPI": "Healthcare prices", "Health_CPI_Medicines": "Medicines prices",
    "Health_CPI_Outpatient": "Outpatient prices", "Health_CPI_Inpatient": "Inpatient prices",
    "Health_CPI_Other": "Other health prices", "Health_CPI_Insurance": "Insurance prices",
    "Gov_Health_per_Capita": "Gov health / person", "Gov_Health_%_GDP": "Gov health % GDP",
}


# ---------------------------------------------------------------- data
@st.cache_data
def master() -> pd.DataFrame:
    return pd.read_csv(PROC / "team_master.csv")


@st.cache_data
def dictionary() -> pd.DataFrame:
    return pd.read_csv(PROC / "team_dictionary.csv").set_index("column")


@st.cache_data
def table(name: str) -> pd.DataFrame:
    return pd.read_csv(PROC / name)


def label(c):
    d = dictionary()
    return d.loc[c, "label"] if c in d.index else c


def short(c):
    return SHORT.get(c, label(c))


def src(*cols):
    d = dictionary()
    ids = sorted({str(d.loc[c, "source_table"]) for c in cols if c in d.index})
    return "Source: SingStat " + ", ".join(ids) if ids else ""


def years():
    return st.session_state.get("years", (2010, 2025))


def data() -> pd.DataFrame:
    y0, y1 = years()
    df = master()
    return df[(df.Year >= y0) & (df.Year <= y1)].copy()


def tfr_index(df):
    """TFR rebased to 2010 = 100 (same base as every price/income index)."""
    base = master().set_index("Year").loc[BASE_YEAR, "TFR"]
    return df.TFR / base * 100


# ---------------------------------------------------------------- small text helpers
def trend(df, col, fmt="{:,.2f}"):
    s = df.set_index("Year")[col].dropna()
    if len(s) < 2:
        return ""
    a, b = s.iloc[0], s.iloc[-1]
    pct = f" ({(b / a - 1) * 100:+.0f}%)" if a else ""
    return f"{s.index[0]}→{s.index[-1]}: {fmt.format(a)} → {fmt.format(b)}{pct}"


def note(*parts):
    st.caption("  ·  ".join(p for p in parts if p).replace("$", "\\$"))   # stop "$a → $b" rendering as LaTeX


# ---------------------------------------------------------------- layout pieces
def header(title, subtitle=""):
    y0, y1 = years()
    st.markdown(f"""<div class="pagehead"><div><h1>{title}</h1><p>{subtitle}</p></div>
        <div class="chips"><span class="chip">Data · SingStat Table Builder</span>
        <span class="chip">Years · {y0}–{y1}</span><span class="chip">Indices · 2010 = 100</span></div></div>""",
                unsafe_allow_html=True)


def kpis(items):
    """items: dicts with label, value, delta (optional), sub (optional). Rendered as bordered cards."""
    html = "".join(
        f'<div class="kpi"><div class="kpi-l">{it["label"]}</div><div class="kpi-v">{it["value"]}</div>'
        + (f'<span class="pill {it.get("tone", "")}">{it["delta"]}</span>' if it.get("delta") else "")
        + (f'<div class="kpi-s">{it["sub"]}</div>' if it.get("sub") else "") + "</div>" for it in items)
    st.markdown(f'<div class="kpis">{html}</div>', unsafe_allow_html=True)


def kpi(df, col, fmt="{:,.0f}", prefix="", suffix="", pct=True, title=None, dsuffix=""):
    """Latest value in the selected years, with the change since the first year that has data."""
    s = df.set_index("Year")[col].dropna()
    if s.empty:
        return dict(label=title or label(col), value="–")
    y0, v0, y1, v1 = int(s.index[0]), s.iloc[0], int(s.index[-1]), s.iloc[-1]
    if pct and v0:
        ch = (v1 / v0 - 1) * 100
        delta = f"{'▲' if ch >= 0 else '▼'} {abs(ch):.0f}% vs {y0}"
    else:
        ch = v1 - v0
        delta = f"{'▲' if ch >= 0 else '▼'} {f'{abs(ch):,.2f}'.rstrip('0').rstrip('.')}{dsuffix} vs {y0}"
    return dict(label=title or label(col), value=prefix + fmt.format(v1) + suffix, delta=delta, sub=f"{y1}",
                tone="up" if ch >= 0 else "down")


def card(title):
    c = st.container(border=True)
    c.markdown(f'<div class="card-t">{title}</div>', unsafe_allow_html=True)
    return c


def style(fig, h=380, legend=True, hover="x unified"):
    fig.update_layout(height=h, template="plotly_white", margin=dict(l=8, r=8, t=12, b=8), font=dict(size=12, color=INK),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, title=None, font=dict(size=11)),
                      showlegend=legend, hovermode=hover, hoverlabel=dict(bgcolor="white", font_size=12),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(showgrid=False, linecolor=LIGHT)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def show(fig, key=None):
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=key)


def add_tfr_index(fig, df, **kw):
    """TFR on the same 2010 = 100 scale as an index chart (no second axis needed)."""
    d = df.dropna(subset=["TFR"])
    fig.add_trace(go.Scatter(x=d.Year, y=tfr_index(d), name="TFR (2010 = 100)", mode="lines+markers",
                             line=dict(color=BLUE, width=3.2, dash="dot"), marker=dict(size=5),
                             customdata=d.TFR, hovertemplate="TFR %{customdata:.2f} (index %{y:.0f})<extra></extra>"), **kw)
    return fig


def index_lines(df, cols, colors, names, h=320):
    """Indices (2010 = 100) plus TFR on the same scale."""
    fig = go.Figure()
    for c in cols:
        d = df.dropna(subset=[c])
        fig.add_trace(go.Scatter(x=d.Year, y=d[c], name=names.get(c, short(c)), mode="lines+markers",
                                 line=dict(color=colors[c], width=2.5), marker=dict(size=5),
                                 hovertemplate=names.get(c, short(c)) + " %{y:.0f}<extra></extra>"))
    add_tfr_index(fig, df)
    fig.add_hline(y=100, line=dict(color="#bdbbb4", width=1))
    fig.update_yaxes(title_text="index, 2010 = 100")
    return style(fig, h)


def tfr_stack(df, h=330, ratio=0.3):
    """Two panels sharing the year axis: TFR on top, the factor below (avoids a second y-axis)."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[ratio, 1 - ratio], vertical_spacing=0.06)
    d = df.dropna(subset=["TFR"])
    fig.add_trace(go.Scatter(x=d.Year, y=d.TFR, name="TFR", mode="lines+markers", line=dict(color=BLUE, width=2.6),
                             marker=dict(size=4), hovertemplate="TFR %{y:.2f}<extra></extra>"), row=1, col=1)
    fig.update_yaxes(title_text="TFR", row=1, col=1, nticks=3)
    style(fig, h)
    return fig


# ---------------------------------------------------------------- link chart (factor vs TFR)
MODES = ["Levels", "Detrended", "Yearly change"]
MODE_SHORT = {"Levels": "Levels", "Detrended": "Detrend", "Yearly change": "Yearly Δ"}
MODE_HELP = ("Levels: the raw values. Detrended: each series minus its own straight-line trend over the selected "
             "years, so only ups and downs around the trend are compared. Yearly change: the change from the previous "
             "year. If a link only shows in Levels, the two series mainly share a long-run trend.")


def transform(d, cols, mode):
    """d indexed by Year. Returns the transformed columns (consecutive years only for yearly change)."""
    d = d[cols].dropna()
    if mode == "Detrended":
        t = d.index.values.astype(float)
        out = d.copy()
        for c in cols:
            b, a = np.polyfit(t, d[c].values, 1)
            out[c] = d[c].values - (a + b * t)
        return out
    if mode == "Yearly change":
        full = d.reindex(range(d.index.min(), d.index.max() + 1))
        return full.diff().dropna()
    return d


def pair(df, x, mode, lag=0):
    """Factor x in year t against TFR in year t + lag, after the chosen transformation."""
    s = df.set_index("Year")
    xs, ys = s[x].dropna(), s["TFR"].dropna()
    if lag:
        xs = xs.copy(); xs.index = xs.index + lag
    d = pd.concat([xs.rename("x"), ys.rename("y")], axis=1).dropna()
    if len(d) < 4:
        return d
    return transform(d, ["x", "y"], mode)


def r_of(df, x, mode, lag=0):
    d = pair(df, x, mode, lag)
    return (pearsonr(d.x, d.y)[0], len(d)) if len(d) >= 4 and d.x.std() > 0 else (np.nan, len(d))


def link(df, x_opts, key, h=300):
    """Scatter of one series from this page against TFR. Dots are years (darker = more recent)."""
    c1, c2 = st.columns([1.15, 3])
    with c1:
        x = st.selectbox("Series", x_opts, format_func=short, key=f"{key}_x")
        mode = st.segmented_control("Compare", MODES, default="Levels", key=f"{key}_m", help=MODE_HELP,
                                    format_func=MODE_SHORT.get) or "Levels"
        lag = st.selectbox("Lag", [0, 1, 2, 3], key=f"{key}_lag",
                           format_func=lambda v: "Same year" if v == 0 else f"TFR {v} yr later",
                           help="Compare the series in one year with TFR one or more years later.")
        rs = {m: r_of(df, x, m, lag)[0] for m in MODES}
        st.markdown('<div class="rtable">' + "".join(
            f'<div class="{"on" if m == mode else ""}"><span>{m}</span><b>{"–" if np.isnan(v) else f"{v:+.2f}"}</b></div>'
            for m, v in rs.items()) + "</div>", unsafe_allow_html=True)
        st.caption("Pearson r with TFR")
    d = pair(df, x, mode, lag)
    with c2:
        if len(d) < 4:
            st.info("Not enough overlapping years for this series - widen the year range.")
            return
        yrs = d.index.values
        fig = go.Figure()
        if mode != "Levels":
            fig.add_hline(y=0, line=dict(color=LIGHT, width=1)); fig.add_vline(x=0, line=dict(color=LIGHT, width=1))
        b, a = np.polyfit(d.x, d.y, 1)
        xs = np.linspace(d.x.min(), d.x.max(), 30)
        fig.add_trace(go.Scatter(x=xs, y=a + b * xs, mode="lines", line=dict(color=MUTED, dash="dash", width=1.4),
                                 hoverinfo="skip", showlegend=False))
        ends = {yrs.min(), yrs.max()}
        fig.add_trace(go.Scatter(
            x=d.x, y=d.y, mode="markers+text", text=[str(y) if y in ends else "" for y in yrs],
            textposition="top center", textfont=dict(size=10, color=MUTED), customdata=yrs,
            marker=dict(size=13, color=yrs, colorscale="Blues", cmin=yrs.min() - 0.5 * (yrs.max() - yrs.min() + 1),
                        cmax=yrs.max(), line=dict(color="white", width=1.5)),
            hovertemplate="<b>%{customdata}</b><br>" + short(x) + ": %{x:,.2f}<br>TFR: %{y:,.3f}<extra></extra>",
            showlegend=False))
        suffix = {"Levels": "", "Detrended": " (gap from trend)", "Yearly change": " (change vs previous year)"}[mode]
        fig.update_xaxes(title_text=short(x) + suffix, showgrid=True, gridcolor=GRID)
        fig.update_yaxes(title_text=("TFR" + (f" {lag} yr later" if lag else "")) + suffix)
        show(style(fig, h, legend=False, hover="closest"))
        note(f"{len(d)} years · darker dot = more recent year", src(x, "TFR"))


# ---------------------------------------------------------------- page style, bar hover and bar click focus
CSS_EXTRA = """
<style>
.rtable { display:flex; flex-direction:column; gap:4px; margin-top:6px; font-family:'Inter',system-ui,sans-serif; }
.rtable div { display:flex; justify-content:space-between; padding:4px 10px; border-radius:8px; background:#f6f6f4;
  font-size:.82rem; color:#6b6a66; }
.rtable div.on { background:#e8f1fc; color:#1d5fae; }
.rtable b { font-weight:650; color:inherit; }
</style>
"""
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
.pagehead, .kpi, .card-t, .chip, .legend-row { font-family: 'Inter', system-ui, sans-serif; }
[data-testid="stAppViewContainer"] { background: #f6f6f4; }
[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #ececea; }
.block-container { padding-top: 1.6rem; max-width: 1400px; }
.pagehead { display:flex; justify-content:space-between; align-items:flex-end; gap:16px; flex-wrap:wrap; margin-bottom:14px; }
.pagehead h1 { font-size:1.75rem; font-weight:700; margin:0; padding:0; color:#1f1f1d; }
.pagehead p { margin:2px 0 0; color:#6b6a66; font-size:.95rem; }
.chips { display:flex; gap:6px; flex-wrap:wrap; }
.chip { background:#fff; border:1px solid #e3e2de; border-radius:999px; padding:3px 10px; font-size:.78rem; color:#52514e; }
.kpis { display:grid; grid-template-columns:repeat(auto-fit, minmax(135px, 1fr)); gap:12px; margin:4px 0 18px; }
.kpi { background:#fff; border:1px solid #e6e5e1; border-radius:12px; padding:12px 14px; box-shadow:0 1px 2px rgba(0,0,0,.03); }
.kpi-l { font-size:.78rem; color:#6b6a66; text-transform:uppercase; letter-spacing:.03em; }
.kpi-v { font-size:1.6rem; font-weight:650; color:#1f1f1d; margin:2px 0 4px; }
.kpi-s { font-size:.75rem; color:#8a8984; margin-top:4px; }
.pill { display:inline-block; font-size:.74rem; padding:2px 8px; border-radius:999px; background:#efeeea; color:#52514e; }
.pill.up { background:#e8f1fc; color:#1d5fae; } .pill.down { background:#fdecec; color:#b4302f; }
[data-testid="stVerticalBlockBorderWrapper"] { background:#fff; border-radius:12px !important; border-color:#e6e5e1 !important;
  box-shadow:0 1px 2px rgba(0,0,0,.03); }
.card-t { font-weight:600; font-size:1.0rem; color:#1f1f1d; margin-bottom:2px; }
.legend-row { display:flex; gap:18px; justify-content:center; flex-wrap:wrap; font-size:.82rem; color:#52514e; margin:-4px 0 10px; }
.dot { display:inline-block; width:11px; height:11px; border-radius:50%; margin-right:6px; vertical-align:-1px; }
/* bars grow slightly on hover (class added by the script below) */
.js-plotly-plot .barlayer .trace, .js-plotly-plot .barlayer .point { transition: opacity .2s ease; }
.js-plotly-plot .barlayer .point path { transition: transform .16s ease, filter .16s ease; transform-box: fill-box; }
.js-plotly-plot .barlayer .point path.bar-hot { filter: brightness(1.06) drop-shadow(0 2px 4px rgba(0,0,0,.18)); }
</style>
"""

HOVER_JS = """
<script>
// Runs inside a same-origin component iframe and attaches handlers to every Plotly chart on the page.
// Hover: the bar under the mouse grows slightly. Click: keep that bar's series in colour and fade the rest;
// click it again (or double-click the chart) to bring everything back.
const doc = window.parent.document;
function barTraces(gd) { return gd._fullData.filter(t => t.type === 'bar' && t.visible === true); }
function pathOf(gd, pt) {
  const idx = barTraces(gd).findIndex(t => t.index === pt.curveNumber); if (idx < 0) return null;
  const g = gd.querySelectorAll('.barlayer .trace')[idx]; if (!g) return null;
  return g.querySelectorAll('.point')[pt.pointNumber]?.querySelector('path');
}
function clearHot(gd) { gd.querySelectorAll('path.bar-hot').forEach(p => { p.style.transform = ''; p.classList.remove('bar-hot'); }); }
function clearFocus(gd) {
  gd.querySelectorAll('.barlayer .trace, .barlayer .point').forEach(e => e.style.opacity = '');
  delete gd.dataset.focus;
}
function focus(gd, pt) {
  const traces = barTraces(gd), groups = gd.querySelectorAll('.barlayer .trace');
  const single = traces.length === 1;
  const key = single ? 'p' + pt.pointNumber : 't' + pt.curveNumber;
  if (gd.dataset.focus === key) { clearFocus(gd); return; }
  clearFocus(gd); gd.dataset.focus = key;
  if (single) {
    groups[0]?.querySelectorAll('.point').forEach((p, i) => p.style.opacity = i === pt.pointNumber ? '1' : '0.22');
  } else {
    const idx = traces.findIndex(t => t.index === pt.curveNumber);
    groups.forEach((g, i) => g.style.opacity = i === idx ? '1' : '0.18');
  }
}
function bind(gd) {
  if (!gd.on || gd.dataset.barHover) return;
  gd.dataset.barHover = '1';
  gd.on('plotly_hover', ev => {
    clearHot(gd);
    ev.points.forEach(pt => {
      if (pt.data.type !== 'bar') return;
      const path = pathOf(gd, pt); if (!path) return;
      path.style.transformOrigin = 'center center';
      path.style.transform = pt.data.orientation === 'h' ? 'scale(1.04, 1.18)' : 'scale(1.18, 1.04)';
      path.classList.add('bar-hot');
    });
  });
  gd.on('plotly_unhover', () => clearHot(gd));
  gd.on('plotly_click', ev => { const pt = ev.points[0]; if (pt && pt.data.type === 'bar') focus(gd, pt); });
  gd.on('plotly_doubleclick', () => clearFocus(gd));
}
setInterval(() => doc.querySelectorAll('.js-plotly-plot').forEach(bind), 700);
</script>
"""
