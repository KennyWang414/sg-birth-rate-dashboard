"""Singapore Birth Rate Explorer (v8) - the whole Streamlit dashboard in one file.

Run from this folder:   streamlit run SB_Group8_Dashboard.py
Data: Fertility_Combined_Data.csv (same folder) - every member's cleaned SingStat series, one row per year.

Conventions
- Births are measured by the total fertility rate (TFR, SingStat M810091).
- Price indices are 2010 = 100. Income is in S$: nominal as published (17872) and real = nominal / CPI x 100,
  with CPI All Items (M213751) expressed as 2010 = 100 (SingStat's method; reproduces 17885 exactly).
- Statistics: Pearson r with a two-sided p-value (scipy.stats.pearsonr), n = years used.
    Levels          x[t]            vs TFR[t+L]
    Yearly change   x[t] - x[t-1]   vs TFR[t+L] - TFR[t+L-1]          (L = lag in years, 0-3)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots
from scipy.stats import pearsonr, spearmanr

DATA = Path(__file__).resolve().parent / "Fertility_Combined_Data.csv"

# ======================================================================== palette (entity -> colour, never by rank)
BLUE, ORANGE, AQUA, VIOLET, YELLOW, PINK, GREEN, RED = ("#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#eda100",
                                                        "#e87ba4", "#008300", "#e34948")
GOLD = "#c98a00"
SERIES = [ORANGE, AQUA, VIOLET, YELLOW, PINK, GREEN]
INK, MUTED, GRID, LIGHT = "#1f1f1d", "#6b6a66", "#ececea", "#d4d3cd"
DIVERGING = [[0, BLUE], [0.5, "#efeeea"], [1, RED]]
BASE_YEAR = 2010

# ======================================================================== what every column is
# column: (label, factor, unit, SingStat table)
META = {
    "TFR": ("Total fertility rate", "Births", "children per woman", "M810091"),
    **{f"ASFR_{a}_{a + 4}": (f"Births per 1,000 women aged {a}-{a + 4}", "Births", "per 1,000 women", "M810091")
       for a in range(15, 45, 5)},
    **{f"F_Marriage_Rate_{a}_{a + 4}": (f"Female marriage rate, age {a}-{a + 4}", "Marriage", "per 1,000 unmarried women",
                                        "M830102") for a in range(20, 45, 5)},
    "Later_Marriage_Shift_Index": ("Later-marriage shift index: rate 30-34 / rate 25-29", "Marriage", "ratio", "M830102"),
    "Median_Age_First_Birth": ("Median age of mothers at first birth", "Marriage", "years", "M810741"),
    "Brides_25_29": ("Brides aged 25-29", "Marriage", "marriages", "M830163"),
    "Brides_30_34": ("Brides aged 30-34", "Marriage", "marriages", "M830163"),
    "PM_Median": ("Median income per household member, nominal", "Income", "S$ per month", "17872"),
    "PM_Median_Real_2010": ("Median income per household member, real (2010 S$)", "Income", "2010 S$ per month",
                            "17872, M213751"),
    "PM_Avg": ("Average income per household member, nominal", "Income", "S$ per month", "17872"),
    "PM_Avg_Real_2010": ("Average income per household member, real (2010 S$)", "Income", "2010 S$ per month",
                         "17872, M213751"),
    "CPI_All_Items_2010_100": ("CPI All Items (2010 = 100)", "Income", "2010 = 100", "M213751"),
    "Real_Growth_%": ("Real change in average income per member", "Income", "% vs previous year", "17885"),
    **{f"Real_Growth_D{i}": (f"Real change in average income per member, decile {i}", "Income", "% vs previous year",
                             "17885") for i in range(1, 10)},
    "Real_Growth_D10_M810361": ("Real change in average income per member, top 10%", "Income", "% vs previous year",
                                "M810361"),
    "Real_Growth_D1_M810361": ("Real change in average income per member, lowest 10%", "Income", "% vs previous year",
                               "M810361"),
    "Gini_Before_Transfers": ("Gini before government transfers & taxes", "Income", "0 = equal, 1 = unequal", "17892"),
    "Gini_After_Transfers": ("Gini after government transfers & taxes", "Income", "0 = equal, 1 = unequal", "17892"),
    "HDB_Resale_Price_Index": ("HDB resale price index", "Housing", "2010 = 100", "M212161"),
    "Private_Landed_Index": ("Private landed price index", "Housing", "2010 = 100", "M212261"),
    "Private_Non_Landed_Index": ("Private non-landed price index", "Housing", "2010 = 100", "M212261"),
    "HDB_YoY_%": ("HDB resale price change", "Housing", "% vs previous year", "M212161"),
    "Landed_YoY_%": ("Landed price change", "Housing", "% vs previous year", "M212261"),
    "NonLanded_YoY_%": ("Non-landed price change", "Housing", "% vs previous year", "M212261"),
    "Edu_CPI": ("Education price index", "Education", "2010 = 100", "M213801"),
    "Edu_CPI_General": ("General education price index", "Education", "2010 = 100", "M213801"),
    "Edu_CPI_Tuition": ("Private tuition & courses price index", "Education", "2010 = 100", "M213801"),
    "Edu_CPI_Textbooks": ("Textbooks & study guides price index", "Education", "2010 = 100", "M213801"),
    "Edu_Public_Total": ("Government education spending", "Education", "S$ million (FY)", "M850011"),
    "Edu_Primary_per_Student": ("Government spending per primary student", "Education", "S$ (FY)", "M850291"),
    "Edu_Secondary_per_Student": ("Government spending per secondary student", "Education", "S$ (FY)", "M850291"),
    "Edu_Op_Primary": ("Operating: primary", "Education", "S$ million (FY)", "M850011"),
    "Edu_Op_Secondary_JC": ("Operating: secondary & JC", "Education", "S$ million (FY)", "M850011"),
    "Edu_Op_Tertiary": ("Operating: tertiary", "Education", "S$ million (FY)", "M850011"),
    "Edu_Op_ITE": ("Operating: ITE", "Education", "S$ million (FY)", "M850011"),
    "Edu_Op_Other": ("Operating: other", "Education", "S$ million (FY)", "M850011"),
    "Edu_Development": ("Development spending", "Education", "S$ million (FY)", "M850011"),
    "Health_CPI": ("Healthcare price index", "Healthcare", "2010 = 100", "M213801"),
    "Health_CPI_Medicines": ("Medicines & health products price index", "Healthcare", "2010 = 100", "M213801"),
    "Health_CPI_Outpatient": ("Outpatient care price index", "Healthcare", "2010 = 100", "M213801"),
    "Health_CPI_Inpatient": ("Inpatient care price index", "Healthcare", "2010 = 100", "M213801"),
    "Health_CPI_Other": ("Other health services price index", "Healthcare", "2010 = 100", "M213801"),
    "Health_CPI_Insurance": ("Health insurance price index", "Healthcare", "2010 = 100", "M213801"),
    "HES_Health_Monthly": ("Household health spending", "Healthcare", "S$ per household per month", "M212981"),
    "Gov_Health_Total": ("Government health spending", "Healthcare", "S$ million (FY)", "M870391"),
    "Gov_Health_per_Capita": ("Government health spending per person", "Healthcare", "S$ (FY)", "M870391"),
    "Gov_Health_%_GDP": ("Government health spending, % of GDP", "Healthcare", "% of GDP", "M870391"),
}
NO_CORR = {"CPI_All_Items_2010_100", "Real_Growth_%", "Real_Growth_D10_M810361", "Real_Growth_D1_M810361",
           *[f"Real_Growth_D{i}" for i in range(1, 10)], *[c for c in META if c.startswith("Edu_Op_")], "Edu_Development"}

# short names for crowded places (legends, matrix axes, selectors)
SHORT = {
    "TFR": "TFR", **{f"F_Marriage_Rate_{a}_{a + 4}": f"Marriage rate {a}-{a + 4}" for a in range(20, 45, 5)},
    "Later_Marriage_Shift_Index": "Later-marriage shift index", "Median_Age_First_Birth": "Age at 1st birth",
    "Brides_25_29": "Brides 25-29", "Brides_30_34": "Brides 30-34",
    "PM_Median": "Median income / member", "PM_Median_Real_2010": "Real median income / member",
    "PM_Avg": "Average income / member", "PM_Avg_Real_2010": "Real average income / member",
    "Gini_Before_Transfers": "Gini before transfers", "Gini_After_Transfers": "Gini after transfers",
    "HDB_Resale_Price_Index": "HDB resale prices", "Private_Non_Landed_Index": "Private non-landed prices",
    "Private_Landed_Index": "Private landed prices", "HDB_YoY_%": "HDB price change",
    "Edu_CPI": "Education prices", "Edu_CPI_Tuition": "Tuition prices", "Edu_CPI_General": "General education prices",
    "Edu_CPI_Textbooks": "Textbook prices", "Edu_Primary_per_Student": "Gov spend / primary pupil",
    "Edu_Secondary_per_Student": "Gov spend / secondary pupil", "Edu_Public_Total": "Gov education spending",
    "Health_CPI": "Healthcare prices", "Health_CPI_Medicines": "Medicines prices",
    "Health_CPI_Outpatient": "Outpatient prices", "Health_CPI_Inpatient": "Inpatient prices",
    "Health_CPI_Other": "Other health prices", "Health_CPI_Insurance": "Health insurance prices",
    "Gov_Health_per_Capita": "Gov health spend / person", "Gov_Health_%_GDP": "Gov health % of GDP",
}

# one headline series per factor (Overview) - named by the actual variable
FACTORS = {
    "Marriage": dict(icon="💍", rep="F_Marriage_Rate_25_29", name="Marriage rate, women 25-29", colour=PINK),
    "Income": dict(icon="💰", rep="PM_Median_Real_2010", name="Real median income per member", colour=AQUA),
    "Housing": dict(icon="🏠", rep="HDB_Resale_Price_Index", name="HDB resale prices", colour=ORANGE),
    "Education": dict(icon="🎓", rep="Edu_CPI_Tuition", name="Tuition prices", colour=VIOLET),
    "Healthcare": dict(icon="🏥", rep="Health_CPI", name="Healthcare prices", colour=YELLOW),
}

# Census 2020 (SingStat CT/17486): married-couple households by housing type; private = condominiums + landed
CENSUS = pd.DataFrame({"Housing_Type": ["HDB flats", "Private housing"],
                       "With_Children": [526014, 120714 + 43865], "Without_Children": [181701, 36881 + 10970]})


# ======================================================================== data
@st.cache_data
def master() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    df["Later_Marriage_Shift_Index"] = df.F_Marriage_Rate_30_34 / df.F_Marriage_Rate_25_29   # marriage member's index
    return df


def years():
    return st.session_state.get("years", (2010, 2025))


def data() -> pd.DataFrame:
    y0, y1 = years()
    df = master()
    return df[(df.Year >= y0) & (df.Year <= y1)].copy()


def label(c):
    return META[c][0] if c in META else c


def short(c):
    return SHORT.get(c, label(c))


def src(*cols):
    ids = sorted({t.strip() for c in cols if c in META for t in META[c][3].split(",")})
    return "Source: SingStat " + ", ".join(ids) if ids else ""


def trend(df, col, fmt="{:,.2f}"):
    s = df.set_index("Year")[col].dropna()
    if len(s) < 2:
        return ""
    a, b = s.iloc[0], s.iloc[-1]
    pct = f" ({(b / a - 1) * 100:+.0f}%)" if a else ""
    return f"{s.index[0]}→{s.index[-1]}: {fmt.format(a)} → {fmt.format(b)}{pct}"


def note(*parts):
    st.caption("  ·  ".join(p for p in parts if p).replace("$", "\\$"))   # stop "$a → $b" rendering as LaTeX


# ======================================================================== layout pieces
def header(title, subtitle=""):
    y0, y1 = years()
    st.markdown(f"""<div class="pagehead"><div><h1>{title}</h1><p>{subtitle}</p></div>
        <div class="chips"><span class="chip">Data · SingStat Table Builder</span>
        <span class="chip">Years · {y0}–{y1}</span></div></div>""", unsafe_allow_html=True)


def kpis(items):
    html = "".join(
        f'<div class="kpi"><div class="kpi-l">{it["label"]}</div><div class="kpi-v">{it["value"]}</div>'
        + (f'<span class="pill {it.get("tone", "")}">{it["delta"]}</span>' if it.get("delta") else "")
        + (f'<div class="kpi-s">{it["sub"]}</div>' if it.get("sub") else "") + "</div>" for it in items)
    st.markdown(f'<div class="kpis">{html}</div>', unsafe_allow_html=True)


def kpi(df, col, fmt="{:,.0f}", prefix="", suffix="", pct=True, title=None, dfmt="{:,.2f}"):
    """Latest value in the selected years, and the change since the first year that has data."""
    s = df.set_index("Year")[col].dropna()
    if s.empty:
        return dict(label=title or label(col), value="–")
    y0, v0, y1, v1 = int(s.index[0]), s.iloc[0], int(s.index[-1]), s.iloc[-1]
    if pct and v0:
        ch = (v1 / v0 - 1) * 100
        delta = f"{'▲' if ch >= 0 else '▼'} {abs(ch):.0f}% vs {y0}"
    else:
        ch = v1 - v0
        delta = f"{'▲' if ch >= 0 else '▼'} {dfmt.format(abs(ch))} vs {y0}"
    return dict(label=title or label(col), value=prefix + fmt.format(v1) + suffix, delta=delta, sub=f"{y1}",
                tone="up" if ch >= 0 else "down")


def card(title):
    c = st.container(border=True)
    c.markdown(f'<div class="card-t">{title}</div>', unsafe_allow_html=True)
    return c


def finding(text):
    st.markdown(f'<div class="finding">{text}</div>', unsafe_allow_html=True)


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


def add_tfr_axis(fig, df):
    """TFR in its own units (children per woman) on a right-hand axis. Call after style()."""
    d = df.dropna(subset=["TFR"])
    fig.add_trace(go.Scatter(x=d.Year, y=d.TFR, name="TFR (right axis)", yaxis="y2", mode="lines+markers",
                             line=dict(color=BLUE, width=2.8, dash="dot"), marker=dict(size=5, symbol="diamond"),
                             hovertemplate="TFR %{y:.2f}<extra></extra>"))
    lo, hi = d.TFR.min(), d.TFR.max()
    pad = max((hi - lo) * 0.15, 0.03)
    fig.update_layout(
        yaxis2=dict(title=dict(text="TFR, children per woman", font=dict(color=BLUE, size=11)), overlaying="y",
                    side="right", showgrid=False, zeroline=False, tickformat=".2f", tickfont=dict(color=BLUE),
                    range=[lo - pad, hi + pad], linecolor=LIGHT),
        margin=dict(r=58))
    return fig


def index_lines(df, cols, colors, names, h=320):
    """Indices (2010 = 100) on the left axis, TFR on the right axis."""
    fig = go.Figure()
    for c in cols:
        d = df.dropna(subset=[c])
        fig.add_trace(go.Scatter(x=d.Year, y=d[c], name=names.get(c, short(c)), mode="lines+markers",
                                 line=dict(color=colors[c], width=2.5), marker=dict(size=5),
                                 hovertemplate=names.get(c, short(c)) + " %{y:.0f}<extra></extra>"))
    fig.add_hline(y=100, line=dict(color="#bdbbb4", width=1))
    fig.update_yaxes(title_text="index, 2010 = 100")
    return add_tfr_axis(style(fig, h), df)


# ======================================================================== statistics
MODES = ["Levels", "Yearly change"]
MODE_SHORT = {"Levels": "Levels", "Yearly change": "Yearly Δ"}
MODE_HELP = ("Levels: the values as they are. Both series trend over time, so levels can look strongly linked "
             "just because one rises while the other falls.\n\nYearly change: this year minus last year for both "
             "series. It asks: in years when this factor moved more than usual, did TFR move too?")
LAG_HELP = ("Compare this year's value with TFR one or more years later. A decision to have a child shows up in "
            "births about a year later (pregnancy), so a cause should show up with a lag.")


def formula(mode, lag=0):
    t = f"t+{lag}" if lag else "t"
    return (f"r of x[t] and TFR[{t}]" if mode == "Levels"
            else f"r of Δx[t] = x[t] − x[t−1] and ΔTFR[{t}]")


def transform(d, cols, mode):
    """d indexed by Year. Levels as they are, or the change from the previous year (consecutive years only)."""
    d = d[cols].dropna()
    if mode == "Yearly change":
        full = d.reindex(range(int(d.index.min()), int(d.index.max()) + 1))
        return full.diff().dropna()
    return d


def corr_test(x, y, method="pearson"):
    """r, two-sided p-value and n."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n < 4 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan, n
    r, p = (pearsonr if method == "pearson" else spearmanr)(x, y)
    return float(r), float(p), n


def stars(p):
    return "" if np.isnan(p) else "**" if p < 0.01 else "*" if p < 0.05 else ""


def p_text(p):
    return "–" if np.isnan(p) else "<0.001" if p < 0.001 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def corr_matrix(m, method="pearson", min_n=6):
    """Pairwise r, p and n for every column pair (each pair uses the years both have)."""
    cols = list(m.columns)
    R, P, N = (pd.DataFrame(np.nan, index=cols, columns=cols) for _ in range(3))
    for i, a in enumerate(cols):
        for b in cols[i:]:
            if a == b:
                r, p, n = 1.0, 0.0, int(m[a].notna().sum())
            else:
                d = m[[a, b]].dropna()
                r, p, n = corr_test(d[a].values, d[b].values, method) if len(d) >= min_n else (np.nan, np.nan, len(d))
            R.loc[a, b] = R.loc[b, a] = r; P.loc[a, b] = P.loc[b, a] = p; N.loc[a, b] = N.loc[b, a] = n
    return R, P, N


def pair(df, x, mode, lag=0):
    """Series x in year t against TFR in year t + lag, after the chosen transformation."""
    s = df.set_index("Year")
    xs, ys = s[x].dropna(), s["TFR"].dropna()
    if lag:
        xs = xs.copy(); xs.index = xs.index + lag
    d = pd.concat([xs.rename("x"), ys.rename("y")], axis=1).dropna()
    if len(d) < 4:
        return d
    return transform(d, ["x", "y"], mode)


def r_of(df, x, mode, lag=0):
    """(r, p, n) for series x against TFR."""
    d = pair(df, x, mode, lag)
    return corr_test(d.x.values, d.y.values) if len(d) >= 4 else (np.nan, np.nan, len(d))


def sig_word(p):
    return "" if np.isnan(p) else "significant" if p < 0.05 else "not significant"


def rp(r, p, n):
    """'r +0.65, p 0.009, n 15' for text."""
    return f"r {r:+.2f}, p {p_text(p)}, n {n}"


def controls(key, lag=True):
    mode = st.segmented_control("Compare", MODES, default="Levels", key=f"{key}_m", help=MODE_HELP,
                                format_func=MODE_SHORT.get) or "Levels"
    L = st.selectbox("Lag", [0, 1, 2, 3], key=f"{key}_lag", help=LAG_HELP,
                     format_func=lambda v: "Same year" if v == 0 else f"TFR {v} yr later") if lag else 0
    return mode, L


def rtable(df, x, mode, lag):
    rows = ""
    for m in MODES:
        r, p, n = r_of(df, x, m, lag)
        v = sig_word(p)
        rows += (f'<div class="{"on" if m == mode else ""} {"sig" if v == "significant" else ""}">'
                 f'<span>{MODE_SHORT[m]}</span><b>r {"–" if np.isnan(r) else f"{r:+.2f}"}</b>'
                 f'<em>p {p_text(p)} · n {n}<br>{v}</em></div>')
    st.markdown(f'<div class="rtable">{rows}</div>', unsafe_allow_html=True)


def scatter(d, xname, mode, lag, h=300, colorscale="Blues", all_labels=False, ref_x=None):
    """Dots = years, coloured by year; dashed = least-squares line."""
    yrs = d.index.values.astype(int)
    fig = go.Figure()
    if mode != "Levels":
        fig.add_hline(y=0, line=dict(color=LIGHT, width=1)); fig.add_vline(x=0, line=dict(color=LIGHT, width=1))
    elif ref_x is not None:
        fig.add_vline(x=ref_x, line=dict(color="#9a9893", width=1.2, dash="dot"))
    b, a = np.polyfit(d.x, d.y, 1)
    xs = np.linspace(d.x.min(), d.x.max(), 30)
    fig.add_trace(go.Scatter(x=xs, y=a + b * xs, mode="lines", line=dict(color=MUTED, dash="dash", width=1.4),
                             hoverinfo="skip", showlegend=False))
    ends = set(yrs) if all_labels else {yrs.min(), yrs.max()}
    fig.add_trace(go.Scatter(
        x=d.x, y=d.y, mode="markers+text", text=[str(y) if y in ends else "" for y in yrs],
        textposition="top center", textfont=dict(size=9 if all_labels else 10, color=MUTED), customdata=yrs,
        marker=dict(size=13, color=yrs, colorscale=colorscale,
                    cmin=yrs.min() if all_labels else yrs.min() - 0.5 * (yrs.max() - yrs.min() + 1),
                    cmax=yrs.max(), line=dict(color="white", width=1.5),
                    showscale=all_labels, colorbar=dict(title="Year", thickness=10, tickformat="d")),
        hovertemplate="<b>%{customdata}</b><br>" + xname + ": %{x:,.3f}<br>TFR: %{y:,.3f}<extra></extra>",
        showlegend=False))
    suffix = {"Levels": "", "Yearly change": " (change vs previous year)"}[mode]
    fig.update_xaxes(title_text=xname + suffix, showgrid=True, gridcolor=GRID)
    fig.update_yaxes(title_text=("TFR" + (f" {lag} yr later" if lag else "")) + suffix)
    return style(fig, h, legend=False, hover="closest")


def link(df, x_opts, key, h=300, ref_x=None, extra=None):
    """One series from this page against TFR: Levels or Yearly change, lag 0-3, with r, p and n."""
    c1, c2 = st.columns([1.15, 3])
    with c1:
        x = st.selectbox("Series", x_opts, format_func=short, key=f"{key}_x")
        mode, lag = controls(key)
        rtable(df, x, mode, lag)
        st.caption("p < 0.05 = significant")
    d = pair(df, x, mode, lag)
    with c2:
        if len(d) < 4:
            st.info("Not enough overlapping years for this series - widen the year range.")
            return
        show(scatter(d, short(x), mode, lag, h, ref_x=(ref_x or {}).get(x)))
        note((extra or {}).get(x, ""), formula(mode, lag), f"{len(d)} years", src(x, "TFR"))


# ======================================================================== pages
def page_overview():
    df = data()
    base = master().set_index("Year")
    header("What moves Singapore's birth rate?", "The total fertility rate, and one headline series per factor · "
                                                 "click a bar to open that factor")
    kpis([
        kpi(df, "TFR", "{:.2f}", pct=False, title="Total fertility rate"),
        kpi(df, "F_Marriage_Rate_25_29", "{:.0f}", title="Marriage rate, women 25-29"),
        kpi(df, "PM_Median_Real_2010", prefix="S$", title="Real median income / member (2010 S$)"),
        kpi(df, "HDB_Resale_Price_Index", "{:.0f}", title="HDB resale prices (2010 = 100)"),
        kpi(df, "Edu_CPI_Tuition", "{:.0f}", title="Tuition prices (2010 = 100)"),
        kpi(df, "Health_CPI", "{:.0f}", title="Healthcare prices (2010 = 100)"),
    ])

    c1, c2 = st.columns([1.2, 1])
    with c1, card("Total fertility rate"):
        d = df.dropna(subset=["TFR"])
        fig = go.Figure(go.Scatter(x=d.Year, y=d.TFR, mode="lines+markers+text", line=dict(color=BLUE, width=3),
                                   marker=dict(size=6), fill="tozeroy", fillcolor="rgba(42,120,214,.07)", name="TFR",
                                   text=[f"{v:.2f}" if y in (d.Year.min(), d.Year.max()) else "" for y, v in zip(d.Year, d.TFR)],
                                   textposition="top center", textfont=dict(color=BLUE, size=12),
                                   hovertemplate="%{x}: %{y:.2f}<extra></extra>"))
        fig.add_hline(y=2.1, line=dict(color="#9a9893", dash="dash"), annotation_text="replacement level 2.1",
                      annotation_position="top right")
        fig.update_yaxes(title_text="children per woman", range=[0, 2.35])
        fig.update_xaxes(range=[d.Year.min() - 0.7, d.Year.max() + 0.7])
        show(style(fig, 340, legend=False))
        note(trend(df, "TFR"), src("TFR"))

    with c2, card("How much did each one change?"):
        rows = []
        t = df.dropna(subset=["TFR"])
        rows.append(("", "👶 Total fertility rate", (t.TFR.iloc[-1] / t.TFR.iloc[0] - 1) * 100, int(t.Year.iloc[0]),
                     int(t.Year.iloc[-1]), BLUE))
        for name, f in FACTORS.items():
            d = df.dropna(subset=[f["rep"]])
            if len(d) > 1:
                rows.append((name, f'{f["icon"]} {f["name"]}', (d[f["rep"]].iloc[-1] / d[f["rep"]].iloc[0] - 1) * 100,
                             int(d.Year.iloc[0]), int(d.Year.iloc[-1]), f["colour"]))
        rows = rows[::-1]
        fig = go.Figure(go.Bar(
            y=[r[1] for r in rows], x=[r[2] for r in rows], orientation="h", marker_color=[r[5] for r in rows],
            text=[f"{r[2]:+.0f}%" for r in rows], textposition="outside", cliponaxis=False,
            customdata=[[r[0], r[3], r[4]] for r in rows],
            hovertemplate="%{y}<br>%{customdata[1]}→%{customdata[2]}: %{x:+.1f}%<extra></extra>"))
        fig.add_vline(x=0, line=dict(color="#bdbbb4", width=1))
        lo, hi = min(r[2] for r in rows), max(r[2] for r in rows)
        fig.update_xaxes(title_text="% change, first to last year", ticksuffix="%", range=[min(lo, 0) * 1.6 - 8, hi * 1.25 + 5])
        fig = style(fig, 340, legend=False, hover="closest")
        fig.update_layout(margin=dict(l=8, r=40, t=12, b=8), dragmode=False, clickmode="event+select")
        key = f"ov_{st.session_state.get('ov_n', 0)}"
        ev = st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode="points", key=key,
                             config={"displayModeBar": False})
        pts = ev.selection.points if ev and ev.selection else []
        if pts:
            cd = pts[0].get("customdata")
            target = cd[0] if isinstance(cd, list) else cd
            st.session_state["ov_n"] = st.session_state.get("ov_n", 0) + 1
            if target in PAGES:
                st.switch_page(PAGES[target])
            st.rerun()
        note("% change = last ÷ first − 1", src(*[f["rep"] for f in FACTORS.values()], "TFR"))

    with card("Headline series on one scale (2010 = 100), TFR on the right"):
        fig = go.Figure()
        for name, f in FACTORS.items():
            d = df.dropna(subset=[f["rep"]])
            y = d[f["rep"]] / base.loc[BASE_YEAR, f["rep"]] * 100
            fig.add_trace(go.Scatter(x=d.Year, y=y, name=f'{f["icon"]} {f["name"]}', mode="lines",
                                     line=dict(color=f["colour"], width=2.3), customdata=d[f["rep"]],
                                     hovertemplate=f["name"] + " %{y:.0f} (%{customdata:,.1f})<extra></extra>"))
        fig.add_hline(y=100, line=dict(color="#bdbbb4", width=1))
        fig.update_yaxes(title_text="2010 = 100")
        show(add_tfr_axis(style(fig, 370), df))
        note("Index = value ÷ 2010 value × 100", src(*[f["rep"] for f in FACTORS.values()], "TFR"))


def page_marriage():
    df = data()
    RATES = {f"F_Marriage_Rate_{a}_{a + 4}": f"{a}-{a + 4}" for a in range(20, 45, 5)}
    s = df.set_index("Year")
    peak = s[list(RATES)].dropna(how="all").idxmax(axis=1).map(RATES)          # age group with the highest rate
    PEAK_COL = {"25-29": BLUE, "30-34": ORANGE}

    header("Marriage", "Who marries at what age, and how that lines up with the birth rate")
    kpis([
        kpi(df, "F_Marriage_Rate_25_29", "{:.0f}", title="Marriage rate, women 25-29"),
        kpi(df, "F_Marriage_Rate_30_34", "{:.0f}", title="Marriage rate, women 30-34"),
        kpi(df, "Later_Marriage_Shift_Index", "{:.2f}", pct=False, title="Later-marriage shift index"),
        dict(label="Peak marriage age (women)", value=peak.iloc[-1], sub=f"{peak.index[-1]} · was {peak.iloc[0]} in {peak.index[0]}"),
        kpi(df, "Brides_30_34", title="Brides aged 30-34"),
        kpi(df, "Median_Age_First_Birth", "{:.1f}", suffix=" yrs", pct=False, title="Mother's age, 1st birth", dfmt="{:.1f}"),
    ])

    c1, c2 = st.columns(2)
    with c1, card("Women's marriage rate by age"):
        z = s[list(RATES)].dropna(how="all").T
        fig = go.Figure(go.Heatmap(z=z.values, x=z.columns, y=list(RATES.values()), colorscale="Purples", xgap=1, ygap=2,
                                   text=np.round(z.values, 0), texttemplate="%{text:.0f}", textfont=dict(size=9),
                                   colorbar=dict(title="per 1,000", thickness=10),
                                   hovertemplate="Age %{y}, %{x}: %{z:.1f}<extra></extra>"))
        fig.update_yaxes(autorange="reversed", title_text="age group")
        show(style(fig, 330, legend=False, hover="closest"))
        note("Marriages per 1,000 unmarried women", src("F_Marriage_Rate_25_29"))

    with c2, card("Peak marriage age and first-birth timing"):
        fig = go.Figure()
        for yr, g in peak.items():
            fig.add_vrect(x0=yr - 0.5, x1=yr + 0.5, fillcolor=PEAK_COL.get(g, "#ffffff"), opacity=0.13, layer="below",
                          line_width=0)
        d = df.dropna(subset=["Median_Age_First_Birth"]).set_index("Year")
        cnt = [d.loc[y, "Brides_25_29"] if peak.get(y) == "25-29" else d.loc[y, "Brides_30_34"] for y in d.index]
        fig.add_trace(go.Scatter(x=d.index, y=d.Median_Age_First_Birth, name="Median age at first birth",
                                 mode="lines+markers+text", text=[f"{v:.1f}" for v in d.Median_Age_First_Birth],
                                 textposition="top center", textfont=dict(size=9, color="#52514e"),
                                 line=dict(color=VIOLET, width=2.6), marker=dict(size=6),
                                 customdata=np.c_[peak.reindex(d.index).values, cnt],
                                 hovertemplate="%{x}: first birth at %{y:.1f}<br>peak marriage age %{customdata[0]} "
                                               "(%{customdata[1]:,.0f} brides)<extra></extra>"))
        for g, c in PEAK_COL.items():
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=f"Peak marriage age {g}",
                                     marker=dict(size=12, symbol="square", color=c, opacity=0.35)))
        fig.update_yaxes(title_text="mother's age at first birth (years)")
        show(add_tfr_axis(style(fig, 350, hover="closest"), df))
        note("Age at first birth " + trend(df, "Median_Age_First_Birth", "{:.1f}"),
             src("Median_Age_First_Birth", "F_Marriage_Rate_25_29", "TFR"))

    with card("Brides aged 25-29 vs 30-34"):
        d = df.dropna(subset=["Brides_25_29"])
        fig = go.Figure()
        for c, n, col in [("Brides_25_29", "25-29", BLUE), ("Brides_30_34", "30-34", ORANGE)]:
            fig.add_trace(go.Bar(x=d.Year, y=d[c], name=n, marker_color=col,
                                 hovertemplate="%{x}, brides " + n + ": %{y:,.0f}<extra></extra>"))
        fig.update_layout(barmode="group", bargap=0.2)
        fig.update_yaxes(title_text="marriages (brides)")
        show(style(fig, 300, hover="closest"))
        note("Brides 30-34 " + trend(df, "Brides_30_34", "{:,.0f}"), src("Brides_25_29"))

    with card("Marriage vs TFR"):
        link(df, ["Later_Marriage_Shift_Index"] + list(RATES) + ["Median_Age_First_Birth", "Brides_25_29", "Brides_30_34"],
             key="mar", ref_x={"Later_Marriage_Shift_Index": 1},
             extra={"Later_Marriage_Shift_Index": "Shift index = marriage rate 30–34 ÷ marriage rate 25–29 (> 1: more marriages at 30–34)"})


def page_income():
    df = data()
    header("Income", "Household income per member, before and after inflation, and how evenly it is shared")
    kpis([
        kpi(df, "PM_Median", prefix="S$", title="Median income / member"),
        kpi(df, "PM_Median_Real_2010", prefix="S$", title="Median, real (2010 S$)"),
        kpi(df, "PM_Avg", prefix="S$", title="Average income / member"),
        kpi(df, "PM_Avg_Real_2010", prefix="S$", title="Average, real (2010 S$)"),
        kpi(df, "Gini_Before_Transfers", "{:.3f}", pct=False, title="Gini before transfers", dfmt="{:.3f}"),
        kpi(df, "Gini_After_Transfers", "{:.3f}", pct=False, title="Gini after transfers", dfmt="{:.3f}"),
    ])

    c1, c2 = st.columns(2)
    with c1, card("Income per household member: nominal and real"):
        which = st.segmented_control("Show", ["Median", "Average", "Both"], default="Median", key="inc_which",
                                     label_visibility="collapsed") or "Median"
        lines = {"Median": [("PM_Median", "Median, nominal", ORANGE, "solid"),
                            ("PM_Median_Real_2010", "Median, real (2010 S$)", AQUA, "solid")],
                 "Average": [("PM_Avg", "Average, nominal", ORANGE, "dash"),
                             ("PM_Avg_Real_2010", "Average, real (2010 S$)", AQUA, "dash")]}
        use = lines["Median"] + lines["Average"] if which == "Both" else lines[which]
        fig = go.Figure()
        for c, n, col, dash in use:
            d = df.dropna(subset=[c])
            fig.add_trace(go.Scatter(x=d.Year, y=d[c], name=n, mode="lines+markers", line=dict(color=col, width=2.6, dash=dash),
                                     marker=dict(size=5), hovertemplate=n + " S$%{y:,.0f}<extra></extra>"))
            fig.add_annotation(x=d.Year.iloc[-1], y=d[c].iloc[-1], text=f"<b>S${d[c].iloc[-1]:,.0f}</b>", showarrow=False,
                               xanchor="left", xshift=6, yshift=-7 if "real" in n else 7,
                               font=dict(color=col, size=11))
        fig.update_yaxes(title_text="S$ per member per month", tickprefix="$", tickformat=",.0f")
        fig.update_xaxes(range=[df.Year.min() - 0.5, df.Year.max() + 1.6])
        show(add_tfr_axis(style(fig, 330), df))
        main = "PM_Median" if which != "Average" else "PM_Avg"
        nom, real = df.dropna(subset=[main]), df.dropna(subset=[main + "_Real_2010"])
        lab = "median" if main == "PM_Median" else "average"
        note("Real = nominal ÷ CPI (2010 = 100) × 100",
             f"{int(nom.Year.iloc[0])}→{int(nom.Year.iloc[-1])} {lab}: S${nom[main].iloc[0]:,.0f} → S${nom[main].iloc[-1]:,.0f} "
             f"nominal ({(nom[main].iloc[-1] / nom[main].iloc[0] - 1) * 100:+.0f}%), "
             f"S${real[main + '_Real_2010'].iloc[-1]:,.0f} real "
             f"({(real[main + '_Real_2010'].iloc[-1] / real[main + '_Real_2010'].iloc[0] - 1) * 100:+.0f}%)",
             src("PM_Median_Real_2010", "TFR"))

    with c2, card("Real income growth by income group"):
        cols = ["Real_Growth_%"] + [f"Real_Growth_D{i}" for i in range(1, 10)]
        g = df.set_index("Year")[cols].dropna(how="all")
        z = g.T
        names = ["<b>All households</b>", "D1 lowest"] + [f"D{i}" for i in range(2, 10)]
        fig = go.Figure(go.Heatmap(z=z.values, x=z.columns, y=names,
                                   colorscale=[[0, RED], [7 / 18, "#efeeea"], [1, BLUE]], zmin=-7, zmax=11,
                                   xgap=1, ygap=1, text=np.round(z.values, 1), texttemplate="%{text:.1f}",
                                   textfont=dict(size=8.5), colorbar=dict(title="%", thickness=10),
                                   hovertemplate="%{y}, %{x}: %{z:+.1f}%<extra></extra>"))
        fig.add_hline(y=0.5, line=dict(color="white", width=4))
        fig.update_yaxes(autorange="reversed")
        show(style(fig, 330, legend=False, hover="closest"))
        neg = [int(c) for c in z.columns if (z[c] < 0).all()]
        note("Real % change in average income per member vs previous year; D1–D9 = income deciles",
             ("All groups fell in " + ", ".join(map(str, neg))) if neg else "", src("Real_Growth_%"))

    with card("Gini coefficient before and after transfers & taxes"):
        d = df.dropna(subset=["Gini_Before_Transfers", "Gini_After_Transfers", "TFR"])
        yr = d.Year.values
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.3, 0.7], vertical_spacing=0.06)
        fig.add_trace(go.Scatter(x=yr, y=d.TFR, name="TFR", mode="lines+markers", line=dict(color=BLUE, width=2.4),
                                 marker=dict(size=5, symbol="diamond"), hovertemplate="TFR %{y:.2f}<extra></extra>"),
                      row=1, col=1)
        for c, n, col in [("Gini_Before_Transfers", "Before transfers & taxes", VIOLET),
                          ("Gini_After_Transfers", "After transfers & taxes", GOLD)]:
            fig.add_trace(go.Scatter(x=yr, y=d[c], name=n, mode="lines+markers", line=dict(color=col, width=3),
                                     marker=dict(size=6, line=dict(color="white", width=1.2)),
                                     hovertemplate=n.split(" ")[0] + " %{y:.3f}<extra></extra>"), row=2, col=1)
            for i, anc in [(0, "right"), (len(yr) - 1, "left")]:
                fig.add_annotation(x=yr[i], y=d[c].iloc[i], text=f"<b>{d[c].iloc[i]:.3f}</b>", showarrow=False,
                                   xanchor=anc, xshift=-7 if anc == "right" else 7, font=dict(color=col, size=11),
                                   row=2, col=1)
        for i, anc in [(0, "right"), (len(yr) - 1, "left")]:
            fig.add_annotation(x=yr[i], y=d.TFR.iloc[i], text=f"<b>{d.TFR.iloc[i]:.2f}</b>", showarrow=False,
                               xanchor=anc, xshift=-7 if anc == "right" else 7, font=dict(color=BLUE, size=11), row=1, col=1)
        fig = style(fig, 400)
        fig.update_layout(margin=dict(l=8, r=30, t=12, b=8))
        fig.update_xaxes(range=[yr.min() - 1.6, yr.max() + 1.6])
        fig.update_yaxes(title_text="TFR", tickformat=".2f", row=1, col=1)
        fig.update_yaxes(title_text="Gini (0 = equal, 1 = unequal)", tickformat=".2f", row=2, col=1)
        fig.update_layout(hovermode="x unified")
        show(fig)
        b, a = d.Gini_Before_Transfers, d.Gini_After_Transfers
        note(f"{int(yr[0])}→{int(yr[-1])}: before {b.iloc[0]:.3f} → {b.iloc[-1]:.3f}, after {a.iloc[0]:.3f} → {a.iloc[-1]:.3f}",
             src("Gini_After_Transfers", "TFR"))

    with card("Income vs TFR"):
        link(df, ["PM_Median", "PM_Median_Real_2010", "PM_Avg", "PM_Avg_Real_2010", "Gini_Before_Transfers",
                  "Gini_After_Transfers"], key="inc")


def page_housing():
    df = data()
    census = CENSUS.copy()
    census["Share"] = census.With_Children / (census.With_Children + census.Without_Children) * 100
    hdb_share = census.set_index("Housing_Type").loc["HDB flats", "Share"]
    yoy = df.dropna(subset=["HDB_YoY_%"])

    header("Housing", "Public and private home prices, and where families with children live")
    kpis([
        kpi(df, "HDB_Resale_Price_Index", "{:.0f}", title="HDB resale prices"),
        kpi(df, "Private_Non_Landed_Index", "{:.0f}", title="Private non-landed prices"),
        kpi(df, "Private_Landed_Index", "{:.0f}", title="Private landed prices"),
        dict(label="HDB price change, latest year", value=f"{yoy['HDB_YoY_%'].iloc[-1]:+.1f}%",
             sub=f"{int(yoy.Year.iloc[-1])} vs previous year"),
        dict(label="HDB couples with children", value=f"{hdb_share:.0f}%", sub="of married couples in HDB flats, 2020"),
    ])

    def trend_pct(c):
        d = df.dropna(subset=[c])
        return f"{(d[c].iloc[-1] / d[c].iloc[0] - 1) * 100:+.0f}%"

    H = {"HDB_Resale_Price_Index": ("HDB resale", ORANGE), "Private_Non_Landed_Index": ("Private non-landed", AQUA),
         "Private_Landed_Index": ("Private landed", VIOLET)}
    with card("HDB vs private prices"):
        show(index_lines(df, list(H), {c: col for c, (n, col) in H.items()}, {c: n for c, (n, col) in H.items()}, h=330))
        note(f"{int(df.Year.min())}→{int(df.Year.max())}: " + ", ".join(f"{n} {trend_pct(c)}" for c, (n, _) in H.items()),
             src(*H, "TFR"))

    c2, c3 = st.columns(2)
    with c2, card("Yearly price change"):
        fig = go.Figure()
        for c, n, col in [("HDB_YoY_%", "HDB resale", ORANGE), ("NonLanded_YoY_%", "Private non-landed", AQUA),
                          ("Landed_YoY_%", "Private landed", VIOLET)]:
            fig.add_trace(go.Bar(x=yoy.Year, y=yoy[c], name=n, marker_color=col,
                                 hovertemplate="%{x} " + n + ": %{y:+.1f}%<extra></extra>"))
        fig.add_hline(y=0, line=dict(color=MUTED, width=1))
        fig.update_layout(barmode="group", bargap=0.18)
        fig.update_yaxes(title_text="% vs previous year", ticksuffix="%")
        show(style(fig, 330, hover="closest"))
        note("% change = index[t] ÷ index[t−1] − 1", src("HDB_YoY_%", "NonLanded_YoY_%"))

    with c3, card("Married couples with and without children, 2020"):
        pct = st.toggle("Show as %", value=True, key="cen_pct")
        fig = go.Figure()
        tot = census.With_Children + census.Without_Children
        for c, n, col in [("With_Children", "With children", ORANGE), ("Without_Children", "Without children", "#bdbbb4")]:
            v = census[c] / tot * 100 if pct else census[c]
            fig.add_trace(go.Bar(y=census.Housing_Type, x=v, name=n, orientation="h", marker_color=col,
                                 text=[f"{t:.0f}%" if pct else f"{t:,.0f}" for t in v], textposition="inside",
                                 hovertemplate="%{y} · " + n + ": %{x:,.1f}" + ("%" if pct else "") + "<extra></extra>"))
        fig.update_layout(barmode="stack", bargap=0.35)
        fig.update_xaxes(title_text="% of married-couple households" if pct else "married-couple households")
        show(style(fig, 300, hover="closest"))
        note("Share = with children ÷ all married couples", "Source: SingStat Census 2020, CT/17486")

    with card("Housing vs TFR"):
        link(df, list(H) + ["HDB_YoY_%"], key="hou")


def page_education():
    df = data()
    e = df.dropna(subset=["Edu_CPI"])
    header("Education", "What families pay for education, and what the government spends")
    kpis([
        kpi(df, "Edu_CPI_Tuition", "{:.0f}", title="Tuition prices"),
        kpi(df, "Edu_CPI_General", "{:.0f}", title="General education prices"),
        kpi(df, "Edu_Primary_per_Student", prefix="S$", title="Gov spend / primary pupil"),
        kpi(df, "Edu_Secondary_per_Student", prefix="S$", title="Gov spend / secondary pupil"),
        kpi(df, "Edu_Public_Total", prefix="S$", suffix="m", title="Gov education spending"),
    ])

    CPI = {"Edu_CPI_Tuition": ORANGE, "Edu_CPI_General": AQUA, "Edu_CPI_Textbooks": VIOLET}
    c1, c2 = st.columns(2)
    with c1, card("Education prices"):
        show(index_lines(df, list(CPI), CPI, {c: short(c).replace(" prices", "") for c in CPI}, h=320))
        note("Tuition " + trend(df, "Edu_CPI_Tuition", "{:.0f}"), src("Edu_CPI", "TFR"))

    with c2, card("When prices rose or fell"):
        z = e.set_index("Year")[list(CPI)].pct_change().mul(100).dropna(how="all").T
        fig = go.Figure(go.Heatmap(z=z.values, x=z.columns, y=[short(c).replace(" prices", "") for c in CPI],
                                   colorscale=[[0, BLUE], [0.5, "#efeeea"], [1, RED]], zmin=-7, zmax=7, xgap=2, ygap=4,
                                   text=np.round(z.values, 1), texttemplate="%{text:.1f}", textfont=dict(size=9),
                                   colorbar=dict(title="%", thickness=10), hovertemplate="%{y}, %{x}: %{z:+.1f}%<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        show(style(fig, 320, legend=False, hover="closest"))
        note("% change = index[t] ÷ index[t−1] − 1", src("Edu_CPI"))

    c3, c4 = st.columns(2)
    with c3, card("Government spending per student"):
        d = df.dropna(subset=["Edu_Primary_per_Student"])
        fig = go.Figure()
        for c, n, col in [("Edu_Primary_per_Student", "Primary", AQUA), ("Edu_Secondary_per_Student", "Secondary", ORANGE)]:
            fig.add_trace(go.Bar(x=d.Year, y=d[c], name=n, marker_color=col,
                                 hovertemplate="FY%{x} " + n + ": S$%{y:,.0f}<extra></extra>"))
        fig.update_layout(barmode="group", bargap=0.2)
        fig.update_yaxes(title_text="S$ per student (FY)", tickprefix="$")
        show(style(fig, 300, hover="closest"))
        note("Primary " + trend(df, "Edu_Primary_per_Student", "${:,.0f}"), src("Edu_Primary_per_Student"))

    with c4, card("Where education funding goes"):
        OPS = {"Edu_Op_Primary": "Primary", "Edu_Op_Secondary_JC": "Secondary & JC", "Edu_Op_ITE": "ITE",
               "Edu_Op_Tertiary": "Tertiary", "Edu_Op_Other": "Other", "Edu_Development": "Development"}
        yrs = df.dropna(subset=list(OPS)).Year.astype(int).tolist()
        yr = st.select_slider("Year", yrs, value=yrs[-1], key="edu_tree", label_visibility="collapsed")
        v = df.set_index("Year").loc[yr, list(OPS)] / 1000
        fig = go.Figure(go.Treemap(labels=list(OPS.values()), parents=[""] * len(OPS), values=v.values,
                                   marker=dict(colors=SERIES[:6], line=dict(color="white", width=2)),
                                   texttemplate="<b>%{label}</b><br>S$%{value:.1f}bn<br>%{percentRoot:.0%}",
                                   hovertemplate="%{label}: S$%{value:.2f}bn<extra></extra>"))
        show(style(fig, 280, legend=False, hover="closest").update_layout(margin=dict(l=0, r=0, t=0, b=0)))
        note(f"FY{yr} total S${v.sum():.1f}bn", src("Edu_Public_Total"))

    with card("Education vs TFR"):
        link(df, list(CPI) + ["Edu_Primary_per_Student", "Edu_Secondary_per_Student", "Edu_Public_Total"], key="edu")


def page_healthcare():
    df = data()
    h = df.dropna(subset=["Health_CPI"])
    hes = df.dropna(subset=["HES_Health_Monthly"])
    header("Healthcare", "What healthcare costs households, and what the government spends")
    items = [
        kpi(df, "Health_CPI", "{:.0f}", title="Healthcare prices"),
        kpi(df, "Health_CPI_Insurance", "{:.0f}", title="Health insurance prices"),
        kpi(df, "Gov_Health_per_Capita", prefix="S$", title="Gov health spend / person"),
    ]
    if len(hes):
        items.append(kpi(df, "HES_Health_Monthly", prefix="S$", title="Household health spend / month"))
    kpis(items)

    CPI = {"Health_CPI": INK, "Health_CPI_Insurance": ORANGE, "Health_CPI_Inpatient": VIOLET, "Health_CPI_Outpatient": AQUA,
           "Health_CPI_Other": PINK, "Health_CPI_Medicines": YELLOW}
    NAME = {c: ("All healthcare" if c == "Health_CPI" else short(c).replace(" prices", "")) for c in CPI}
    c1, c2 = st.columns(2)
    with c1, card("Healthcare prices by type"):
        show(index_lines(df, list(CPI), CPI, NAME, h=330))
        note("All healthcare " + trend(df, "Health_CPI", "{:.0f}"), src("Health_CPI", "TFR"))

    with c2, card("How much prices change in a year"):
        fig = go.Figure()
        for c, col in CPI.items():
            ch = h.set_index("Year")[c].pct_change().mul(100).dropna()
            fig.add_trace(go.Box(x=ch.values, name=NAME[c], marker_color=col, boxpoints="all", jitter=0.4, pointpos=0,
                                 customdata=ch.index, line=dict(width=1.5),
                                 hovertemplate="%{customdata}: %{x:+.1f}%<extra></extra>", orientation="h"))
        fig.add_vline(x=0, line=dict(color="#bdbbb4", width=1))
        fig.update_xaxes(title_text="% change vs previous year", ticksuffix="%", showgrid=True, gridcolor=GRID)
        fig.update_yaxes(autorange="reversed")
        show(style(fig, 330, legend=False, hover="closest"))
        note("% change = index[t] ÷ index[t−1] − 1; one dot per year", src("Health_CPI"))

    c3, c4 = st.columns(2)
    with c3, card("Household health spending (survey years)"):
        fig = go.Figure(go.Bar(x=hes.Year.astype(str), y=hes.HES_Health_Monthly, marker_color=AQUA,
                               text=[f"S${v:,.0f}" for v in hes.HES_Health_Monthly], textposition="outside",
                               hovertemplate="%{x}: S$%{y:,.0f} per month<extra></extra>", width=0.5))
        fig.update_yaxes(title_text="S$ per household per month", tickprefix="$", range=[0, hes.HES_Health_Monthly.max() * 1.25])
        show(style(fig, 300, legend=False, hover="closest"))
        note(trend(df, "HES_Health_Monthly", "${:,.0f}"), "Household Expenditure Survey", src("HES_Health_Monthly"))

    with c4, card("Government health spending per person"):
        g = df.dropna(subset=["Gov_Health_per_Capita"])
        fig = go.Figure(go.Bar(x=g.Year, y=g.Gov_Health_per_Capita,
                               marker_color=[ORANGE if 2020 <= y <= 2022 else BLUE for y in g.Year],
                               hovertemplate="FY%{x}: S$%{y:,.0f}<extra></extra>"))
        fig.update_yaxes(title_text="S$ per person (FY)", tickprefix="$")
        show(style(fig, 300, legend=False, hover="closest"))
        note(trend(df, "Gov_Health_per_Capita", "${:,.0f}"), "orange = FY2020–22 (COVID-19)", src("Gov_Health_per_Capita"))

    with card("Healthcare vs TFR"):
        link(df, list(CPI) + ["Gov_Health_per_Capita"], key="hea")


def page_correlations():
    df = data()
    header("Correlations", "The five factors side by side, against TFR and against each other")
    avail = [c for c in META if META[c][1] != "Births" and c not in NO_CORR and c in df and df[c].notna().sum() >= 6
             and not c.endswith("_YoY_%")]
    default = [f["rep"] for f in FACTORS.values()] + ["Later_Marriage_Shift_Index", "Median_Age_First_Birth",
                                                      "Gini_After_Transfers", "Private_Non_Landed_Index"]
    with st.container(border=True):
        pick = st.multiselect("Indicators", avail, default=[c for c in default if c in avail],
                              format_func=lambda c: f"{META[c][1]} · {short(c)}")
        c1, c2, _ = st.columns([1, 1.3, 2])
        method = c1.segmented_control("Correlation", ["Pearson", "Spearman"], default="Pearson") or "Pearson"
        mode = c2.segmented_control("Compare", MODES, default="Levels", help=MODE_HELP) or "Levels"
    if len(pick) < 2:
        st.info("Pick at least two indicators."); st.stop()

    cols = ["TFR"] + pick
    m = df.set_index("Year")[cols]
    mm = pd.DataFrame(index=m.index)
    for c in m:                                         # each series on its own years (series end in different years)
        s = m[c].dropna()
        if mode == "Yearly change":
            s = s.reindex(range(int(s.index.min()), int(s.index.max()) + 1)).diff()
        mm[c] = s
    cm, pm, nm = corr_matrix(mm, method.lower())
    names = [short(c) for c in cols]
    H = max(380, 34 * len(cols) + 120)

    c3, c4 = st.columns([1.35, 1])
    with c3, card("Everything on one scale"):
        z = (m - m.mean()) / m.std()
        fig = go.Figure()
        for i, c in enumerate(pick):
            fig.add_trace(go.Scatter(x=z.index, y=z[c], name=short(c), connectgaps=True,
                                     line=dict(color=SERIES[i % 6], width=1.7), opacity=0.9))
        fig.add_trace(go.Scatter(x=z.index, y=z["TFR"], name="TFR", line=dict(color=BLUE, width=4)))
        fig.add_hline(y=0, line=dict(color="#bdbbb4", width=1))
        fig.update_yaxes(title_text="z-score")
        show(style(fig, 380))
        note("z = (value − own mean) ÷ own standard deviation", src(*cols))

    with c4, card("How much did each indicator change?"):
        yrs = df.Year.astype(int).tolist()
        a, b = st.select_slider("Between", yrs, value=(2019 if 2019 in yrs else yrs[0], yrs[-1]), key="per",
                                label_visibility="collapsed")
        rows = []
        for c in cols:
            s = m[c].loc[a:b].dropna()
            if len(s) >= 2 and s.iloc[0]:
                rows.append((c, short(c), (s.iloc[-1] / s.iloc[0] - 1) * 100, f"{s.index[0]}→{s.index[-1]}"))
        t = pd.DataFrame(rows, columns=["col", "name", "chg", "span"]).sort_values("chg")
        fig = go.Figure(go.Bar(x=t.chg, y=t.name, orientation="h", customdata=t.span,
                               marker_color=[BLUE if c == "TFR" else (ORANGE if v >= 0 else VIOLET) for c, v in zip(t.col, t.chg)],
                               text=[f"{v:+.0f}%" for v in t.chg], textposition="outside", cliponaxis=False,
                               hovertemplate="%{y}: %{x:+.1f}% (%{customdata})<extra></extra>"))
        fig.add_vline(x=0, line=dict(color="#bdbbb4", width=1))
        lo, hi = min(t.chg.min(), 0), max(t.chg.max(), 0)
        fig.update_xaxes(title_text=f"% change, {a} → {b}", ticksuffix="%", range=[lo * 1.5 - 8, hi * 1.25 + 5])
        fig = style(fig, 338, legend=False, hover="closest")
        fig.update_layout(margin=dict(l=4, r=40, t=8, b=4))
        show(fig)
        note("% change = last ÷ first − 1")

    c1, c2 = st.columns([1.35, 1])
    with c1, card("Correlation matrix"):
        k = len(cols)
        txt = [[f"{cm.iloc[i, j]:.2f}{stars(pm.iloc[i, j]) if i != j else ''}" if not np.isnan(cm.iloc[i, j]) else ""
                for j in range(k)] for i in range(k)]
        hov = [[f"p = {p_text(pm.iloc[i, j])}, {int(nm.iloc[i, j])} years" for j in range(k)] for i in range(k)]
        fig = go.Figure(go.Heatmap(z=cm.values, x=names, y=names, zmin=-1, zmax=1, colorscale=DIVERGING, xgap=2, ygap=2,
                                   text=txt, texttemplate="%{text}", textfont=dict(size=11), customdata=hov,
                                   colorbar=dict(thickness=10, len=0.8),
                                   hovertemplate="%{y} × %{x}: r = %{z:.2f}<br>%{customdata}<extra></extra>"))
        fig.update_xaxes(tickangle=-35, tickfont=dict(size=11), side="bottom")
        fig.update_yaxes(autorange="reversed", tickfont=dict(size=11))
        fig = style(fig, H, legend=False, hover="closest")
        fig.update_layout(margin=dict(l=4, r=4, t=8, b=4))
        show(fig)
        note(f"{method} r, {mode.lower()} · * p < 0.05, ** p < 0.01")

    with c2, card("Link with TFR, strongest first"):
        r = cm.loc[pick, "TFR"].dropna().sort_values(key=lambda s: -s.abs())[::-1]
        p = pm.loc[r.index, "TFR"]
        fig = go.Figure(go.Bar(y=[short(c) for c in r.index], x=r.values, orientation="h",
                               marker=dict(color=r.values, colorscale=DIVERGING, cmin=-1, cmax=1,
                                           opacity=[1 if q < 0.05 else 0.4 for q in p.values]),
                               text=[f"{v:+.2f}{stars(q)}" for v, q in zip(r.values, p.values)], textposition="outside",
                               cliponaxis=False,
                               customdata=[[META[c][1], p_text(p[c]), int(nm.loc[c, "TFR"])] for c in r.index],
                               hovertemplate="%{y} (%{customdata[0]}): r = %{x:+.2f}<br>p = %{customdata[1]}, "
                                             "%{customdata[2]} years<extra></extra>"))
        fig.add_vline(x=0, line=dict(color=MUTED, width=1))
        fig.update_xaxes(range=[-1.55, 1.55], tickvals=[-1, -0.5, 0, 0.5, 1], title_text=f"{method} r with TFR ({mode.lower()})")
        fig = style(fig, H, legend=False, hover="closest")
        fig.update_layout(margin=dict(l=4, r=30, t=8, b=4))
        show(fig)
        note(f"{int((p < 0.05).sum())} of {len(p)} significant at p < 0.05 · faded = p ≥ 0.05")

    with st.expander(f"Each indicator vs TFR ({mode.lower()})"):
        ncol = 4
        nrow = int(np.ceil(len(pick) / ncol))
        titles, sets = [], []
        for c in pick:
            d = mm[[c, "TFR"]].dropna()
            rr, pp, nn = corr_test(d[c].values, d.TFR.values, method.lower()) if len(d) >= 5 else (np.nan, np.nan, len(d))
            sets.append((c, d)); titles.append(f"{short(c)}<br><sup>r = {rr:+.2f}, p = {p_text(pp)} · {nn} yrs</sup>")
        fig = make_subplots(rows=nrow, cols=ncol, subplot_titles=titles, vertical_spacing=0.32 / nrow, horizontal_spacing=0.06)
        for i, (c, d) in enumerate(sets):
            if len(d) < 3:
                continue
            yrs_ = d.index.values
            fig.add_trace(go.Scatter(x=d[c], y=d.TFR, mode="markers", text=yrs_,
                                     marker=dict(size=7, color=yrs_, colorscale="Blues", cmin=yrs_.min() - 8, cmax=yrs_.max(),
                                                 line=dict(color="white", width=1)),
                                     hovertemplate="%{text}: %{x:,.2f} / TFR %{y:,.3f}<extra></extra>"),
                          row=i // ncol + 1, col=i % ncol + 1)
            b_, a_ = np.polyfit(d[c], d.TFR, 1); xs = np.linspace(d[c].min(), d[c].max(), 20)
            fig.add_trace(go.Scatter(x=xs, y=a_ + b_ * xs, mode="lines", line=dict(color=MUTED, dash="dash", width=1.3),
                                     hoverinfo="skip"), row=i // ncol + 1, col=i % ncol + 1)
        fig.update_annotations(font_size=11)
        fig = style(fig, 230 * nrow + 40, legend=False, hover="closest")
        fig.update_layout(margin=dict(t=48))
        show(fig)


def page_data():
    df = data()
    header("Data & sources", "Every series, where it comes from, and how it was checked")
    dic = pd.DataFrame([dict(column=c, label=v[0], factor=v[1], unit=v[2], source_table=v[3],
                             first_year=int(master().dropna(subset=[c]).Year.min()),
                             last_year=int(master().dropna(subset=[c]).Year.max())) for c, v in META.items()])
    t1, t2, t3 = st.tabs(["Data", "Sources", "Checks"])
    with t1:
        groups = list(dict.fromkeys(dic.factor))
        c1, c2 = st.columns([1, 3])
        g = c1.multiselect("Factors", groups, default=groups)
        cols = dic[dic.factor.isin(g)].column.tolist()
        pick = c2.multiselect("Columns", cols, default=cols[:8], format_func=label)
        view = df[["Year"] + pick]
        st.dataframe(view, hide_index=True, width="stretch", height=420)
        d1, d2 = st.columns(2)
        d1.download_button("Selected columns (CSV)", view.to_csv(index=False), "selection.csv", "text/csv")
        d2.download_button("Full combined table (CSV)", DATA.read_bytes(), DATA.name, "text/csv")
    with t2:
        st.dataframe(dic[["factor", "label", "unit", "source_table", "first_year", "last_year", "column"]],
                     hide_index=True, width="stretch", height=520)
        st.caption("Search a table ID at tablebuilder.singstat.gov.sg")
    with t3:
        M = master().set_index("Year")
        real_chk = ((M.PM_Avg_Real_2010.pct_change() * 100).round(1) - M["Real_Growth_%"]).loc[2011:2024].abs().max()
        sh = (M.F_Marriage_Rate_30_34 / M.F_Marriage_Rate_25_29 - M.Later_Marriage_Shift_Index).abs().max()
        checks = pd.DataFrame([
            dict(check="Real income = 17872 ÷ CPI (2010 = 100) × 100 reproduces 17885's published real % change, 2011-2024",
                 result=f"max difference {real_chk:.1f} pp", ok=bool(real_chk == 0)),
            dict(check="CPI All Items equals 100 in 2010", result=f"{M.loc[2010, 'CPI_All_Items_2010_100']:.3f}",
                 ok=bool(abs(M.loc[2010, "CPI_All_Items_2010_100"] - 100) < 1e-9)),
            dict(check="Later-marriage shift index = rate 30-34 ÷ rate 25-29", result=f"max difference {sh:.1e}",
                 ok=bool(sh < 1e-12)),
            dict(check="Education: operating + development spending = total", ok=bool(
                ((M[[c for c in META if c.startswith("Edu_Op_")] + ["Edu_Development"]].sum(axis=1) - M.Edu_Public_Total)
                 .loc[2010:2025].abs().max() < 0.01)), result="all years 2010-2025"),
        ])
        st.dataframe(checks, hide_index=True, width="stretch")
        st.markdown("**Series with gaps inside 2010-2025** (the source table stops earlier or is a survey)")
        MM = M.loc[2010:2025]
        gaps = [dict(factor=META[c][1], series=META[c][0], years=int(MM[c].notna().sum()),
                     missing=", ".join(str(y) for y in MM.index[MM[c].isna()]))
                for c in META if c in MM and 0 < MM[c].notna().sum() < len(MM)]
        st.dataframe(pd.DataFrame(gaps), hide_index=True, width="stretch")


# ======================================================================== style, bar hover and bar click focus
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
.pagehead, .kpi, .card-t, .chip, .finding, .rtable { font-family: 'Inter', system-ui, sans-serif; }
[data-testid="stAppViewContainer"] { background: #f6f6f4; }
[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #ececea; }
.block-container { padding-top: 1.6rem; max-width: 1400px; }
.pagehead { display:flex; justify-content:space-between; align-items:flex-end; gap:16px; flex-wrap:wrap; margin-bottom:14px; }
.pagehead h1 { font-size:1.75rem; font-weight:700; margin:0; padding:0; color:#1f1f1d; }
.pagehead p { margin:2px 0 0; color:#6b6a66; font-size:.95rem; }
.chips { display:flex; gap:6px; flex-wrap:wrap; }
.chip { background:#fff; border:1px solid #e3e2de; border-radius:999px; padding:3px 10px; font-size:.78rem; color:#52514e; }
.kpis { display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:12px; margin:4px 0 18px; }
.kpi { background:#fff; border:1px solid #e6e5e1; border-radius:12px; padding:12px 14px; box-shadow:0 1px 2px rgba(0,0,0,.03); }
.kpi-l { font-size:.76rem; color:#6b6a66; text-transform:uppercase; letter-spacing:.03em; }
.kpi-v { font-size:1.55rem; font-weight:650; color:#1f1f1d; margin:2px 0 4px; }
.kpi-s { font-size:.75rem; color:#8a8984; margin-top:4px; }
.pill { display:inline-block; font-size:.74rem; padding:2px 8px; border-radius:999px; background:#efeeea; color:#52514e; }
.pill.up { background:#e8f1fc; color:#1d5fae; } .pill.down { background:#fdecec; color:#b4302f; }
[data-testid="stVerticalBlockBorderWrapper"] { background:#fff; border-radius:12px !important; border-color:#e6e5e1 !important;
  box-shadow:0 1px 2px rgba(0,0,0,.03); }
.card-t { font-weight:600; font-size:1.0rem; color:#1f1f1d; margin-bottom:2px; }
[data-testid="stCaptionContainer"] { font-size:.8rem; color:#6b6a66; }
.finding { background:#f6f4ee; border-radius:10px; padding:10px 14px; font-size:.86rem; line-height:1.5; color:#2b2a27; margin:4px 0 6px; }
table.rp { width:100%; border-collapse:collapse; font-family:'Inter', system-ui, sans-serif; font-size:.9rem; margin:6px 0 10px; }
table.rp th, table.rp td { border:1px solid #e6e5e1; padding:7px 10px; text-align:right; color:#2b2a27; }
table.rp tr:first-child th { background:#f3f2ee; font-weight:600; }
table.rp tr th:first-child { text-align:left; font-weight:500; }
table.rp td.sig { color:#c2352f; font-weight:650; }
.rtable { display:flex; flex-direction:column; gap:4px; margin-top:6px; }
.rtable div { display:grid; grid-template-columns:auto 1fr; column-gap:8px; padding:6px 10px; border-radius:8px;
  background:#f6f6f4; font-size:.82rem; color:#6b6a66; border:1px solid transparent; }
.rtable div span { font-weight:600; }
.rtable div b { font-weight:650; color:inherit; text-align:right; }
.rtable div em { grid-column:1 / 3; font-style:normal; font-size:.74rem; text-align:right; line-height:1.35; }
.rtable div.on { background:#e8f1fc; color:#1d5fae; border-color:#c9dcf5; }
.rtable div.sig em { color:#12805a; font-weight:600; }
.js-plotly-plot .barlayer .trace, .js-plotly-plot .barlayer .point { transition: opacity .2s ease; }
.js-plotly-plot .barlayer .point path { transition: transform .16s ease, filter .16s ease; transform-box: fill-box; }
.js-plotly-plot .barlayer .point path.bar-hot { filter: brightness(1.06) drop-shadow(0 2px 4px rgba(0,0,0,.18)); }
</style>
"""

HOVER_JS = """
<script>
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

# ======================================================================== app
st.set_page_config(page_title="Singapore Birth Rate Explorer", page_icon="👶", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
if hasattr(st, "iframe"):          # Streamlit >= 1.5x
    st.iframe(HOVER_JS, height=1)
else:
    components.html(HOVER_JS, height=0)

with st.sidebar:
    st.markdown("### 👶 Birth Rate Explorer")
    st.caption("Singapore · SingStat Table Builder")
    st.session_state["years"] = st.slider("Years", 2010, 2025, (2010, 2025), help="Applies to every chart.")
    st.caption("Click a bar to focus on it; click again to reset.")

PAGES = {
    "Overview": st.Page(page_overview, title="Overview", icon="🧭", default=True, url_path="overview"),
    "Marriage": st.Page(page_marriage, title="Marriage", icon="💍", url_path="marriage"),
    "Income": st.Page(page_income, title="Income", icon="💰", url_path="income"),
    "Housing": st.Page(page_housing, title="Housing", icon="🏠", url_path="housing"),
    "Education": st.Page(page_education, title="Education", icon="🎓", url_path="education"),
    "Healthcare": st.Page(page_healthcare, title="Healthcare", icon="🏥", url_path="healthcare"),
    "Correlations": st.Page(page_correlations, title="Correlations", icon="🧩", url_path="correlations"),
    "Data": st.Page(page_data, title="Data & sources", icon="🗂️", url_path="data"),
}
st.navigation({"": [PAGES["Overview"]],
               "Factors": [PAGES[k] for k in FACTORS],
               "Together": [PAGES["Correlations"], PAGES["Data"]]}).run()
