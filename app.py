"""From Pilots to Agents: How US Banks Talk About AI (2019-2026) — DATS 6401 mid-semester project.
Run:  streamlit run app.py   (all data files are in the same folder; they are made by ai_banks_project.ipynb)"""
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="AI in US Bank Reports", page_icon="🏦", layout="wide")
HERE = Path(__file__).parent

# ---------------- settings (same as in the notebook) ----------------
SIZE_LABELS = ["Community (< $10 bn)", "Regional ($10–100 bn)", "Large (> $100 bn)"]
GROUPS = ["AI (any)", "Generative AI", "Agentic AI"]
COLORS = {"AI (any)": "#0072B2", "Generative AI": "#56B4E9", "Agentic AI": "#D55E00",
          "Large (> $100 bn)": "#0072B2", "Regional ($10–100 bn)": "#E69F00", "Community (< $10 bn)": "#009E73",
          "context": "#999999"}
GROUP_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#882255", "#332288"]   # no orange: orange = agentic ring
THEMES = ["Risk", "Cybersecurity & fraud", "Regulation & law", "Third-party vendors",
          "Customers & service", "Efficiency & operations"]
MEASURES = {"genai": "% of banks mentioning Generative AI", "ai": "% of banks mentioning AI",
            "agentic": "% of banks mentioning Agentic AI", "ai_sentences": "AI sentences per bank report"}
MIN_BANKS = 3
NO_DATA = f"No data / fewer than {MIN_BANKS} banks"
LAYOUT = dict(template="simple_white", font_size=14)


@st.cache_data
def load():
    """Read all files made by the notebook."""
    files = ["filings.parquet", "timeseries.csv", "term_counts.parquet", "ai_sentences.parquet",
             "network_nodes.csv", "network_edges.csv", "network_groups.csv", "vendor_edges.csv", "states.csv"]
    data = {f.split(".")[0]: (pd.read_parquet(HERE / f) if f.endswith(".parquet") else pd.read_csv(HERE / f))
            for f in files}
    narr = json.loads((HERE / "narrative.json").read_text(encoding="utf-8"))
    return data, narr


D, NARR = load()


# ---------------- works with old and new Streamlit versions ----------------
NEW_ST = "alt" in inspect.signature(st.plotly_chart).parameters      # Streamlit 1.5x+ has alt text
STRETCH = {"width": "stretch"} if NEW_ST else {"use_container_width": True}


def chart(fig, alt=""):
    """Show a plotly chart full width; add alt text if this Streamlit version supports it."""
    st.plotly_chart(fig, **STRETCH, **({"alt": alt} if NEW_ST else {}))


def box():
    """A box with a border (plain container on very old Streamlit)."""
    return st.container(border=True) if "border" in inspect.signature(st.container).parameters else st.container()


def picture(path, alt=""):
    """Show an image full width; add alt text if this Streamlit version supports it."""
    params = inspect.signature(st.image).parameters
    if "alt" in params:
        st.image(path, width="stretch", alt=alt)
    elif "use_container_width" in params:
        st.image(path, use_container_width=True)
    else:
        st.image(path, use_column_width=True)


# ---------------- chart functions (same as in the notebook) ----------------
def wilson(k, n, z=1.96):
    """95% interval (low, high) in % for k out of n."""
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * (p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5 / (1 + z**2 / n)
    return 100 * (centre - half), 100 * (centre + half)


def trend_chart(df, line_col, title):
    """% per year with 95% intervals and labels at the line ends."""
    df = df.assign(up=df["high"] - df["pct"], down=df["pct"] - df["low"])
    fig = px.line(df, x="filing_year", y="pct", color=line_col, markers=True, error_y="up", error_y_minus="down",
                  color_discrete_map=COLORS, title=title, labels={"filing_year": "Filing year", "pct": "% of 10-K reports"})
    label_y = -10
    for r in df[df["filing_year"] == df["filing_year"].max()].sort_values("pct").itertuples():
        name = getattr(r, line_col)
        label_y = max(r.pct, label_y + 7)
        fig.add_annotation(x=r.filing_year, y=label_y, text=f"  {name.split(' (')[0]}: {r.pct:.0f}%", showarrow=False,
                           xanchor="left", font=dict(color=COLORS[name], size=13))
    if df["filing_year"].min() <= 2022 < df["filing_year"].max():
        fig.add_vline(x=2022.5, line_dash="dot", line_color=COLORS["context"])
        fig.add_annotation(x=2022.5, y=100, text="ChatGPT (Nov 2022)", showarrow=False, xanchor="right",
                           font=dict(color="#555555", size=12))
    fig.update_layout(**LAYOUT, showlegend=False, yaxis_range=[0, 105], xaxis_dtick=1, margin=dict(r=150),
                      xaxis_range=[df["filing_year"].min() - 0.3, df["filing_year"].max() + 0.3])
    return fig


def keyness(tc_a, tc_b, n=10, min_count=10):
    """Words typical for A (log_ratio > 0) and for B (< 0). +1 = two times more often."""
    a, b = tc_a.groupby("term")["count"].sum(), tc_b.groupby("term")["count"].sum()
    k = pd.concat([a, b], axis=1, keys=["a", "b"]).fillna(0)
    k = k[(k["a"] + k["b"]) >= min_count]
    k["log_ratio"] = np.log2(((k["a"] + 0.5) / a.sum()) / ((k["b"] + 0.5) / b.sum()))
    return pd.concat([k.nlargest(n, "log_ratio"), k.nsmallest(n, "log_ratio")]).rename_axis("term").reset_index()


def keyness_chart(k, name_a, name_b, color_a, color_b, title):
    k = k.assign(side=np.where(k["log_ratio"] > 0, name_a, name_b)).sort_values("log_ratio")
    fig = px.bar(k, x="log_ratio", y="term", orientation="h", color="side", title=title,
                 color_discrete_map={name_a: color_a, name_b: color_b},
                 labels={"log_ratio": f"← more {name_b}   |   more {name_a} →  (log2 ratio)", "term": "", "side": ""})
    fig.update_layout(**LAYOUT, height=560, legend=dict(orientation="h", y=1.08))
    return fig


def network_chart(nodes, edges, groups, color_by="group", min_weight=0.25, title="", highlight=None):
    """Bank similarity network with one place per group. color_by = 'group' or 'size_class'."""
    xy = nodes.set_index("node")
    e = edges[edges["weight"] >= min_weight]
    same = xy.loc[e["source"], "group"].values == xy.loc[e["target"], "group"].values
    fig = go.Figure()
    styles = [("Link between groups", ~same, "#D5D5D5", 0.8),
              ("Same group, similarity < 0.5", same & (e["weight"] < 0.5).values, "#A6A6A6", 1.2),
              ("Same group, 0.5–0.8", same & e["weight"].between(0.5, 0.8, inclusive="left").values, "#5F5F5F", 2.0),
              ("Same group, ≥ 0.8 (almost the same text)", same & (e["weight"] >= 0.8).values, "#1A1A1A", 3.2)]
    for name, mask, color, width in styles:
        part = e[mask]
        gap = np.full(len(part), np.nan)                  # NaN = break between two lines
        fig.add_trace(go.Scatter(
            x=np.column_stack([xy.loc[part["source"], "x"], xy.loc[part["target"], "x"], gap]).ravel(),
            y=np.column_stack([xy.loc[part["source"], "y"], xy.loc[part["target"], "y"], gap]).ravel(),
            mode="lines", line=dict(color=color, width=width), hoverinfo="skip", name=name))

    q_of = dict(zip(groups["group"], groups["q"]))
    if color_by == "group":
        cats = list(groups["group"])
        cmap = {g: (GROUP_COLORS[i] if g != "Smaller groups" else COLORS["context"]) for i, g in enumerate(cats)}
        names = {g: f"{g} (Q = {q_of[g]:.2f})" for g in cats}
    else:
        cats = [s for s in SIZE_LABELS if s in set(nodes["size_class"])]
        cmap, names = {s: COLORS[s] for s in cats}, {s: s for s in cats}

    for cat in cats:
        n = nodes[nodes[color_by] == cat]
        fig.add_trace(go.Scatter(
            x=n["x"], y=n["y"], mode="markers", name=names[cat],
            marker=dict(size=9 + 2 * n["n_sentences"] ** 0.5, color=cmap[cat], opacity=0.95,
                        line=dict(color=np.where(n["agentic"], COLORS["Agentic AI"], "white").tolist(),
                                  width=np.where(n["agentic"], 3, 1.2).tolist())),
            text=[f"<b>{r.bank_name}</b><br>{r.size_class}<br>{r.n_sentences} AI sentences · {r.degree} links"
                  f"<br>{r.group} (Q = {q_of[r.group]:.2f})" + ("<br><b>mentions agentic AI</b>" if r.agentic else "")
                  for r in n.itertuples()],
            hoverinfo="text"))

    for i, g in enumerate(groups.itertuples()):            # a title above every group
        head = g.group.split(":")[0]
        words_g = g.group.split(": ")[1] if ":" in g.group else "many small groups"
        fig.add_annotation(x=g.cx, y=g.cy + g.r + 0.45, showarrow=False, align="center",
                           text=f"<b>{head}</b> · {g.banks} banks · Q = {g.q:.2f}<br><i>{words_g}</i>",
                           font=dict(size=12, color=GROUP_COLORS[i] if g.group != "Smaller groups" else "#555555"))

    lab = nodes.loc[nodes.groupby("group")["degree"].idxmax()]    # most connected bank of each group
    if highlight is not None:
        h = nodes[nodes["bank_name"] == highlight]
        fig.add_trace(go.Scatter(x=h["x"], y=h["y"], mode="markers", name="Selected bank", hoverinfo="skip",
                                 marker=dict(size=30, color="rgba(0,0,0,0)", line=dict(color="black", width=3))))
        lab = pd.concat([lab, h])
    fig.add_trace(go.Scatter(x=lab["x"], y=lab["y"], mode="text", text=lab["bank_name"].str[:24], hoverinfo="skip",
                             textposition="bottom center", textfont=dict(size=11, color="#222222"), showlegend=False))
    fig.update_layout(title=title, template="simple_white", height=850, font_size=13,
                      margin=dict(l=10, r=10, t=70, b=10), legend=dict(orientation="h", y=-0.02, font_size=12),
                      xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x", scaleratio=1))
    return fig


def network_chart_3d(nodes, edges, groups, color_by="group", min_weight=0.25, title="", highlight=None, vendors=None):
    """3D bank network: drag = rotate, scroll = zoom. Optional vendor nodes (purple squares)."""
    xyz = nodes.set_index("node")
    e = edges[edges["weight"] >= min_weight]
    same = xyz.loc[e["source"], "group"].values == xyz.loc[e["target"], "group"].values
    fig = go.Figure()

    def lines(a, b, name, color, width, dash="solid"):
        gap = np.full(len(a), np.nan)
        fig.add_trace(go.Scatter3d(
            x=np.column_stack([a[:, 0], b[:, 0], gap]).ravel(), y=np.column_stack([a[:, 1], b[:, 1], gap]).ravel(),
            z=np.column_stack([a[:, 2], b[:, 2], gap]).ravel(), mode="lines", name=name, hoverinfo="skip",
            line=dict(color=color, width=width, dash=dash)))

    for name, mask, color, width in [("Link between groups", ~same, "#CFCFCF", 1),
                                     ("Same group, similarity < 0.8", same & (e["weight"] < 0.8).values, "#8C8C8C", 2.5),
                                     ("Same group, ≥ 0.8 (almost the same text)", same & (e["weight"] >= 0.8).values, "#1A1A1A", 5)]:
        part = e[mask]
        lines(xyz.loc[part["source"], ["x3", "y3", "z3"]].values, xyz.loc[part["target"], ["x3", "y3", "z3"]].values,
              name, color, width)

    q_of = dict(zip(groups["group"], groups["q"]))
    if color_by == "group":
        cats = list(groups["group"])
        cmap = {g: (GROUP_COLORS[i] if g != "Smaller groups" else COLORS["context"]) for i, g in enumerate(cats)}
        names = {g: f"{g} (Q = {q_of[g]:.2f})" for g in cats}
    else:
        cats = [s for s in SIZE_LABELS if s in set(nodes["size_class"])]
        cmap, names = {s: COLORS[s] for s in cats}, {s: s for s in cats}
    for cat in cats:
        n = nodes[nodes[color_by] == cat]
        fig.add_trace(go.Scatter3d(
            x=n["x3"], y=n["y3"], z=n["z3"], mode="markers", name=names[cat],
            marker=dict(size=5 + 1.2 * n["n_sentences"] ** 0.5, color=cmap[cat], opacity=1,
                        symbol=np.where(n["agentic"], "diamond", "circle").tolist(), line=dict(color="white", width=1)),
            text=[f"<b>{r.bank_name}</b><br>{r.size_class}<br>{r.n_sentences} AI sentences · {r.degree} links"
                  f"<br>{r.group} (Q = {q_of[r.group]:.2f})" + ("<br><b>mentions agentic AI</b>" if r.agentic else "")
                  for r in n.itertuples()], hoverinfo="text"))

    fig.add_trace(go.Scatter3d(                               # group titles
        x=groups["cx3"], y=groups["cy3"], z=groups["cz3"] + groups["r3"] + 0.35, mode="text", showlegend=False,
        text=[f"<b>{g.split(':')[0]}</b> (Q = {q:.2f})" for g, q in zip(groups["group"], groups["q"])],
        textfont=dict(size=13, color="#222222"), hoverinfo="skip"))

    if vendors is not None and len(vendors):
        v = vendors.merge(nodes[["cik", "x3", "y3", "z3"]], on="cik")      # only banks that are in the network
        if len(v):
            vpos = v.groupby("vendor")[["x3", "y3", "z3"]].mean()
            vpos = vpos * (4.6 / np.linalg.norm(vpos.values, axis=1, keepdims=True).clip(min=0.1))  # just outside the sphere
            vcount = v.groupby("vendor")["cik"].nunique()
            lines(v[["x3", "y3", "z3"]].values, vpos.loc[v["vendor"]].values, "Bank names this vendor", "rgba(123,50,148,0.55)", 1.5)
            fig.add_trace(go.Scatter3d(
                x=vpos["x3"], y=vpos["y3"], z=vpos["z3"], mode="markers+text", name="Technology vendor",
                text=vpos.index, textposition="top center", textfont=dict(size=13, color="#7B3294"),
                marker=dict(size=8 + 3 * vcount.loc[vpos.index] ** 0.5, color="#7B3294", symbol="square"),
                hovertext=[f"{name}: named by {vcount[name]} banks in the network" for name in vpos.index], hoverinfo="text"))

    if highlight is not None:
        h = nodes[nodes["bank_name"] == highlight]
        fig.add_trace(go.Scatter3d(x=h["x3"], y=h["y3"], z=h["z3"], mode="markers+text", name="Selected bank",
                                   text=h["bank_name"], textposition="top center", hoverinfo="skip",
                                   marker=dict(size=16, color="rgba(0,0,0,0)", symbol="circle-open",
                                               line=dict(color="black", width=4))))
    axis = dict(visible=False, showbackground=False)
    fig.update_layout(title=title, height=820, font_size=13, margin=dict(l=0, r=0, t=60, b=0),
                      legend=dict(orientation="h", y=0, font_size=12),
                      scene=dict(xaxis=axis, yaxis=axis, zaxis=axis, aspectmode="data",
                                 camera=dict(eye=dict(x=0.95, y=0.95, z=0.55))))
    return fig


def classify(values, scheme="Quantiles", k=5):
    """Class edges. Quantiles = same number of states per class; Equal interval = same width."""
    v = values.dropna()
    edges = np.unique(np.quantile(v, np.linspace(0, 1, k + 1))) if scheme == "Quantiles" \
        else np.unique(np.linspace(v.min(), v.max(), k + 1))
    return edges if len(edges) > 1 else np.array([v.min() - 0.5, v.max() + 0.5])


def state_map(states, measure="genai", scheme="Quantiles", k=5, title=""):
    """Classed choropleth (equal-area Albers USA). Grey = no data / too few banks."""
    d = states.copy()
    ok = d["banks"] >= MIN_BANKS
    edges = classify(d.loc[ok, measure], scheme, k)
    unit = "%" if measure != "ai_sentences" else ""
    dec = 0 if (edges[-1] - edges[0]) / (len(edges) - 1) >= 2 else 1          # 1 decimal for narrow classes
    labels = [f"{a:.{dec}f}–{b:.{dec}f}{unit}" for a, b in zip(edges[:-1], edges[1:])]
    d["class"] = NO_DATA
    d.loc[ok, "class"] = pd.cut(d.loc[ok, measure], edges, labels=labels, include_lowest=True, ordered=False).astype(str)
    colors = dict(zip(labels, px.colors.sample_colorscale("YlGnBu", list(np.linspace(0.15, 1, len(labels))))))
    colors[NO_DATA] = "#D9D9D9"
    d["hover"] = np.where(ok, d["biz_state"] + ": " + d[measure].astype(str) + unit + " (" + d["banks"].astype(str) + " banks)",
                          d["biz_state"] + ": " + d["banks"].astype(str) + " banks — too few")
    fig = px.choropleth(d, locations="biz_state", locationmode="USA-states", scope="usa", color="class",
                        category_orders={"class": labels + [NO_DATA]}, color_discrete_map=colors, hover_name="hover",
                        hover_data={"biz_state": False, "class": False}, labels={"class": MEASURES[measure]}, title=title)
    fig.update_traces(marker_line_color="white", marker_line_width=0.8)
    fig.update_geos(bgcolor="rgba(0,0,0,0)", lakecolor="white")
    fig.update_layout(height=620, font_size=14, margin=dict(l=0, r=0, t=60, b=0),
                      legend=dict(orientation="h", y=-0.02, title_font_size=14))
    return fig


def note(key):
    """Grey text under a chart = the same finding in words (alt text)."""
    st.caption(NARR.get(key, ""))


# ---------------- sidebar ----------------
f_all = D["filings"]
y_min, y_max = int(f_all["filing_year"].min()), int(f_all["filing_year"].max())
st.sidebar.title("🏦 Filters")
years = st.sidebar.slider("Filing years", y_min, y_max, (y_min, y_max))
sizes = st.sidebar.multiselect("Bank size", SIZE_LABELS, default=SIZE_LABELS)
group = st.sidebar.radio("AI word group", GROUPS, help="Used in the bank-size chart (Over time).")
st.sidebar.caption("Years and bank size change the *Over time* and *What banks say* tabs. "
                   "Network and map use the latest report of each bank (2025–2026).")
if not sizes:
    st.warning("Please choose at least one bank size in the sidebar.")
    st.stop()

in_years = lambda df: df[df["filing_year"].between(*years)]
in_sizes = lambda df: df[df["size_class"].isin(sizes)]

tabs = st.tabs(["🏠 Overview", "⏱ Over time", "📝 What banks say", "🕸 Network", "🗺 Map", "📄 Notes & data"])

# ---------------- 1. Overview ----------------
with tabs[0]:
    st.markdown(NARR["overview_title"])
    st.markdown(NARR["overview_intro"])
    ts_all = D["timeseries"][D["timeseries"]["size_class"] == "All banks"]
    pct = ts_all.pivot(index="filing_year", columns="group", values="pct")
    c = st.columns(4)
    c[0].metric(f"Reports mentioning AI, {y_max}", f"{pct.loc[y_max, 'AI (any)']:.0f}%",
                f"{pct.loc[y_max, 'AI (any)'] - pct.loc[y_min, 'AI (any)']:+.0f} pts since {y_min}")
    c[1].metric(f"Generative AI, {y_max}", f"{pct.loc[y_max, 'Generative AI']:.0f}%")
    c[2].metric(f"Agentic AI, {y_max}", f"{pct.loc[y_max, 'Agentic AI']:.0f}%")
    c[3].metric("Reports read", f"{len(f_all):,}", f"{f_all['cik'].nunique()} banks", delta_color="off")
    left, right = st.columns([1, 1])
    left.markdown(NARR["overview_tabs"])
    right.markdown(NARR.get("overview_findings", ""))
    st.info(NARR["overview_how_to_read"])

# ---------------- 2. Over time ----------------
with tabs[1]:
    st.subheader("When did AI words appear in bank reports?")
    ts = in_years(D["timeseries"])
    ts = ts[ts["size_class"].isin(sizes)]
    # add up the chosen bank sizes, then compute % and 95% range again
    waves = ts.groupby(["filing_year", "group"])[["mentions", "reports"]].sum().reset_index()
    waves["pct"] = 100 * waves["mentions"] / waves["reports"]
    waves["low"], waves["high"] = wilson(waves["mentions"], waves["reports"])
    a = waves[waves["group"] == "AI (any)"].set_index("filing_year")["pct"]
    chart(trend_chart(waves, "group",
                                f"AI in bank reports: from {a.iloc[0]:.0f}% ({a.index[0]}) to {a.iloc[-1]:.0f}% ({a.index[-1]})"),
                    alt=NARR["caption_waves"])
    note("caption_waves")

    by_size = ts[ts["group"] == group].copy()
    w = by_size.pivot(index="filing_year", columns="size_class", values="pct")
    lead = w.iloc[-1].idxmax()
    chart(trend_chart(by_size, "size_class",
                                f"{group}: {lead.split(' (')[0]} banks lead in {w.index[-1]} ({w.iloc[-1].max():.0f}%)"),
                    alt=NARR["caption_size"])
    note("caption_size")
    with box():
        st.markdown(NARR["ts_findings"])

# ---------------- 3. What banks say ----------------
with tabs[2]:
    st.subheader("What do banks write about AI?")
    tc = in_sizes(in_years(D["term_counts"]))
    sent = in_sizes(in_years(D["ai_sentences"]))
    if sent.empty:
        st.warning("No AI sentences for this choice — please choose more years.")
    else:
        left, right = st.columns([1.1, 1])
        with left:
            picture(str(HERE / "wordcloud.png"), alt=NARR["caption_cloud"])
            st.caption("Word cloud (all years, all banks): bigger word = used more often. Position and colour mean nothing.")
        with right:
            top = tc.groupby("term")["count"].sum().nlargest(20).rename_axis("term").reset_index()
            fig = px.bar(top.sort_values("count"), x="count", y="term", orientation="h", text="count",
                         title=f"Top words next to AI: {', '.join(top['term'].head(3))}",
                         labels={"count": "Times used in AI sentences", "term": ""})
            fig.update_traces(marker_color=COLORS["AI (any)"], textposition="outside")
            fig.update_layout(**LAYOUT, height=560)
            chart(fig, alt=NARR["caption_top"])

        themes = (sent[THEMES].mean().mul(100).rename_axis("theme").reset_index(name="pct").sort_values("pct"))
        lead_theme = themes.iloc[-1]
        fig = px.bar(themes, x="pct", y="theme", orientation="h", text="pct",
                     title=f"{lead_theme.pct:.0f}% of AI sentences talk about {lead_theme.theme.lower()}",
                     labels={"pct": "% of AI sentences", "theme": ""})
        fig.update_traces(texttemplate="%{text:.0f}%", textposition="outside",
                          marker_color=[COLORS["AI (any)"] if t == lead_theme.theme else COLORS["context"] for t in themes["theme"]])
        fig.update_layout(**LAYOUT, xaxis_range=[0, 100], height=380)
        chart(fig, alt=NARR["caption_themes"])
        note("caption_themes")

        st.subheader("Keywords: what is different?")
        mode = st.radio("Compare", ["Two years", "Large vs community banks"], horizontal=True)
        if mode == "Two years":
            yrs = sorted(D["term_counts"]["filing_year"].unique())
            c1, c2 = st.columns(2)
            ya = c1.selectbox("Year A", yrs, index=len(yrs) - 1)
            yb = c2.selectbox("Year B", yrs, index=max(0, len(yrs) - 3))
            k = keyness(D["term_counts"][D["term_counts"].filing_year == ya], D["term_counts"][D["term_counts"].filing_year == yb])
            fig = keyness_chart(k, str(ya), str(yb), COLORS["Agentic AI"], COLORS["context"],
                                f"Words typical for {ya}: {', '.join(k[k.log_ratio > 0]['term'].head(3))}")
        else:
            big, small = SIZE_LABELS[2], SIZE_LABELS[0]
            tcy = in_years(D["term_counts"])
            k = keyness(tcy[tcy.size_class == big], tcy[tcy.size_class == small])
            fig = keyness_chart(k, "Large banks", "Community banks", COLORS[big], COLORS[small],
                                f"Large banks write {', '.join(k[k.log_ratio > 0]['term'].head(2))}; "
                                f"community banks {', '.join(k.sort_values('log_ratio')['term'].head(2))}")
        chart(fig, alt=NARR["caption_keyness"])
        note("caption_keyness")

        with box():
            st.markdown(NARR["text_findings"])

# ---------------- 4. Network ----------------
with tabs[3]:
    st.subheader("Which banks write about AI in the same way?")
    nodes, edges = D["network_nodes"], D["network_edges"]
    c0, c1, c2, c3 = st.columns([1, 1, 1, 1.3])
    view = c0.radio("View", ["3D (drag to rotate, scroll to zoom)", "2D"])
    color_by = {"Group (similar text)": "group", "Bank size": "size_class"}[
        c1.radio("Colour by", ["Group (similar text)", "Bank size"])]
    min_w = c2.slider("Show links with similarity ≥", 0.25, 0.95, 0.25, 0.05)
    pick = c3.selectbox("Find a bank", ["—"] + sorted(nodes["bank_name"]))
    groups = D["network_groups"]
    big = groups[groups["main"]].iloc[0]
    q_all = groups["q"].sum()
    title = (f"{len(nodes)} banks, {int(groups['main'].sum())} main groups (Q = {q_all:.2f}); "
             f"the largest ({big.banks / len(nodes):.0%}) uses: {big.group.split(': ')[-1]}")
    highlight = None if pick == "—" else pick
    if view.startswith("3D"):
        show_vendors = st.checkbox("Show technology vendors (purple squares)", value=True)
        chart(network_chart_3d(nodes, edges, groups, color_by, min_w, title, highlight,
                               D["vendor_edges"] if show_vendors else None), alt=NARR["caption_net"])
        st.caption("Drag to rotate, scroll to zoom. Diamond = mentions agentic AI. Purple square = technology company.")
    else:
        chart(network_chart(nodes, edges, groups, color_by, min_w, title, highlight=highlight), alt=NARR["caption_net"])
    note("caption_net")
    if pick != "—":
        me = nodes[nodes["bank_name"] == pick].iloc[0]
        nb = edges[(edges.source == me.node) | (edges.target == me.node)]
        partners = nodes.set_index("node").loc[np.where(nb.source == me.node, nb.target, nb.source), "bank_name"]
        st.success(f"**{pick}** — {me.group}. Most similar banks: {', '.join(partners)}.")
    with box():
        st.markdown(NARR["network_findings"])

    vc = D["vendor_edges"]["vendor"].value_counts().rename_axis("vendor").reset_index(name="banks")
    fig = px.bar(vc.sort_values("banks"), x="banks", y="vendor", orientation="h", text="banks",
                 title=f"Few banks name a technology company; {vc.iloc[0].vendor} leads",
                 labels={"banks": "Number of banks (latest 10-K)", "vendor": ""})
    fig.update_traces(marker_color=COLORS["context"], textposition="outside")
    fig.update_layout(**LAYOUT, height=380)
    chart(fig, alt=NARR["caption_vendors"])
    note("caption_vendors")

# ---------------- 5. Map ----------------
with tabs[4]:
    st.subheader("Where are the banks that talk about AI?")
    states = D["states"]
    c1, c2, c3 = st.columns([1.4, 1, 1])
    label = c1.selectbox("Measure", list(MEASURES.values()), index=0)        # Generative AI: shows the clearest differences
    measure = {v: k for k, v in MEASURES.items()}[label]
    scheme = c2.radio("Classes", ["Quantiles", "Equal interval"], horizontal=True,
                      help="Quantiles: same number of states per colour. Equal interval: same width per colour.")
    k = c3.slider("Number of classes", 3, 7, 5)
    ok = states[states["banks"] >= MIN_BANKS]
    hi, lo = ok.nlargest(1, measure).iloc[0], ok.nsmallest(1, measure).iloc[0]
    unit = "%" if measure != "ai_sentences" else ""
    chart(state_map(states, measure, scheme, k,
                              f"{MEASURES[measure]}: from {lo[measure]:.0f}{unit} ({lo.biz_state}) to "
                              f"{hi[measure]:.0f}{unit} ({hi.biz_state})"), alt=NARR["caption_map"])
    st.caption(f"Albers USA (equal-area) · {scheme}, {k} classes · YlGnBu · latest report of each bank (2025–2026) · "
               f"grey = no public bank or fewer than {MIN_BANKS} banks.")
    with box():
        st.markdown(NARR["map_findings"])

# ---------------- 6. Notes & data ----------------
with tabs[5]:
    st.markdown(NARR["data_filings"])
    per_year = f_all.groupby("filing_year").size().reset_index(name="reports")
    fig = px.bar(per_year, x="filing_year", y="reports", text="reports", title="Number of bank 10-K reports per year",
                 labels={"filing_year": "Filing year", "reports": "Reports"})
    fig.update_traces(marker_color=COLORS["context"], textposition="outside", cliponaxis=False)
    fig.update_layout(**LAYOUT, height=380, xaxis_dtick=1, margin=dict(t=70),
                      yaxis_range=[0, per_year["reports"].max() * 1.2])   # room for the numbers on top
    chart(fig, alt=NARR["caption_filings"])
    note("caption_filings")
    st.markdown(NARR["data_method"])
    st.markdown(NARR["data_banks"])
    st.markdown(NARR.get("notes_limits", ""))
    st.markdown("**Download the data**")
    c = st.columns(3)
    for col, (name, label) in zip(c, [("timeseries", "Time series"), ("states", "States"),
                                      ("network_nodes", "Network banks")]):
        col.download_button(label, D[name].to_csv(index=False).encode("utf-8"), f"{name}.csv", "text/csv")
