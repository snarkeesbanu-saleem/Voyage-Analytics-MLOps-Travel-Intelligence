"""
Voyage Analytics — MLOps Travel Intelligence Dashboard
Premium Streamlit Dashboard with Dark Theme
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.inference_pipeline import InferencePipeline

# ─────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Voyage Analytics",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
ASSETS_DIR = Path(__file__).resolve().parent / "assets"

# ─────────────────────────────────────────────────────────────
# Color Palette
# ─────────────────────────────────────────────────────────────
NAVY = "#0a1628"
TEAL = "#00d4aa"
GOLD = "#f5a623"
CORAL = "#ff6b6b"
SOFT_WHITE = "#f0f2f5"
LIGHT_NAVY = "#132040"
MID_NAVY = "#1a2d50"
PURPLE = "#8b5cf6"
CYAN = "#06b6d4"

PALETTE = [TEAL, GOLD, CORAL, PURPLE, CYAN, "#f472b6", "#34d399", "#fbbf24", "#a78bfa", "#fb923c"]
PLOTLY_TEMPLATE = "plotly_dark"

# ─────────────────────────────────────────────────────────────
# Custom CSS — Premium Dark Theme
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Import Google Fonts ───────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Root Vars ─────────────────────────────────────────── */
    :root {
        --navy: #0a1628;
        --light-navy: #132040;
        --mid-navy: #1a2d50;
        --teal: #00d4aa;
        --gold: #f5a623;
        --coral: #ff6b6b;
        --soft-white: #f0f2f5;
        --glass-bg: rgba(19, 32, 64, 0.7);
        --glass-border: rgba(0, 212, 170, 0.15);
        --card-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    /* ── Main App Background ───────────────────────────────── */
    .stApp {
        background: linear-gradient(135deg, #0a1628 0%, #132040 40%, #0d1b33 70%, #0a1628 100%);
        font-family: 'Inter', sans-serif;
    }

    /* ── Sidebar ───────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b33 0%, #0a1628 50%, #091422 100%);
        border-right: 1px solid rgba(0, 212, 170, 0.1);
    }

    section[data-testid="stSidebar"] .stRadio label {
        color: var(--soft-white) !important;
        font-weight: 500;
        padding: 8px 12px;
        border-radius: 10px;
        transition: all 0.3s ease;
    }

    section[data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(0, 212, 170, 0.08);
    }

    /* ── Headers ───────────────────────────────────────────── */
    h1, h2, h3, h4 {
        font-family: 'Inter', sans-serif !important;
        color: var(--soft-white) !important;
    }

    h1 {
        font-weight: 800 !important;
        background: linear-gradient(135deg, #00d4aa, #06b6d4, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.2rem !important;
        letter-spacing: -0.02em;
    }

    /* ── Metric Cards ──────────────────────────────────────── */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, var(--glass-bg), rgba(26, 45, 80, 0.5));
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 20px 24px;
        backdrop-filter: blur(20px);
        box-shadow: var(--card-shadow);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0, 212, 170, 0.15);
    }

    div[data-testid="stMetric"] label {
        color: rgba(240, 242, 245, 0.6) !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.75rem !important;
    }

    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: var(--soft-white) !important;
        font-weight: 700 !important;
        font-size: 1.8rem !important;
    }

    div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {
        color: var(--teal) !important;
    }

    /* ── Tabs ──────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(19, 32, 64, 0.5);
        border-radius: 12px;
        padding: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        color: rgba(240, 242, 245, 0.6);
        font-weight: 500;
        padding: 10px 20px;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0, 212, 170, 0.2), rgba(6, 182, 212, 0.15));
        color: var(--teal) !important;
        border-bottom: none;
    }

    /* ── Containers / Cards ────────────────────────────────── */
    div[data-testid="stExpander"] {
        background: var(--glass-bg);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        backdrop-filter: blur(20px);
    }

    /* ── Selectbox & Inputs ────────────────────────────────── */
    .stSelectbox, .stMultiSelect, .stDateInput, .stNumberInput, .stSlider {
        color: var(--soft-white);
    }

    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background: var(--glass-bg) !important;
        border-color: var(--glass-border) !important;
        border-radius: 10px !important;
    }

    /* ── Plotly Chart Containers ────────────────────────────── */
    .js-plotly-plot {
        border-radius: 16px;
        overflow: hidden;
    }

    /* ── Data Tables ───────────────────────────────────────── */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* ── Scrollbar ─────────────────────────────────────────── */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }

    ::-webkit-scrollbar-track {
        background: var(--navy);
    }

    ::-webkit-scrollbar-thumb {
        background: var(--mid-navy);
        border-radius: 3px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--teal);
    }

    /* ── Divider ───────────────────────────────────────────── */
    hr {
        border-color: var(--glass-border) !important;
    }

    /* ── Image Banner ──────────────────────────────────────── */
    .banner-img img {
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }

    /* ── Sidebar Title ─────────────────────────────────────── */
    .sidebar-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #00d4aa;
        padding: 10px 0;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        border-bottom: 1px solid rgba(0,212,170,0.2);
        margin-bottom: 20px;
    }

    /* ── Glowing accent line ───────────────────────────────── */
    .glow-line {
        height: 3px;
        background: linear-gradient(90deg, transparent, #00d4aa, #06b6d4, #8b5cf6, transparent);
        border-radius: 2px;
        margin: 10px 0 25px 0;
    }

    /* ── Info boxes ─────────────────────────────────────────── */
    .info-box {
        background: linear-gradient(135deg, rgba(0,212,170,0.08), rgba(6,182,212,0.05));
        border: 1px solid rgba(0,212,170,0.15);
        border-radius: 12px;
        padding: 16px 20px;
        color: #b0bec5;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    /* ── Button ─────────────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #00d4aa, #06b6d4) !important;
        color: #0a1628 !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        border: none !important;
        padding: 10px 24px !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(0,212,170,0.3) !important;
    }

    /* ── Hide Streamlit Footer ─────────────────────────────── */
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_users():
    df = pd.read_csv(DATA_DIR / "users.csv")
    return df


@st.cache_data
def load_flights():
    df = pd.read_csv(DATA_DIR / "flights.csv")
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["route"] = df["from"] + " → " + df["to"]
    return df


@st.cache_data
def load_hotels():
    df = pd.read_csv(DATA_DIR / "hotels.csv")
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["price_per_night"] = df["total"] / df["days"].replace(0, np.nan)
    return df


# ─────────────────────────────────────────────────────────────
# Plotly layout defaults
# ─────────────────────────────────────────────────────────────
def apply_chart_layout(fig, title="", height=500):
    """Apply consistent premium dark styling to all Plotly figures."""
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, color=SOFT_WHITE, family="Inter"),
            x=0.02,
            xanchor="left",
        ),
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,22,40,0.6)",
        font=dict(family="Inter", color="rgba(240,242,245,0.7)", size=12),
        height=height,
        margin=dict(l=60, r=30, t=60, b=60),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,212,170,0.15)",
            borderwidth=1,
            font=dict(size=11, color="rgba(240,242,245,0.7)"),
        ),
        hoverlabel=dict(
            bgcolor=LIGHT_NAVY,
            bordercolor=TEAL,
            font=dict(family="Inter", size=13, color=SOFT_WHITE),
        ),
    )
    fig.update_xaxes(
        gridcolor="rgba(255,255,255,0.04)",
        zerolinecolor="rgba(255,255,255,0.06)",
    )
    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.04)",
        zerolinecolor="rgba(255,255,255,0.06)",
    )
    return fig


def chart_container(fig):
    """Wrap a Plotly figure in a styled container."""
    st.plotly_chart(fig, use_column_width=True, config={"displayModeBar": False})


def banner_image(path):
    """Display a hero banner image."""
    if path.exists():
        st.image(str(path), use_column_width=True)


def section_header(icon, text):
    """Render a styled section header with glow line."""
    st.markdown(f"## {icon} {text}")
    st.markdown('<div class="glow-line"></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">✈️ Voyage Analytics</div>', unsafe_allow_html=True)
    st.markdown("")

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "✈️ Flights Analytics",
            "🏨 Hotels Analytics",
            "👥 Users Analytics",
            "🤖 ML Predictions",
            "📊 Model Performance",
            "🔄 Churn Analysis",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        '<div class="info-box">🚀 <b>MLOps Travel Intelligence</b><br>'
        "Real-time analytics & ML predictions for travel data.</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────────────────────────
users = load_users()
flights = load_flights()
hotels = load_hotels()


# =============================================================
#  PAGE 1 — HOME / OVERVIEW
# =============================================================
if page == "🏠 Home":
    # Logo
    logo_path = ASSETS_DIR / "voyage_logo.jpg"
    if logo_path.exists():
        col_logo = st.columns([1, 2, 1])
        with col_logo[1]:
            st.image(str(logo_path), use_column_width=True)

    st.markdown("")
    section_header("🏠", "Voyage Analytics — MLOps Travel Intelligence")

    st.markdown(
        '<div class="info-box">'
        "Welcome to <b>Voyage Analytics</b> — your premium travel intelligence dashboard. "
        "Explore flight patterns, hotel bookings, user demographics, and ML-powered predictions "
        "across a rich dataset of travel transactions."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    # ── Key Metrics ───────────────────────────────────────────
    total_users = len(users)
    total_flights = len(flights)
    total_hotels = len(hotels)
    total_flight_rev = flights["price"].sum()
    total_hotel_rev = hotels["total"].sum()
    total_revenue = total_flight_rev + total_hotel_rev

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Users", f"{total_users:,}")
    c2.metric("Total Flights", f"{total_flights:,}")
    c3.metric("Total Hotels", f"{total_hotels:,}")
    c4.metric("Total Revenue", f"${total_revenue:,.0f}")

    st.markdown("")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Avg Flight Price", f"${flights['price'].mean():,.2f}")
    c6.metric("Avg Hotel/Night", f"${hotels['price_per_night'].mean():,.2f}")
    c7.metric("Unique Routes", f"{flights['route'].nunique():,}")
    c8.metric("Unique Hotels", f"{hotels['name'].nunique()}")

    st.markdown("")

    # ── Quick Charts ──────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        monthly = flights.groupby("month").size().reset_index(name="bookings")
        fig = px.area(
            monthly, x="month", y="bookings",
            color_discrete_sequence=[TEAL],
        )
        fig.update_traces(fill="tozeroy", line=dict(width=2))
        apply_chart_layout(fig, "Monthly Flight Bookings Overview", 340)
        chart_container(fig)

    with col_b:
        hotel_monthly = hotels.groupby("month").size().reset_index(name="bookings")
        fig = px.area(
            hotel_monthly, x="month", y="bookings",
            color_discrete_sequence=[GOLD],
        )
        fig.update_traces(fill="tozeroy", line=dict(width=2))
        apply_chart_layout(fig, "Monthly Hotel Bookings Overview", 340)
        chart_container(fig)

    # ── Revenue Breakdown ─────────────────────────────────────
    fig = go.Figure(data=[go.Pie(
        labels=["Flight Revenue", "Hotel Revenue"],
        values=[total_flight_rev, total_hotel_rev],
        hole=0.65,
        marker=dict(colors=[TEAL, GOLD], line=dict(color=NAVY, width=3)),
        textinfo="label+percent",
        textfont=dict(size=14, color=SOFT_WHITE),
        hoverinfo="label+value+percent",
    )])
    apply_chart_layout(fig, "Revenue Distribution", 380)
    chart_container(fig)


# =============================================================
#  PAGE 2 — FLIGHTS ANALYTICS
# =============================================================
elif page == "✈️ Flights Analytics":
    banner_image(ASSETS_DIR / "flights_banner.jpg")
    section_header("✈️", "Flights Analytics")

    # ── Filters ───────────────────────────────────────────────
    with st.expander("🔍 Filters", expanded=True):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            date_min = flights["date"].min()
            date_max = flights["date"].max()
            if pd.notna(date_min) and pd.notna(date_max):
                date_range = st.date_input(
                    "Date Range",
                    value=(date_min.date(), date_max.date()),
                    min_value=date_min.date(),
                    max_value=date_max.date(),
                )
            else:
                date_range = None
        with fc2:
            flight_types = ["All"] + sorted(flights["flightType"].dropna().unique().tolist())
            sel_flight_type = st.selectbox("Flight Type", flight_types)
        with fc3:
            agencies = ["All"] + sorted(flights["agency"].dropna().unique().tolist())
            sel_agency = st.selectbox("Agency", agencies)

    # Apply filters
    fdf = flights.copy()
    if date_range and len(date_range) == 2:
        fdf = fdf[(fdf["date"].dt.date >= date_range[0]) & (fdf["date"].dt.date <= date_range[1])]
    if sel_flight_type != "All":
        fdf = fdf[fdf["flightType"] == sel_flight_type]
    if sel_agency != "All":
        fdf = fdf[fdf["agency"] == sel_agency]

    # Quick metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Filtered Flights", f"{len(fdf):,}")
    m2.metric("Avg Price", f"${fdf['price'].mean():,.2f}" if len(fdf) > 0 else "N/A")
    m3.metric("Total Revenue", f"${fdf['price'].sum():,.0f}" if len(fdf) > 0 else "N/A")
    m4.metric("Unique Routes", f"{fdf['route'].nunique():,}" if len(fdf) > 0 else "N/A")
    st.markdown("")

    # ── Chart 1: Bookings Over Time ───────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        monthly_trend = fdf.groupby("month").size().reset_index(name="bookings")
        fig = px.line(
            monthly_trend, x="month", y="bookings",
            markers=True, color_discrete_sequence=[TEAL],
        )
        fig.update_traces(line=dict(width=3), marker=dict(size=8))
        apply_chart_layout(fig, "Flight Bookings — Monthly Trend", 420)
        fig.update_xaxes(title_text="Month")
        fig.update_yaxes(title_text="Number of Bookings")
        chart_container(fig)

    # ── Chart 2: Sunburst — Agency → FlightType ──────────────
    with col2:
        sun_df = fdf.groupby(["agency", "flightType"]).size().reset_index(name="count")
        fig = px.sunburst(
            sun_df, path=["agency", "flightType"], values="count",
            color="count", color_continuous_scale=["#0a1628", TEAL, GOLD],
        )
        apply_chart_layout(fig, "Flight Distribution: Agency → Type", 420)
        chart_container(fig)

    # ── Chart 3: Heatmap — Avg Price by Route ─────────────────
    route_price = fdf.groupby(["from", "to"])["price"].mean().reset_index()
    top_froms = fdf["from"].value_counts().head(15).index.tolist()
    top_tos = fdf["to"].value_counts().head(15).index.tolist()
    hm_df = route_price[(route_price["from"].isin(top_froms)) & (route_price["to"].isin(top_tos))]
    if len(hm_df) > 0:
        pivot = hm_df.pivot_table(index="from", columns="to", values="price", aggfunc="mean")
        fig = px.imshow(
            pivot, color_continuous_scale=["#0a1628", TEAL, GOLD, CORAL],
            aspect="auto",
        )
        apply_chart_layout(fig, "Average Flight Price by Route (Top Origins × Destinations)", 500)
        fig.update_xaxes(title_text="Destination", tickangle=45)
        fig.update_yaxes(title_text="Origin")
        chart_container(fig)

    # ── Charts 4 & 5 ─────────────────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        # Chart 4: Violin — Price by Flight Type
        fig = px.violin(
            fdf, x="flightType", y="price", color="flightType",
            box=True, points=False, color_discrete_sequence=PALETTE,
        )
        apply_chart_layout(fig, "Price Distribution by Flight Type", 420)
        fig.update_xaxes(title_text="Flight Type")
        fig.update_yaxes(title_text="Price ($)")
        chart_container(fig)

    with col4:
        # Chart 5: Top 10 Busiest Routes
        top_routes = fdf["route"].value_counts().head(10).reset_index()
        top_routes.columns = ["route", "count"]
        fig = px.bar(
            top_routes, x="count", y="route", orientation="h",
            color="count", color_continuous_scale=[NAVY, TEAL],
        )
        apply_chart_layout(fig, "Top 10 Busiest Routes", 420)
        fig.update_xaxes(title_text="Number of Flights")
        fig.update_yaxes(title_text="Route")
        chart_container(fig)

    # ── Charts 6 & 7 ─────────────────────────────────────────
    col5, col6 = st.columns(2)

    with col5:
        # Chart 6: Scatter — Distance vs Price
        sample = fdf.sample(min(5000, len(fdf)), random_state=42) if len(fdf) > 5000 else fdf
        fig = px.scatter(
            sample, x="distance", y="price", color="flightType",
            opacity=0.6, color_discrete_sequence=PALETTE,
        )
        apply_chart_layout(fig, "Distance vs Price by Flight Type", 420)
        fig.update_xaxes(title_text="Distance (km)")
        fig.update_yaxes(title_text="Price ($)")
        chart_container(fig)

    with col6:
        # Chart 7: Revenue Trend by Agency
        rev_trend = fdf.groupby(["month", "agency"])["price"].sum().reset_index()
        fig = px.area(
            rev_trend, x="month", y="price", color="agency",
            color_discrete_sequence=PALETTE,
        )
        apply_chart_layout(fig, "Revenue Trend Over Time by Agency", 420)
        fig.update_xaxes(title_text="Month")
        fig.update_yaxes(title_text="Revenue ($)")
        chart_container(fig)


# =============================================================
#  PAGE 3 — HOTELS ANALYTICS
# =============================================================
elif page == "🏨 Hotels Analytics":
    banner_image(ASSETS_DIR / "hotels_banner.jpg")
    section_header("🏨", "Hotels Analytics")

    # ── Filters ───────────────────────────────────────────────
    with st.expander("🔍 Filters", expanded=True):
        hc1, hc2, hc3 = st.columns(3)
        with hc1:
            hotel_names = ["All"] + sorted(hotels["name"].dropna().unique().tolist())
            sel_hotel = st.selectbox("Hotel Name", hotel_names)
        with hc2:
            places = ["All"] + sorted(hotels["place"].dropna().unique().tolist())
            sel_place = st.selectbox("Place", places)
        with hc3:
            h_date_min = hotels["date"].min()
            h_date_max = hotels["date"].max()
            if pd.notna(h_date_min) and pd.notna(h_date_max):
                h_date_range = st.date_input(
                    "Date Range ",
                    value=(h_date_min.date(), h_date_max.date()),
                    min_value=h_date_min.date(),
                    max_value=h_date_max.date(),
                    key="hotel_date_range",
                )
            else:
                h_date_range = None

    hdf = hotels.copy()
    if sel_hotel != "All":
        hdf = hdf[hdf["name"] == sel_hotel]
    if sel_place != "All":
        hdf = hdf[hdf["place"] == sel_place]
    if h_date_range and len(h_date_range) == 2:
        hdf = hdf[(hdf["date"].dt.date >= h_date_range[0]) & (hdf["date"].dt.date <= h_date_range[1])]

    # Quick metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Filtered Bookings", f"{len(hdf):,}")
    m2.metric("Avg Stay", f"{hdf['days'].mean():.1f} days" if len(hdf) > 0 else "N/A")
    m3.metric("Total Revenue", f"${hdf['total'].sum():,.0f}" if len(hdf) > 0 else "N/A")
    m4.metric("Avg Price/Night", f"${hdf['price_per_night'].mean():,.2f}" if len(hdf) > 0 else "N/A")
    st.markdown("")

    # ── Chart 1: Treemap — Spending by Hotel → Place ──────────
    tree_df = hdf.groupby(["name", "place"])["total"].sum().reset_index()
    if len(tree_df) > 0:
        fig = px.treemap(
            tree_df, path=["name", "place"], values="total",
            color="total", color_continuous_scale=[NAVY, TEAL, GOLD],
        )
        apply_chart_layout(fig, "Hotel Spending: Hotel → Place", 500)
        chart_container(fig)

    col1, col2 = st.columns(2)

    with col1:
        # Chart 2: Avg Stay Duration by Hotel
        avg_stay = hdf.groupby("name")["days"].mean().sort_values(ascending=True).tail(15).reset_index()
        fig = px.bar(
            avg_stay, x="days", y="name", orientation="h",
            color="days", color_continuous_scale=[NAVY, TEAL],
        )
        apply_chart_layout(fig, "Average Stay Duration by Hotel", 420)
        fig.update_xaxes(title_text="Average Days")
        fig.update_yaxes(title_text="Hotel")
        chart_container(fig)

    with col2:
        # Chart 3: Price Per Night Box Plot
        fig = px.box(
            hdf, x="name", y="price_per_night", color="name",
            color_discrete_sequence=PALETTE,
        )
        apply_chart_layout(fig, "Price Per Night by Hotel", 420)
        fig.update_xaxes(title_text="Hotel", tickangle=45)
        fig.update_yaxes(title_text="Price Per Night ($)")
        fig.update_layout(showlegend=False)
        chart_container(fig)

    col3, col4 = st.columns(2)

    with col3:
        # Chart 4: Booking Distribution Pie
        booking_dist = hdf["name"].value_counts().reset_index()
        booking_dist.columns = ["name", "count"]
        fig = px.pie(
            booking_dist, names="name", values="count",
            hole=0.4, color_discrete_sequence=PALETTE,
        )
        apply_chart_layout(fig, "Booking Distribution by Hotel", 420)
        fig.update_traces(textinfo="label+percent", textfont_size=11)
        chart_container(fig)

    with col4:
        # Chart 5: Hotel Bookings Monthly Trend
        h_monthly = hdf.groupby("month").size().reset_index(name="bookings")
        fig = px.line(
            h_monthly, x="month", y="bookings",
            markers=True, color_discrete_sequence=[GOLD],
        )
        fig.update_traces(line=dict(width=3), marker=dict(size=8))
        apply_chart_layout(fig, "Hotel Bookings — Monthly Trend", 420)
        fig.update_xaxes(title_text="Month")
        fig.update_yaxes(title_text="Bookings")
        chart_container(fig)

    # Chart 6: Days vs Total Cost Scatter
    sample_h = hdf.sample(min(5000, len(hdf)), random_state=42) if len(hdf) > 5000 else hdf
    fig = px.scatter(
        sample_h, x="days", y="total", color="name",
        opacity=0.6, size="price", size_max=15,
        color_discrete_sequence=PALETTE,
    )
    apply_chart_layout(fig, "Days Stayed vs Total Cost", 450)
    fig.update_xaxes(title_text="Days Stayed")
    fig.update_yaxes(title_text="Total Cost ($)")
    chart_container(fig)


# =============================================================
#  PAGE 4 — USERS ANALYTICS
# =============================================================
elif page == "👥 Users Analytics":
    banner_image(ASSETS_DIR / "users_banner.jpg")
    section_header("👥", "Users Analytics")

    # Quick metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Users", f"{len(users):,}")
    m2.metric("Avg Age", f"{users['age'].mean():.1f}")
    m3.metric("Companies", f"{users['company'].nunique()}")
    m4.metric("Gender Split", f"{(users['gender'].value_counts(normalize=True).iloc[0]*100):.0f}% / {(users['gender'].value_counts(normalize=True).iloc[1]*100):.0f}%")
    st.markdown("")

    col1, col2 = st.columns(2)

    with col1:
        # Chart 1: Gender Donut
        gender_counts = users["gender"].value_counts().reset_index()
        gender_counts.columns = ["gender", "count"]
        fig = go.Figure(data=[go.Pie(
            labels=gender_counts["gender"],
            values=gender_counts["count"],
            hole=0.6,
            marker=dict(colors=[TEAL, CORAL], line=dict(color=NAVY, width=3)),
            textinfo="label+percent+value",
            textfont=dict(size=14, color=SOFT_WHITE),
        )])
        apply_chart_layout(fig, "Gender Distribution", 420)
        chart_container(fig)

    with col2:
        # Chart 2: Age Histogram
        fig = px.histogram(
            users, x="age", nbins=30, color_discrete_sequence=[TEAL],
            marginal="violin",
        )
        apply_chart_layout(fig, "Age Distribution", 420)
        fig.update_xaxes(title_text="Age")
        fig.update_yaxes(title_text="Count")
        chart_container(fig)

    col3, col4 = st.columns(2)

    with col3:
        # Chart 3: Users per Company (Top 15)
        company_counts = users["company"].value_counts().head(15).reset_index()
        company_counts.columns = ["company", "count"]
        fig = px.bar(
            company_counts, x="count", y="company", orientation="h",
            color="count", color_continuous_scale=[NAVY, TEAL, GOLD],
        )
        apply_chart_layout(fig, "Users per Company — Top 15", 420)
        fig.update_xaxes(title_text="Number of Users")
        fig.update_yaxes(title_text="Company")
        chart_container(fig)

    with col4:
        # Chart 4: Age vs Total Spending
        user_flight_spend = flights.groupby("userCode")["price"].sum().reset_index()
        user_flight_spend.columns = ["code", "flight_spend"]
        user_hotel_spend = hotels.groupby("userCode")["total"].sum().reset_index()
        user_hotel_spend.columns = ["code", "hotel_spend"]
        user_spend = users.merge(user_flight_spend, on="code", how="left").merge(
            user_hotel_spend, on="code", how="left"
        )
        user_spend["total_spend"] = user_spend["flight_spend"].fillna(0) + user_spend["hotel_spend"].fillna(0)
        fig = px.scatter(
            user_spend, x="age", y="total_spend", color="gender",
            opacity=0.6, color_discrete_sequence=[TEAL, CORAL],
        )
        apply_chart_layout(fig, "Age vs Total Spending", 420)
        fig.update_xaxes(title_text="Age")
        fig.update_yaxes(title_text="Total Spending ($)")
        chart_container(fig)

    # ── Chart 5: Avg Spend by Gender and Age Group ────────────
    user_spend["age_group"] = pd.cut(
        user_spend["age"],
        bins=[0, 25, 35, 45, 55, 65, 100],
        labels=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"],
    )
    avg_gender_age = user_spend.groupby(["age_group", "gender"], observed=True)["total_spend"].mean().reset_index()
    fig = px.bar(
        avg_gender_age, x="age_group", y="total_spend", color="gender",
        barmode="group", color_discrete_sequence=[TEAL, CORAL],
    )
    apply_chart_layout(fig, "Average Spending by Gender & Age Group", 420)
    fig.update_xaxes(title_text="Age Group")
    fig.update_yaxes(title_text="Average Spend ($)")

    col5, col6 = st.columns(2)
    with col5:
        chart_container(fig)

    with col6:
        # Chart 6: Funnel — User Activity Levels
        flight_counts = flights.groupby("userCode").size().reset_index(name="flight_count")
        hotel_counts = hotels.groupby("userCode").size().reset_index(name="hotel_count")
        activity = users[["code"]].merge(
            flight_counts.rename(columns={"userCode": "code"}), on="code", how="left"
        ).merge(
            hotel_counts.rename(columns={"userCode": "code"}), on="code", how="left"
        )
        activity["total_trips"] = activity["flight_count"].fillna(0) + activity["hotel_count"].fillna(0)

        def classify_activity(x):
            if x >= activity["total_trips"].quantile(0.75):
                return "🔥 High Travelers"
            elif x >= activity["total_trips"].quantile(0.25):
                return "✈️ Medium Travelers"
            else:
                return "🏠 Low Travelers"

        activity["level"] = activity["total_trips"].apply(classify_activity)
        level_counts = activity["level"].value_counts().reset_index()
        level_counts.columns = ["level", "count"]
        # Sort in funnel order
        order = ["🔥 High Travelers", "✈️ Medium Travelers", "🏠 Low Travelers"]
        level_counts["sort_key"] = level_counts["level"].map({v: i for i, v in enumerate(order)})
        level_counts = level_counts.sort_values("sort_key")

        fig = go.Figure(go.Funnel(
            y=level_counts["level"],
            x=level_counts["count"],
            marker=dict(color=[TEAL, GOLD, CORAL]),
            textinfo="value+percent total",
            textfont=dict(size=14, color=SOFT_WHITE),
        ))
        apply_chart_layout(fig, "User Activity Levels", 420)
        chart_container(fig)


# =============================================================
#  PAGE 5 — ML PREDICTIONS
# =============================================================
elif page == "🤖 ML Predictions":
    section_header("🤖", "ML Predictions")

    # Check for models using inference pipeline check
    try:
        pipeline = InferencePipeline()
        model_errors = pipeline.load_errors
        models_exist = not model_errors
    except Exception as exc:
        model_errors = {"critical": str(exc)}
        models_exist = False

    if not models_exist:
        st.warning(
            "⚠️ **No trained models found.** Please run the training pipeline first "
            "to generate model files in the `models/` directory."
        )
        st.markdown(
            '<div class="info-box">'
            "💡 <b>Tip:</b> Run <code>python pipelines/training_pipeline.py</code> "
            "to train and save all three models."
            "</div>",
            unsafe_allow_html=True,
        )
        if model_errors:
            st.error(f"Loader details: {model_errors}")
    else:
        tab1, tab2, tab3 = st.tabs([
            "✈️ Flight Price Prediction",
            "🧑 Gender Prediction",
            "📉 Churn Prediction",
        ])

        with tab1:
            st.markdown("### Predict Flight Price")
            st.markdown(
                '<div class="info-box">Enter flight details to get an estimated price prediction.</div>',
                unsafe_allow_html=True,
            )
            st.markdown("")

            pc1, pc2, pc3 = st.columns(3)
            with pc1:
                pred_from = st.selectbox("From", sorted(flights["from"].unique()), key="pred_from")
                pred_type = st.selectbox("Flight Type", sorted(flights["flightType"].unique()), key="pred_type")
            with pc2:
                pred_to = st.selectbox("To", sorted(flights["to"].unique()), key="pred_to")
                pred_agency = st.selectbox("Agency", sorted(flights["agency"].unique()), key="pred_agency")
            with pc3:
                pred_distance = st.number_input("Distance (km)", min_value=0.0, value=500.0, key="pred_dist")
                pred_time = st.number_input("Time (hrs)", min_value=0.0, value=2.0, key="pred_time")

            if st.button("🔮 Predict Price", key="btn_price"):
                try:
                    payload = {
                        "from": pred_from,
                        "to": pred_to,
                        "flightType": pred_type,
                        "agency": pred_agency,
                        "distance": float(pred_distance),
                        "time": float(pred_time)
                    }
                    result = pipeline.predict_flight_price(payload)
                    prediction = result["predicted_price"]
                    ci = result.get("confidence_interval", {"lower": prediction * 0.95, "upper": prediction * 1.05})

                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=prediction,
                        number=dict(prefix="$", font=dict(size=48, color=TEAL)),
                        title=dict(text="Predicted Flight Price", font=dict(size=18, color=SOFT_WHITE)),
                        gauge=dict(
                            axis=dict(range=[0, 3000], tickcolor=SOFT_WHITE),
                            bar=dict(color=TEAL),
                            bgcolor=LIGHT_NAVY,
                            bordercolor=TEAL,
                            steps=[
                                dict(range=[0, 1000], color="rgba(0,212,170,0.1)"),
                                dict(range=[1000, 2000], color="rgba(245,166,35,0.1)"),
                                dict(range=[2000, 3000], color="rgba(255,107,107,0.1)"),
                            ],
                        ),
                    ))
                    apply_chart_layout(fig, "", 350)
                    chart_container(fig)
                    st.write(f"Confidence Interval: **${ci['lower']:.2f}** to **${ci['upper']:.2f}**")
                except Exception as e:
                    st.error(f"Prediction error: {e}")

        with tab2:
            st.markdown("### Predict User Gender")
            st.markdown(
                '<div class="info-box">Enter travel behavior features to predict user gender.</div>',
                unsafe_allow_html=True,
            )
            st.markdown("")

            gc1, gc2, gc3 = st.columns(3)
            with gc1:
                g_age = st.number_input("Age", min_value=18, max_value=100, value=30, key="g_age")
                g_flight_type = st.selectbox("Flight Type", ["economic", "firstClass", "premium"], key="g_flight_type")
                g_price = st.number_input("Flight Price ($)", min_value=0.0, value=500.0, key="g_price")
            with gc2:
                g_distance = st.number_input("Flight Distance (km)", min_value=0.0, value=1000.0, key="g_dist")
                g_time = st.number_input("Flight Time (hrs)", min_value=0.0, value=2.0, key="g_time")
                g_agency = st.selectbox("Agency", ["FlyingDrops", "CloudFy", "Rainbow"], key="g_agency")
            with gc3:
                g_days = st.number_input("Hotel Stay Days", min_value=0, value=3, key="g_days")
                g_hotel_price = st.number_input("Hotel Daily Price ($)", min_value=0.0, value=150.0, key="g_hotel_price")
                g_hotel_total = st.number_input("Hotel Total Price ($)", min_value=0.0, value=450.0, key="g_hotel_total")

            if st.button("🔮 Predict Gender", key="btn_gender"):
                try:
                    payload = {
                        "age": int(g_age),
                        "flightType": g_flight_type,
                        "price": float(g_price),
                        "distance": float(g_distance),
                        "time": float(g_time),
                        "agency": g_agency,
                        "days": int(g_days),
                        "hotel_price": float(g_hotel_price),
                        "hotel_total": float(g_hotel_total)
                    }
                    result = pipeline.predict_gender(payload)
                    pred = result["predicted_gender"]
                    proba = result["probability"]

                    st.markdown(f"### Prediction: **{pred.upper()}**")

                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=proba * 100,
                        number=dict(suffix="%", font=dict(size=42, color=TEAL)),
                        title=dict(text="Prediction Confidence", font=dict(size=16, color=SOFT_WHITE)),
                        gauge=dict(
                            axis=dict(range=[50, 100]),
                            bar=dict(color=TEAL),
                            bgcolor=LIGHT_NAVY,
                            bordercolor=TEAL,
                        ),
                    ))
                    apply_chart_layout(fig, "", 300)
                    chart_container(fig)
                except Exception as e:
                    st.error(f"Prediction error: {e}")

        with tab3:
            st.markdown("### Predict Churn Risk")
            st.markdown(
                '<div class="info-box">Enter user activity features to assess churn risk level.</div>',
                unsafe_allow_html=True,
            )
            st.markdown("")

            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                c_total_flights = st.number_input("Total Flights", min_value=0, value=5, key="c_total_flights")
                c_total_hotel_bookings = st.number_input("Total Hotel Bookings", min_value=0, value=2, key="c_total_hotel_bookings")
                c_total_spend = st.number_input("Total Spend ($)", min_value=0.0, value=3000.0, key="c_total_spend")
            with cc2:
                c_avg_flight_price = st.number_input("Avg Flight Price ($)", min_value=0.0, value=600.0, key="c_avg_flight_price")
                c_avg_hotel_price = st.number_input("Avg Hotel Price ($)", min_value=0.0, value=150.0, key="c_avg_hotel_price")
                c_unique_destinations = st.number_input("Unique Destinations", min_value=0, value=3, key="c_unique_destinations")
            with cc3:
                c_pref_flight_type = st.selectbox("Preferred Flight Type", ["economic", "firstClass", "premium"], key="c_pref_flight_type")
                c_pref_agency = st.selectbox("Preferred Agency", ["FlyingDrops", "CloudFy", "Rainbow"], key="c_pref_agency")
                c_age = st.number_input("Age", min_value=18, max_value=100, value=40, key="c_age")
                c_days_since_last_trip = st.number_input("Days Since Last Trip", min_value=0.0, value=100.0, key="c_days_since_last_trip")

            if st.button("🔮 Predict Churn", key="btn_churn"):
                try:
                    payload = {
                        "total_flights": int(c_total_flights),
                        "total_hotel_bookings": int(c_total_hotel_bookings),
                        "total_spend": float(c_total_spend),
                        "avg_flight_price": float(c_avg_flight_price),
                        "avg_hotel_price": float(c_avg_hotel_price),
                        "unique_destinations": int(c_unique_destinations),
                        "preferred_flight_type": c_pref_flight_type,
                        "preferred_agency": c_pref_agency,
                        "age": int(c_age),
                        "days_since_last_trip": float(c_days_since_last_trip)
                    }
                    result = pipeline.predict_churn(payload)
                    churn_prob = result["churn_probability"] * 100

                    risk = "🟢 Low" if churn_prob < 30 else ("🟡 Medium" if churn_prob < 70 else "🔴 High")
                    st.markdown(f"### Churn Risk: **{risk}**")

                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=churn_prob,
                        number=dict(suffix="%", font=dict(size=48, color=CORAL if churn_prob > 70 else GOLD if churn_prob > 30 else TEAL)),
                        title=dict(text="Churn Probability", font=dict(size=18, color=SOFT_WHITE)),
                        gauge=dict(
                            axis=dict(range=[0, 100], tickcolor=SOFT_WHITE),
                            bar=dict(color=CORAL if churn_prob > 70 else GOLD if churn_prob > 30 else TEAL),
                            bgcolor=LIGHT_NAVY,
                            bordercolor="rgba(0,212,170,0.2)",
                            steps=[
                                dict(range=[0, 30], color="rgba(0,212,170,0.15)"),
                                dict(range=[30, 70], color="rgba(245,166,35,0.15)"),
                                dict(range=[70, 100], color="rgba(255,107,107,0.15)"),
                            ],
                            threshold=dict(
                                line=dict(color=CORAL, width=4),
                                thickness=0.8,
                                value=churn_prob,
                            ),
                        ),
                    ))
                    apply_chart_layout(fig, "", 350)
                    chart_container(fig)
                except Exception as e:
                    st.error(f"Prediction error: {e}")


# =============================================================
#  PAGE 6 — MODEL PERFORMANCE
# =============================================================
elif page == "📊 Model Performance":
    section_header("📊", "Model Performance")

    # Look for JSON reports in REPORTS_DIR
    report_files = list(REPORTS_DIR.glob("*report*.json")) + list(REPORTS_DIR.glob("*metrics*.json"))

    if not report_files:
        st.warning(
            "⚠️ **No model reports found.** Train the models first to generate performance metrics."
        )
        st.markdown(
            '<div class="info-box">'
            "💡 After training, model metrics will be saved as JSON files in the <code>reports/</code> directory."
            "</div>",
            unsafe_allow_html=True,
        )

        # Still show placeholder structure
        st.markdown("### Expected Model Performance Views")
        placeholder_tabs = st.tabs([
            "📊 Metrics Overview",
            "🎯 Feature Importance",
            "🔲 Confusion Matrix",
            "📈 Predictions vs Actual",
        ])

        with placeholder_tabs[0]:
            st.info("Model accuracy, precision, recall, F1, and RMSE metrics will appear here after training.")
            # Demo metrics table
            demo_df = pd.DataFrame({
                "Model": ["Flight Price (Random Forest)", "Gender (Random Forest)", "Churn (Random Forest)"],
                "Type": ["Regression", "Classification", "Classification"],
                "Primary Metric": ["RMSE", "Accuracy", "Accuracy"],
                "Status": ["⏳ Not Trained", "⏳ Not Trained", "⏳ Not Trained"],
            })
            st.dataframe(demo_df, use_container_width=True)

        with placeholder_tabs[1]:
            st.info("Feature importance charts will appear here after training.")

        with placeholder_tabs[2]:
            st.info("Confusion matrices for classification models will appear here.")

        with placeholder_tabs[3]:
            st.info("Predicted vs Actual scatter plots for regression models will appear here.")

    else:
        # Load and display reports
        reports = {}
        for f in report_files:
            try:
                with open(f, "r") as fp:
                    reports[f.stem] = json.load(fp)
            except Exception:
                pass

        if reports:
            perf_tabs = st.tabs([
                "📊 Metrics Overview",
                "🎯 Feature Importance",
                "🔲 Confusion Matrix",
                "📈 Predictions vs Actual",
            ])

            with perf_tabs[0]:
                st.markdown("### Model Metrics Summary")
                for name, report in reports.items():
                    with st.expander(f"📋 {name.replace('_metrics', '').replace('_', ' ').title()}", expanded=True):
                        if isinstance(report, dict):
                            metrics_to_show = {k: v for k, v in report.items()
                                                if isinstance(v, (int, float, str))}
                            if metrics_to_show:
                                cols = st.columns(min(len(metrics_to_show), 4))
                                for i, (k, v) in enumerate(metrics_to_show.items()):
                                    with cols[i % len(cols)]:
                                        display_val = f"{v:.4f}" if isinstance(v, float) else str(v)
                                        st.metric(k.replace("_", " ").title(), display_val)

            with perf_tabs[1]:
                st.markdown("### Feature Importance Plots")
                for model_key in ["flight_price", "gender", "churn"]:
                    img_path = REPORTS_DIR / f"{model_key}_feature_importance.png"
                    if img_path.exists():
                        st.markdown(f"#### {model_key.replace('_', ' ').title()}")
                        st.image(str(img_path), use_column_width=True)
                    else:
                        st.info(f"No feature importance chart found for {model_key}")

            with perf_tabs[2]:
                st.markdown("### Confusion Matrices")
                for model_key in ["gender", "churn"]:
                    img_path = REPORTS_DIR / f"{model_key}_confusion_matrix.png"
                    if img_path.exists():
                        st.markdown(f"#### {model_key.replace('_', ' ').title()}")
                        st.image(str(img_path), use_column_width=True)
                    else:
                        st.info(f"No confusion matrix chart found for {model_key}")

            with perf_tabs[3]:
                st.markdown("### Predicted vs Actual")
                img_path = REPORTS_DIR / "flight_price_actual_vs_predicted.png"
                if img_path.exists():
                    st.image(str(img_path), use_column_width=True)
                else:
                    st.info("No predicted vs actual chart found for flight_price")


# =============================================================
#  PAGE 7 — CHURN ANALYSIS
# =============================================================
elif page == "🔄 Churn Analysis":
    section_header("🔄", "Churn Analysis")

    st.markdown(
        '<div class="info-box">'
        "📊 <b>Churn Analysis</b> identifies users who may become inactive. "
        "We use RFM (Recency, Frequency, Monetary) segmentation to classify user engagement levels."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    # ── Build RFM ─────────────────────────────────────────────
    last_date = flights["date"].max()

    # Recency: days since last flight
    recency = flights.groupby("userCode")["date"].max().reset_index()
    recency.columns = ["code", "last_flight"]
    recency["recency"] = (last_date - recency["last_flight"]).dt.days

    # Frequency: number of bookings (flights + hotels)
    freq_flights = flights.groupby("userCode").size().reset_index(name="f_flights")
    freq_hotels = hotels.groupby("userCode").size().reset_index(name="f_hotels")
    freq = freq_flights.rename(columns={"userCode": "code"}).merge(
        freq_hotels.rename(columns={"userCode": "code"}), on="code", how="outer"
    )
    freq["frequency"] = freq["f_flights"].fillna(0) + freq["f_hotels"].fillna(0)

    # Monetary: total spend
    mon_flights = flights.groupby("userCode")["price"].sum().reset_index()
    mon_flights.columns = ["code", "m_flights"]
    mon_hotels = hotels.groupby("userCode")["total"].sum().reset_index()
    mon_hotels.columns = ["code", "m_hotels"]
    monetary = mon_flights.merge(mon_hotels, on="code", how="outer")
    monetary["monetary"] = monetary["m_flights"].fillna(0) + monetary["m_hotels"].fillna(0)

    # Merge RFM
    rfm = users[["code", "gender", "age"]].merge(recency[["code", "recency"]], on="code", how="left")
    rfm = rfm.merge(freq[["code", "frequency"]], on="code", how="left")
    rfm = rfm.merge(monetary[["code", "monetary"]], on="code", how="left")
    rfm = rfm.fillna(0)

    # Churn definition: high recency = likely churned
    recency_threshold = rfm["recency"].quantile(0.7)
    freq_threshold = rfm["frequency"].quantile(0.3)
    rfm["churned"] = ((rfm["recency"] > recency_threshold) & (rfm["frequency"] < freq_threshold)).astype(int)
    churn_rate = rfm["churned"].mean() * 100

    # ── Gauge Chart ───────────────────────────────────────────
    col_gauge, col_metrics = st.columns([1, 1])

    with col_gauge:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=churn_rate,
            number=dict(suffix="%", font=dict(size=52, color=CORAL if churn_rate > 30 else GOLD if churn_rate > 15 else TEAL)),
            title=dict(text="Overall Churn Rate", font=dict(size=20, color=SOFT_WHITE)),
            gauge=dict(
                axis=dict(range=[0, 50], tickcolor=SOFT_WHITE),
                bar=dict(color=CORAL if churn_rate > 30 else GOLD if churn_rate > 15 else TEAL),
                bgcolor=LIGHT_NAVY,
                bordercolor="rgba(0,212,170,0.2)",
                steps=[
                    dict(range=[0, 15], color="rgba(0,212,170,0.12)"),
                    dict(range=[15, 30], color="rgba(245,166,35,0.12)"),
                    dict(range=[30, 50], color="rgba(255,107,107,0.12)"),
                ],
            ),
        ))
        apply_chart_layout(fig, "", 380)
        chart_container(fig)

    with col_metrics:
        st.markdown("")
        st.markdown("")
        mc1, mc2 = st.columns(2)
        mc1.metric("Total Users", f"{len(rfm):,}")
        mc2.metric("Churned Users", f"{rfm['churned'].sum():,}")
        mc3, mc4 = st.columns(2)
        mc3.metric("Active Users", f"{(rfm['churned'] == 0).sum():,}")
        mc4.metric("Churn Rate", f"{churn_rate:.1f}%")

    st.markdown("")

    # ── Churn by Demographics ─────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        # Churn by age group
        rfm["age_group"] = pd.cut(
            rfm["age"], bins=[0, 25, 35, 45, 55, 65, 100],
            labels=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"],
        )
        churn_age = rfm.groupby("age_group", observed=True)["churned"].mean().reset_index()
        churn_age["churn_pct"] = churn_age["churned"] * 100
        fig = px.bar(
            churn_age, x="age_group", y="churn_pct",
            color="churn_pct", color_continuous_scale=[TEAL, GOLD, CORAL],
        )
        apply_chart_layout(fig, "Churn Rate by Age Group", 400)
        fig.update_xaxes(title_text="Age Group")
        fig.update_yaxes(title_text="Churn Rate (%)")
        chart_container(fig)

    with col2:
        # Churn by gender
        churn_gender = rfm.groupby("gender")["churned"].agg(["sum", "count"]).reset_index()
        churn_gender["churn_pct"] = churn_gender["sum"] / churn_gender["count"] * 100
        churn_gender["active"] = churn_gender["count"] - churn_gender["sum"]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=churn_gender["gender"], y=churn_gender["sum"],
            name="Churned", marker_color=CORAL,
        ))
        fig.add_trace(go.Bar(
            x=churn_gender["gender"], y=churn_gender["active"],
            name="Active", marker_color=TEAL,
        ))
        fig.update_layout(barmode="stack")
        apply_chart_layout(fig, "Churn by Gender", 400)
        fig.update_xaxes(title_text="Gender")
        fig.update_yaxes(title_text="Count")
        chart_container(fig)

    # ── Churn by Travel Behavior ──────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        fig = px.box(
            rfm, x="churned", y="frequency",
            color="churned", color_discrete_sequence=[TEAL, CORAL],
            labels={"churned": "Churned (0=No, 1=Yes)"},
        )
        apply_chart_layout(fig, "Booking Frequency: Churned vs Active", 400)
        fig.update_xaxes(title_text="Churned")
        fig.update_yaxes(title_text="Total Bookings")
        chart_container(fig)

    with col4:
        fig = px.box(
            rfm, x="churned", y="monetary",
            color="churned", color_discrete_sequence=[TEAL, CORAL],
            labels={"churned": "Churned (0=No, 1=Yes)"},
        )
        apply_chart_layout(fig, "Spending: Churned vs Active", 400)
        fig.update_xaxes(title_text="Churned")
        fig.update_yaxes(title_text="Total Spend ($)")
        chart_container(fig)

    # ── 3D RFM Scatter ────────────────────────────────────────
    st.markdown("### 🧊 RFM Segmentation — 3D View")
    fig = px.scatter_3d(
        rfm, x="recency", y="frequency", z="monetary",
        color=rfm["churned"].map({0: "Active", 1: "Churned"}),
        opacity=0.6, color_discrete_sequence=[TEAL, CORAL],
        labels={"color": "Status"},
    )
    apply_chart_layout(fig, "", 600)
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="Recency (days)", backgroundcolor="rgba(10,22,40,0.8)", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Frequency", backgroundcolor="rgba(10,22,40,0.8)", gridcolor="rgba(255,255,255,0.05)"),
            zaxis=dict(title="Monetary ($)", backgroundcolor="rgba(10,22,40,0.8)", gridcolor="rgba(255,255,255,0.05)"),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    chart_container(fig)

    # ── Churn Risk Distribution ───────────────────────────────
    rfm["risk_score"] = (
        rfm["recency"] / rfm["recency"].max() * 40
        + (1 - rfm["frequency"] / rfm["frequency"].max()) * 30
        + (1 - rfm["monetary"] / rfm["monetary"].max()) * 30
    )
    fig = px.histogram(
        rfm, x="risk_score", nbins=40,
        color_discrete_sequence=[TEAL], marginal="box",
    )
    apply_chart_layout(fig, "Churn Risk Score Distribution", 420)
    fig.update_xaxes(title_text="Risk Score (0 = Low, 100 = High)")
    fig.update_yaxes(title_text="Number of Users")
    chart_container(fig)
