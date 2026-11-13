"""Overview page: Complete Green + Black redesign matching Master Prompt Section 13."""

from datetime import datetime, time
from zoneinfo import ZoneInfo
import math
import plotly.graph_objects as go
import streamlit as st

from config import (
    ALL_ZONES,
    COLOR_BORDER,
    COLOR_CARD,
    COLOR_CLOSED,
    COLOR_DANGER,
    COLOR_MUTED_GREEN,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_TEXT_MAIN,
    COLOR_TEXT_MUTED,
    COLOR_WARNING,
    SUB_ZONES,
    ZONE_NAMES,
)
from src.forecast import day_curve, free_seats, peak_hours, predict, status
from ui import PLOT_CONFIG, note, section, style_fig, zone_card

KTH_TZ = ZoneInfo("Europe/Stockholm")


def render_planner(bundle: dict):
    """
    Render Left Panel (60% width on desktop) of the Overview Page:
    Comprehensive spatial telemetry, zone comparison cards, and hourly trends.
    """
    now = datetime.now(KTH_TZ)

    st.markdown(
        f"""
        <div style="margin-bottom: 1.25rem;">
          <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.25rem;">
            <span style="background: rgba(0,208,132,0.15); color: var(--green); font-size: 0.78rem; font-weight: 800; padding: 2px 8px; border-radius: 9999px; text-transform: uppercase; border: 1px solid var(--green);">
              Spatial Telemetry
            </span>
            <span style="font-size: 0.82rem; color: var(--green);">KTH Main Campus Library (KTHB)</span>
          </div>
          <h1 style="font-size: 2.1rem; font-weight: 800; color: var(--ink); margin: 0 0 0.35rem 0; letter-spacing: -0.025em; line-height: 1.25;">
            Library Overview & Spatial Intelligence
          </h1>
          <p style="font-size: 1rem; color: var(--muted); margin: 0; line-height: 1.55;">
            Compare model-estimated occupancy across all 5 study wings. Review historical patterns and forecast limitations.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Temporal Controls Bar
    with st.container():
        date_col, time_col, exam_col = st.columns([1.3, 1.3, 1.1], gap="small", vertical_alignment="bottom")
        with date_col:
            selected_date = st.date_input("Compare Date", value=now.date(), key="overview_date")
        with time_col:
            time_options = [time(h, 0) for h in range(24)]
            default_time_idx = min(max(now.hour, 0), 23)
            selected_time = st.selectbox(
                "Compare Time",
                options=time_options,
                index=default_time_idx,
                format_func=lambda t: f"{t:%H}:00",
                key="overview_time_select",
            )
        with exam_col:
            exam_period = st.toggle("Use exam-period model profile", value=False, key="overview_exam_period", help="Sets the model's exam-period feature. It does not add a fixed 18% occupancy adjustment.")

    selected_dt = datetime.combine(selected_date, selected_time, tzinfo=KTH_TZ)

    # Calculate overall library & per-zone values
    rows = []
    curves = {}
    for zone in ALL_ZONES:
        occupancy = predict(bundle, selected_dt, zone, exam=exam_period)
        zone_status = status(occupancy)
        seats = free_seats(occupancy, zone, bundle)
        rows.append({
            "zone": zone,
            "name": ZONE_NAMES[zone],
            "occupancy": occupancy,
            "status": zone_status,
            "free_seats": seats,
        })
        curves[zone] = day_curve(bundle, selected_date, zone, exam=exam_period)

    lib_row = next((r for r in rows if r["zone"] == "library"), rows[0])
    lib_occ = (
        float(lib_row["occupancy"])
        if lib_row["occupancy"] is not None and math.isfinite(float(lib_row["occupancy"]))
        else None
    )
    lib_seats = lib_row["free_seats"]
    lib_occ_display = f"{lib_occ:.0f}%" if lib_occ is not None else "Unavailable"
    lib_seats_display = f"~{lib_seats}" if lib_occ is not None else "—"
    note(
        "Values on this page are model forecasts from historical data, not live sensor measurements. "
        "Zone capacities are assumptions used only to estimate available seats.",
        "info",
    )

    # 2. Executive Summary Metrics Strip
    m_cols = st.columns(3, gap="small")
    with m_cols[0]:
        st.markdown(
            f"""
            <div class="saas-card" style="padding: 1.1rem;">
              <div style="font-size: 0.78rem; color: var(--green); text-transform: uppercase; font-weight: 700;">Overall Occupancy</div>
              <div style="font-size: 1.85rem; font-weight: 800; color: var(--green); margin: 0.2rem 0;">{lib_occ_display}</div>
              <div style="font-size: 0.82rem; color: var(--muted);">Whole Building Aggregate</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_cols[1]:
        st.markdown(
            f"""
            <div class="saas-card" style="padding: 1.1rem;">
              <div style="font-size: 0.78rem; color: var(--green); text-transform: uppercase; font-weight: 700;">Available Desks</div>
              <div style="font-size: 1.85rem; font-weight: 800; color: var(--ink); margin: 0.2rem 0;">{lib_seats_display}</div>
              <div style="font-size: 0.82rem; color: var(--muted);">Of 420 Assumed Seats</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_cols[2]:
        crowd_color = "#718078" if lib_occ is None else ("var(--green)" if lib_occ < 50 else ("#F5B942" if lib_occ <= 80 else "#FF6464"))
        st.markdown(
            f"""
            <div class="saas-card" style="padding: 1.1rem;">
              <div style="font-size: 0.78rem; color: var(--green); text-transform: uppercase; font-weight: 700;">Current Crowd Status</div>
              <div style="font-size: 1.85rem; font-weight: 800; color: {crowd_color}; margin: 0.2rem 0;">{lib_row['status']}</div>
              <div style="font-size: 0.82rem; color: var(--muted);">Model estimate · {selected_time:%H:%M} Stockholm</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 3. Quiet Sanctuary Callout Banner
    open_sub_rows = [r for r in rows if r["zone"] != "library" and r["occupancy"] is not None and not math.isnan(r["occupancy"])]
    if open_sub_rows:
        quietest = min(open_sub_rows, key=lambda r: r["occupancy"])
        note(
            f"🎯 <strong>Optimal Study Sanctuary:</strong> At <strong>{selected_time:%H:%M}</strong>, "
            f"<strong>{quietest['name']}</strong> is the quietest zone at <strong>{quietest['occupancy']:.0f}% occupancy</strong> "
            f"(~{quietest['free_seats']} seats available).",
            "success",
        )
    else:
        note("The library is currently forecast closed at this hour.", "warning")

    # 4. Availability Across the 5 Wings
    section("Availability Across Study Wings", f"Predicted demand at {selected_time:%H:%M} · {selected_date:%A, %b %d}")
    sub_cards = st.columns(len(SUB_ZONES), gap="small")
    sub_rows = [r for r in rows if r["zone"] in SUB_ZONES]
    for idx, r in enumerate(sub_rows):
        with sub_cards[idx]:
            zone_card(r["name"], r["occupancy"], r["status"], r["free_seats"], r["zone"])

    # 5. Hourly Comparison Chart (Clean Green + Black Plotly)
    section("Hourly Demand Forecast (00:00 – 23:00)", "Temporal trajectory across all study wings throughout the selected day")

    fig = go.Figure()
    palette = {
        "library": "#1D6949",
        "angdomen": "#25845C",
        "south_east": "#1D6949",
        "newton": "#F5B942",
        "south": "#5A6860",
        "north": "#FF6464",
    }

    for row in rows:
        z = row["zone"]
        curve = curves[z]
        fig.add_trace(go.Scatter(
            x=curve["hour"],
            y=curve["occupancy"],
            mode="lines",
            name=row["name"],
            line=dict(color=palette.get(z, "#1D6949"), width=3.5 if z == "library" else 2),
            hovertemplate=f"<b>{row['name']}</b><br>%{{x:02d}}:00 · %{{y:.1f}}%<extra></extra>",
        ))

    fig.add_vline(x=selected_time.hour, line_dash="dash", line_color="#1D6949", opacity=0.8)
    fig.update_layout(
        height=320,
        xaxis=dict(title="Hour of Day (Stockholm Time)", tickmode="linear", tick0=0, dtick=2, range=[0, 23]),
        yaxis=dict(title="Estimated Occupancy (%)", range=[0, 100]),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    style_fig(fig, height=320)
    st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)
