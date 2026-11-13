"""Dataset and model transparency page for students and reviewers.
Master Prompt Section 7 & 15: Clean Green + Black styling & transparent ML telemetry.
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
import streamlit as st

from config import ALL_ZONES, ZONE_NAMES
from src.live_data import fetch_latest_occupancy
from src.data_quality import is_fresh_timestamp
from ui import note, section

HF_DATASET_URL = "https://huggingface.co/datasets/davnas/real-time-library-occupancy"


def render_dataset(df_history: pd.DataFrame, metrics: dict | None):
    """
    Render Left Panel (60% width on desktop) of the Dataset Page:
    Real data sources, freshness, feature engineering schema, and measured model error.
    """
    st.markdown(
        """
        <div style="margin-bottom: 1.25rem;">
          <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.25rem;">
            <span style="background: rgba(0,208,132,0.15); color: var(--green); font-size: 0.78rem; font-weight: 800; padding: 2px 8px; border-radius: 9999px; text-transform: uppercase; border: 1px solid var(--green);">
              Data Provenance & ML Audit
            </span>
            <span style="font-size: 0.8rem; color: var(--green);">Historical Dataset & Model Evaluation</span>
          </div>
          <h1 style="font-size: 2.1rem; font-weight: 800; color: var(--ink); margin: 0 0 0.35rem 0; letter-spacing: -0.025em; line-height: 1.25;">
            Dataset & Predictive Model Intelligence
          </h1>
          <p style="font-size: 1rem; color: var(--muted); margin: 0; line-height: 1.55;">
            See where occupancy information originates, how features are engineered, and audit chronological holdout model error against historical baselines.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Archived Dataset Freshness
    source_row, source_error = fetch_latest_occupancy()
    now = datetime.now(ZoneInfo("Europe/Stockholm"))
    if source_row:
        observed_at = source_row["observed_at"]
        age = now - observed_at
        fresh = is_fresh_timestamp(observed_at, now)
        status_label = "Recent dataset observation" if fresh else "Historical archive · stale for live occupancy"
        status_level = "success" if fresh else "warning"
        observed_label = observed_at.strftime("%d %b %Y, %H:%M Stockholm")
    else:
        status_label = "Source currently unavailable"
        status_level = "warning"
        observed_label = "No recent observation received"

    period_label = "Not available"
    row_count = 0
    if df_history is not None and not df_history.empty:
        row_count = len(df_history)
        period_label = f"{df_history.index.min():%d %b %Y} – {df_history.index.max():%d %b %Y}"

    # Metric Cards
    m1, m2, m3 = st.columns(3, gap="small")
    with m1:
        st.html(
            f"""
            <div class="saas-card" style="padding: 1.1rem; margin-bottom: 0.5rem;">
              <div style="font-size: 0.72rem; color: var(--green); text-transform: uppercase; font-weight: 700;">Processed Hourly Records</div>
              <div style="font-size: 1.8rem; font-weight: 800; color: var(--green); margin: 0.2rem 0;">{row_count:,}</div>
              <div style="font-size: 0.78rem; color: var(--muted);">Cleaned Time-Series</div>
            </div>
            """
        )
    with m2:
        st.html(
            f"""
            <div class="saas-card" style="padding: 1.1rem; margin-bottom: 0.5rem;">
              <div style="font-size: 0.72rem; color: var(--green); text-transform: uppercase; font-weight: 700;">Historical Period</div>
              <div style="font-size: 1.15rem; font-weight: 800; color: var(--ink); margin: 0.35rem 0;">{period_label}</div>
              <div style="font-size: 0.78rem; color: var(--muted);">Training Window</div>
            </div>
            """
        )
    with m3:
        st.html(
            f"""
            <div class="saas-card" style="padding: 1.1rem; margin-bottom: 0.5rem;">
              <div style="font-size: 0.72rem; color: var(--green); text-transform: uppercase; font-weight: 700;">Latest Observation</div>
              <div style="font-size: 1.15rem; font-weight: 800; color: var(--green-hover); margin: 0.35rem 0;">{observed_label}</div>
              <div style="font-size: 0.78rem; color: var(--muted);">Archived dataset source</div>
            </div>
            """
        )

    note(status_label + (f" · {source_error}" if source_error and not source_row else ""), status_level)

    # Action Links
    b_col1, b_col2, b_col3 = st.columns([1.6, 1.4, 1.2], gap="small")
    with b_col1:
        st.link_button("🌐 Hugging Face Occupancy Dataset", HF_DATASET_URL, type="primary", use_container_width=True)
    with b_col2:
        st.link_button("📚 Open Library API Docs", "https://openlibrary.org/developers/api", type="secondary", use_container_width=True)
    with b_col3:
        st.link_button("🏛️ KTH Primo Catalog", "https://kth-ch.primo.exlibrisgroup.com/", type="secondary", use_container_width=True)

    # 2. What Data The App Uses
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    section("Feature Engineering & Architecture", "How the predictive intelligence model is constructed")
    st.markdown(
        """
        <div class="saas-card" style="padding: 1.25rem;">
          <ul style="margin: 0; padding-left: 1.25rem; color: var(--ink); font-size: 0.95rem; line-height: 1.65;">
            <li style="margin-bottom: 0.5rem;"><strong style="color: var(--green);">Historical Occupancy Dataset:</strong> Timestamped Library occupancy records distributed through Hugging Face; source freshness and sensor provenance are not independently verified. The retrieved source is an archive and is not treated as a verified live sensor feed. Forecast values come from the saved model.</li>
            <li style="margin-bottom: 0.5rem;"><strong style="color: var(--green);">Calendar & Seasonal Features:</strong> Local hour, day of week, Swedish public holiday flags, weekend indicators, and academic exam crunch flags drive the gradient boosted trees.</li>
            <li style="margin-bottom: 0.5rem;"><strong style="color: var(--green);">Reading & Literature Demand:</strong> Powered by Open Library public search. Reader want-to-read numbers provide global bibliographic popularity signals, Open Library metadata is bibliographic discovery information and does not establish KTH holdings.</li>
            <li><strong style="color: var(--green);">Spatial Capacity Estimates:</strong> Assumed zone seat capacities convert percentage occupancy into estimated available study desks.</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3. Measured Model Performance & Error Analysis
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    section("Model Performance & MAE Benchmarks", "MAE is the average absolute occupancy error in percentage points (lower is better)")

    if not metrics or not metrics.get("zones"):
        note("Evaluation metrics are missing. Run python src/train.py to generate the backtest report.", "warning")
        return

    summary = metrics.get("summary", {})
    xgb_mae = summary.get("avg_xgb_mae")
    baseline_mae = summary.get("avg_baseline_mae")
    test_samples = summary.get("test_samples")
    if not isinstance(xgb_mae, (int, float)) or not isinstance(baseline_mae, (int, float)):
        note("Model evaluation metrics are incomplete. Retrain and regenerate the evaluation report.", "warning")
        return

    if isinstance(test_samples, (int, float)) and test_samples < 30:
        note(
            f"The saved evaluation contains only {int(test_samples)} held-out samples. "
            "This is too small for a reliable performance claim; treat forecasts as low-confidence.",
            "warning",
        )
    if summary.get("overall_beats_baseline") is False:
        note(
            "The saved XGBoost evaluation is worse than the historical baseline. "
            "The current saved artifact has not yet been retrained with the v7 model-selection pipeline.",
            "warning",
        )

    mae_c1, mae_c2, mae_c3 = st.columns(3, gap="small")
    with mae_c1:
        st.html(
            f"""
            <div class="saas-card" style="padding: 1rem;">
              <div style="font-size: 0.72rem; color: var(--green); text-transform: uppercase; font-weight: 700;">Forecast Model MAE</div>
              <div style="font-size: 1.8rem; font-weight: 800; color: var(--green); margin: 0.2rem 0;">{xgb_mae:.2f} pp</div>
              <div style="font-size: 0.78rem; color: var(--muted);">Average Absolute Error</div>
            </div>
            """
        )
    with mae_c2:
        st.html(
            f"""
            <div class="saas-card" style="padding: 1rem;">
              <div style="font-size: 0.72rem; color: var(--green); text-transform: uppercase; font-weight: 700;">Historical Baseline MAE</div>
              <div style="font-size: 1.8rem; font-weight: 800; color: #F5B942; margin: 0.2rem 0;">{baseline_mae:.2f} pp</div>
              <div style="font-size: 0.78rem; color: var(--muted);">Day-Hour Mean Baseline</div>
            </div>
            """
        )
    with mae_c3:
        st.html(
            f"""
            <div class="saas-card" style="padding: 1rem;">
              <div style="font-size: 0.72rem; color: var(--green); text-transform: uppercase; font-weight: 700;">Held-Out Test Samples</div>
              <div style="font-size: 1.8rem; font-weight: 800; color: var(--ink); margin: 0.2rem 0;">{test_samples:,}</div>
              <div style="font-size: 0.78rem; color: var(--muted);">Chronological Holdout</div>
            </div>
            """
        )

    if xgb_mae is not None and baseline_mae is not None:
        if xgb_mae <= baseline_mae:
            note("The trained XGBoost model beats or matches the historical baseline in the chronological evaluation holdout.", "success")
        else:
            note("The trained model has slightly higher error than the baseline; treat forecasts as planning estimates.", "warning")

    # Zone-level MAE Table
    zone_rows = []
    for zone in ALL_ZONES:
        result = metrics["zones"].get(zone)
        if not result:
            continue
        zone_rows.append({
            "Library Area": result.get("display_name", ZONE_NAMES.get(zone, zone)),
            "Model MAE (pp)": f"{result.get('xgb_mae', 0):.2f}",
            "Baseline MAE (pp)": f"{result.get('baseline_mae', 0):.2f}",
            "Model Beats Baseline": "✅ Yes" if result.get("beats_baseline") else "⚠️ No",
        })

    if zone_rows:
        st.dataframe(
            pd.DataFrame(zone_rows).set_index("Library Area"),
            use_container_width=True,
        )

    with st.expander("Evaluation notes and telemetry limitations"):
        st.markdown(
            f"""
            - The report evaluates the trained regressor on a strict chronological split of **{test_samples}** observations.
            - Forecasts are zone-level and aggregate. The system does not track individuals or reserve physical chairs.
            - Real-world demand may vary due to campus events, course schedules, or sudden weather changes not reflected in calendar features.
            """
        )
