"""Home page: Student Study Planner Form matching Master Prompt Section 3 & 4.
Collects visit time, planned duration (dropdown), preferred environment (radio), and preferences.
"""

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo
import streamlit as st

from config import (
    COLOR_BORDER,
    COLOR_CARD,
    COLOR_ELEVATED_CARD,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_TEXT_MAIN,
    COLOR_TEXT_MUTED,
)
from src.live_data import get_data_status_info
from ui import note, section

KTH_TZ = ZoneInfo("Europe/Stockholm")


def render_home(bundle: dict):
    """
    Render Left Panel (50% width on desktop) of the Home Page:
    The Student Study Planner Form with dropdown duration and radio environment.
    """
    now = datetime.now(KTH_TZ)
    data_info = get_data_status_info()

    # Header with Data Freshness Badge
    st.markdown(
        f"""
        <div style="margin-bottom: 1.15rem;">
          <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.35rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
              <span style="background: rgba(0,208,132,0.18); color: var(--green); font-size: 0.78rem; font-weight: 800; padding: 3px 9px; border-radius: 9999px; text-transform: uppercase; border: 1px solid var(--green);">
                AI Study Planner
              </span>
              <span style="font-size: 0.85rem; color: var(--green);">KTH Main Campus Library</span>
            </div>
            <span style="background: var(--surface); border: 1px solid var(--line); color: var(--green); font-size: 0.76rem; padding: 3px 8px; border-radius: 6px;">
              ⏱️ Data last updated: <strong>{data_info['last_updated']}</strong>
            </span>
          </div>
          <h1 style="font-size: 2.1rem; font-weight: 800; color: var(--ink); margin: 0 0 0.35rem 0; letter-spacing: -0.025em; line-height: 1.25;">
            Plan Your Perfect Study Session
          </h1>
          <p style="font-size: 1rem; color: var(--muted); margin: 0; line-height: 1.55;">
            Tell us when you're arriving and your study preferences. Our AI will analyze predicted occupancy and available desks across all 5 wings to recommend your best workspace.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Temporal Parameters (Date & Arrival Time)
    with st.container():
        st.markdown(
            """
            <div style="font-size: 0.95rem; font-weight: 700; color: var(--green); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.35rem;">
              📅 1. When Are You Coming?
            </div>
            """,
            unsafe_allow_html=True,
        )
        date_col, time_col = st.columns([1.2, 1.2], gap="medium")
        with date_col:
            visit_date = st.date_input("Visit Date *", value=now.date(), key="planner_visit_date")
        with time_col:
            time_hours = list(range(8, 23))
            current_hour = min(max(now.hour, 8), 22)
            default_hour_idx = time_hours.index(current_hour) if current_hour in time_hours else 5
            arr_hour = st.selectbox(
                "Arrival Time *",
                options=time_hours,
                index=default_hour_idx,
                format_func=lambda h: f"{h:02d}:00 ({'Morning' if h < 12 else ('Afternoon' if h < 17 else 'Evening')})",
                key="planner_arrival_time",
            )

    # 2. Planned Study Duration
    st.markdown(
        """
        <div style="font-size: 0.92rem; font-weight: 700; color: var(--green); text-transform: uppercase; letter-spacing: 0.05em; margin: 0.45rem 0 0.25rem;">
          ⏱️ 2. Planned Study Duration *
        </div>
        """,
        unsafe_allow_html=True,
    )
    duration_options = [
        "1 hour",
        "2 hours",
        "3 hours",
        "4 hours",
        "5 hours",
        "6 hours",
        "7+ hours",
    ]
    duration_choice = st.selectbox(
        "Planned study hours",
        options=duration_options,
        index=1,
        label_visibility="collapsed",
        key="planner_duration_select",
        help="Select your total planned stay duration",
    )
    duration_num = int(duration_choice[0]) if duration_choice[0].isdigit() else 7

    # 3. Preferred Environment
    st.markdown(
        """
        <div style="font-size: 0.92rem; font-weight: 700; color: var(--green); text-transform: uppercase; letter-spacing: 0.05em; margin: 0.45rem 0 0.25rem;">
          🎧 3. Preferred Study Environment *
        </div>
        """,
        unsafe_allow_html=True,
    )
    env_choice = st.radio(
        "Preferred study environment",
        options=[
            "Quiet environment",
            "Moderate activity",
            "Collaborative environment",
            "No preference",
        ],
        index=0,
        label_visibility="collapsed",
        key="planner_env_radio",
    )

    # 4. Seating Preference & Study Purpose
    pref_col1, pref_col2 = st.columns(2, gap="medium")
    with pref_col1:
        st.markdown(
            """
            <div style="font-size: 0.92rem; font-weight: 700; color: var(--green); text-transform: uppercase; letter-spacing: 0.05em; margin: 0.35rem 0 0.25rem;">
              🪑 4. Seating Preference *
            </div>
            """,
            unsafe_allow_html=True,
        )
        seating_choice = st.selectbox(
            "Seating preference",
            options=[
                "More available space",
                "Individual seating",
                "Group seating",
                "Near windows",
                "No preference",
            ],
            index=0,
            label_visibility="collapsed",
            key="planner_seating_choice",
        )
    with pref_col2:
        st.markdown(
            """
            <div style="font-size: 0.92rem; font-weight: 700; color: var(--green); text-transform: uppercase; letter-spacing: 0.05em; margin: 0.35rem 0 0.25rem;">
              🎯 5. Study Purpose *
            </div>
            """,
            unsafe_allow_html=True,
        )
        purpose_choice = st.selectbox(
            "Study purpose",
            options=[
                "Exam preparation",
                "Assignment work",
                "Reading",
                "Research",
                "Group project",
                "General studying",
            ],
            index=0,
            label_visibility="collapsed",
            key="planner_purpose_choice",
        )

    # 5. Crowd Tolerance & Additional Requirements
    st.markdown(
        """
        <div style="font-size: 0.92rem; font-weight: 700; color: var(--green); text-transform: uppercase; letter-spacing: 0.05em; margin: 0.45rem 0 0.25rem;">
          👥 6. Crowd Tolerance *
        </div>
        """,
        unsafe_allow_html=True,
    )
    crowd_choice = st.selectbox(
        "Crowd preference",
        options=[
            "Avoid crowded areas",
            "Moderate occupancy is fine",
            "Any occupancy level",
        ],
        index=0,
        label_visibility="collapsed",
        key="planner_crowd_choice",
    )

    st.markdown(
        """
        <div style="font-size: 0.92rem; font-weight: 700; color: var(--green); text-transform: uppercase; letter-spacing: 0.05em; margin: 0.45rem 0 0.25rem;">
          📝 7. Additional Requirements (Optional)
        </div>
        """,
        unsafe_allow_html=True,
    )
    extra_notes = st.text_area(
        "Additional requirements",
        placeholder="Anything else we should consider? (e.g. need dual monitors, power outlets, quietest desk...)",
        label_visibility="collapsed",
        key="planner_notes",
        height=70,
    )

    # Preferences Summary Box
    summary_chips = [
        f"📅 {visit_date:%a, %b %d}",
        f"🕒 {arr_hour:02d}:00 ({duration_choice})",
        f"🎧 {env_choice}",
        f"🪑 {seating_choice}",
        f"🎯 {purpose_choice}",
    ]
    chips_html = "".join(
        [
            f'<span style="background: var(--surface); border: 1px solid var(--line); color: var(--ink); font-size: 0.85rem; '
            f'padding: 4px 10px; border-radius: 6px; font-weight: 600;">{c}</span>'
            for c in summary_chips
        ]
    )
    st.markdown(
        f"""
        <div style="background: var(--paper); border: 1px dashed var(--line); border-radius: 8px; padding: 0.75rem 1rem; margin: 0.85rem 0; display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center;">
          <span style="font-size: 0.82rem; color: var(--green); font-weight: 700; text-transform: uppercase; margin-right: 0.2rem;">Summary:</span>
          {chips_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Prominent Green Submission Button
    submitted = st.button("🚀 Find My Best Study Spot", key="btn_find_best_study_spot", type="primary", use_container_width=True)

    if submitted:
        # Validate and package payload
        form_payload = {
            "date": visit_date,
            "arrival_hour": arr_hour,
            "duration": duration_num,
            "duration_str": duration_choice,
            "environment": env_choice,
            "seating": seating_choice,
            "purpose": purpose_choice,
            "crowd": crowd_choice,
            "notes": extra_notes,
            "exam": "Exam" in purpose_choice,
        }
        # Sets pending submission for chatbot to process WITHOUT clearing history
        st.session_state.pending_planner_data = form_payload
        st.session_state.mascot_state = "thinking"
        st.rerun()
