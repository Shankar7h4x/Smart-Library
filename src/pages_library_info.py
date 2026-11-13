"""Evidence-aware KTH library guide with explicitly labelled model assumptions."""
import pandas as pd
import streamlit as st

from config import SUB_ZONES, ZONE_CAPACITIES, ZONE_NAMES

# Approximate campus map marker used only for orientation; verify in official KTH maps.
KTHB_LATITUDE = 59.3479
KTHB_LONGITUDE = 18.0728
OFFICIAL_LIBRARY_URL = "https://www.kth.se/en/biblioteket"


def render_library_info():
    st.markdown(
        """
        <div style="margin-bottom: 1.15rem;">
          <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.25rem;">
            <span style="background: rgba(0,208,132,0.15); color: var(--green); font-size: 0.78rem; font-weight: 800; padding: 2px 8px; border-radius: 9999px; text-transform: uppercase; border: 1px solid var(--green);">
              Campus Orientation
            </span>
            <span style="font-size: 0.82rem; color: var(--green);">KTH Royal Institute of Technology</span>
          </div>
          <h1 style="font-size: 2.1rem; font-weight: 800; color: var(--ink); margin: 0 0 0.35rem 0; letter-spacing: -0.025em; line-height: 1.25;">
            Library Guide & Study Spaces
          </h1>
          <p style="font-size: 1rem; color: var(--muted); margin: 0; line-height: 1.55;">
            Explore study zones, environment profiles, installed facilities, and orientation details for KTH Main Campus Library.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    hero, details = st.columns([1.35, 1], gap="large")
    with hero:
        st.markdown(
            """
            <div class="saas-card" style="height:100%">
              <div class="page-kicker">KTH · Stockholm</div>
              <h2 style="margin:.25rem 0 .6rem">KTH Library</h2>
              <p style="color:var(--muted);line-height:1.65">
                Use this page to understand the study-zone labels used by Smart Library.
                For current access, services, facilities and opening hours, consult KTH's
                official library information before making a trip.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with details:
        st.markdown(
            """
            <div class="saas-card" style="height:100%">
              <div class="saas-section-title">Official information</div>
              <p style="color:var(--muted);font-size:.9rem">
                Opening hours, accessibility, services and room availability can change.
                This application does not currently verify those details live.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.link_button("Open KTH Library website ↗", OFFICIAL_LIBRARY_URL, use_container_width=True)

    st.markdown("### Study zones used by the model")
    st.caption("The following names are model zones. Descriptions indicate intended preference profiles only; exact room locations and installed facilities are not verified.")
    cards = st.columns(2, gap="medium")
    zone_profiles = {
        "angdomen": ("Quiet-study preference", "The model profile gives this zone a stronger quiet-study preference score."),
        "north": ("Quiet-study preference", "The model profile gives this zone a stronger quiet-study preference score."),
        "south": ("Moderate activity", "The model profile treats this zone as a moderate-activity option."),
        "south_east": ("Flexible study preference", "The model profile treats this zone as a flexible option, including window-seat preference."),
        "newton": ("Collaborative preference", "The model profile gives this zone a stronger collaborative-study preference score."),
    }
    for index, zone in enumerate(SUB_ZONES):
        label, description = zone_profiles.get(zone, ("Study zone", "No preference profile is configured."))
        cap = ZONE_CAPACITIES.get(zone)
        with cards[index % 2]:
            st.markdown(
                f"""
                <div class="saas-card" style="margin-bottom:.8rem">
                  <div style="display:flex;justify-content:space-between;gap:.8rem;align-items:center">
                    <strong>{ZONE_NAMES.get(zone, zone)}</strong>
                    <span class="status-pill">Model zone</span>
                  </div>
                  <div style="color:var(--green);font-size:.78rem;font-weight:700;margin:.45rem 0">{label}</div>
                  <p style="color:var(--muted);font-size:.88rem;margin:.2rem 0 .6rem">{description}</p>
                  <small style="color:var(--muted)">Assumed capacity: {cap if cap is not None else "not configured"} seats · Not an audited seat count</small>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### Campus orientation")
    st.caption("Map marker is approximate. Use the official KTH map for navigation and accessible routes.")
    map_df = pd.DataFrame([{"lat": KTHB_LATITUDE, "lon": KTHB_LONGITUDE}])
    st.map(map_df, zoom=15, use_container_width=True)
    st.link_button(
        "Open directions in Google Maps ↗",
        "https://maps.google.com/?q=KTH+Biblioteket+Stockholm",
        use_container_width=True,
    )
    st.info(
        "This guide intentionally omits unverified claims about noise levels, desk equipment, "
        "room counts, lockers, power outlets and opening hours. Check the official KTH source "
        "for current details."
    )
