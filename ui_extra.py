"""Extra UI rendering components for cards, strips, tips, and legends."""

import streamlit as st


def stat_card(label: str, value: str, subtext: str = ""):
    st.markdown(
        f"""
        <div class="saas-card" style="padding: 1rem;">
            <div style="font-size: 0.72rem; color: var(--green); text-transform: uppercase; font-weight: 700;">{label}</div>
            <div style="font-size: 1.6rem; font-weight: 800; color: var(--ink); margin: 0.2rem 0;">{value}</div>
            <div style="font-size: 0.78rem; color: var(--muted);">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def slot_card(date_str: str, time_str: str, duration: int, occupancy: float, seats: int):
    st.markdown(
        f"""
        <div class="saas-card" style="padding: 0.8rem; margin-bottom: 0.5rem;">
            <div style="font-weight: 700; color: var(--ink);">{date_str} · {time_str} ({duration}h)</div>
            <div style="font-size: 0.85rem; color: var(--green);">Predicted: {occupancy:.0f}% · ~{seats} free seats</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def tip_card(title: str, text: str, category: str = "Tip"):
    st.markdown(
        f"""
        <div class="saas-card" style="padding: 0.8rem; border-left: 3px solid var(--green);">
            <div style="font-size: 0.72rem; color: var(--green); text-transform: uppercase; font-weight: 700;">{category}: {title}</div>
            <div style="font-size: 0.85rem; color: var(--ink);">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def hour_strip(hours_data: list):
    cells = "".join([
        f'<div style="flex: 1; text-align: center; font-size: 0.7rem; padding: 2px; '
        f'background: {"#00D084" if h.get("occupancy", 0) < 50 else "#F5B942"}; margin: 1px;">{h.get("hour")}:00</div>'
        for h in hours_data
    ])
    st.markdown(f'<div style="display: flex; width: 100%;">{cells}</div>', unsafe_allow_html=True)


def legend():
    st.markdown(
        """
        <div style="display: flex; gap: 1rem; font-size: 0.8rem; margin-top: 0.5rem;">
            <span>🟢 Free (&lt;50%)</span>
            <span>🟡 Busy (50-80%)</span>
            <span>🔴 Full (&gt;80%)</span>
            <span>⚪ Closed</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
