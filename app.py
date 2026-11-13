"""Smart Library Intelligence | fixed-shell application."""
import json
import streamlit as st

from config import BACKTEST_PATH, METRICS_PATH, MODEL_PATH, PROCESSED_DATA_PATH
from src.chatbot import render_chatbot_panel
from ui import footer, inject_css

st.set_page_config(
    page_title="Smart Library | Study smarter",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PAGES = {
    "Study planner": ("📅", "Find your best study window"),
    "Occupancy": ("📊", "Explore library activity"),
    "Book discovery": ("📚", "Search the collection"),
    "Library guide": ("🧭", "Spaces, facilities and access"),
    "Data & model": ("📈", "Sources and model performance"),
    "Settings": ("⚙️", "AI provider and app preferences"),
}
if "active_page" not in st.session_state or st.session_state.active_page not in PAGES:
    st.session_state.active_page = "Study planner"
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True


def navigate(page):
    st.session_state.active_page = page


def toggle_theme():
    st.session_state.dark_mode = not bool(st.session_state.get("dark_mode", True))


@st.cache_resource
def load_bundle():
    import joblib

    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


@st.cache_data
def load_history():
    import pandas as pd

    return pd.read_parquet(PROCESSED_DATA_PATH) if PROCESSED_DATA_PATH.exists() else None


@st.cache_data
def load_evaluation():
    if not METRICS_PATH.exists():
        return None, None
    import pandas as pd

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    backtest = pd.read_parquet(BACKTEST_PATH) if BACKTEST_PATH.exists() else None
    return metrics, backtest


inject_css("Dark" if st.session_state.dark_mode else "Light")

# Full-bleed navigation bar: Brand lockup on left, 6 centered page links, Theme toggle on right
with st.container(key="main_navigation"):
    nav_brand, c1, c2, c3, c4, c5, c6, nav_theme = st.columns(
        [2.35, 1.05, 0.95, 1.15, 1.05, 1.02, 0.88, 0.78],
        gap="small",
        vertical_alignment="center",
    )
    with nav_brand:
        st.markdown(
            """
            <div class="nav-brand-lockup">
                <div class="nav-brand-icon">📚</div>
                <div class="nav-brand-text">
                    <span class="nav-brand-title">Smart Library</span>
                    <span class="nav-brand-sub">AI-Powered Study & Library Assistant</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    page_buttons = [c1, c2, c3, c4, c5, c6]
    for col, (page_name, (glyph, _)) in zip(page_buttons, PAGES.items()):
        with col:
            st.button(
                f"{glyph}  {page_name}",
                key=f"nav_{page_name}",
                type="primary" if st.session_state.active_page == page_name else "secondary",
                use_container_width=True,
                on_click=navigate,
                args=(page_name,),
            )
    with nav_theme:
        theme_label = "☀ Light" if st.session_state.dark_mode else "☾ Dark"
        st.button(
            theme_label,
            key="theme_nav_toggle",
            on_click=toggle_theme,
            use_container_width=True,
            help="Switch between light and dark appearance",
        )

page = st.session_state.active_page
kicker, descriptions = {
    "Study planner": ("PLAN WITH CONFIDENCE", "Build a study session around your schedule, preferred environment and predicted library activity."),
    "Occupancy": ("LIBRARY PULSE", "Compare predicted occupancy across the study areas and find the quieter parts of the day."),
    "Book discovery": ("THE COLLECTION", "Find useful titles, explore subjects and check availability through the library catalogue."),
    "Library guide": ("YOUR CAMPUS GUIDE", "Get familiar with the library's study spaces, facilities and practical details."),
    "Data & model": ("BEHIND THE FORECAST", "Understand the data, model evaluation and limitations behind the occupancy estimates."),
    "Settings": ("PERSONALIZE YOUR ASSISTANT", "Connect an AI provider and manage your local application settings."),
}[page]

with st.container(key="page_content"):
    bundle = load_bundle()
    model_ready = bundle is not None

    if page == "Study planner":
        if not model_ready:
            st.info("Study recommendations are unavailable because the forecast model is missing.")
        else:
            from src.pages_home import render_home

            left, right = st.columns([0.55, 0.45], gap="small")
            with left:
                with st.container(key="primary_work_panel", border=True):
                    render_home(bundle)
            with right:
                render_chatbot_panel(bundle, "Home")
    elif page == "Occupancy":
        if not model_ready:
            st.info("Occupancy forecasts are unavailable because the forecast model is missing.")
        else:
            from src.pages_planner import render_planner

            left, right = st.columns([0.55, 0.45], gap="small")
            with left:
                with st.container(key="primary_work_panel", border=True):
                    render_planner(bundle)
            with right:
                render_chatbot_panel(bundle, "Overview")
    elif page == "Book discovery":
        from src.pages_resources import render_resources_page

        render_resources_page(bundle)
    elif page == "Library guide":
        from src.pages_library_info import render_library_info

        left, right = st.columns([0.55, 0.45], gap="small")
        with left:
            with st.container(key="primary_work_panel", border=True):
                render_library_info()
        with right:
            render_chatbot_panel(bundle, "Library Info")
    elif page == "Data & model":
        history = load_history()
        metrics, backtest = load_evaluation() if history is not None else (None, None)
        if history is None:
            st.warning("Processed historical data is unavailable. Prepare the dataset to view model diagnostics.")
            st.code("python src/prepare_data.py")
        else:
            from src.pages_dataset import render_dataset

            left, right = st.columns([0.55, 0.45], gap="small")
            with left:
                with st.container(key="primary_work_panel", border=True):
                    render_dataset(history, metrics)
            with right:
                render_chatbot_panel(bundle, "Dataset")
    else:
        left, right = st.columns([0.55, 0.45], gap="small")
        with left:
            with st.container(key="primary_work_panel", border=True):
                from src.pages_settings import render_settings_page
                render_settings_page()
        with right:
            render_chatbot_panel(bundle, "Settings")

footer()
