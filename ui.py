"""UI components and design system helpers for Smart Library Intelligence.
Master Theme: Premium Green + Black AI SaaS.
"""

import math
import plotly.graph_objects as go
import streamlit as st
from config import STATUS_BUSY, STATUS_CLOSED, STATUS_FREE

PLOT_CONFIG = {
    "displayModeBar": False,
    "responsive": True,
    "staticPlot": False,
}


def _theme_color(token: str) -> str:
    """Return a semantic chart color that remains legible in the selected theme."""
    dark = bool(st.session_state.get("dark_mode", False))
    palette = {
        "warning": "#F5B942" if dark else "#A66E12",
        "danger": "#FF7777" if dark else "#BD4E4E",
        "closed": "#A0AFA5" if dark else "#718078",
        "green": "#55C58A" if dark else "#1D6949",
    }
    return palette[token]


def section(title: str, subtitle: str = None):
    """Render a compact modern section header."""
    sub_html = f'<p class="saas-section-sub">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f"""
        <div class="saas-section">
            <h3 class="saas-section-title">{title}</h3>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def note(text: str, level: str = "info"):
    """Render an styled callout banner."""
    icons = {
        "info": "💡",
        "warning": "⚠️",
        "danger": "🚨",
        "success": "✅",
    }
    icon = icons.get(level, "💡")
    st.markdown(
        f"""
        <div class="saas-note saas-note-{level}">
            <span style="font-size: 1.1rem; line-height: 1;">{icon}</span>
            <div>{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def zone_card(title: str, occupancy, status: str, seats: int, zone_key: str):
    """Render a compact study wing card with green+black visual hierarchy."""
    if status == STATUS_CLOSED or occupancy is None or (isinstance(occupancy, float) and math.isnan(occupancy)):
        badge_bg = "rgba(90, 104, 96, 0.2)"
        badge_color = "var(--muted)"
        val_display = "Closed"
        val_color = "var(--muted)"
        bar_width = 0
        seats_text = "Closed"
    else:
        val_display = f"{occupancy:.0f}%"
        bar_width = max(0, min(100, occupancy))
        if status == STATUS_FREE:
            badge_bg = "rgba(0, 208, 132, 0.15)"
            badge_color = "var(--green)"
            val_color = "var(--green)"
        elif status == STATUS_BUSY:
            badge_bg = "rgba(245, 185, 66, 0.15)"
            badge_color = "var(--warning)"
            val_color = "var(--warning)"
        else:
            badge_bg = "rgba(255, 100, 100, 0.15)"
            badge_color = "var(--danger)"
            val_color = "var(--danger)"
        seats_text = f"~{seats} free seats"

    st.markdown(
        f"""
        <div class="saas-card zone-card" style="margin-bottom: 0.75rem; padding: 0.8rem 0.75rem !important;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 4px;">
                <div style="min-width: 0; flex: 1 1 auto;">
                    <div style="font-weight: 700; font-size: clamp(0.78rem, 0.85vw, 0.92rem); color: var(--ink); line-height: 1.25; word-break: break-word;">{title}</div>
                    <div style="font-size: 0.72rem; color: var(--muted); margin-top: 2px;">{seats_text}</div>
                </div>
                <span style="background-color: {badge_bg}; color: {badge_color}; border: 1px solid {badge_color}; padding: 1px 7px; border-radius: 9999px; font-weight: 700; font-size: 0.68rem; white-space: nowrap; flex-shrink: 0;">
                    {status}
                </span>
            </div>
            <div style="font-size: 1.45rem; font-weight: 800; color: {val_color}; margin: 0.35rem 0 0.2rem;">
                {val_display}
            </div>
            <div style="width: 100%; height: 5px; background: var(--paper); border-radius: 9999px; overflow: hidden; margin-top: 0.4rem;">
                <div style="width: {bar_width}%; height: 100%; background: {val_color}; border-radius: 9999px;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_fig(fig: go.Figure, height: int = 300) -> go.Figure:
    """Style Plotly charts to perfectly match the Green + Black SaaS theme with readable typography."""
    dark = st.session_state.get("dark_mode", False)
    bg = "#101713" if dark else "#f4f7f3"
    card = "#17211b" if dark else "#ffffff"
    ink = "#edf4ee" if dark else "#17251e"
    muted = "#a0afa5" if dark else "#718078"
    border = "#2b3b31" if dark else "#dfe8e1"
    fig.update_layout(
        template="plotly_dark" if dark else "plotly_white",
        paper_bgcolor=bg,
        plot_bgcolor=card,
        margin=dict(l=15, r=15, t=25, b=25),
        height=height,
        font=dict(color=ink, family="DM Sans, sans-serif", size=13),
        xaxis=dict(
            showgrid=True,
            gridcolor=border,
            zeroline=False,
            tickfont=dict(color=muted, size=12),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=border,
            zeroline=False,
            tickfont=dict(color=muted, size=12),
        ),
        legend=dict(
            font=dict(color=ink, size=12),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig


def footer():
    """Render unified dashboard footer component."""
    st.markdown(
        """
        <div style="margin-top: 2rem; padding: 1.25rem clamp(14px, 2vw, 32px); border-top: 1px solid var(--line); color: var(--muted); font-size: 0.82rem; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem; background: var(--paper);">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="font-size: 1rem;">📚</span>
                <span>Smart Library Intelligence & Occupancy Forecasting · KTH Royal Institute of Technology</span>
            </div>
            <div style="color: var(--green); font-weight: 700; display: flex; align-items: center; gap: 0.4rem;">
                <span style="display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: var(--green);"></span>
                <span>Gemini 2.5 Active</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# New interface layer: editorial light canvas, forest accents, compact surfaces.
# Kept here so existing analytical components inherit the new design system.
# ---------------------------------------------------------------------------
def inject_css(theme: str = "Light"):
    """Responsive full-width design system with session-selectable themes."""
    dark = str(theme).lower() == "dark"
    palette = ({
        "bg": "#06120b", "surface": "#0a1b12", "surface2": "#0f261b",
        "ink": "#f3f7f4", "muted": "#8ea597", "line": "#183d29",
        "green": "#10b981", "green_hover": "#34d399", "soft": "#143523",
        "shadow": "rgba(0,0,0,.45)", "input": "#08160f", "plot": "#0a1b12",
        "nav_bg": "linear-gradient(180deg, #092015 0%, #06160e 100%)",
        "chatbot_border": "#1b4a31",
        "chatbot_border_subtle": "rgba(16, 185, 129, 0.35)",
        "chatbot_glow": "rgba(16, 185, 129, 0.18)",
    } if dark else {
        "bg": "#f0f5f2", "surface": "#ffffff", "surface2": "#edf4ef",
        "ink": "#112419", "muted": "#5d7568", "line": "#cfe0d5",
        "green": "#167a4d", "green_hover": "#11623d", "soft": "#e2ede5",
        "shadow": "rgba(16,40,26,.08)", "input": "#ffffff", "plot": "#ffffff",
        "nav_bg": "linear-gradient(180deg, #0d3824 0%, #0a2e1d 100%)",
        "chatbot_border": "#8ec4a5",
        "chatbot_border_subtle": "rgba(22, 122, 77, 0.35)",
        "chatbot_glow": "rgba(22, 122, 77, 0.14)",
    })
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    :root {{
      color-scheme: {'dark' if dark else 'light'};
      --paper:{palette['bg']}; --surface:{palette['surface']}; --surface-2:{palette['surface2']};
      --ink:{palette['ink']}; --muted:{palette['muted']}; --line:{palette['line']};
      --green:{palette['green']}; --green-hover:{palette['green_hover']}; --soft:{palette['soft']};
      --warning:{"#F5B942" if dark else "#A66E12"}; --danger:{"#FF7777" if dark else "#BD4E4E"};
      --shadow:{palette['shadow']}; --input:{palette['input']}; --plot:{palette['plot']};
      --nav-bg:{palette['nav_bg']};
      --primary-color:{palette['green']}; --background-color:{palette['bg']};
      --secondary-background-color:{palette['surface']}; --text-color:{palette['ink']};
      --st-primary-color:{palette['green']}; --st-background-color:{palette['bg']};
      --st-secondary-background-color:{palette['surface']}; --st-text-color:{palette['ink']};
      --chatbot-border:{"#1b4a31" if dark else "#8ec4a5"};
      --chatbot-border-subtle:{"rgba(16, 185, 129, 0.35)" if dark else "rgba(22, 122, 77, 0.35)"};
      --chatbot-glow:{"rgba(16, 185, 129, 0.18)" if dark else "rgba(22, 122, 77, 0.14)"};
    }}
    :root, html, body, #root, .stApp, [data-testid="stAppViewContainer"],
    [data-testid="stMain"], [data-testid="stMainBlockContainer"],
    [data-testid="stBottomBlockContainer"] {{
      background-color:var(--paper) !important;
      color:var(--ink) !important;
      color-scheme:{'dark' if dark else 'light'} !important;
    }}
    html, body, [class*="css"] {{ font-family:'DM Sans',sans-serif !important; }}
    [data-testid="stAppViewContainer"] > .main,
    [data-testid="stMain"] > div,
    [data-testid="stMainBlockContainer"] > div {{
      background:var(--paper) !important;
      color:var(--ink) !important;
    }}
    [data-testid="stVerticalBlock"], [data-testid="stHorizontalBlock"],
    [data-testid="stElementContainer"] {{ color:var(--ink) !important; }}
    /* CHATBOT */
    [data-testid="stChatMessage"] {{
      display: flex !important;
      flex-direction: row !important;
      align-items: flex-start !important;
      gap: 0.75rem !important;
      padding: 0.75rem 1rem !important;
      margin: 0.4rem 0 !important;
      border-radius: 14px !important;
      border: 1px solid var(--line) !important;
      background: var(--surface-2) !important;
      width: 100% !important;
      max-width: 100% !important;
      box-sizing: border-box !important;
    }}
    [data-testid="stChatMessage"][data-testid="stChatMessageUser"] {{
      background: rgba(0, 208, 132, 0.12) !important;
      border: 1px solid rgba(0, 208, 132, 0.3) !important;
    }}
    [data-testid="stChatMessage"][data-testid="stChatMessageAssistant"] {{
      background: var(--surface-2) !important;
      border: 1px solid var(--line) !important;
    }}
    [data-testid="stChatMessage"] [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessage"] [data-testid="stChatMessageAvatarAssistant"],
    [data-testid="stChatMessage"] [data-testid="chatAvatarIcon-user"],
    [data-testid="stChatMessage"] [data-testid="chatAvatarIcon-assistant"] {{
      width: 32px !important;
      height: 32px !important;
      min-width: 32px !important;
      min-height: 32px !important;
      flex-shrink: 0 !important;
      border-radius: 8px !important;
    }}
    [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] {{
      flex: 1 1 auto !important;
      min-width: 0 !important;
    }}
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {{
      line-height: 1.55 !important;
      overflow-wrap: break-word !important;
      word-break: break-word !important;
      color: var(--ink) !important;
    }}
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {{
      margin: 0 0 0.4rem 0 !important;
      color: var(--ink) !important;
    }}
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p:last-child {{
      margin-bottom: 0 !important;
    }}
    [data-testid="stChatMessage"] .stCaptionContainer {{
      font-size: 0.68rem !important;
      opacity: 0.75 !important;
      margin-top: 0.2rem !important;
      color: var(--muted) !important;
    }}
    [data-testid="stChatMessage"] code {{
      background: rgba(255,255,255,0.08);
      border-radius: 4px;
      padding: 0.12rem 0.35rem;
    }}
    [data-testid="stChatMessage"] a {{
      color: var(--green) !important;
      text-decoration: underline !important;
    }}
    [data-testid="stChatMessage"] pre {{
      overflow-x: auto !important;
      margin: 0.5rem 0 !important;
    }}
    [data-testid="stChatMessage"] table {{
      overflow-x: auto !important;
      display: block !important;
    }}
    [class*="st-key-chatbot_panel_"] {{
      display: flex !important;
      flex-direction: column !important;
      height: clamp(580px, calc(100vh - 110px), 860px) !important;
      overflow: hidden !important;
    }}
    [class*="st-key-chat_messages_"] {{
      flex: 1 1 0 !important;
      min-height: 200px !important;
      overflow-y: auto !important;
      scrollbar-width: thin !important;
      scrollbar-color: var(--green) var(--surface-2) !important;
      padding: 0.4rem 0.2rem !important;
    }}
    .assistant-suggestions-label {{
      flex: 0 0 auto !important;
      color: var(--muted);
      font-size: 0.68rem;
      font-weight: 800;
      letter-spacing: 0.12em;
      margin: 0.5rem 0 0.3rem !important;
    }}
    .assistant-status-dot.is-neutral {{ background: #a0afa5 !important; }}

    /* Clean Composer Form with inline button */
    [class*="st-key-assistant_composer_form_"] {{
      border: 1px solid var(--line) !important;
      border-radius: 12px !important;
      background: var(--surface-2) !important;
      padding: 0.4rem !important;
      margin-top: auto !important;
    }}
    [class*="st-key-assistant_composer_form_"] [data-testid="stHorizontalBlock"] {{
      align-items: center !important;
      gap: 0.4rem !important;
    }}
    [class*="st-key-assistant_composer_form_"] [data-testid="stTextInput"] {{
      margin-bottom: 0 !important;
    }}
    [class*="st-key-assistant_composer_form_"] [data-testid="stTextInput"] input {{
      min-height: 2.6rem !important;
      border: 1px solid var(--line) !important;
      border-radius: 8px !important;
      padding: 0.45rem 0.8rem !important;
      font-size: 0.88rem !important;
      background: var(--surface) !important;
      color: var(--ink) !important;
    }}
    [class*="st-key-assistant_composer_form_"] [data-testid="stFormSubmitButton"] button {{
      min-height: 2.6rem !important;
      height: 2.6rem !important;
      border-radius: 8px !important;
      font-size: 0.88rem !important;
      font-weight: 700 !important;
      padding: 0 1.2rem !important;
      background: var(--green) !important;
      color: #ffffff !important;
      border: none !important;
      cursor: pointer !important;
    }}
    [class*="st-key-assistant_composer_form_"] [data-testid="stFormSubmitButton"] button:hover {{
      background: var(--green-hover) !important;
    }}
    /* Complete Streamlit chrome reset so navbar sits at absolute top (0, 0) without clipping */
    header[data-testid="stHeader"],
    [data-testid="stHeader"],
    [data-testid="stDecoration"],
    [data-testid="stToolbar"],
    #MainMenu,
    footer {{
      display:none !important;
      visibility:hidden !important;
      height:0 !important;
      min-height:0 !important;
      max-height:0 !important;
      padding:0 !important;
      margin:0 !important;
      opacity:0 !important;
      pointer-events:none !important;
    }}
    [data-testid="stElementContainer"]:has(style:only-child),
    [data-testid="stElementContainer"]:empty,
    div:has(> [data-testid="stMarkdownContainer"] > style:only-child) {{
      display:none !important;
      height:0 !important;
      min-height:0 !important;
      margin:0 !important;
      padding:0 !important;
    }}
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] > .main,
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"],
    .block-container {{
      padding-top:0 !important;
      margin-top:0 !important;
      max-width:none !important;
      width:100% !important;
    }}
    .block-container,
    [data-testid="stMainBlockContainer"] {{
      padding:0 !important;
      margin:0 !important;
    }}
    [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] {{
      gap:0 !important;
      padding-top:0 !important;
      margin-top:0 !important;
    }}
    .stApp {{ overflow-x:hidden !important; }}
    h1,h2,h3,h4 {{ font-family:'Manrope',sans-serif !important; color:var(--ink) !important; letter-spacing:-.035em !important; }}
    h1 {{ font-size:clamp(1.85rem,2.8vw,2.65rem) !important; font-weight:800 !important; line-height:1.16 !important; }}
    h2 {{ font-size:clamp(1.35rem,2vw,1.7rem) !important; font-weight:800 !important; }}
    h3 {{ font-size:1.12rem !important; font-weight:750 !important; }}
    p, label, [data-testid="stMarkdownContainer"] {{ color:var(--ink); }}
    [data-testid="stCaptionContainer"] p, small {{ color:var(--muted) !important; }}
    .topbar-brand {{ display:flex; align-items:center; gap:.8rem; min-height:44px; }}
    .brand-lockup {{ display:flex; align-items:center; gap:.7rem; }}
    .brand-mark {{ display:grid; place-items:center; flex:0 0 40px; width:40px; height:40px; border-radius:13px; background:var(--green); color:#fff; font-size:1.05rem; box-shadow:0 5px 14px var(--shadow); }}
    .brand-name {{ color:var(--ink); font-family:'Manrope',sans-serif; font-size:1.12rem; font-weight:800; letter-spacing:-.04em; }}
    .brand-meta {{ color:var(--muted); font-size:.72rem; font-weight:600; letter-spacing:.015em; }}
    .institution-meta {{ text-align:right; color:var(--muted); font-size:.78rem; font-weight:600; padding-top:.4rem; }}
    .masthead-rule {{ height:1px; background:var(--line); margin:.55rem 0 .75rem; }}
    /* Edge-to-edge persistent top navigation bar */
    div[data-testid="stElementContainer"]:has([class*="st-key-main_navigation"]),
    div[data-testid="stElementContainer"].st-key-main_navigation {{
      position:sticky !important;
      top:0 !important;
      left:0 !important;
      right:0 !important;
      z-index:1000 !important;
      width:100% !important;
      max-width:100% !important;
      margin:0 !important;
      padding:0 !important;
    }}
    [class*="st-key-main_navigation"] {{
      position:sticky !important;
      top:0 !important;
      left:0 !important;
      right:0 !important;
      z-index:1000 !important;
      width:100vw !important;
      max-width:100vw !important;
      margin:0 0 0 calc(50% - 50vw) !important;
      padding:.42rem clamp(12px,2vw,30px) !important;
      border:0 !important;
      border-bottom:1px solid var(--line) !important;
      border-radius:0 !important;
      box-shadow:0 6px 22px var(--shadow) !important;
      background:var(--nav-bg) !important;
      box-sizing:border-box !important;
    }}
    [class*="st-key-main_navigation"]::before {{ display:none !important; }}
    [class*="st-key-main_navigation"] [data-testid="stHorizontalBlock"] {{ gap:.5rem !important; align-items:center !important; }}
    [class*="st-key-main_navigation"] .nav-brand-lockup {{
      display:flex !important;
      align-items:center !important;
      gap:.65rem !important;
      user-select:none !important;
      min-width:0 !important;
    }}
    [class*="st-key-main_navigation"] .nav-brand-icon {{
      display:grid !important;
      place-items:center !important;
      width:34px !important;
      height:34px !important;
      border-radius:9px !important;
      background:linear-gradient(135deg,var(--green),#059669) !important;
      color:#fff !important;
      font-size:1.05rem !important;
      box-shadow:0 3px 10px rgba(16,185,129,.35) !important;
      flex-shrink:0 !important;
    }}
    [class*="st-key-main_navigation"] .nav-brand-text {{
      display:flex !important;
      flex-direction:column !important;
      min-width:0 !important;
      line-height:1.18 !important;
    }}
    [class*="st-key-main_navigation"] .nav-brand-title {{
      font-family:'Manrope',sans-serif !important;
      font-size:.96rem !important;
      font-weight:800 !important;
      color:var(--ink) !important;
      letter-spacing:-.03em !important;
      white-space:nowrap !important;
    }}
    [class*="st-key-main_navigation"] .nav-brand-sub {{
      font-size:.63rem !important;
      font-weight:600 !important;
      color:var(--muted) !important;
      letter-spacing:.01em !important;
      white-space:nowrap !important;
      overflow:hidden !important;
      text-overflow:ellipsis !important;
    }}
    [class*="st-key-main_navigation"] [data-testid="stButton"] {{ min-width:0 !important; }}
    [class*="st-key-main_navigation"] [data-testid="stButton"] button {{
      position:relative; min-height:2.65rem !important; border-radius:10px !important;
      font-size:clamp(.72rem,.76vw,.86rem) !important; font-weight:700 !important;
      letter-spacing:-.015em; border:1px solid transparent !important;
      background:transparent !important; color:var(--muted) !important;
      box-shadow:none !important; transform:translateY(0);
      transition:background .2s ease,color .2s ease,border-color .2s ease,
                 box-shadow .2s ease,transform .2s ease !important;
      white-space:nowrap !important;
      padding:.3rem .55rem !important;
    }}
    [class*="st-key-main_navigation"] [data-testid="stButton"] button:hover {{
      background:var(--soft) !important; color:var(--ink) !important;
      border-color:var(--line) !important; transform:translateY(-1px);
      box-shadow:0 5px 14px var(--shadow) !important;
    }}
    [class*="st-key-main_navigation"] [data-testid="stButton"] button[kind="primary"] {{
      background:linear-gradient(135deg,var(--green),var(--green-hover)) !important;
      color:#fff !important; border-color:rgba(255,255,255,.18) !important;
      box-shadow:0 4px 16px var(--chatbot-glow),inset 0 1px 0 rgba(255,255,255,.2) !important;
    }}
    [class*="st-key-main_navigation"] [data-testid="stButton"] button[kind="primary"]::after {{
      content:""; position:absolute; left:20%; right:20%; bottom:3px; height:2px;
      border-radius:4px; background:rgba(255,255,255,.85);
    }}
    [class*="st-key-main_navigation"] [data-testid="stButton"] button[kind="primary"]:hover {{
      background:linear-gradient(135deg,var(--green-hover),var(--green)) !important;
      color:#fff !important; transform:translateY(-1px);
    }}
    .availability-guide {{
      margin-top:.85rem; padding:.9rem 1rem; border:1px solid var(--line);
      border-radius:13px; background:var(--surface-2); color:var(--ink);
    }}
    .availability-guide-title {{ font-size:.88rem; font-weight:800; color:var(--ink); margin-bottom:.25rem; }}
    .availability-guide-copy {{ font-size:.74rem; line-height:1.45; color:var(--muted); margin-bottom:.65rem; }}
    .availability-guide-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:.55rem .8rem; }}
    .availability-guide-grid > div {{ display:grid; grid-template-columns:9px 1fr; column-gap:.45rem; align-items:center; }}
    .availability-guide-grid b {{ font-size:.72rem; color:var(--ink); }}
    .availability-guide-grid small {{ grid-column:2; font-size:.65rem; color:var(--muted); }}
    .availability-dot {{ width:8px; height:8px; border-radius:50%; grid-row:span 2; }}
    .availability-dot.available {{ background:#229765; }}
    .availability-dot.reserve {{ background:#d49a36; }}
    .availability-dot.checked {{ background:#c65d5d; }}
    .availability-dot.digital {{ background:#4d91c7; }}
    .page-kicker {{ color:var(--green); font-size:.72rem; font-weight:800; letter-spacing:.13em; text-transform:uppercase; margin:.4rem 0 .45rem; }}
    .page-intro {{ color:var(--muted) !important; font-size:.98rem; line-height:1.65; max-width:1200px; margin:.2rem 0 1.35rem; }}
    [data-testid="stVerticalBlockBorderWrapper"] {{ background:var(--surface) !important; border:1px solid var(--line) !important; border-radius:16px !important; box-shadow:0 4px 18px var(--shadow) !important; }}
    [data-testid="stVerticalBlockBorderWrapper"] > div,
    [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"] {{
      background:transparent !important; color:var(--ink) !important;
    }}
    [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li, [data-testid="stWidgetLabel"] label {{
      color:var(--ink) !important;
    }}
    [data-testid="stCaptionContainer"] p {{ color:var(--muted) !important; }}
    [data-testid="stMetric"] {{ background:var(--surface); border:1px solid var(--line); border-radius:15px; padding:1rem 1.1rem; }}
    [data-testid="stMetricLabel"] {{ color:var(--muted) !important; font-size:.78rem !important; font-weight:700 !important; }}
    [data-testid="stMetricValue"] {{ color:var(--ink) !important; font-family:'Manrope',sans-serif !important; font-weight:800 !important; }}
    [data-testid="stButton"] button {{ border-radius:11px !important; border:1px solid var(--line) !important; min-height:2.65rem; font-weight:700 !important; color:var(--ink) !important; background:var(--surface) !important; transition:all .18s ease !important; }}
    [data-testid="stButton"] button[kind="primary"] {{ background:var(--green) !important; color:#fff !important; border-color:var(--green) !important; box-shadow:0 5px 13px var(--shadow) !important; }}
    [data-testid="stButton"] button:hover {{ border-color:var(--green) !important; color:var(--green) !important; transform:translateY(-1px); }}
    [data-testid="stButton"] button[kind="primary"]:hover {{ background:var(--green-hover) !important; color:#fff !important; }}
    [data-testid="stTextInput"] input,[data-testid="stTextArea"] textarea,
    [data-baseweb="select"] > div,[data-baseweb="input"] > div,
    [data-testid="stDateInput"] input,[data-testid="stTimeInput"] input,
    [data-testid="stNumberInput"] input {{
      background:var(--input) !important; color:var(--ink) !important;
      border:1px solid var(--line) !important; border-radius:10px !important;
      -webkit-text-fill-color:var(--ink) !important;
    }}
    [data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"],
    [data-baseweb="select"] [role="option"] {{
      background:var(--surface) !important; color:var(--ink) !important;
    }}
    [data-baseweb="popover"] *, [data-baseweb="menu"] *, [role="option"] {{
      color:var(--ink) !important;
    }}
    [data-testid="stRadio"] label,[data-testid="stCheckbox"] label,[data-testid="stToggle"] label {{ color:var(--ink) !important; }}
    [data-testid="stTabs"] [data-baseweb="tab-list"] {{ gap:.4rem; border-bottom:1px solid var(--line); }}
    [data-testid="stTabs"] button[role="tab"] {{ border-radius:9px 9px 0 0; font-weight:700; color:var(--muted) !important; }}
    [data-testid="stPlotlyChart"] {{ border-radius:14px; overflow:hidden; }}
    [data-testid="stAlert"] {{ border-radius:12px !important; }}
    [data-testid="stNotification"], [data-testid="stExpander"],
    [data-testid="stForm"], [data-testid="stPopoverBody"] {{
      background:var(--surface) !important; color:var(--ink) !important;
      border-color:var(--line) !important;
    }}
    [data-testid="stForm"] {{ border:1px solid var(--line) !important; border-radius:15px !important; padding:1.1rem !important; background:var(--surface) !important; }}
    [data-testid="stExpander"] {{ border-color:var(--line) !important; border-radius:12px !important; }}
    [data-testid="stToggle"] {{ padding:.25rem .5rem; border:1px solid var(--line); border-radius:12px; background:var(--surface); }}
    .saas-footer {{ border-top:1px solid var(--line) !important; color:var(--muted) !important; margin-top:2.5rem !important; }}
    .saas-card {{
      background:var(--surface) !important; color:var(--ink) !important;
      border:1px solid var(--line) !important; border-radius:18px !important;
      padding:clamp(1rem,1.4vw,1.35rem) !important; box-shadow:0 8px 24px var(--shadow) !important;
      box-sizing:border-box !important;
      transition:transform .18s ease,box-shadow .18s ease !important;
    }}
    .stHorizontalBlock > div > div > .saas-card,
    .stHorizontalBlock > div > .saas-card {{
      height:100% !important; display:flex !important; flex-direction:column !important;
      justify-content:space-between !important;
    }}
    .saas-section {{ margin:.4rem 0 .8rem; }}
    .saas-section-title {{ color:var(--ink) !important; font-family:'Manrope',sans-serif !important; font-weight:800 !important; }}
    .saas-section-sub {{ color:var(--muted) !important; font-size:.84rem !important; }}
    .saas-note {{
      display:flex; gap:.7rem; align-items:flex-start; padding:.85rem 1rem;
      border:1px solid var(--line); border-radius:12px; background:var(--surface);
      color:var(--ink); margin:.6rem 0;
    }}
    .saas-note-info {{ border-left:3px solid var(--green); }}
    .saas-note-warning {{ border-left:3px solid #d99a28; }}
    .saas-note-danger {{ border-left:3px solid #d95b5b; }}
    .saas-note-success {{ border-left:3px solid var(--green); }}
    #book-search-results-anchor {{ scroll-margin-top:1.25rem; }}
    .book-search-results-heading {{ scroll-margin-top:1.25rem; padding:.35rem 0 .25rem; }}
    .assistant-heading {{ display:flex; flex-direction:column; gap:.18rem; }}
    .assistant-eyebrow {{ color:var(--green); font-size:.62rem; font-weight:800; letter-spacing:.12em; }}
    .assistant-title {{ color:var(--ink); font-family:'Manrope',sans-serif; font-size:1.12rem; font-weight:800; letter-spacing:-.03em; }}
    .assistant-subtitle {{ color:var(--muted); font-size:.76rem; line-height:1.35; }}
    .assistant-status {{ display:inline-flex; align-items:center; gap:.35rem; color:var(--muted); font-size:.67rem; font-weight:700; margin-top:.25rem; }}
    .assistant-status-dot {{ width:7px; height:7px; border-radius:50%; background:var(--green); display:inline-block; box-shadow:0 0 0 3px var(--soft); }}
    .assistant-suggestions-label {{ color:var(--muted); font-size:.68rem; font-weight:850; letter-spacing:.13em; margin:.85rem 0 .45rem; }}
    [class*="st-key-chatbot_panel_"] [data-testid="stChatInput"] {{ width:100% !important; }}
    [class*="st-key-chatbot_panel_"] [data-testid="stChatInput"] textarea {{ min-height:3.15rem !important; }}
    [class*="st-key-chatbot_panel_"] [data-testid="stHorizontalBlock"] {{ min-width:0 !important; }}
    [class*="st-key-chatbot_panel_"] [data-testid="stButton"] button {{ min-height:2.25rem; font-size:.72rem !important; padding:.35rem .55rem !important; }}
    .assistant-status-dot {{ background:var(--assistant-status-color, var(--green)) !important; }}
    .assistant-status-dot.is-error {{ background:#e05252 !important; box-shadow:0 0 0 3px rgba(224,82,82,.16) !important; }}
    .assistant-status-dot.is-ok {{ background:#28b779 !important; box-shadow:0 0 0 3px rgba(40,183,121,.16) !important; }}
    /* Streamlit/BaseWeb nested controls sometimes retain their own default colors. */
    [data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="select"],
    [data-testid="stDateInput"] [data-baseweb="input"],
    [data-testid="stNumberInput"] [data-baseweb="input"] {{
      background:var(--input) !important; color:var(--ink) !important; border-color:var(--line) !important;
    }}
    [data-testid="stMarkdownContainer"] div, [data-testid="stMarkdownContainer"] span,
    [data-testid="stMarkdownContainer"] li {{ color:inherit; }}
    [data-testid="stVerticalBlockBorderWrapper"] *, [data-testid="stExpander"] * {{
      scrollbar-color:var(--green) var(--surface-2);
    }}
    .stApp, .stApp p, .stApp label, .stApp [data-testid="stMarkdownContainer"] {{
      transition:background-color .18s ease,color .18s ease,border-color .18s ease;
    }}
    .chatbot-robot {{ display:flex; align-items:center; justify-content:flex-end; }}
    .chatbot-robot img {{ width:94px !important; height:94px !important; object-fit:contain; }}
     /* Theme-aware Netflix shelves and book cards. */
     .movie-book-card {{ background:var(--surface-2) !important; border-color:var(--line) !important; box-shadow:0 4px 14px var(--shadow) !important; }}
     .movie-cover-wrap, .movie-cover-placeholder {{ background:var(--surface) !important; }}
     .movie-card-title {{ color:var(--ink) !important; }}
     .movie-card-author, .movie-card-shelf {{ color:var(--muted) !important; }}
     .movie-primo-btn {{ background:var(--surface) !important; border-color:var(--line) !important; color:var(--green) !important; }}
     .movie-carousel-row {{ scrollbar-color:var(--green) var(--surface-2) !important; }}
     [data-testid="stVerticalBlockBorderWrapper"] {{ background:var(--surface) !important; }}

    @media(prefers-reduced-motion:reduce) {{
      *,*::before,*::after {{scroll-behavior:auto !important;animation-duration:.01ms !important;animation-iteration-count:1 !important;transition-duration:.01ms !important;}}
    }}
    button:focus-visible,input:focus-visible,textarea:focus-visible,[role="combobox"]:focus-visible {{
      outline:3px solid var(--green) !important;outline-offset:2px !important;
    }}

    /* v7.3 fixed application frame: edge-to-edge navigation and contained page gutters. */
    .block-container, [data-testid="stMainBlockContainer"] {{
      max-width:none !important; width:100% !important; padding:0 !important; margin:0 !important;
    }}
    [class*="st-key-page_content"] {{
      width:100% !important; max-width:none !important;
      padding:.55rem clamp(12px,1.8vw,28px) 1.2rem !important;
      margin:0 !important; box-sizing:border-box !important;
    }}
    /* Connected Two-Column Layout (55% left / 45% right) with Single Shared Divider */
    [data-testid="stHorizontalBlock"]:has([class*="st-key-primary_work_panel"]),
    [data-testid="stHorizontalBlock"]:has([class*="st-key-resources_search_panel"]),
    [data-testid="stHorizontalBlock"]:has([class*="st-key-chatbot_panel_"]),
    [class*="st-key-page_content"] > [data-testid="stVerticalBlock"] > div > [data-testid="stHorizontalBlock"] {{
      display:flex !important;
      flex-direction:row !important;
      align-items:stretch !important;
      width:100% !important;
      box-sizing:border-box !important;
      gap:0 !important;
      margin:0 !important;
      padding:0 !important;
    }}

    /* Ensure both columns stretch to the full row height */
    [data-testid="stHorizontalBlock"]:has([class*="st-key-primary_work_panel"]) > [data-testid="stColumn"],
    [data-testid="stHorizontalBlock"]:has([class*="st-key-resources_search_panel"]) > [data-testid="stColumn"],
    [data-testid="stHorizontalBlock"]:has([class*="st-key-chatbot_panel_"]) > [data-testid="stColumn"],
    [class*="st-key-page_content"] > [data-testid="stVerticalBlock"] > div > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
      display:flex !important;
      flex-direction:column !important;
      align-self:stretch !important;
      height:100% !important;
      min-height:100% !important;
      box-sizing:border-box !important;
      padding:0 !important;
      margin:0 !important;
    }}

    [data-testid="stHorizontalBlock"]:has([class*="st-key-chatbot_panel_"]) > [data-testid="stColumn"]:first-child,
    [data-testid="stHorizontalBlock"]:has([class*="st-key-primary_work_panel"]) > [data-testid="stColumn"]:first-child,
    [data-testid="stHorizontalBlock"]:has([class*="st-key-resources_search_panel"]) > [data-testid="stColumn"]:first-child {{
      flex:0 0 55% !important;
      width:55% !important;
      max-width:55% !important;
      min-width:0 !important;
    }}

    [data-testid="stHorizontalBlock"]:has([class*="st-key-chatbot_panel_"]) > [data-testid="stColumn"]:last-child,
    [data-testid="stHorizontalBlock"]:has([class*="st-key-primary_work_panel"]) > [data-testid="stColumn"]:last-child,
    [data-testid="stHorizontalBlock"]:has([class*="st-key-resources_search_panel"]) > [data-testid="stColumn"]:last-child {{
      flex:0 0 45% !important;
      width:45% !important;
      max-width:45% !important;
      min-width:0 !important;
    }}

    /* Intermediate Streamlit element container stretching */
    [data-testid="stColumn"] > div[data-testid="stElementContainer"]:has([class*="st-key-primary_work_panel"]),
    [data-testid="stColumn"] > div[data-testid="stElementContainer"]:has([class*="st-key-resources_search_panel"]),
    [data-testid="stColumn"] > div[data-testid="stElementContainer"]:has([class*="st-key-chatbot_panel_"]),
    div[data-testid="stElementContainer"].st-key-primary_work_panel,
    div[data-testid="stElementContainer"].st-key-resources_search_panel,
    div[data-testid="stElementContainer"][class*="st-key-chatbot_panel_"] {{
      display:flex !important;
      flex-direction:column !important;
      flex:1 1 100% !important;
      height:100% !important;
      min-height:100% !important;
      width:100% !important;
      border:none !important;
      background:transparent !important;
      box-shadow:none !important;
      padding:0 !important;
      margin:0 !important;
      box-sizing:border-box !important;
    }}

    /* Left work panel */
    div[data-testid="stVerticalBlock"][class*="st-key-primary_work_panel"],
    div.stVerticalBlock[class*="st-key-primary_work_panel"],
    [class*="st-key-primary_work_panel"] [data-testid="stVerticalBlockBorderWrapper"],
    div[data-testid="stVerticalBlock"][class*="st-key-resources_search_panel"],
    div.stVerticalBlock[class*="st-key-resources_search_panel"],
    [class*="st-key-resources_search_panel"] [data-testid="stVerticalBlockBorderWrapper"],
    [class*="st-key-primary_work_panel"],
    [class*="st-key-resources_search_panel"] {{
      display:flex !important;
      flex-direction:column !important;
      flex:1 1 100% !important;
      height:100% !important;
      min-height:580px !important;
      max-height:none !important;
      overflow:visible !important;
      padding:clamp(1.2rem, 1.8vw, 1.75rem) !important;
      border:2px solid var(--chatbot-border) !important;
      border-right:none !important;
      border-radius:18px 0 0 18px !important;
      background:var(--surface) !important;
      box-shadow:none !important;
      box-sizing:border-box !important;
      width:100% !important;
      min-width:0 !important;
    }}

    /* Right chatbot panel extends toward the bottom of the viewport. */
    div[data-testid="stVerticalBlock"][class*="st-key-chatbot_panel_"],
    div.stVerticalBlock[class*="st-key-chatbot_panel_"],
    [class*="st-key-chatbot_panel_"] > [data-testid="stVerticalBlockBorderWrapper"],
    [class*="st-key-chatbot_panel_"] [data-testid="stVerticalBlockBorderWrapper"],
    [data-testid="stVerticalBlockBorderWrapper"]:has([class*="st-key-chatbot_panel_"]),
    div[class*="st-key-chatbot_panel_"]:not([data-testid="stElementContainer"]) {{
      display:flex !important;
      flex-direction:column !important;
      justify-content:flex-start !important;
      flex:1 1 100% !important;
      height:100% !important;
      min-height:max(580px, calc(100vh - 90px)) !important;
      max-height:none !important;
      overflow:hidden !important;
      padding:clamp(1rem, 1.4vw, 1.35rem) !important;
      border:2px solid var(--chatbot-border) !important;
      border-radius:18px !important;
      border-top-right-radius:18px !important;
      border-bottom-right-radius:18px !important;
      background:var(--surface) !important;
      box-shadow:0 10px 30px var(--shadow), 0 0 16px var(--chatbot-glow) !important;
      box-sizing:border-box !important;
      width:100% !important;
      min-width:0 !important;
      transition:border-color .2s ease,box-shadow .2s ease !important;
    }}

    /* Keep a small gutter between the work panel and the full-height assistant. */
    [data-testid="stHorizontalBlock"]:has([class*="st-key-chatbot_panel_"]) {{
      gap:.75rem !important;
    }}
    [data-testid="stHorizontalBlock"]:has([class*="st-key-chatbot_panel_"]) > [data-testid="stColumn"]:first-child {{
      flex:0 0 calc(55% - .375rem) !important;
      width:calc(55% - .375rem) !important;
      max-width:calc(55% - .375rem) !important;
    }}
    [data-testid="stHorizontalBlock"]:has([class*="st-key-chatbot_panel_"]) > [data-testid="stColumn"]:last-child {{
      flex:0 0 calc(45% - .375rem) !important;
      width:calc(45% - .375rem) !important;
      max-width:calc(45% - .375rem) !important;
    }}
    [data-testid="stHorizontalBlock"]:has([class*="st-key-chatbot_panel_"]) [class*="st-key-primary_work_panel"],
    [data-testid="stHorizontalBlock"]:has([class*="st-key-chatbot_panel_"]) [class*="st-key-resources_search_panel"] {{
      border-right:2px solid var(--chatbot-border) !important;
      border-radius:18px !important;
    }}
    [class*="st-key-chatbot_panel_"] {{
      border-top-left-radius:18px !important;
      border-bottom-left-radius:18px !important;
    }}

    /* Inner vertical block inside chatbot border wrapper */
    [class*="st-key-chatbot_panel_"] [data-testid="stVerticalBlockBorderWrapper"] > [data-testid="stVerticalBlock"],
    [class*="st-key-chatbot_panel_"] > [data-testid="stVerticalBlock"],
    [class*="st-key-chatbot_panel_"] > div > [data-testid="stVerticalBlock"] {{
      display:flex !important;
      flex-direction:column !important;
      flex:1 1 100% !important;
      height:100% !important;
      min-height:100% !important;
      justify-content:flex-start !important;
    }}

    /* Let the transcript wrapper consume spare height so suggestions and composer dock below it. */
    [class*="st-key-chatbot_panel_"] > [data-testid="stLayoutWrapper"]:has([class*="st-key-chat_messages_"]) {{
      display:flex !important;
      flex:1 1 0 !important;
      min-height:0 !important;
      overflow:hidden !important;
    }}
    [class*="st-key-chatbot_panel_"] > [data-testid="stLayoutWrapper"]:has([class*="st-key-chat_messages_"]) > [data-testid="stVerticalBlock"] {{
      flex:1 1 0 !important;
      height:100% !important;
      min-height:0 !important;
    }}

    /* Chatbot internal layout: header at top, messages flexes, composer at bottom */
    [class*="st-key-chatbot_panel_"] .assistant-heading,
    [class*="st-key-chatbot_panel_"] > [data-testid="stHorizontalBlock"]:first-child {{
      flex:0 0 auto !important;
    }}

    /* Chat transcript container takes available vertical space */
    [class*="st-key-chatbot_panel_"] div:has(> [class*="st-key-chat_messages_"]),
    [class*="st-key-chatbot_panel_"] [data-testid="stElementContainer"]:has([class*="st-key-chat_messages_"]),
    [class*="st-key-chatbot_panel_"] [data-testid="stElementContainer"]:has([class*="st-key-chat_messages_"]) > div {{
      display:flex !important;
      flex-direction:column !important;
      flex:1 1 auto !important;
      min-height:260px !important;
      height:auto !important;
      overflow:hidden !important;
    }}

    [class*="st-key-chat_messages_"] {{
      flex:1 1 auto !important;
      height:clamp(260px, calc(100vh - 510px), 480px);
      height:auto !important;
      min-height:260px !important;
      max-height:none !important;
      overflow-y:auto !important;
      overflow-x:hidden !important;
      overscroll-behavior:contain !important;
      scrollbar-width:thin !important;
      scrollbar-color:var(--green) var(--surface-2) !important;
      padding:.25rem .35rem .6rem !important;
      border-radius:12px !important;
      scroll-behavior:smooth !important;
    }}
    [class*="st-key-chat_messages_"] > div {{
      min-height:100% !important;
      align-content:start !important;
    }}
    [class*="st-key-chat_messages_"] > [data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"] {{
      gap:.25rem !important;
      justify-content:flex-start !important;
    }}
    [class*="st-key-chat_messages_"] > [data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"] {{
      flex:0 0 auto !important;
      height:auto !important;
      min-height:0 !important;
    }}

    /* Suggestions chips */
    [class*="st-key-chatbot_panel_"] .assistant-suggestions-label {{
      flex:0 0 auto !important;
      margin:.55rem 0 .35rem !important;
    }} 

    div[data-testid="stVerticalBlock"][class*="st-key-resources_books_panel"],
    div.stVerticalBlock[class*="st-key-resources_books_panel"],
    [class*="st-key-resources_books_panel"] [data-testid="stVerticalBlockBorderWrapper"],
    [class*="st-key-resources_books_panel"] {{
      border-radius:18px !important;
      border:1px solid var(--line) !important;
      background:var(--surface) !important;
      box-shadow:0 8px 26px var(--shadow) !important;
      padding:clamp(1rem,1.5vw,1.5rem) !important;
      box-sizing:border-box !important;
      scroll-margin-top:5.5rem !important;
      width:100% !important;
    }}
    #book-search-results-section,
    #book-search-results-anchor {{ scroll-margin-top:5.5rem !important; }}
    .book-search-results-heading {{
      border-left:4px solid var(--green);
      padding:.35rem .8rem;
      background:var(--surface-2);
      border-radius:0 10px 10px 0;
      margin:.6rem 0;
      scroll-margin-top:5.5rem !important;
    }}
    .book-search-results-heading h3 {{ margin:0 !important; }}

    @media(max-width:900px) {{
      .block-container {{ padding:0 !important; }}
      [class*="st-key-page_content"] {{ padding:.85rem !important; }}
      [data-testid="stHorizontalBlock"]:has([class*="st-key-chatbot_panel_"]),
      [class*="st-key-page_content"] > [data-testid="stVerticalBlock"] > div > [data-testid="stHorizontalBlock"] {{
        flex-direction:column !important;
        gap:1.25rem !important;
      }}
      [data-testid="stHorizontalBlock"]:has([class*="st-key-chatbot_panel_"]) > [data-testid="stColumn"],
      [class*="st-key-page_content"] > [data-testid="stVerticalBlock"] > div > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
        width:100% !important;
        max-width:100% !important;
        flex:1 1 100% !important;
      }}
      div[data-testid="stVerticalBlock"][class*="st-key-primary_work_panel"],
      div[data-testid="stVerticalBlock"][class*="st-key-resources_search_panel"],
      div[data-testid="stVerticalBlock"][class*="st-key-chatbot_panel_"],
      div.stVerticalBlock[class*="st-key-chatbot_panel_"] {{
        min-height:auto !important;
        height:auto !important;
      }}
      div[data-testid="stVerticalBlock"][class*="st-key-primary_work_panel"],
      div[data-testid="stVerticalBlock"][class*="st-key-resources_search_panel"] {{
        border-right:2px solid var(--chatbot-border) !important;
        border-radius:18px !important;
      }}
      div[data-testid="stVerticalBlock"][class*="st-key-chatbot_panel_"],
      div.stVerticalBlock[class*="st-key-chatbot_panel_"] {{
        border-radius:18px !important;
        height:640px !important;
      }}
      [class*="st-key-chat_messages_"] {{ height:340px !important; min-height:280px !important; }}
    }}
    @media(max-width:650px) {{
      [class*="st-key-main_navigation"] {{ overflow-x:auto !important; }}
      [class*="st-key-main_navigation"] [data-testid="stHorizontalBlock"] {{ min-width:0 !important; }}
      [class*="st-key-page_content"] {{ padding:.75rem !important; }}
      [class*="st-key-chatbot_panel_"] {{ min-height:0 !important; height: min(76vh, 760px) !important; }}
      [class*="st-key-chat_messages_"] {{ height:auto !important; min-height:0 !important; }}
      [class*="st-key-chatbot_panel_"] [data-testid="stHorizontalBlock"] {{ flex-wrap:wrap !important; }}
      [class*="st-key-chatbot_panel_"] .assistant-title {{ font-size:1rem !important; }}
      [class*="st-key-chatbot_panel_"] .assistant-subtitle {{ max-width:34ch; }}
      [class*="st-key-chatbot_panel_"] [data-testid="stButton"] button {{ white-space:normal !important; min-height:2.5rem !important; }}
    }}

    /* Chat layout contract: one bounded panel, one independently scrolling transcript,
       and a composer that remains in the bottom row. These rules intentionally come
       after legacy declarations so earlier auto-height overrides cannot win. */
    [class*="st-key-chatbot_panel_"] {{
      display:flex !important; flex-direction:column !important;
      height:clamp(620px, calc(100vh - 116px), 900px) !important;
      min-height:620px !important; max-height:900px !important;
      overflow:hidden !important; box-sizing:border-box !important;
    }}
    [class*="st-key-chatbot_panel_"] > [data-testid="stVerticalBlockBorderWrapper"],
    [class*="st-key-chatbot_panel_"] [data-testid="stVerticalBlockBorderWrapper"] > [data-testid="stVerticalBlock"] {{
      display:flex !important; flex:1 1 auto !important; flex-direction:column !important;
      min-height:0 !important; height:100% !important; overflow:hidden !important;
    }}
    [class*="st-key-chatbot_panel_"] [data-testid="stElementContainer"]:has([class*="st-key-chat_messages_"]) {{
      display:flex !important; flex:1 1 0 !important; flex-direction:column !important;
      min-height:0 !important; height:0 !important; overflow:hidden !important;
    }}
    [class*="st-key-chat_messages_"] {{
      display:flex !important; flex:1 1 0 !important;
      height:100% !important; min-height:0 !important; max-height:none !important;
      overflow-y:auto !important; overflow-x:hidden !important;
      overscroll-behavior:contain !important; scrollbar-gutter:stable;
      padding:.45rem .5rem .8rem !important; box-sizing:border-box !important;
      scroll-behavior:smooth !important;
    }}
    [class*="st-key-chat_messages_"] > [data-testid="stVerticalBlock"],
    [class*="st-key-chat_messages_"] > div {{
      min-height:0 !important; width:100% !important; max-width:100% !important;
      align-content:start !important; box-sizing:border-box !important;
    }}
    @media(max-width:900px) {{
      [class*="st-key-chatbot_panel_"] {{ height: min(78vh, 760px) !important; min-height:560px !important; }}
      [class*="st-key-chat_messages_"] {{ height:100% !important; min-height:0 !important; }}
    }}
    @media(max-width:650px) {{
      [class*="st-key-chatbot_panel_"] {{ height: min(76vh, 700px) !important; min-height:520px !important; }}
      [class*="st-key-chat_messages_"] {{ height:100% !important; min-height:0 !important; }}
    }}
    </style>
    """, unsafe_allow_html=True)
