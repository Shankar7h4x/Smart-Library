"""Global AI Chatbot & Animated Toy Mascot Engine for Smart Library Intelligence.
Multi-provider AI service (OpenAI, Groq, Gemini) with fallback to historical model estimates.
Page-aware greetings, concise decisive recommendations, book specialist mode, and campus guide.
"""

import html
import json
import logging
import math
import os
import re
from pathlib import Path
import time
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo
import streamlit as st
import streamlit.components.v1 as components

from config import (
    COLOR_BACKGROUND,
    COLOR_BORDER,
    COLOR_CARD,
    COLOR_DANGER,
    COLOR_ELEVATED_CARD,
    COLOR_MUTED_GREEN,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_SECONDARY_BG,
    COLOR_SUCCESS,
    COLOR_TEXT_MAIN,
    COLOR_TEXT_MUTED,
    COLOR_WARNING,
    STATUS_BUSY,
    STATUS_CLOSED,
    STATUS_FREE,
    STATUS_FULL,
    SUB_ZONES,
    ZONE_CAPACITIES,
    ZONE_NAMES,
)
from src.ai_service import generate_ai_response_stream, get_configured_providers
from src.books import kth_primo_search_url
from src.book_data import search_local_books
from src.forecast import free_seats, is_open, peak_hours, predict, status
from src.knowledge_service import KnowledgeService

KTH_TZ = ZoneInfo("Europe/Stockholm")
logger = logging.getLogger(__name__)

PAGE_INTRODUCTIONS = {
    "Home": (
        "Welcome to Smart Library! I can help you find the best study location based on your "
        "preferences and predicted occupancy."
    ),
    "Overview": (
        "You're viewing Library Overview. I can explain occupancy, crowd levels, historical trends "
        "and predictions."
    ),
    "Resources": (
        "Welcome to Book Discovery! I can help you find books by title, author, subject or topic. "
        "Tell me what you're interested in and I'll search the available library catalogue."
    ),
    "Library Info": (
        "Welcome to Library Information! I can help you explore the library, understand its study "
        "areas, find facilities and locate different sections."
    ),
    "Dataset": (
        "Welcome to Dataset Explorer! I can explain the time-series dataset, column schemas, "
        "data quality health, and preprocessing pipeline for the library intelligence system."
    ),
    "Model Center": (
        "Welcome to Model Center! I can explain model performance metrics (MAE, RMSE, R²), "
        "algorithm tournament comparisons, residual errors, and feature importance weights."
    ),
    "Settings": (
        "Welcome to Settings! I can guide you through connecting Google AI Studio (Gemini), "
        "selecting supported models, and managing your local environment."
    ),
}

PAGE_SUGGESTIONS = {
    "Home": [
        "Find my best study spot",
        "Which area is least crowded?",
        "How many seats are available?",
    ],
    "Overview": [
        "Explain current occupancy",
        "When is the library busiest?",
        "Compare occupancy trends",
    ],
    "Resources": [
        "Find books about AI",
        "Recommend programming books",
        "Find books by author",
        "Show available computer science books",
    ],
    "Library Info": [
        "Where is the library?",
        "What facilities are available?",
        "Where is the Newton area?",
        "What are the library opening hours?",
    ],
    "Dataset": [
        "Explain the training dataset",
        "What features does the dataset contain?",
        "How is data quality validated?",
    ],
    "Model Center": [
        "What is the model MAE across zones?",
        "Which algorithm was selected per zone?",
        "What are the most predictive features?",
    ],
    "Settings": [
        "How do I connect Google AI Studio?",
        "Which Gemini models are supported?",
        "Where is my API key stored?",
    ],
}

ZONE_PROFILES = {
    "angdomen": {
        "name": "Ångdomen",
        "level": "Model zone · exact location unverified",
        "style": "Quiet environment",
        "description": "The recommendation model assigns this zone a strong quiet-study preference profile.",
        "quiet_score": 10,
        "collab_score": 1,
        "seats": 60,
    },
    "south_east": {
        "name": "South-East Gallery",
        "level": "Model zone · exact location unverified",
        "style": "Moderate activity / Window seating",
        "description": "The recommendation model treats this as a flexible study option with a window-seat preference profile.",
        "quiet_score": 7,
        "collab_score": 6,
        "seats": 100,
    },
    "newton": {
        "name": "Newton Area",
        "level": "Model zone · exact location unverified",
        "style": "Collaborative environment",
        "description": "The recommendation model assigns this zone a strong collaborative-study preference profile.",
        "quiet_score": 3,
        "collab_score": 10,
        "seats": 40,
    },
    "south": {
        "name": "South Gallery",
        "level": "Model zone · exact location unverified",
        "style": "Moderate activity",
        "description": "The recommendation model treats this as a moderate-activity study option.",
        "quiet_score": 7,
        "collab_score": 4,
        "seats": 100,
    },
    "north": {
        "name": "North Gallery",
        "level": "Model zone · exact location unverified",
        "style": "Quiet environment",
        "description": "The recommendation model assigns this zone a quiet-study preference profile.",
        "quiet_score": 9,
        "collab_score": 2,
        "seats": 120,
    },
}


def render_toy_mascot(state: str = "idle") -> str:
    """Return the CSS and HTML for the cute animated toy mascot in the chatbot corner."""
    return f"""
    <div class="toy-mascot-wrapper {state}">
      <div class="mascot-stage">
        <!-- Floating Antenna with Pulsing Green Orb -->
        <div class="mascot-antenna">
          <div class="antenna-stalk"></div>
          <div class="antenna-orb"></div>
        </div>
        <!-- Head / Robot Toy Face -->
        <div class="mascot-head">
          <div class="mascot-ear left"></div>
          <div class="mascot-ear right"></div>
          <div class="mascot-screen">
            <div class="mascot-eye left">
              <div class="eye-pupil"></div>
              <div class="eye-highlight"></div>
            </div>
            <div class="mascot-eye right">
              <div class="eye-pupil"></div>
              <div class="eye-highlight"></div>
            </div>
            <div class="mascot-blush left"></div>
            <div class="mascot-blush right"></div>
            <div class="mascot-mouth"></div>
          </div>
        </div>
      </div>
    </div>
    """


def evaluate_recommendations(bundle: dict, form_data: dict) -> List[dict]:
    """Rank zones only when every requested hour is open and has a valid forecast."""
    now = datetime.now(KTH_TZ)
    visit_date = form_data.get("date", now.date())
    if isinstance(visit_date, datetime):
        visit_date = visit_date.date()
    if not isinstance(visit_date, date):
        return []
    arr_hour = int(form_data.get("arrival_hour", 13))
    duration = int(form_data.get("duration", 2))
    if not 0 <= arr_hour <= 23 or not 1 <= duration <= 7:
        return []
    env_pref = form_data.get("environment", "No preference")
    seat_pref = form_data.get("seating", "No preference")
    crowd_pref = form_data.get("crowd", "Avoid crowded areas")
    purpose = form_data.get("purpose", "General studying")
    exam = bool(form_data.get("exam", False))
    start = datetime.combine(visit_date, datetime.min.time(), tzinfo=KTH_TZ).replace(hour=arr_hour)
    visit_hours = [start + timedelta(hours=i) for i in range(duration)]

    results = []
    capacities = bundle.get("capacities", ZONE_CAPACITIES)
    for zone_key in SUB_ZONES:
        profile = ZONE_PROFILES.get(zone_key, {})
        cap = int(capacities.get(zone_key, ZONE_CAPACITIES.get(zone_key, 0)))
        if cap <= 0 or not all(is_open(bundle, dt) for dt in visit_hours):
            continue
        occ_readings = [predict(bundle, dt, zone_key, exam=exam) for dt in visit_hours]
        if len(occ_readings) != duration or any(
            value is None or not math.isfinite(float(value)) for value in occ_readings
        ):
            continue
        avg_occ = float(sum(occ_readings) / duration)
        avg_seats = free_seats(avg_occ, zone_key, bundle)
        score = 100.0 - avg_occ
        if env_pref == "Quiet environment":
            score += profile.get("quiet_score", 5) * 8.0
        elif env_pref == "Collaborative environment":
            score += profile.get("collab_score", 5) * 8.0
        elif env_pref == "Moderate activity":
            score += 15.0 if zone_key in ["south_east", "south"] else 0.0
        if seat_pref == "Individual seating" and zone_key in ["angdomen", "south", "north"]:
            score += 20.0
        elif seat_pref == "Near windows" and zone_key == "south_east":
            score += 25.0
        elif seat_pref == "Group seating" and zone_key in ["newton", "south_east"]:
            score += 25.0
        elif seat_pref == "More available space":
            score += (avg_seats / cap) * 40.0
        if crowd_pref == "Avoid crowded areas":
            score += 20.0 if avg_occ < 40.0 else (-35.0 if avg_occ > 65.0 else 0.0)
        elif crowd_pref == "Moderate occupancy is fine" and avg_occ <= 75.0:
            score += 15.0

        if purpose in {"Group project"} and zone_key in {"newton", "south_east"}:
            score += 15.0
        elif purpose in {"Exam preparation", "Reading", "Research"} and zone_key in {"angdomen", "north"}:
            score += 10.0
        elif purpose == "Assignment work" and zone_key in {"south", "south_east"}:
            score += 7.0

        if avg_occ < 50.0:
            badge_class, crowd_text = "green", "Low"
        elif avg_occ <= 80.0:
            badge_class, crowd_text = "yellow", "Moderate"
        else:
            badge_class, crowd_text = "red", "High"
        results.append({
            "key": zone_key,
            "name": profile.get("name", ZONE_NAMES.get(zone_key, zone_key)),
            "level": profile.get("level", ""),
            "occupancy": avg_occ,
            "hourly_occupancy": [float(v) for v in occ_readings],
            "free_seats": avg_seats,
            "capacity": cap,
            "status": status(avg_occ),
            "badge_class": badge_class,
            "crowd_level": crowd_text,
            "score": score,
            "description": profile.get("description", ""),
        })
    results.sort(key=lambda item: item["score"], reverse=True)
    return results


def format_concise_recommendation(top_results: List[dict], form_data: dict, prev_data: Optional[dict] = None) -> Dict[str, Any]:
    """
    Generate short, clear and decisive study recommendations matching Master Prompt Section 4.
    Maximum ~80-120 words. Puts best recommendation first with bullet points and clear alternative.
    """
    if not top_results:
        return {
            "text": "I couldn't find a zone with a valid forecast for every hour of your requested visit. Check the opening schedule, choose a later time, or shorten the duration.",
            "card": None,
        }
    top = top_results[0]
    alt = top_results[1] if len(top_results) > 1 else None
    env_pref = form_data.get("environment", "Quiet environment")
    seat_pref = form_data.get("seating", "No preference")
    arr_hour = form_data.get("arrival_hour", 13)

    # Check for parameter change context
    change_intro = ""
    if prev_data:
        old_hour = prev_data.get("arrival_hour")
        old_env = prev_data.get("environment")
        if old_hour != arr_hour:
            change_intro = f"Your arrival time changed to {arr_hour:02d}:00. I've recalculated the occupancy.\n\n"
        elif old_env != env_pref:
            change_intro = f"Your environment preference changed to {env_pref}. I've updated my recommendation.\n\n"

    # Match reason
    if "Quiet" in env_pref:
        reason = f"Its modelled preference profile best matches your request for a {env_pref.lower()}."
    elif "Collab" in env_pref:
        reason = "Its modelled preference profile best matches collaborative study."
    elif seat_pref == "Group seating":
        reason = "Its modelled profile is a better match for group seating."
    elif seat_pref == "Near windows":
        reason = "Its modelled profile best matches your window-seat preference."
    else:
        reason = f"It offers the highest estimated free-seat count (~{top['free_seats']}) for your visit."

    alternative_text = (
        f"**Alternative:** {alt['name']} (`{alt['occupancy']:.0f}%` occupancy, ~{alt['free_seats']} seats free)."
        if alt else "**Alternative:** No second zone has a valid forecast for the full visit."
    )
    text = (
        f"{change_intro}"
        f"Based on your preferences, I recommend:\n\n"
        f"**{top['name']}**\n\n"
        f"- Occupancy: `{top['occupancy']:.0f}%`\n"
        f"- Available seats: `~{top['free_seats']}` of {top['capacity']}\n"
        f"- Crowd level: {top['crowd_level']}\n\n"
        f"**Why?**\n"
        f"{reason}\n\n"
        f"{alternative_text}\n\n"
        f"My recommendation: **{top['name']}**."
    )

    card = {
        "name": top["name"],
        "level": top["level"],
        "occupancy": top["occupancy"],
        "free_seats": top["free_seats"],
        "badge_class": top["badge_class"],
        "description": top["description"],
    }

    return {"text": text, "card": card}


def build_system_prompt(current_page: str, bundle: dict) -> str:
    """Build structured system prompt for real AI providers (OpenAI, Groq, Gemini)."""
    now = datetime.now(KTH_TZ)

    # Precalculate zone values for prompt context
    zone_status_lines = []
    models_ready = bool(bundle and isinstance(bundle, dict) and bundle.get("models"))
    for z in SUB_ZONES:
        occ = None
        if models_ready:
            try:
                occ = predict(bundle, now, z)
            except Exception:
                occ = None
        if occ is None or not math.isfinite(float(occ)):
            zone_status_lines.append(f"- {ZONE_NAMES[z]}: Closed or forecast unavailable; free seats unknown")
        else:
            seats = free_seats(occ, z, bundle)
            zone_status_lines.append(f"- {ZONE_NAMES[z]}: {float(occ):.0f}% model forecast, approximately {seats} free seats")
    zones_ctx = "\n".join(zone_status_lines)

    base = (
        KnowledgeService.get_verified_system_context() + "\n"
        "You are the KTH Smart Library AI Assistant, an intelligent, concise, and helpful campus companion.\n"
        f"Current Date/Time: {now.strftime('%A, %d %B %Y, %H:%M')} (Stockholm Time).\n"
        "Data Status: Occupancy values are model forecasts based on historical data, not live readings.\n"
        f"Library Zones & Modelled Occupancy Estimate:\n{zones_ctx}\n\n"
        "STRICT GUIDELINES:\n"
        "1. Be concise, direct, and conversational. Do not write essays or repeat the user's form.\n"
        "2. Put the best recommendation first. Use short bullet points.\n"
        "3. Never invent fake library areas, books, seat numbers, or opening hours.\n"
        "4. SENSOR HARDWARE: Never claim or guess sensor hardware (Wi-Fi, PIR, cameras, turnstiles). If asked, explicitly state: 'This information is not documented or available in the current project files.'\n"
        "5. DATASET FACTS: Do not invent millions of rows or multi-year datasets. Cite the verified row count (95,164 raw, 5,615 processed) and dates (Dec 9, 2024 to Jul 31, 2025).\n"
    )

    if current_page == "Home":
        return base + (
            "Current Page Role: STUDY SPACE ADVISOR.\n"
            "Your main purpose is to answer: WHERE SHOULD THE STUDENT STUDY?\n"
            "Keep recommendations around 80-120 words. Give one top spot, model-estimated occupancy, estimated free seats, one short 'Why', one alternative, and a final verdict. Never call estimates live measurements."
        )
    elif current_page == "Overview":
        return base + (
            "Current Page Role: OCCUPANCY ANALYST.\n"
            "Explain model-estimated occupancy, historical trends, data freshness and model limitations. Never describe archive-derived forecasts as live sensor readings."
        )
    elif current_page == "Resources":
        return base + (
            "Current Page Role: BOOK SPECIALIST.\n"
            "Help students find books in the KTH library catalogue. Search by title, author, or subject. Mention shelf locations and KTH Primo holdings."
        )
    elif current_page == "Library Info":
        return base + (
            "Current Page Role: CAMPUS LIBRARY GUIDE.\n"
            "Help students navigate the library. Do not invent opening hours, facilities, equipment or room locations. Direct users to the official KTH library website when current details are requested."
        )
    elif current_page == "Dataset":
        return base + (
            "Current Page Role: DATASET EXPLORER ADVISOR.\n"
            "Help students understand the historical library dataset, schemas, quality metrics, and cleaning operations. Be honest about data limits."
        )
    elif current_page == "Model Center":
        return base + (
            "Current Page Role: ML MODEL CENTER & METRICS ADVISOR.\n"
            "Help students understand the trained models, MAE/RMSE/R² metrics, algorithm tournaments, and feature importances. Be transparent about prediction confidence."
        )
    elif current_page == "Settings":
        return base + (
            "Current Page Role: APPLICATION & AI SETTINGS ADVISOR.\n"
            "Guide users on configuring Google AI Studio, verifying credentials, selecting Gemini models, and managing private settings."
        )
    return base


def _word_boundary_search(text: str, term: str) -> bool:
    """Return True when term appears as a full word in the query, ignoring punctuation."""
    if not term or not text:
        return False
    return bool(re.search(rf"\b{re.escape(term)}\b", text, flags=re.IGNORECASE))


def _strip_command_prefixes(text: str) -> str:
    """Remove common leading command phrases without stripping words inside titles."""
    stripped = text.strip()
    prefixes = [
        "find me", "find", "show me", "show", "recommend", "suggest", "search for", "search",
        "what is", "tell me", "can you", "could you", "please",
    ]
    for prefix in prefixes:
        if re.match(rf"^\s*{re.escape(prefix)}\b", stripped, flags=re.IGNORECASE):
            stripped = re.sub(rf"^\s*{re.escape(prefix)}\b\s*", "", stripped, flags=re.IGNORECASE)
            break
    return stripped.strip()


def handle_deterministic_query(q: str, current_page: str, bundle: dict) -> Dict[str, Any]:
    """Reliable local intent routing when a cloud model is absent or unavailable."""
    q_raw = q or ""
    q_lower = q_raw.casefold().strip()
    q_clean = re.sub(r"\s+", " ", q_raw.strip())
    page_label = {
        "Home": "Study planner",
        "Overview": "Occupancy",
        "Resources": "Book discovery",
        "Library Info": "Library guide",
        "Model Center": "Model Center",
        "Dataset": "Dataset Explorer",
        "Settings": "Settings",
    }.get(current_page, current_page)
    now = datetime.now(KTH_TZ)

    small_talk = [
        "hi", "hello", "hey", "thanks", "thank you", "help", "what can you do",
        "what can i ask", "what can i ask you", "good morning", "good evening",
    ]
    if any(_word_boundary_search(q_lower, term) for term in ["hi", "hello", "hey"]) or q_lower in {"thanks", "thank you", "help"}:
        return {
            "text": (
                f"Hi! I can help on {page_label}. Ask about study spots, quiet areas, book searches, library facilities, or the model and dataset."
            ),
            "cards": None,
        }
    if "what can you do" in q_lower or "what can i ask" in q_lower:
        examples = [
            "Which area is the quietest right now?",
            "Where is the library and what facilities are nearby?",
            "Find books by author or subject.",
        ]
        return {"text": f"I can help with {page_label}. For example: {examples[0]} • {examples[1]} • {examples[2]}", "cards": None}

    if any(_word_boundary_search(q_lower, term) for term in ["opening hours", "opening hour", "open hours", "when is it open", "what time does it open", "closing time", "open now"]):
        return {
            "text": "I don’t have a verified live opening-hours source here. Please check the official KTH Library site before travelling: https://www.lib.kth.se/",
            "cards": None,
        }

    if any(_word_boundary_search(q_lower, term) for term in ["where is", "location", "address", "directions", "where is the library", "where is the newton area", "newton area", "where is it"]):
        return {
            "text": (
                "KTH Main Campus Library (KTH Biblioteket)\n\n"
                "- Address: Osquars backe 31, 100 44 Stockholm\n"
                "- Campus: Valhallavägen, next to Nymble\n"
                "- Metro: Tekniska Högskolan (Red Line 14)\n\n"
                "For facilities and study areas, use the Library guide page."
            ),
            "cards": None,
        }

    if any(_word_boundary_search(q_lower, term) for term in ["facilities", "study rooms", "equipment", "printer", "locker", "computer", "wifi", "toilets", "quiet room"]):
        return {
            "text": "The library guide covers study spaces, equipment, access, and practical facilities. I can point you to the right area or explain the layout without inventing details.",
            "cards": None,
        }

    if _word_boundary_search(q_lower, "hours") or _word_boundary_search(q_lower, "open") and "when" in q_lower:
        return {
            "text": "I don’t have a verified live opening-hours source here. Please check the official KTH Library site before travelling: https://www.lib.kth.se/",
            "cards": None,
        }

    if any(_word_boundary_search(q_lower, term) for term in ["least crowded", "quietest", "quiet", "calm", "peaceful", "lowest occupancy", "least busy"]):
        if not is_open(bundle, now):
            next_open = next((dt for dt in [now + timedelta(hours=i) for i in range(1, 25)] if is_open(bundle, dt)), None)
            if next_open is not None:
                return {"text": f"The library is closed right now. The quietest time to return is {next_open.strftime('%A %H:%M')} (Stockholm time).", "cards": None}
            return {"text": "The library is closed right now. I can suggest the quietest zone for the next open window when you want to return.", "cards": None}
        zone_occs = [(z, predict(bundle, now, z)) for z in SUB_ZONES]
        valid = [(z, occ) for z, occ in zone_occs if occ is not None and math.isfinite(float(occ))]
        if valid:
            best_z, best_occ = min(valid, key=lambda x: x[1])
            seats = free_seats(best_occ, best_z, bundle)
            return {"text": f"{ZONE_NAMES[best_z]} is the quietest area right now. Occupancy is about {best_occ:.0f}% and there are roughly {seats} seats available. It suits focused individual study.", "cards": None}

    if any(_word_boundary_search(q_lower, term) for term in ["busiest", "peak", "most crowded", "crowdedest", "heaviest"]):
        zone_occs = [(z, predict(bundle, now, z)) for z in SUB_ZONES]
        valid = [(z, occ) for z, occ in zone_occs if occ is not None and math.isfinite(float(occ))]
        if valid:
            best_z, best_occ = max(valid, key=lambda x: x[1])
            return {"text": f"{ZONE_NAMES[best_z]} is likely the busiest area right now, with about {best_occ:.0f}% occupancy. If you want a calmer study spot, choose a quieter zone or come a little later.", "cards": None}

    if any(_word_boundary_search(q_lower, term) for term in ["available seats", "free seats", "how many seats", "seat availability", "seat available", "how many free seats"]):
        total_seats = 0
        details = []
        for zone in SUB_ZONES:
            occ = predict(bundle, now, zone)
            if occ is None or not math.isfinite(float(occ)):
                details.append(f"- {ZONE_NAMES[zone]}: closed or forecast unavailable")
                continue
            seats = free_seats(occ, zone, bundle)
            total_seats += seats
            details.append(f"- {ZONE_NAMES[zone]}: ~{seats} seats ({float(occ):.0f}% forecast)")
        return {"text": f"There are about {total_seats} free seats across the library.\n\n" + "\n".join(details) + "\n\nThese are model estimates, not live counter readings.", "cards": None}

    # 1. Training Dataset Explanation
    if any(_word_boundary_search(q_lower, term) for term in [
        "explain the training dataset", "explain the dataset", "training dataset",
        "about the dataset", "dataset summary", "dataset overview", "describe the dataset"
    ]) or ("dataset" in q_lower and any(w in q_lower for w in ["explain", "overview", "summary", "training", "describe"])):
        summary = KnowledgeService.get_dataset_summary()
        return {
            "text": (
                "**Training Dataset Overview (Verified Ground Truth)**:\n\n"
                f"- **Facility**: {summary['source_facility']}\n"
                f"- **Raw Observations**: Exactly **{summary['raw_rows']:,} rows** across {summary['raw_columns']} columns (~3.17 min sampling interval).\n"
                f"- **Processed Dataset**: Exactly **{summary['processed_rows']:,} hourly rows** across {summary['processed_columns']} columns "
                f"({summary['open_operating_hours']:,} open operating hours and {summary['closed_night_hours']:,} closed night hours).\n"
                f"- **Date Range**: {summary['date_range_start'][:10]} to {summary['date_range_end'][:10]} ({summary['date_range_days']} continuous days, Dec 2024 – Jul 2025).\n"
                f"- **Monitored Zones**: 6 zones ({', '.join(summary['monitored_zones'])}).\n"
                "- **Target Metric**: Desk occupancy percentage [0.0%, 100.0%].\n"
                "- **Source**: Hugging Face community repository (`davnas/real-time-library-occupancy`)."
            ),
            "cards": None,
        }

    # 2. Row count query
    if any(_word_boundary_search(q_lower, term) for term in [
        "how many rows", "row count", "number of rows", "total rows", "how many records", "how many samples", "how many observations"
    ]) or ("rows" in q_lower and ("dataset" in q_lower or "raw" in q_lower or "processed" in q_lower or "how many" in q_lower or "count" in q_lower)):
        raw_count = KnowledgeService.get_dataset_row_count("raw")
        proc_count = KnowledgeService.get_dataset_row_count("processed")
        summary = KnowledgeService.get_dataset_summary()
        return {
            "text": (
                "**Dataset Record Counts (Verified Ground Truth)**:\n\n"
                f"- **Raw Dataset**: Exactly **{raw_count:,} rows** (7 columns).\n"
                f"- **Processed Dataset**: Exactly **{proc_count:,} hourly rows** (29 columns).\n"
                f"  - **Open Operating Hours**: {summary['open_operating_hours']:,} hours\n"
                f"  - **Closed Night Hours**: {summary['closed_night_hours']:,} hours\n\n"
                "Note: The project contains exactly 233 days of data, not millions of rows or multi-year archives."
            ),
            "cards": None,
        }

    # 3. Date range query
    if any(_word_boundary_search(q_lower, term) for term in [
        "date range", "what dates", "time period", "start date", "end date", "how many days", "dates covered", "dataset period", "what is the date range"
    ]) or ("date" in q_lower and any(w in q_lower for w in ["range", "start", "end", "period", "span", "cover"])):
        dr = KnowledgeService.get_dataset_date_range()
        return {
            "text": (
                "**Dataset Date Range (Verified Ground Truth)**:\n\n"
                f"- **Start Date**: {dr['start_date_human']} ({dr['start_date']})\n"
                f"- **End Date**: {dr['end_date_human']} ({dr['end_date']})\n"
                f"- **Total Duration**: Exactly **{dr['days_count']} continuous days** (~7.5 months).\n"
                f"- **Timezone**: {dr['timezone']} (Europe/Stockholm).\n\n"
                "The observations cover Dec 9, 2024 through Jul 31, 2025. There are no multi-year historical archives."
            ),
            "cards": None,
        }

    # 4. Sensor hardware query
    if any(_word_boundary_search(q_lower, term) for term in [
        "sensor", "sensors", "hardware", "wifi", "wi-fi", "pir", "optical", "turnstile", "turnstiles", "camera", "cameras", "how is data collected"
    ]) or ("sensor" in q_lower or "sensors" in q_lower):
        sources = KnowledgeService.get_dataset_sources()
        return {
            "text": (
                f"**Occupancy Sensor Hardware**: {sources['sensor_hardware']}\n\n"
                f"{sources['hardware_disclaimer']}\n\n"
                "The source dataset contains recorded zone occupancy percentages directly. Smart Library does not fabricate or guess hardware details."
            ),
            "cards": None,
        }

    # 5. Model MAE / Accuracy query
    if any(_word_boundary_search(q_lower, term) for term in [
        "model mae", "what is the mae", "mae", "model accuracy", "evaluation metrics", "rmse", "r2", "r-squared", "error reduction", "how accurate"
    ]) or ("mae" in q_lower and ("model" in q_lower or "baseline" in q_lower or "what" in q_lower)):
        m = KnowledgeService.get_model_metrics()
        return {
            "text": (
                "**Model Evaluation Metrics (Verified Ground Truth)**:\n\n"
                f"- **Average Selected MAE**: **{m['average_selected_mae']:.2f} percentage points** across the 6 zones.\n"
                f"- **Baseline Comparison**: Baseline MAE is {m['average_baseline_mae']:.2f} pp (a **{m['error_reduction_percentage']:.1f}% error reduction**).\n"
                f"- **Whole Library Building**: MAE = {m['whole_library_mae']:.2f} pp, RMSE = {m['whole_library_rmse']:.2f} pp, R² = {m['whole_library_r2']:.3f}, Bias = {m['whole_library_bias']:+.2f} pp.\n"
                f"- **Best Performing Zone**: **{m['best_zone']}** with an MAE of **{m['best_zone_mae']:.2f} pp** and R² = {m['best_zone_r2']:.3f}.\n"
                f"- **Tolerance Accuracy (Holdout)**: {m['tolerance_accuracy_within_5pp']:.1f}% within ±5 pp, {m['tolerance_accuracy_within_10pp']:.1f}% within ±10 pp, {m['tolerance_accuracy_within_15pp']:.1f}% within ±15 pp."
            ),
            "cards": None,
        }

    # 6. Algorithm selection query
    if any(_word_boundary_search(q_lower, term) for term in [
        "which algorithm", "what algorithm", "algorithm", "algorithms", "xgboost", "random forest", "extra trees"
    ]) or ("algorithm" in q_lower and ("newton" in q_lower or "select" in q_lower or "use" in q_lower or "which" in q_lower or "what" in q_lower)):
        algos = KnowledgeService.get_model_algorithms()
        if "newton" in q_lower:
            return {
                "text": (
                    f"**Selected Algorithm for Newton**: **{algos['newton']}**.\n\n"
                    "Extra Trees achieved superior out-of-sample generalization on Newton study wing by using randomized split thresholds "
                    "that stabilize variance across collaborative study groups."
                ),
                "cards": None,
            }
        algo_lines = "\n".join(f"- **{ZONE_NAMES.get(k, k)}**: {v}" for k, v in algos.items())
        return {
            "text": f"**Selected Machine Learning Algorithms by Zone**:\n\n{algo_lines}",
            "cards": None,
        }

    # 7. Features query
    if any(_word_boundary_search(q_lower, term) for term in [
        "how many features", "features does the model use", "features use", "feature count", "engineered features", "what features"
    ]) or ("feature" in q_lower and ("many" in q_lower or "count" in q_lower or "model" in q_lower or "engineered" in q_lower)):
        features = KnowledgeService.get_feature_information()
        cats = {}
        for f in features:
            cats.setdefault(f["category"], []).append(f["feature"])
        cat_lines = "\n".join(f"- **{cat}** ({len(fl)}): {', '.join(fl)}" for cat, fl in cats.items())
        return {
            "text": (
                f"**Engineered Model Features**: Exactly **{len(features)} features** are used in model training:\n\n"
                f"{cat_lines}\n\n"
                "All autoregressive lags (1h, 24h, 168h) are strictly shifted by 1 hour (shift(1)) with zero future data leakage."
            ),
            "cards": None,
        }

    # 8. How is occupancy predicted
    if any(_word_boundary_search(q_lower, term) for term in [
        "how is occupancy predicted", "how do you predict", "how does the model predict", "prediction logic", "forecasting engine", "how does forecasting work"
    ]) or ("how" in q_lower and "occupancy" in q_lower and "predict" in q_lower):
        return {
            "text": f"**How Occupancy is Predicted**:\n\n{KnowledgeService.get_forecast_information()}",
            "cards": None,
        }

    # 9. Unavailable / undocumented information query
    if any(_word_boundary_search(q_lower, term) for term in [
        "what information is not available", "information is not available", "what is not available", "what is not documented", "unavailable information", "what information is unavailable", "missing information"
    ]) or ("not available" in q_lower or "unavailable" in q_lower or "not documented" in q_lower):
        return {
            "text": (
                "**Information Not Documented or Available in Project Files**:\n\n"
                "1. **Sensor Hardware**: The specific sensor types, brands, or network telemetry (Wi-Fi APs, PIR detectors, optical counters) are not documented in the raw dataset.\n"
                "2. **Personal Data (PII)**: No student names, student IDs, card-swipe transactions, or individual identities exist.\n"
                "3. **Live Hardware Telemetry**: All real-time values are machine-learning model forecasts based on historical patterns, not live sensor counters.\n"
                "4. **Live Physical Door Feeds / Opening Hours**: Opening hours are not fetched via a live API; users are referred to the official KTH Library portal."
            ),
            "cards": None,
        }

    # 10. Predict next week query
    if any(_word_boundary_search(q_lower, term) for term in [
        "predict occupancy for next week", "predict for next week", "predict next week", "forecast next week", "can the model predict occupancy for next week", "can the model predict for next week", "future weeks"
    ]) or ("next week" in q_lower and ("predict" in q_lower or "forecast" in q_lower or "can" in q_lower)):
        return {
            "text": (
                "**Next Week Occupancy Forecast**: Yes! The model can forecast occupancy for next week.\n\n"
                "The feature engineering pipeline computes future calendar indicators, trigonometric cyclical sine/cosine "
                "encodings of the hour and day of week, and seasonal lag features (such as 168-hour / 7-day lags), enabling "
                "accurate out-of-sample forward forecasts for any upcoming open library operating hours."
            ),
            "cards": None,
        }

    if any(_word_boundary_search(q_lower, term) for term in ["book", "books", "author", "subject", "title", "textbook", "catalogue", "catalog", "programming", "computer science"]):
        query = _strip_command_prefixes(q_raw)
        query = re.sub(r"\b(?:book|books|author|subject|title|textbook|catalogue|catalog|find|show|search|recommend|about|for|by|on|in)\b", " ", query, flags=re.IGNORECASE)
        query = re.sub(r"\s+", " ", query).strip(" ,.:;!?-_")
        query_aliases = {"ai": "artificial intelligence", "ml": "machine learning", "cs": "computer science", "programming": "python programming"}
        query = query_aliases.get(query.casefold(), query)
        if len(query) < 3:
            return {"text": "Tell me a subject, title, or author and I’ll search the library catalogue.", "cards": None}
        docs = search_local_books(query, limit=5)
        if docs:
            lines = []
            for book in docs:
                title = str(book.get("title", "Untitled"))
                authors = book.get("author_name", book.get("authors", [])) or []
                if not isinstance(authors, (list, tuple)):
                    authors = [str(authors)]
                author = ", ".join(str(name) for name in authors[:2]) or "Author not listed"
                year = book.get("first_publish_year", "Year unavailable")
                link = kth_primo_search_url(title)
                lines.append(f"- {title} ({year}) · {author}\n  [KTH Primo holdings]({link})")
            return {"text": "I found these matches.\n\n" + "\n\n".join(lines) + "\n\nUse the Primo link to confirm current holdings.", "cards": None}
        return {"text": "I couldn’t find a strong match in the local catalogue. Try a simpler title, author, or subject keyword.", "cards": None}

    if current_page == "Overview":
        return {"text": "I can compare study zones, explain the forecast, and highlight quieter times. Ask about current occupancy, busiest periods, or the quietest place right now.", "cards": None}
    if current_page == "Dataset":
        metrics_path = Path(__file__).resolve().parent.parent / "models" / "metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            summary = metrics.get("summary", {})
            avg_mae = summary.get("avg_xgb_mae", "unknown")
            return {"text": f"The model MAE is about {avg_mae} across the evaluated zones. I can explain the feature set, dataset date range, and error measurements in the Data & model page.", "cards": None}
    if current_page == "Settings":
        return {"text": "I can help you add a Gemini API key, choose a supported model, and confirm where the key is stored locally in the project .env file. Never paste or display a secret value in chat.", "cards": None}

    return {"text": f"I’m ready to help with {page_label}. You can ask about study spaces, occupancy, books, library facilities, or the model and dataset.", "cards": None}


def render_recommendation_card_html(card: dict) -> str:
    """Render a compact green-and-black recommendation card for display inside chat bubbles."""
    if not isinstance(card, dict) or not card:
        return ""
    name = html.escape(str(card.get("name") or "Study Spot"))
    level = html.escape(str(card.get("level") or ""))
    occupancy = card.get("occupancy")
    seats = card.get("free_seats")
    if occupancy is None or (isinstance(occupancy, float) and math.isnan(occupancy)):
        occupancy_display = "—"
    else:
        occupancy_display = f"{float(occupancy):.0f}%"
    if seats is None or (isinstance(seats, float) and math.isnan(seats)):
        seats_display = "—"
    else:
        seats_display = f"~{seats} seats free"
    return f"""
    <div style="background: var(--surface); border: 2px solid var(--green); border-radius: 10px; padding: 0.85rem 1rem; margin: 0.6rem 0; box-shadow: 0 0 14px rgba(0, 208, 132, 0.25);">
      <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.35rem;">
        <span style="font-weight: 800; font-size: 1.1rem; color: var(--ink);">{name}</span>
        <span style="background: var(--green); color: var(--paper); font-size: 0.72rem; font-weight: 800; padding: 2px 8px; border-radius: 9999px; text-transform: uppercase;">
          ★ Best Match
        </span>
      </div>
      <div style="font-size: 0.82rem; color: var(--muted); margin-bottom: 0.4rem;">{level}</div>
      <div style="display: flex; align-items: center; gap: 0.5rem; font-size: 0.82rem; flex-wrap: wrap;">
        <span style="background: rgba(0, 208, 132, 0.15); color: var(--green); border: 1px solid var(--green); font-weight: 700; padding: 2px 7px; border-radius: 4px;">
          {occupancy_display} Occupancy
        </span>
        <span style="color: var(--ink); font-weight: 700;">{seats_display}</span>
      </div>
    </div>
    """


def init_chat_session(current_page: str):
    """Keep a separate, persistent conversation for each app page."""
    if "chat_sessions" not in st.session_state:
        previous_history = list(st.session_state.get("chat_history", []))
        previous_page = st.session_state.get("chat_last_page") or "Home"
        st.session_state.chat_sessions = {}
        if previous_history:
            st.session_state.chat_sessions[previous_page] = {"history": previous_history}

    sessions = st.session_state.chat_sessions
    if current_page not in sessions:
        sessions[current_page] = {"history": []}

    history = sessions[current_page].setdefault("history", [])
    if not history:
        intro_text = PAGE_INTRODUCTIONS.get(current_page, PAGE_INTRODUCTIONS["Home"])
        history.append({
            "id": f"intro_{current_page}_{time.time()}",
            "role": "assistant",
            "content": intro_text,
            "cards": None,
            "timestamp": datetime.now(KTH_TZ).strftime("%H:%M"),
        })

    # Backward-compatible alias used by the planner integration.
    st.session_state.chat_history = history
    st.session_state.chat_last_page = current_page
    st.session_state.mascot_state = st.session_state.get("mascot_state", "idle")
    st.session_state.last_planner_data = st.session_state.get("last_planner_data")


def _stream_local_text(text: str):
    """Yield local text word-by-word with small delay for a smooth realistic typing effect."""
    value = str(text or "").strip()
    if not value:
        value = "I couldn't prepare a useful answer. Please try asking in a different way."
    words = value.split(" ")
    for idx, word in enumerate(words):
        suffix = " " if idx < len(words) - 1 else ""
        yield word + suffix
        time.sleep(0.02)



def _render_aligned_chat_message(role: str, content: str, cards=None, timestamp=None):
    """Render chat message with clear avatar and neat layout."""
    with st.chat_message(role):
        st.markdown(content or "")
        for card in cards or []:
            if card:
                st.html(render_recommendation_card_html(card))
        if timestamp:
            st.caption(timestamp)


def _render_chat_history(history, page_key: str):
    for msg in history[-40:]:
        _render_aligned_chat_message(
            msg.get("role", "assistant"),
            msg.get("content", ""),
            cards=msg.get("cards"),
            timestamp=msg.get("timestamp"),
        )


def _prepare_chat_reply(user_text: str, current_page: str, bundle: dict, transcript, page_key: str):
    """Stream and persist one assistant turn inside the transcript placeholder."""
    timestamp = datetime.now(KTH_TZ).strftime("%H:%M")
    user_message = {
        "id": f"user_{time.time()}",
        "role": "user",
        "content": user_text,
        "cards": None,
        "timestamp": timestamp,
    }
    history = st.session_state.chat_history
    history.append(user_message)
    st.session_state.mascot_state = "thinking"
    system_prompt = build_system_prompt(current_page, bundle)

    def local_fallback():
        fallback = handle_deterministic_query(user_text, current_page, bundle)
        fallback_result.update(fallback)
        yield from _stream_local_text(fallback.get("text", ""))

    fallback_result = {}
    stream, _provider_label = generate_ai_response_stream(
        system_prompt,
        history,
        user_text,
        fallback_generator=local_fallback(),
    )

    with transcript:
        with st.chat_message("user"):
            st.markdown(user_message["content"])
            if user_message.get("timestamp"):
                st.caption(user_message["timestamp"])
        with st.chat_message("assistant"):
            output = st.empty()
            output.markdown("_Thinking…_")
            try:
                response_text = output.write_stream(stream, cursor="▍")
            except Exception:
                logger.exception("Unexpected error while streaming a chatbot reply")
                response_text = "I'm having trouble connecting right now. Please try again."
                output.markdown(response_text)
    if not isinstance(response_text, str):
        response_text = "".join(str(part) for part in response_text)
    response_text = response_text.strip() or "I couldn't prepare a useful answer. Please try again."
    history.append({
        "id": f"assistant_{time.time()}",
        "role": "assistant",
        "content": response_text,
        "cards": fallback_result.get("cards"),
        "timestamp": datetime.now(KTH_TZ).strftime("%H:%M"),
    })
    st.session_state.mascot_state = "idle"


@st.fragment
def render_chatbot_panel(bundle: dict, current_page: str):
    """Render a fixed-height assistant with its own scrollable transcript and bottom composer."""
    init_chat_session(current_page)
    page_key = current_page.lower().replace(" ", "_")
    providers = get_configured_providers()
    primary_provider = providers[0][0].lower() if providers else "gemini"
    provider_text = f"{primary_provider.capitalize()} configured"
    provider_dot_class = "is-ok" if providers else "is-neutral"

    if st.session_state.get("ai_notice"):
        st.caption(st.session_state.pop("ai_notice"))

    pending = st.session_state.pop("pending_planner_data", None)
    if pending:
        prev_data = st.session_state.get("last_planner_data")
        user_msg = (
            f"Find my best study spot for {pending.get('date')} at "
            f"{pending.get('arrival_hour'):02d}:00 ({pending.get('duration_str')}, "
            f"{pending.get('environment')})."
        )
        done = False
        for idx, item in enumerate(st.session_state.chat_history):
            if item.get("role") == "user" and item.get("content") == user_msg:
                done = True
                break
        if not done:
            recs = evaluate_recommendations(bundle, pending)
            rec_res = format_concise_recommendation(recs, pending, prev_data=prev_data)
            st.session_state.chat_history.extend([
                {"id": f"planner_user_{time.time()}", "role": "user", "content": user_msg, "cards": None, "timestamp": datetime.now(KTH_TZ).strftime("%H:%M")},
                {"id": f"planner_ai_{time.time()}", "role": "assistant", "content": rec_res["text"], "cards": [rec_res["card"]] if rec_res.get("card") else None, "timestamp": datetime.now(KTH_TZ).strftime("%H:%M")},
            ])
            st.session_state.last_planner_data = pending

    panel_key = f"chatbot_panel_{page_key}"
    with st.container(key=panel_key, border=True):
        header_row = st.columns([6.5, 1.2], vertical_alignment="center")
        with header_row[0]:
            st.markdown(
                f"""
                <div class="assistant-heading">
                  <div class="assistant-eyebrow">SMART LIBRARY · AI ASSISTANT</div>
                  <div class="assistant-title">Your library companion</div>
                  <div class="assistant-subtitle">Ask about study spaces, occupancy, books and library services</div>
                  <span class="assistant-status"><span class="assistant-status-dot {provider_dot_class}"></span>{html.escape(provider_text)}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with header_row[1]:
            if st.button("✚", key=f"new_chat_{page_key}", help="New chat", use_container_width=True):
                st.session_state.chat_sessions[current_page]["history"] = [{
                    "id": f"intro_{current_page}_{time.time()}",
                    "role": "assistant",
                    "content": PAGE_INTRODUCTIONS.get(current_page, PAGE_INTRODUCTIONS["Home"]),
                    "cards": None,
                    "timestamp": datetime.now(KTH_TZ).strftime("%H:%M"),
                }]
                st.session_state.chat_history = st.session_state.chat_sessions[current_page]["history"]
                st.rerun()

        transcript = st.container(height=560, key=f"chat_messages_{page_key}")
        with transcript:
            _render_chat_history(st.session_state.chat_history, page_key)

        suggestions = PAGE_SUGGESTIONS.get(current_page, PAGE_SUGGESTIONS["Home"])[:3]
        if len(st.session_state.chat_history) <= 2:
            st.markdown('<div class="assistant-suggestions-label">TRY ASKING</div>', unsafe_allow_html=True)
            suggestion_row = st.container()
            with suggestion_row:
                for index, suggestion in enumerate(suggestions):
                    if st.button(suggestion, key=f"assistant_suggestion_{page_key}_{index}", use_container_width=False):
                        st.session_state[f"pending_assistant_prompt_{page_key}"] = suggestion
                        st.rerun()

        prompt = st.session_state.pop(f"pending_assistant_prompt_{page_key}", None)
        if prompt:
            _prepare_chat_reply(prompt, current_page, bundle, transcript, page_key)
            st.rerun()

        with st.form(key=f"assistant_composer_form_{page_key}", clear_on_submit=True):
            input_col, button_col = st.columns([5.5, 1], gap="small", vertical_alignment="center")
            with input_col:
                user_input = st.text_input(
                    "Message your library assistant",
                    key=f"assistant_chat_input_{page_key}",
                    placeholder="Ask about study spots, books, facilities…",
                    label_visibility="collapsed",
                )
            with button_col:
                submitted = st.form_submit_button("Send", type="primary", use_container_width=True)
        if submitted and user_input is not None:
            cleaned = str(user_input).strip()
            if cleaned:
                _prepare_chat_reply(cleaned, current_page, bundle, transcript, page_key)
                st.rerun()
