"""Automated Test Suite for Knowledge Service, Anti-Hallucination Ground Truth & Chat Layout.

Tests:
1. KnowledgeService ground truth accuracy (95,164 raw rows, 5,615 processed rows, 233 days, 22 features).
2. Sensor hardware disclaimer ("This information is not documented or available in the current project files.").
3. Model evaluation metrics (13.65 pp avg MAE, 13.01 pp whole library, 4.07 pp Ångdomen).
4. Deterministic query routing for all 10 verified evaluation questions.
5. Chatbot system prompt verification injection.
6. CSS layout contract: non-overlapping message bubbles, independent scrolling transcript, static timestamps.
"""

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]

from src.knowledge_service import KnowledgeService
from src.chatbot import build_system_prompt, handle_deterministic_query


def test_knowledge_service_dataset_summary():
    """Verify exact dataset volume and date range ground truth."""
    summary = KnowledgeService.get_dataset_summary()
    assert summary["raw_rows"] == 95164, "Raw rows must be exactly 95,164"
    assert summary["processed_rows"] == 5615, "Processed rows must be exactly 5,615"
    assert summary["raw_columns"] == 7
    assert summary["processed_columns"] == 29
    assert summary["open_operating_hours"] == 1621
    assert summary["closed_night_hours"] == 3994
    assert summary["date_range_days"] == 233
    assert summary["monitored_zones_count"] == 6


def test_knowledge_service_sensor_hardware_disclaimer():
    """Verify sensor hardware explicitly refuses to hallucinate Wi-Fi APs or PIR sensors."""
    sources = KnowledgeService.get_dataset_sources()
    assert "This information is not documented or available in the current project files." in sources["sensor_hardware"]
    assert "optical" in sources["hardware_disclaimer"].lower()
    assert "turnstiles" in sources["hardware_disclaimer"].lower()


def test_knowledge_service_model_metrics():
    """Verify verified model evaluation metrics match actual test holdout results."""
    m = KnowledgeService.get_model_metrics()
    assert abs(m["average_selected_mae"] - 13.65) < 0.05
    assert abs(m["average_baseline_mae"] - 27.35) < 0.05
    assert abs(m["whole_library_mae"] - 13.01) < 0.05
    assert abs(m["best_zone_mae"] - 4.07) < 0.05
    assert m["best_zone"] == "Ångdomen (Silent Sanctuary)"


def test_knowledge_service_algorithms_and_features():
    """Verify selected algorithms and 22 engineered features."""
    algos = KnowledgeService.get_model_algorithms()
    assert "Extra Trees" in algos["newton"]
    assert "XGBoost" in algos["library"]
    assert "Random Forest" in algos["angdomen"]

    features = KnowledgeService.get_feature_information()
    assert len(features) == 22, f"Expected 22 features, found {len(features)}"


def test_chatbot_ground_truth_question_1_dataset_explanation():
    """Q1: 'Explain the training dataset'"""
    res = handle_deterministic_query("Explain the training dataset", "Home", {})
    text = res["text"]
    assert "95,164" in text
    assert "5,615" in text
    assert "233" in text
    assert "6 zones" in text


def test_chatbot_ground_truth_question_2_row_counts():
    """Q2: 'How many rows are in the raw and processed datasets?'"""
    res = handle_deterministic_query("How many rows are in the raw and processed datasets?", "Home", {})
    text = res["text"]
    assert "95,164" in text
    assert "5,615" in text
    assert "millions" in text.lower() or "not" in text.lower()


def test_chatbot_ground_truth_question_3_date_range():
    """Q3: 'What is the date range of the dataset?'"""
    res = handle_deterministic_query("What is the date range of the dataset?", "Home", {})
    text = res["text"]
    assert "December 9, 2024" in text
    assert "July 31, 2025" in text
    assert "233" in text


def test_chatbot_ground_truth_question_4_sensor_hardware():
    """Q4: 'What sensors are used to collect occupancy data?'"""
    res = handle_deterministic_query("What sensors are used to collect occupancy data?", "Home", {})
    text = res["text"]
    assert "This information is not documented or available in the current project files." in text


def test_chatbot_ground_truth_question_5_model_mae():
    """Q5: 'What is the model MAE?'"""
    res = handle_deterministic_query("What is the model MAE?", "Home", {})
    text = res["text"]
    assert "13.65" in text
    assert "27.35" in text
    assert "4.07" in text


def test_chatbot_ground_truth_question_6_newton_algorithm():
    """Q6: 'Which algorithm is selected for Newton?'"""
    res = handle_deterministic_query("Which algorithm is selected for Newton?", "Home", {})
    text = res["text"]
    assert "Extra Trees" in text


def test_chatbot_ground_truth_question_7_features_count():
    """Q7: 'How many features does the model use?'"""
    res = handle_deterministic_query("How many features does the model use?", "Home", {})
    text = res["text"]
    assert "22 features" in text
    assert "shift(1)" in text


def test_chatbot_ground_truth_question_8_how_occupancy_predicted():
    """Q8: 'How is occupancy predicted?'"""
    res = handle_deterministic_query("How is occupancy predicted?", "Home", {})
    text = res["text"]
    assert "supervised" in text.lower()
    assert "regression" in text.lower()
    assert "capacity" in text.lower() or "desk" in text.lower()


def test_chatbot_ground_truth_question_9_unavailable_information():
    """Q9: 'What information is not available in the project?'"""
    res = handle_deterministic_query("What information is not available in the project?", "Home", {})
    text = res["text"]
    assert "Sensor Hardware" in text
    assert "Personal Data" in text or "PII" in text
    assert "Live" in text


def test_chatbot_ground_truth_question_10_predict_next_week():
    """Q10: 'Can the model predict occupancy for next week?'"""
    res = handle_deterministic_query("Can the model predict occupancy for next week?", "Home", {})
    text = res["text"]
    assert "Yes" in text
    assert "cyclical" in text.lower() or "168" in text.lower() or "calendar" in text.lower()


def test_system_prompt_includes_strict_ground_truth():
    """Verify LLM system prompt is injected with KnowledgeService ground truth context."""
    prompt = build_system_prompt("Home", {})
    assert "VERIFIED SMART LIBRARY PROJECT KNOWLEDGE (STRICT GROUND TRUTH)" in prompt
    assert "95,164" in prompt
    assert "5,615" in prompt
    assert "This information is not documented or available in the current project files." in prompt


def test_chat_layout_css_rules_prevent_message_overlap():
    """Verify CSS layout rules enforce non-overlapping bubbles, word wrapping, and clear captions."""
    ui_source = (ROOT / "ui.py").read_text(encoding="utf-8")

    # Message bubble rules
    assert "position: relative !important;" in ui_source
    assert "height: auto !important;" in ui_source
    assert "min-height: min-content !important;" in ui_source
    assert "overflow: visible !important;" in ui_source
    assert "flex: 0 0 auto !important;" in ui_source
    assert "flex-shrink: 0 !important;" in ui_source

    # Text wrapping rules
    assert "word-break: break-word !important;" in ui_source
    assert "overflow-wrap: break-word !important;" in ui_source

    # Caption / timestamp rules
    assert "display: block !important;" in ui_source
    assert "position: static !important;" in ui_source
    assert "clear: both !important;" in ui_source
