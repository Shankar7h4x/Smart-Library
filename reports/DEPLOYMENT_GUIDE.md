# Smart Library — Production Deployment Guide

**Target Environment:** Streamlit Community Cloud / Docker Container / Linux Server  
**Python Runtime:** Python 3.10 – 3.12  
**Framework:** Streamlit 1.40+  

---

## 1. Architecture Overview

The Smart Library application is structured as a zero-latency cached analytical application:
- **Model Bundle (`models/occupancy.joblib`):** Loaded once at startup using `@st.cache_resource`. In-memory inference takes under 1 millisecond.
- **Processed History (`data/processed/hourly.parquet`):** Loaded via `@st.cache_data` for rapid analytics rendering.
- **Evaluation Telemetry (`models/metrics.json`, `models/backtest.parquet`):** Loaded lazily on the "Data & model" page.
- **External Integrations:** Optional LLM assistants (Google Gemini, OpenAI, Groq) with graceful fallbacks if keys are unconfigured.

---

## 2. Prerequisites & Dependencies

### Python Environment
- Python 3.10, 3.11, or 3.12 (tested and verified on Python 3.12).
- Pip package dependencies pinned in `requirements.txt`:

```text
streamlit>=1.40,<2
streamlit-option-menu>=0.4,<1
pandas>=2.2,<4
numpy>=1.26,<3
scikit-learn>=1.4,<2
xgboost>=2.0,<4
plotly>=5.24,<7
holidays>=0.57,<1
joblib>=1.4,<2
pyarrow>=17,<25
requests>=2.32,<3
datasets>=3,<5
openai>=1.40,<3
groq>=0.11,<2
```

---

## 3. Environment Variables & Secrets Management

The application looks for configuration in two places:
1. Streamlit Secrets (`.streamlit/secrets.toml`) on Streamlit Cloud.
2. Local `.env` file (loaded automatically if present via `config.py`).

### Supported Environment Variables:
```ini
# AI Provider Selection (options: gemini, openai, groq)
AI_PROVIDER=gemini

# Google AI Studio API Key (Recommended default)
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash

# Optional: OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini

# Optional: Groq API Key
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

> **Security Note:** Never commit `.env` or API keys to public repositories. `.gitignore` is pre-configured to strictly ignore all `.env` files while preserving `.env.example`.

---

## 4. Deploying to Streamlit Community Cloud

1. **Push Repository to GitHub:**
   Ensure `models/occupancy.joblib`, `models/metrics.json`, and `data/processed/hourly.parquet` are committed so the deployed app runs instantly without needing retraining on cold start.

2. **Connect Streamlit Cloud:**
   - Log into [share.streamlit.io](https://share.streamlit.io/).
   - Select your repository and branch (`main`).
   - Set **Main file path:** `app.py`.

3. **Configure Secrets in Streamlit Cloud Dashboard:**
   In the App Settings → Secrets, paste:
   ```toml
   AI_PROVIDER = "gemini"
   GEMINI_API_KEY = "your_actual_key"
   GEMINI_MODEL = "gemini-2.5-flash"
   ```

4. **Deploy:**
   Click **Deploy**. Streamlit Cloud will install dependencies from `requirements.txt` and launch the application.

---

## 5. Local Development & Self-Hosting

### Quick Start (PowerShell / Windows):
```powershell
# 1. Clone repository
cd project/old

# 2. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 3. Run automated tests to verify environment
python tests/run_all_tests.py

# 4. Launch Streamlit application
streamlit run app.py
```

### Quick Start (Linux / macOS / Bash):
```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run automated test suite
python tests/run_all_tests.py

# 4. Launch application
streamlit run app.py --server.port 8501
```

---

## 6. Model Retraining in Production

To update the model with new sensor data:
```bash
# 1. Run data preparation pipeline (validates raw data, generates 22 features)
python src/prepare_data.py

# 2. Run model training & time-series CV selection
python src/train.py

# 3. Run test suite to verify model contract
python tests/run_all_tests.py
```
`src/train.py` automatically backs up the existing production model to `models/occupancy_prev.joblib` before saving the new bundle.

---

## 7. Operational Health Checks & Verification

Verify these three checks post-deployment:
1. **Model Loading:** Navigate to "Study planner" — ensure zone recommendation cards render without warning banners.
2. **Telemetry Page:** Navigate to "Data & model" — ensure the metrics cards show `Selected Model MAE: 13.65 pp` with the actual vs predicted Plotly chart rendered.
3. **Automated Suite:** Run `python tests/run_all_tests.py` — ensure all 69 tests pass.
