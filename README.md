# Smart Library Occupancy & Resource Intelligence System

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent, student-facing dashboard providing occupancy forecasts, visit planning, and library resource intelligence for KTH Royal Institute of Technology Library in Stockholm, Sweden. It uses a historical occupancy archive for model estimates and does not claim to have a verified live sensor feed.

---

## 1. Overview

**Interface refresh:** The application now uses a clean editorial layout with a compact horizontal navigation, warm light canvas, forest-green accents, and a consistent responsive component style. See `REDESIGN_NOTES.md` for the new page map and design decisions.

Finding quiet study spots in university libraries during peak semesters and exam periods is a perennial challenge. **Smart Library Occupancy & Resource Intelligence System** solves this by converting historical sensor telemetry into predictive intelligence.

Students can:
- Review model-estimated crowd levels for the whole library and individual study zones.
- Estimate available seats across 5 distinct library galleries.
- Plan optimal 7-day visit times to avoid peak congestion.
- Search real online book records and view public reader-interest and digital lending signals.
- Analyze historical usage patterns and machine learning prediction accuracy.

---

## 2. Main Features

### 🏠 Home — Plan a Study Session
- **Visit inputs**: Select a date, arrival time, study duration, preferred area, and exam-period flag.
- **Session timeline**: See the hourly Free, Busy, Full, or Closed forecast for the complete planned stay.
- **Data freshness**: Shows the latest archived dataset observation separately from saved-model forecasts; the archive is not treated as a verified live sensor feed.

### 📊 Overview — Compare All Areas
- **All-zone status**: Compare estimated occupancy and available seats across the whole library and each study wing.
- **Hourly comparison**: Review all selected area forecasts throughout a day and highlight the selected visit time.

### 📚 Resources — Online Book Search
- **Book search**: Search the bundled discovery dataset and Open Library when you submit a search.
- **Reader Interest**: Show Open Library want-to-read and currently-reading counts when available; these are global reader signals, not KTH borrowing demand.
- **Supply Check**: Open KTH Primo to check current local copies and reservations. Open Library records do not establish current KTH physical-copy availability.

### 🧾 Data & Model — Sources & Model Performance
- **Data source and freshness**: Show the processed history period and latest Hugging Face observation.
- **Transparent evaluation**: Report model and baseline MAE in occupancy percentage points, per-area scores, and the held-out sample count.
- **Limitations**: Explain stale telemetry, assumed seat capacities, and the difference between Open Library interest and KTH borrowing demand.

---

## 3. Technology Stack

- **Frontend & App Framework**: [Streamlit](https://streamlit.io/) with custom SaaS CSS, modern typography (`DM Sans` and `Manrope`), and [`streamlit-option-menu`](https://github.com/victoryhb/streamlit-option-menu).
- **Machine Learning**: [XGBoost Regressor](https://xgboost.readthedocs.io/) and [scikit-learn](https://scikit-learn.org/).
- **Data Engineering**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [PyArrow](https://arrow.apache.org/docs/python/), and [python-holidays](https://github.com/vacanza/python-holidays).
- **Interactive Visualizations**: [Plotly](https://plotly.com/python/).
- **Model Serialization**: [Joblib](https://joblib.readthedocs.io/).

---

## 4. Dataset Source

The application can retrieve the newest row from the public Hugging Face Dataset Viewer API for [`davnas/real-time-library-occupancy`](https://huggingface.co/datasets/davnas/real-time-library-occupancy). This dataset is an archive and is not treated as a verified live sensor service:
- **Location**: KTH Royal Institute of Technology Library, Stockholm, Sweden.
- **Timeline**: December 2024 to July 2025 (~95,000 raw 3-minute readings).
- **Zones Tracked**: Whole Library, South-East Gallery, North Gallery, South Gallery, Ångdomen, Newton.
- **Preprocessed**: Aggregated to 5,616 hourly observations with holiday calendars and academic exam periods.

---

## 5. Project Structure

```text
smart-library/
│
├── app.py                      # Main entrypoint and redesigned navigation controller
├── config.py                   # Central paths, zone mappings, colors, and constants
├── ui.py                       # Design system, CSS injection, hero, gauge, notes, chart styling
├── ui_extra.py                 # Hour strips, slot cards, stat cards, tip cards, and legends
├── requirements.txt            # Production dependencies for deployment
├── README.md                   # Complete documentation
├── REDESIGN_NOTES.md           # New UI direction and navigation map
├── .gitignore                  # Git ignore rules (keeps data & models tracked)
│
├── .streamlit/
│   └── config.toml             # Theme defaults (runtime light/dark palettes are applied by `ui.py`)
│
├── data/
│   ├── raw/
│   │   └── occupancy.parquet   # Local cached raw dataset (95,164 records)
│   └── processed/
│       └── hourly.parquet      # Hourly resampled dataset (5,616 hours)
│
├── models/
│   ├── occupancy.joblib        # Model bundle (models, resid_std, open_profile, capacities)
│   ├── metrics.json            # MAE, RMSE, and baseline comparison metrics
│   └── backtest.parquet        # Out-of-sample backtest predictions
│
├── src/
│   ├── prepare_data.py         # Pipeline: download, clean, resample, feature engineer
│   ├── train.py                # Model training, baseline evaluation, retraining, export
│   ├── forecast.py             # Pure prediction, windowing, slot optimization, smart picker
│   ├── books.py                # Online book lookup service
│   ├── book_data.py            # Open Library API client and KTH Primo links
│   ├── pages_home.py           # Home page UI rendering
│   ├── pages_planner.py        # Overview and all-zone comparison UI
│   ├── pages_resources.py      # Resources page UI rendering
│   └── pages_dataset.py        # Data-source and model-performance UI
│
└── tests/
    └── test_functions.py       # Forecast engine checks
```

---

## 6. Local Setup & Execution

### 1. Clone & create environment
```bash
git clone <your-repo-url>
cd smart-library
python -m venv .venv

# Activate environment:
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
```

### 2. Install production dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Run data preparation & training from scratch
*Note: The repository already includes pre-processed data and trained models. To re-run the pipeline:*
```bash
pip install datasets huggingface_hub
python src/prepare_data.py
python src/train.py
```

### 4. Run the test suite
```bash
python tests/test_functions.py
```

### 5. Launch the Streamlit application
```bash
streamlit run app.py
```

The application will start locally at `http://localhost:8501`.

---

## 7. Streamlit Community Cloud Deployment

1. Push this entire project repository to GitHub.
2. Verify that `data/raw/occupancy.parquet`, `data/processed/hourly.parquet`, and `models/` are committed.
3. Log in to [Streamlit Community Cloud](https://share.streamlit.io/).
4. Click **New app**, select your GitHub repository and branch.
5. Set the Main file path to `app.py`.
6. Click **Deploy!**

The trained models and historical Parquet files are bundled locally. The app also makes public, no-token requests to Hugging Face for the newest occupancy observation and to Open Library when a person searches for a book. If those services are unreachable, occupancy forecasts still use the local model; current KTH physical-copy status is checked through the linked Primo catalog.

---

## 8. Limitations & Methodological Disclaimers

1. **Assumed Capacities**: Zone seating capacities (North: 120, South: 100, South-East: 100, Ångdomen: 60, Newton: 40) are approximate engineering assumptions for visit planning, not architecturally certified headcounts.
2. **No KTH Circulation API**: Hugging Face provides occupancy measurements only. Open Library reader interest is a global proxy, not KTH loan demand. Check KTH Primo for authoritative local copy and reservation status.
3. **Historical Telemetry**: The dataset currently covers December 2024 to July 2025. The app marks this feed stale and shows forecasts unless a future refresh contains recent observations.
4. **Baseline Performance**: In highly cyclical university environments, simple Day×Hour historical averages form a competitive baseline; the Dataset page reports model error and the held-out sample count transparently.

---

## 9. Future Improvements

- Integration of live Wi-Fi probe API feeds when authorized by university infrastructure.
- Group study room reservation system integration.
- Modelled quiet/collaborative preference profiles for the configured study zones.
- Push notifications for quiet hour alerts.


## AI setup (in the app)
1. Start the app with `streamlit run app.py`.
2. Open **Settings** in the top navigation.
3. Choose **openai**, enter your OpenAI API key, choose a model, and click **Save AI settings**.
4. Return to the chatbot and send a message. The key is stored in the local `.env` file, which is excluded from Git.

You can also configure Groq or Gemini from the Optional providers section. API usage is billed by the provider according to your account and selected model.

## Book discovery
Enter a title or keyword (for example, `Harry Potter`) and click **Discover & Search Stacks**. Results are fetched from Open Library and local reserve data. Use each **Check KTH Primo holdings** link to verify whether a physical copy is available at KTH; Open Library metadata does not confirm KTH inventory.

## Theme
Use the **Dark mode** switch in the top bar. The selected theme is retained for the current Streamlit session.


## 8. AI assistant setup and troubleshooting

The assistant can use OpenAI, Groq, or Gemini. Open **Settings → AI connection**, select a provider, enter its API key, and save. The key is stored in the local `.env` file; that file is intentionally excluded from the project archive.

If no cloud key is configured, the assistant still answers supported library questions using the local occupancy model and catalogue. If a configured provider fails during streaming, the app tries the next configured provider and then falls back to the local assistant. A cloud connection failure is shown in the response rather than leaving the chat silently blank.

Book searches from the Resources form update the first horizontal shelf in the lower **Academic Stacks & Digital Course Reserves** section. They do not insert result cards between the search form and the assistant.

For a fresh installation:

```bash
pip install -r requirements.txt
streamlit run app.py
```

See `FIXES_V5.md` for the detailed v5.0 regression and interface changes.


## v6.0 interface update

See `FIXES_V6.md` for the redesigned navigation, theme control, full-width layout, advanced book filters and catalogue availability verification workflow.


## v7.0 reliability and security notes

- The Streamlit application (`streamlit run app.py`) is the canonical application.
- `start_web.py` is an optional static preview, bound to `127.0.0.1` and limited to `web/`.
- Zone capacities in `config.py` are planning assumptions, not verified KTH seat counts.
- The bundled occupancy data is historical/archive data. A successful dataset API response does not establish a live sensor feed.
- Occupancy outputs are model estimates unless an explicitly validated observation source is displayed.
- Bundled book records are discovery metadata only. Current KTH holdings and copy availability must be checked in KTH Primo.
- AI provider keys are stored in `.env`; the settings writer updates managed AI keys while preserving unrelated entries. Cloud prompts are sent to the selected provider.
- Security defaults enable Streamlit CORS and XSRF protection.
- Run the dependency-light v7 regression tests with `python -m pytest -q`. The legacy test scripts are retained for manual use and are not part of pytest discovery.
