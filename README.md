# Lotto Analyzer

**Lotto Analyzer** is a Streamlit-based web app for exploring Korean Lotto 6/45 draw history,
visualizing number statistics, and generating **data-driven number suggestions**.

It focuses on:

- Clean, up-to-date draw data (fetched from the official API)
- Intuitive visualizations (Plotly + Streamlit)
- A few transparent recommendation strategies
- Lightweight fairness diagnostics

<img width="1757" height="676" alt="image" src="https://github.com/user-attachments/assets/d15d4c43-a40e-4f0f-be7d-64b5b43daed5" />

---

## Table of Contents

1. [Key Ideas & Disclaimer](#key-ideas--disclaimer)  
2. [Features at a Glance](#features-at-a-glance)  
3. [High-level Architecture](#high-level-architecture-2)  
4. [UI Overview – Using the App in the Browser](#ui-overview--using-the-app-in-the-browser)  
   - [Data & Overview](#data--overview)  
   - [Frequency & Rolling Trends](#frequency--rolling-trends)  
   - [Recommendations](#recommendations)  
   - [Fairness Checks](#fairness-checks)  
   - [Member Management (Optional)](#member-management-optional)  
5. [Core Modules](#core-modules)  
6. [Requirements & Installation](#requirements--installation-2)  
7. [Running the App](#running-the-app)  
8. [Notes](#notes)

---

## Key Ideas & Disclaimer

- Lotto draws are **random by design**.
- This project does **not claim to predict** future draws or guarantee any gain.
- The goal is to:
  - Provide a practical example of **data pipeline + analytics + UI** in Python.
  - Help users **understand patterns** (frequency, co-occurrence, etc.) in historical data.
  - Demonstrate basic fairness checks and simple recommendation heuristics.

Use it for learning and analysis, **not** as a financial strategy.

---

## Features at a Glance

- **Data ingestion** from the official Korean Lotto 6/45 API.
- **Incremental updates** to a local CSV (no need to re-download everything).
- **Frequency analysis** (overall and rolling windows).
- **Presence matrices** and **co-occurrence** between numbers.
- **Feature engineering** (sum, range, odd/even, last digit, etc.).
- **Recommendation strategies**:
  - Hot / Cold / Balanced
  - Weighted by recent history
- **Fairness diagnostics**:
  - Uniformity (χ² approximation)
  - Pair over-representation with FDR correction
- Streamlit UI with:
  - KPI cards
  - Plotly charts
  - Interactive controls

---

## High-level Architecture

```text
lotto-analyzer/
└─ lotto-analyzer-main/
   ├─ main.py            # Streamlit app entry point
   ├─ lotto_data.py      # Data download, CSV management, frequency helpers
   ├─ rolling.py         # Rolling-window frequency computation
   ├─ features.py        # Draw-level feature engineering
   ├─ recs.py            # Recommendation strategies
   ├─ fairness.py        # Statistical tests (uniformity / pair significance / FDR)
   ├─ viz.py             # Plotly + Streamlit visualization helpers
   ├─ update_data.py     # CLI utility to update data/lotto_draws.csv
   ├─ requirements.txt   # Original dependency pinning (reference)
   ├─ .devcontainer/
   │   └─ devcontainer.json
   └─ streamlit/
       └─ config.toml    # Optional Streamlit configuration
```

---

## UI Overview – Using the App in the Browser

### Data & Overview

When running `streamlit run main.py`, the app opens in your browser:

- On startup:
  - Loads `data/lotto_draws.csv`.
  - If needed, calls `incremental_update()` to fetch missing draws from the official API.
- Overview section (exact layout may vary):
  - Shows total number of draws.
  - Highlights KPIs (e.g., top numbers, last draw summary).
  - Applies a consistent visual style via `viz.apply_global_style()`.

A typical user **does not need to know Python** – they simply:

1. Launch the app.
2. Wait for data to sync.
3. Explore numbers, charts, and suggestions on each tab.

---

### Frequency & Rolling Trends

Backed by:

- `lotto_data.frequency()`
- `rolling.rolling_frequency()`
- Visualization functions in `viz.py`

The UI can show:

- **Overall frequency** (1–45), with bar charts:
  - Horizontal or vertical bars.
  - Top-N charts highlighting the most frequent numbers.
- **Rolling frequency**:
  - Frequency of each number in the last `N` draws across time.
  - Helps visualize “hot” and “cold” phases for each number.

Interactive features:

- Select whether to include the **bonus** number in calculations.
- Adjust rolling window size.
- Hover over Plotly charts for exact values.

---

### Recommendations

Implemented mainly in `recs.py` and surfaced in `main.py`.

Strategies include:

- **Hot** – pick from the most frequent numbers overall.
- **Cold** – pick from the least frequent numbers.
- **Balanced** – mix of hot and cold numbers.
- **Weighted recent** – sample numbers based on their frequency in recent draws.

For each suggestion:

- Results are shown as 6 unique numbers in the 1–45 range.
- A small stats panel based on `composition_metrics()` can show:
  - Sum
  - Range
  - Odd / even split
  - Count of low numbers, etc.

All strategies are **transparent and simple** – good for educational purposes.

---

### Fairness Checks

Backed by `fairness.py`.

Includes:

- **Uniformity check**
  - Uses a χ²-like statistic and Wilson–Hilferty approximation to produce p-values.
  - Helps see whether overall frequencies deviate strongly from a uniform assumption.

- **Pair over-representation**
  - For each pair of numbers, uses a binomial approximation to test whether
    the observed co-occurrence is unusually high under independence.
  - Applies Benjamini–Hochberg FDR correction to control false discovery rate.

Results are presented as:

- KPIs (e.g., χ² value, global p-value).
- Tables or plots highlighting “interesting” pairs.

These checks are approximations (not full-blown statistical packages),
but they provide a good starting point for exploratory analysis.

---

### Member Management (Optional)

`main.py` contains optional hooks for:

- Simple **member registration** (name + phone).
- CSV-based storage (`members.csv`).
- Optional Supabase integration for remote storage (using `requests`).

From a UI perspective:

- Users can enter name and phone.
- The app normalizes phone numbers (E.164) and may hash them for privacy.
- An admin tab can list registered members and provide a CSV download.

> This is optional and can be disabled/ignored if you only care about statistics.

---

## Core Modules

### `lotto_data.py`

Responsibilities:

- Download and update lotto draw data via:

  ```text
  https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo={drwNo}
  ```

- Functions:
  - `find_latest_draw()`
  - `collect_range()`
  - `load_csv(csv_path)`
  - `incremental_update(csv_path)` – atomic CSV updates + deduplication.
  - `frequency(df, include_bonus)`
  - `presence_matrix(df, include_bonus)`
  - `cooccurrence(only_num)`

### `rolling.py`

- `rolling_frequency(df, window, include_bonus)`:
  - For each draw, counts appearances in the previous `window` draws.

### `features.py`

- `build_features(df)`:
  - Adds sum, range, odd/even counts, low-number counts, consecutive flags, last-digit modes.
- `last_digit_hist(df)`:
  - Histogram of last digits (0–9) across all numbers.

### `recs.py`

- Provides recommenders: hot, cold, balanced, weighted recent.
- Uses clean NumPy / Pandas logic, easy to extend with new strategies.

### `fairness.py`

- Implements:
  - χ² uniformity approximation
  - Binomial pair significance (upper-tail)
  - Benjamini–Hochberg FDR

All using **NumPy and math**, without SciPy/Statsmodels.

### `viz.py`

- Houses all visualization style and chart functions.
- Contains:
  - Global CSS / style injection (font, colors, spacing).
  - KPI card generator (`kpi_card`).
  - Plotly-based figures for frequencies, rolling trends, co-occurrence.
  - Optional SciPy-based clustering for heatmaps (if SciPy is installed).

---

## Requirements & Installation

### Python & OS

- **Python**: 3.9+ recommended  
- **OS**: Any OS where Streamlit + required libs work (Windows, macOS, Linux).

### Python dependencies

Based on the included `requirements.txt`:

```text
streamlit==1.37.1
pandas==2.2.3
numpy==2.1.3
requests==2.32.3
plotly==5.23.0
tqdm==4.66.4
```

- `reportlab` is only needed if you add PDF export features (not required by core code).
- `scipy` is optional – only used for more advanced clustering in `viz.py` if present.

Install:

```bash
pip install -r requirements.txt
```

---

## Running the App

From `lotto-analyzer-main`:

```bash
streamlit run main.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`).

### Optional CLI updater

You can also update the CSV beforehand:

```bash
python update_data.py --data-path data/lotto_draws.csv
```

This will:

- Call the official API.
- Append missing draws.
- Deduplicate and sort by draw number.

---

## Notes

- All analysis is retrospective and based on observed draws.
- The app is ideal as:
  - A **demo project** (data download → processing → visualization → web UI).
  - A **playground** for statistics and visualization.
- Use it responsibly; there is no “winning formula” here – only transparent math
  and visualizations over past results.
