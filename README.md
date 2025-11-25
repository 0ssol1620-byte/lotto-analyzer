# Lotto Analyzer

A Streamlit-based web application for exploring Korean Lotto 6/45 draw history,
visualizing number statistics, and generating **data-driven number recommendations**.

It automatically downloads official draw results from the lottery API, maintains a
local CSV, and provides an interactive dashboard for:

- Frequency and rolling-window analysis
- Co-occurrence and presence matrices
- Simple feature engineering (sum/range/odd–even/etc.)
- Recommendation strategies (hot, cold, balanced, weighted recent)
- Basic fairness tests for uniformity and pair over-representation

<img width="1757" height="676" alt="image" src="https://github.com/user-attachments/assets/d15d4c43-a40e-4f0f-be7d-64b5b43daed5" />

---

## Features

### 1. Data ingestion & storage (`lotto_data.py`)

- Fetches Lotto 6/45 draw JSON from:

  ```text
  https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo={drwNo}
  ```

- Functions:
  - `find_latest_draw()` – detects the latest available draw number
  - `collect_range()` – downloads a range of draws and builds a DataFrame
  - `load_csv(csv_path)` – loads existing CSV (with safe defaults if missing)
  - `incremental_update(csv_path)` – appends only missing draws:
    - Fetches from the last stored draw up to the latest draw
    - Deduplicates by `draw_no` and keeps rows sorted
    - Uses an atomic temp-file + move pattern to avoid corrupted CSVs
  - `frequency(df, include_bonus)` – returns a 1–45 frequency Series
  - `presence_matrix(df, include_bonus)` – returns a 0/1 matrix per draw & number
  - `cooccurrence(only_num)` – builds a co-occurrence matrix of numbers

### 2. Rolling frequency analysis (`rolling.py`)

- `rolling_frequency(df, window=100, include_bonus=False)`
  - For each draw, computes the frequency of each number over the last *N* draws
  - Returns a DataFrame indexed by `draw_no`, columns = 1..45
  - Used in the UI to visualize how “hot” each number has been recently

### 3. Feature engineering (`features.py`)

For each draw (6 main numbers only):

- `sum` – sum of the six numbers
- `range` – max − min
- `odd_cnt` – count of odd numbers
- `low_cnt` – count of numbers ≤ 22
- `has_consecutive` – 1 if there is any consecutive pair, else 0
- `last_digit_mode` – most frequent last digit (0–9)

Also provides:

- `last_digit_hist(df)` – histogram of last digits across the history

### 4. Recommendation strategies (`recs.py`)

Implements several ways to suggest candidate numbers:

- `recommend_hot(freq)` – picks from the most frequent numbers
- `recommend_cold(freq)` – picks from the least frequent numbers
- `recommend_balanced(freq)` – tries to mix hot and cold
- `recommend_weighted_recent(df, lookback=200, include_bonus=False, seed=42)`
  - Uses recent draw history with a simple frequency-based weighting
  - Samples numbers according to their recent occurrence counts
- `composition_metrics(picks)` – evaluates a set of picks (sum/range/odd/low/etc.)
- `bonus_candidates(df)` – helper for bonus-number insights

All recommendations output sorted lists of 6 distinct numbers in the 1–45 range.

### 5. Fairness checks (`fairness.py`)

Pure NumPy implementations of basic significance tests:

- **Uniformity (χ² test)** using Wilson–Hilferty normal approximation
- **Pair co-occurrence significance** using a binomial upper-tail normal approximation
- **Benjamini–Hochberg FDR** correction for multiple testing

> SciPy/Statsmodels are intentionally not required; the approximations are designed
> to be “good enough” for dashboard-level diagnostics.

### 6. Visualization & UI helpers (`viz.py`)

Uses **Streamlit** + **Plotly** to create a consistent dashboard look:

- `apply_global_style()` – injects custom CSS (font, colors, cards, etc.)
- `kpi_card(title, value, sub=None)` – KPI tiles for headline metrics
- Top-N frequency charts (horizontal & vertical bars)
- Rolling-frequency line charts
- Co-occurrence heatmaps (with optional SciPy-based hierarchical clustering)

If SciPy is available, the co-occurrence heatmap can be reordered by clustering:

```python
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
```

If SciPy is missing or errors out, the original order is used.

---

## Main app (`main.py`)

The main Streamlit app wires everything together:

- Loads or updates the lotto CSV (via `lotto_data.incremental_update`)
- Caches data in `st.session_state` / `st.cache_data` (depending on implementation)
- Provides multiple tabs, for example:
  - Overview (basic stats, KPIs)
  - Frequency / Rolling analysis
  - Recommendations
  - Fairness checks (uniformity, pair over-representation)
- Uses `viz.py` components and `plotly.express` to render interactive charts
- Optionally integrates with a simple member management CSV and Supabase backend
  (phone numbers are stored in hashed/E.164 format in `members.csv`)

Run it as a Streamlit app:

```bash
streamlit run main.py
```

---

## Project Structure

```text
lotto-analyzer/
└─ lotto-analyzer-main/
   ├─ main.py            # Streamlit app entry point
   ├─ lotto_data.py      # Data download, CSV management, frequency helpers
   ├─ rolling.py         # Rolling-window frequency computation
   ├─ features.py        # Feature engineering per draw
   ├─ recs.py            # Recommendation strategies
   ├─ fairness.py        # Statistical tests (uniformity / pair significance / FDR)
   ├─ viz.py             # Plotly + Streamlit visualization helpers
   ├─ update_data.py     # CLI utility to update the CSV incrementally
   ├─ requirements.txt   # Original dependency pinning (reference)
   ├─ .devcontainer/
   │   └─ devcontainer.json
   └─ streamlit/
       └─ config.toml    # Optional Streamlit config
```

---

## Example `requirements.txt`

Based on the project’s own pinned requirements, the minimal set of runtime
dependencies is:

```text
streamlit==1.37.1
pandas==2.2.3
numpy==2.1.3
requests==2.32.3
plotly==5.23.0
tqdm==4.66.4
```

> Notes  
> - `reportlab` is only needed if you later add PDF export; it is not required by
>   the current Python modules.  
> - SciPy is optional and only used for nicer co-occurrence clustering when present.

---

## Installation

```bash
pip install -r requirements.txt
```

If you place the file in `lotto-analyzer-main/requirements.txt`, this will install
the exact versions used in the original project.

---

## Usage

1. Update or download data (optional, the app can do this on first run as well):

   ```bash
   python update_data.py --data-path data/lotto_draws.csv
   ```

2. Start the Streamlit app:

   ```bash
   streamlit run main.py
   ```

3. Open the URL shown in the terminal (usually `http://localhost:8501`) and
   explore:

   - Number frequencies and trends
   - Rolling-window behavior
   - Suggested number combinations
   - Simple fairness diagnostics of the draw history
