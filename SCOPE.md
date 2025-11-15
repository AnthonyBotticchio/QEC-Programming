
## 1. One-line Project Definition

**EV ChargeBalance**:  
A Python + Streamlit tool to predict future EV charging demand across a city grid, compute under-served areas, recommend new charger placements, and visualize before vs. after shortage heatmaps.

**Domain:** EV charger placement optimization.

---

## 2. In-Scope: Core Functionality (Must Have)

*If any of these are missing, the project is incomplete.*

### 2.1 Data & Model

#### Data

- **Use Only Synthetic Data:** _No real datasets._
- **Two generated files:**
  - `city_grid.csv` – Static table.
  - `historical_charging.csv` – Time series training samples.

##### **city_grid.csv** Columns:
- `cell_id`
- `x`
- `y`
- `land_use_type` (residential, commercial, mall, highway, downtown)
- `charger_capacity` (number of chargers in the cell)

##### **historical_charging.csv** Columns (one row per sample):
- `cell_id`
- `time_of_day` (0–23)
- `day_of_week` (0–6)
- `land_use_type`
- `ev_density`
- `current_utilization` (0–1)
- `future_utilization` (0–1) ← label

#### Modeling

- **Baseline Model (Naive):**  
  `future_util_pred_baseline = current_utilization`  
  Compute & report MAE for baseline.

- **ML Model:**  
  Use **RandomForestRegressor** or **LinearRegression**.  
  - **Features:**  
    - `time_of_day`
    - `day_of_week`
    - One-hot encoded `land_use_type`
    - `ev_density`
    - `current_utilization`
  - **Target:** `future_utilization`
  - Compute MAE; show model MAE < baseline MAE (even slightly).
  - Save model as `demand_model.pkl`.

  > This is sufficient to demonstrate “we forecast future utilization better than a naive baseline.”

---

### 2.2 Shortage Scoring (**Charger Balance Score – CBS**)

- **For each cell in a snapshot:**
  1. Predict `future_utilization` using the trained model.
  2. Normalize charger capacity:  
     `capacity_norm = charger_capacity / max_charger_capacity`
  3. Compute shortage score:  
     `CBS = predicted_future_utilization - capacity_norm`
- **Interpretation:**
  - High CBS → high demand vs. capacity (shortage)
  - Low/negative CBS → surplus
- **Only this metric is needed.**

---

### 2.3 Placement Recommendation (Where to Add Chargers)

- **Given:** CBS per cell and a number k (new chargers)
- **Steps:**
  1. Sort cells by CBS (descending)
  2. Select top k cells as recommended new charger locations
- **Pseudo:**
  ```python
  top_k = cbs_df.sort_values("CBS", ascending=False).head(k)
  ```
- **No advanced optimization—just this.**

---

### 2.4 Before vs. After Simulation

- **Goal:** Simulate how adding chargers improves grid shortage.
- **Process:**
  1. Start with current `charger_capacity`
  2. For top k selected cells, add new chargers:
     ```python
     charger_capacity_after = charger_capacity.copy()
     charger_capacity_after[cell_id_in_top_k] += added_capacity_per_new_station  # e.g. +2
     ```
  3. Recompute:
     - `capacity_norm_after = charger_capacity_after / max(charger_capacity_after)`
     - `CBS_after = predicted_future_utilization - capacity_norm_after`
  4. Compute & report summary metrics:
     - **Average CBS** before vs. after
     - **Number of "high shortage" cells** (CBS > threshold) before vs. after
     - **% improvement** in total shortage (sum of CBS for CBS > 0)

- *Provides clear, quantifiable impact.*

---

### 2.5 UI – Streamlit App (One-Page)

#### **Inputs (Sidebar):**
- **Scenario** (min 1, max 2):  
  - `weekday_peak`
  - `weekend_midday`
- **Slider:** for k (1–5) new chargers
- **Button:** “Run ChargeBalance”

#### **Main View (On Click):**
- **Heatmap BEFORE:**
  - Grid/scatter:  
    - x, y = coordinates  
    - color = CBS_before
- **Heatmap AFTER:**
  - Side-by-side or toggle  
    - color = CBS_after
- **Placement Markers:**
  - Highlight top K cells (different size/color)
- **Metrics section:**
  ```
  Avg CBS before: X
  Avg CBS after:  Y
  High-shortage cells before: A
  High-shortage cells after:  B
  % reduction in shortage:   Z%
  ```
- *All other features unnecessary. Clarity > beauty.*

---

## 3. In-Scope: Codebase Structure

_Simple & clear:_
```
.
├── data_gen.py            # generates city_grid.csv & historical_charging.csv
├── train_model.py         # trains baseline + ML model, saves demand_model.pkl
├── chargebalance_core.py  # snapshot -> predictions -> CBS -> placement -> before/after
├── app.py                 # Streamlit UI using core logic
├── requirements.txt
└── README.md
```

**Responsibilities:**
- `data_gen.py`: Generates synthetic data/city and synthetic charging data.
- `train_model.py`: Model & baseline training, evaluation, and model saving.
- `chargebalance_core.py`: All core logic per this spec.
- `app.py`: Presentation/UI only.

> Code structure maps directly to the written “code overview” section.

---

## 4. In-Scope: Documentation & Presentation

- **Written sections:**
  - Problem definition (misplaced chargers, shortage vs. surplus)
  - System overview (Predict → Score → Place → Compare)
  - Data generation description
  - Model + baseline comparison
  - CBS definition
  - Placement (top-K) method
  - How to run:
    ```
    python data_gen.py
    python train_model.py
    streamlit run app.py
    ```
  - Per-file code descriptions

- **Slides:**
  - Problem (misplaced chargers, dead zones, overload)
  - Approach (map + model)
  - Data & model (simple; MAE chart)
  - CBS & shortage visualization
  - Before/after maps with placements
  - Impact metrics (% shortage reduction for top-K)
  - Responsible AI (equity, fairness)
  - Limitations & future work

---

## 5. Out-of-Scope (“Do NOT Do”)

*Explicitly NOT allowed. Do not attempt!*

- No real-world EV datasets
- No external APIs
- No deep learning / LSTMs / complex models
- No multi-page app
- No complex drag-and-drop map
- No backend services (Docker, Postgres, etc.)
- No real-time streaming data
- No multi-city/national-scale
- _Stay with: static synthetic city, predicted utilization, simulated placements only_

---

## 6. “Nice-to-haves” (ONLY When Core is Done)

Only consider if everything above is stable.

- **Interactive tweak:** E.g., click a cell to add a charger manually and recompute CBS_after.
- **Extra scenario:** “Event Night” with heavy downtown demand.
- **Improved visuals:** Better color scale, labels, tooltips.

_These are not necessary to win. Only polish after all else is done._
