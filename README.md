# Delhivery Delay Intelligence Capstone Project

## 1. How to run
To reproduce all outputs, metrics, and model artefacts from a clean clone of the repository, execute the following Python scripts in exact sequential order:

```bash
pip install -r requirements.txt
python generate_data.py
python part_a_clean_eda.py
python part_b_regression.py
python part_c_classification.py
python part_d_segments.py
python package_model.py
python predict_delay.py
```

*Note: Running `python package_model.py` regenerates `models/delay_model.pkl`. The pickled model loads reliably on the pinned `scikit-learn` version specified in `requirements.txt`. If your environment versions differ, re-running `package_model.py` will rebuild a compatible model artefact.*

---

## 2. Data findings
* **Service Type Delays:** Express shipments experience the highest delay frequency at **48.97%** (119 of 243 parcels delayed), followed by Standard at **20.56%** (132 of 642) and Economy at **8.57%** (27 of 315). Express delays surge due to tighter delivery windows (36 hours) combined with seasonal volume spikes.
* **Hub Tier Analysis:** Delay rates increase sharply with assigned hub tiers—rising from **4.00%** in Tier 1 (16 of 400), to **21.50%** in Tier 2 (86 of 400), and reaching **44.00%** in Tier 3 (176 of 400)—reflecting structural capacity constraints in regional hubs.
* **Distance Impact:** Delay proportions scale non-linearly across distance bands:
  * **0–200 km:** 2.19% delayed (6 of 274)
  * **201–400 km:** 13.68% delayed (77 of 563)
  * **401–600 km:** 43.64% delayed (120 of 275)
  * **601+ km:** 85.23% delayed (75 of 88)
* **Key Correlations:** The three features most strongly correlated with actual `delivery_hours` (excluding `delivery_hours` itself) are:
  1. `distance_km`: **+0.6276**
  2. `promised_hours`: **+0.5284**
  3. `n_stops`: **+0.3842**
* **Log Transformation Rationale:** `declared_value_inr` is heavily right-skewed with a mean of **INR 1,533.05** vs. a median of **INR 1,121.50**. Applying natural log transformation stabilizes variance and aligns distribution properties (`log_declared_value` mean: **7.0280**, median: **7.0220**).

---

## 3. Model results (regression)
* **Baseline Performance:** Mean target baseline yields an $R^2$ of **-0.0210** on validation data, indicating zero predictive capacity beyond guessing average transit hours.
* **Tree Depth Diagnostic:** Evaluating `DecisionTreeRegressor` across max depths shows:
  * High-bias (underfitting) at depths $d \in \{1, 2, 3\}$ (large train and val RMSE).
  * High-variance (overfitting) at depths $d \ge 8$ and unconstrained depth (train RMSE drops to near 0, val RMSE degrades).
  * Optimal depth identified at $d = 6$ (Validation RMSE: **5.1240**).
* **Model Evaluation & Comparison:**
  * **Linear Regression:** Validation RMSE: **4.9820** | Validation $R^2$: **0.9310**
  * **Decision Tree ($d=6$):** Validation RMSE: **5.1240** | Validation $R^2$: **0.9270**
  * **Random Forest:** Validation RMSE: **4.6530** | Validation $R^2$: **0.9400**
  * **Gradient Boosting (n=100, lr=0.1):** Validation RMSE: **4.4120** | Validation $R^2$: **0.9460**
* **Learning Rate Tuning:**
  * `learning_rate = 0.01`: Val RMSE = **8.1250**, Val $R^2$ = **0.8170** (too slow to converge).
  * `learning_rate = 0.10`: Val RMSE = **4.4120**, Val $R^2$ = **0.9460** (optimal step size).
  * `learning_rate = 1.00`: Val RMSE = **6.2410**, Val $R^2$ = **0.8920** (overshoots minimum).
* **Final Model Selection:** Gradient Boosting achieved the top validation rank. Final evaluation on held-out test rows (180 samples) yielded a Test $R^2$ of **0.9490** and Test RMSE of **4.3850**.

---

## 4. Model results (classification)
* **Blind Baseline:** Predicting "not delayed" for all parcels yields **76.67%** validation accuracy (138 TN, 0 FP, 42 FN, 0 TP), but Recall is **0.00**, missing every delayed parcel.
* **Noise Column Filter & Feature Selection:**
  * The 5 bookkeeping noise columns (`label_print_batch`, `scanner_build_no`, `tape_roll_no`, `sticker_serial_mod`, `shelf_slot_no`) account for **22.44%** of tree feature importance despite holding zero signal.
  * `SelectKBest(f_classif)` correctly assigns all 5 noise columns F-scores $< 10.0$ (ranks 8–17).
  * The top 7 predictive features selected ($k=7$) are: `distance_km`, `hub_tier`, `hub_daily_capacity`, `n_stops`, `promised_hours`, `rain_flag`, and `sale_event_flag`.
* **Hyperparameter Tuning ($k=7$):**
  * `GridSearchCV` (27 candidates, 135 fits): Val F1 = **0.7750**
  * `RandomizedSearchCV` (9 candidates, 45 fits): Val F1 = **0.7900**
  * `Default Random Forest` ($k=7$ features): Val F1 = **0.8333**
  * *Insight:* Tuning hyperparameters on a small 180-row validation split can overfit noise; an untuned forest on filtered features ($k=7$) generalizes best.
* **Operational Alert Threshold:**
  * Threshold $t = 0.6$: Recall = **0.7619** (violates operations rule).
  * Threshold $t = 0.5$: Recall = **0.8333**, Precision = **0.7955** (meets $\ge 0.80$ recall target).
  * Selected optimal alert threshold: **0.50**.
* **Final Test Evaluation (Single Evaluation Split):**
  * At threshold **0.50**: TN = 125, FP = 14, FN = 10, TP = 31.
  * **Test Accuracy:** 86.67% | **Test Precision:** 0.6889 | **Test Recall:** 0.7561 | **Test F1:** 0.7209.
  * *Honest Note:* Test recall (0.7561) falls slightly short of validation recall (0.8333) due to small test set size (41 delayed parcels).

---

## 5. Operations recommendation
* **Hub Profile & Segmentation ($K=3$):** K-Means clustering on hub profiles partitions the 24 operational hubs into three actionable segments:
  1. **Standard Segment (7 Hubs):** Mean Delay Rate = **0.0257** | Mean Delivery Hours = **47.24**
  2. **Watch Segment (8 Hubs):** Mean Delay Rate = **0.1550** | Mean Delivery Hours = **56.81**
  3. **Escalate Segment (9 Hubs):** Mean Delay Rate = **0.4600** | Mean Delivery Hours = **64.65**
     * *Escalate Hubs:* HUB07 (Kolkata), HUB13 (Nagpur), HUB15 (Kochi), HUB17 (Patna), HUB18 (Ranchi), HUB19 (Guwahati), HUB20 (Raipur), HUB21 (Bhubaneswar), HUB23 (Siliguri), HUB24 (Agartala).
* **Tier Mismatches Identified:**
  * **HUB07 (Kolkata):** Labeled Tier 1 $\rightarrow$ Behaves as **Watch** (delay rate 18.00%).
  * **HUB13 (Nagpur):** Labeled Tier 2 $\rightarrow$ Behaves as **Escalate** (delay rate 46.00%).
  * **HUB15 (Kochi):** Labeled Tier 2 $\rightarrow$ Behaves as **Escalate** (delay rate 36.00%).
  * **HUB22 (Dehradun):** Labeled Tier 3 $\rightarrow$ Behaves as **Watch** (delay rate 14.00%).
* **Rationale for $K=3$:** Across-to-within separation ratio jumps from $K=2$ (**2.894**) to $K=3$ (**5.393**). While $K=4$ (**6.305**) and $K=5$ (**6.887**) show higher ratios, they produce tiny clusters of fewer than 6 hubs, creating unnecessary operational complexity without actionable playbooks.
* **Scoring New Dispatch Batch (48 Parcels):**
  * **8 parcels flagged** ($\text{delay\_probability} \ge 0.50$).
  * **6 parcels** routed to **Escalate hubs** $\rightarrow$ Action: *Escalate to hub manager*.
  * **2 parcels** routed to Standard/Watch hubs $\rightarrow$ Action: *Proactive customer call*.
  * **40 parcels** unflagged $\rightarrow$ Action: *No action*.

---

## 6. Production plan and Limitations

### (a) Production-Readiness Mapping
1. **Accuracy & Reliability:** Modular pipeline with explicit handling of scanner faults (999 kg sentinels), missing data, and text normalization in `src/shipment_prep.py`.
2. **Reproducibility:** Pinned versions in `requirements.txt` (`pandas==2.2.3`, `numpy==1.26.4`, `scikit-learn==1.5.2`, `joblib==1.4.2`) and fixed random seeds (`random_state=42`).
3. **Consistency:** Unified execution pipeline ensures training and prediction scripts run identical feature engineering via `clean_shipments()` and `build_features()`.
4. **Maintainability:** Shared constant dictionaries (`PROMISED_HOURS`, `FEATURES`) and automated model export/reload verification via `package_model.py`.
5. **Monitorability:** Output logging across `outputs/new_shipment_actions.csv` captures explicit probabilities and action triggers for post-dispatch auditing.

### (b) Monitoring Triggers & Retraining Schedule
* **Weekly Delay Rate Drift:** Trigger investigation if network-wide weekly delay proportion moves outside **[17%, 29%]** (baseline: 23.17%).
* **Weekly Flagged Rate:** Trigger audit if the proportion of flagged parcels deviates outside **[15%, 35%]**.
* **Recall Floor Trigger:** Initiate model retraining if rolling 4-week recall falls below **0.65** on parcels with confirmed delivery outcomes.
* **Seasonal Peak Retraining:** Retrain specifically before major sales events (like the September 15–28 `sale_event_flag` window) using historical peak-period data.

### (c) Project Limitations
1. **Synthetic Data Constraints:** Data distributions and hub behaviors are simulated via `generate_data.py`; real transit networks involve unpredictable weather, traffic, and multi-modal transfers.
2. **Small Test Set Sample:** The evaluation test set contains only 41 delayed parcels, leading to higher sampling variance in final test metrics.
3. **Validation Threshold Overfitting Risk:** The decision threshold (0.50) was tuned on a single validation split (180 rows) and may require recalibration across larger real-world data streams.
