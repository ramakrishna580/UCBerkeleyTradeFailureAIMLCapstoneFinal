# AI-Powered Intelligent Trade Settlement Failure Prediction and Explainable Risk Management System

**UC Berkeley Professional Certificate in Machine Learning & Artificial Intelligence — Capstone Project**

**Author:** Rama Nuti  
**GitHub:** [ramakrishna580](https://github.com/ramakrishna580/)

## Executive summary

This capstone investigates whether pre-settlement trade and operational signals can identify securities transactions at elevated risk of settlement failure, enabling operations teams to prioritize preventive action. It combines supervised failure classification, settlement-delay regression, exploratory and statistical analysis, unsupervised segmentation, anomaly detection, explainability, and an operations-oriented risk queue. The analysis uses **synthetic trade settlement data**, so results are educational demonstrations rather than validated production performance.

## System Architecture

The numbered, color-coded architecture below illustrates the project's data ingestion, EDA and feature preparation, supervised and unsupervised machine learning, evaluation and explainability, operations risk prioritization, and monitoring/reporting. Each box has a distinct background color and a bold title. Solid arrows show principal dependencies; dashed arrows show supporting or proposed connections.

![Machine Learning Pipeline](docs/MachineLearning-PipeLine.jpg)

![Machine Learning Pipeline](docs/Machine_Learning_Pipeline.svg)

![Numbered, color-coded trade settlement failure prediction system architecture](docs/Trade_Failure_System_Architecture_1.png)

[View full-size PNG](docs/Trade_Failure_System_Architecture_1.png) · [View editable Graphviz source](docs/Trade_Failure_System_Architecture.dot) · [View svg](docs/Trade_Failure_System_Architecture_1.svg)

**Scope:** This is a research architecture using synthetic data, not a deployed trading system. Analyst review is a proposed workflow; narrative explanations are template-based rather than an external LLM service. The numbers identify components and branches, not a strict execution sequence.

# AI-Powered Intelligent Trade Settlement Failure Prediction and Explainable Risk Management System

**UC Berkeley Professional Certificate in Machine Learning & Artificial Intelligence — Final Capstone Project**  
**Author:** Rama Nuti

## 1. What Is the Trade Settlement Failure Prediction and Explainable Risk Management System?

Trade settlement is the completion of a securities transaction through the transfer of securities and the corresponding payment. A settlement failure occurs when a transaction does not complete by its contractual settlement deadline. Potential contributing factors include missing or incorrect settlement instructions, delayed confirmation or affirmation, inventory constraints, and operational processing issues.

This capstone explores a machine learning decision-support system that identifies trades at elevated risk **before settlement**. It combines historical trade characteristics, operational signals, and counterparty history to produce a failure-risk score and prioritize trades for investigation.

The explainable risk management component helps an analyst understand which features influenced a prediction. Global explanations describe the model's overall behavior; local explanations describe an individual trade's score. These explanations identify model associations rather than establish the actual cause of a failure.

The intended workflow is:

1. Capture trade and operational information available at a defined prediction time.
2. Validate and transform the information using the fitted preprocessing pipeline.
3. Estimate settlement-failure risk and, in a separate regression task, settlement delay.
4. Rank trades for review and display the main model-based risk drivers.
5. Support analyst investigation, record outcomes, and monitor model performance.

**Project scope:** This is an educational research prototype using synthetic data. It does not demonstrate a deployed trading system, validated production performance, or realized financial savings. The risk narrative demonstration uses grounded rules/templates; it does not call an external large language model.

## 2. Business Understanding

### Business problem

Reactive exception management identifies problems after they have already disrupted settlement. Operations teams need a way to focus limited investigation capacity on trades most likely to require preventive intervention.

Settlement failures can create additional reconciliation work, liquidity pressure, client disruption, and potential financial costs. Predictive prioritization could help teams investigate instruction issues, inventory constraints, or processing delays earlier.

### Business objectives

- Identify potentially failing trades before the contractual settlement deadline.
- Prioritize investigations using failure probability, trade context, and available review capacity.
- Provide understandable explanations for alerts.
- Explore patterns across asset classes, markets, counterparties, and operational processes.
- Evaluate the trade-off between missed failures and unnecessary investigations.

### Analytical questions

- Can pre-settlement information distinguish failures from successful settlements?
- Which features are associated with elevated predicted risk?
- Which models provide the most useful balance of detection and investigation workload?
- Can a separate regression model estimate settlement delay?
- Can clustering and anomaly detection reveal operational segments worth investigating?

### Success criteria

Model evaluation emphasizes failure recall, precision, F1, precision–recall performance, and alert volume. Business evaluation should additionally consider recall among the top-ranked trades, analyst capacity, explanation usefulness, and expected costs under explicitly stated assumptions.

Accuracy alone is insufficient because failures are rare. A model that predicts every trade will settle can achieve high accuracy while detecting no failures. Financial savings and reductions in actual failures would require a controlled real-world evaluation and are not established by this project.

## 3. Data Understanding

### Dataset and provenance

The project uses `data/trade_settlement_synthetic.csv`, supported by `data/data_dictionary.csv`. The existing project README records the following statistics from saved final-notebook outputs:

| Characteristic | Recorded value |
|---|---|
| Observations | 25,000 trades |
| Columns | 37 |
| Trade-date coverage | 2023-01-02 through 2025-11-28 |
| Settlement failure rate | 1.96% |
| Duplicate trade IDs | 0 |
| Classification target | `settlement_failed` |
| Separate regression target | `settlement_delay_hours` |

These values are previously recorded notebook results, not statistics independently recalculated for this documentation. The data dictionary confirms 37 named fields. Synthetic data enables reproducible experimentation without exposing confidential client or transaction information, but its relationships may reflect generation assumptions rather than live market behavior.

### Unit of analysis and prediction timing

Each row represents a synthetic trade. The classifier predicts whether the trade will fail to settle by its expected contractual deadline. The prediction time must be specified before evaluating operational usefulness.

Although the dictionary labels operational fields as pre-prediction, their real-world availability depends on the scoring time. Confirmation, allocation, and affirmation delays, amendment counts, matching attempts, and inventory indicators must represent information observed by that time. A completed processing duration must not be used when that activity would still be pending at scoring.

Historical failure rates and reference averages must use only prior information available at scoring. Their windows must not include the current trade's outcome or future outcomes.

## 4. Attribute Information

The following grouped dictionary covers all 37 original fields. Descriptions follow the project's supplied data dictionary.

| Attribute(s) | Type | Meaning and modeling role |
|---|---|---|
| `trade_id` | Identifier | Unique synthetic trade identifier; retain for traceability and exclude as a predictor. |
| `trade_date`, `settlement_date` | Date | Execution date and expected contractual settlement date; support chronological partitioning and calendar context. |
| `asset_class` | Categorical | Equity, corporate bond, government bond, or ETF. |
| `buy_sell` | Categorical | Transaction direction. |
| `currency`, `market`, `settlement_location` | Categorical | Currency, market geography, and synthetic settlement/depository location. |
| `counterparty_id`, `custodian` | Categorical | Synthetic counterparty and custodian identifiers; evaluate sensitivity to unfamiliar entities. |
| `trade_amount`, `quantity` | Numeric | Synthetic trade value and security quantity; check units and currency comparability. |
| `settlement_days` | Numeric | Expected interval between execution and settlement; verify the generator's day-count convention. |
| `cross_border` | Binary | Whether the trade crosses market borders. |
| `month_end`, `quarter_end` | Binary | Calendar indicators based on the trade date. |
| `market_holiday_mismatch` | Binary | Synthetic cross-market calendar mismatch flag. |
| `ssi_status` | Categorical | Standing settlement instruction status: `VALID`, `CHANGED`, or `MISSING`. |
| `confirmation_delay_minutes` | Numeric | Time between trade execution and confirmation, subject to scoring-time availability. |
| `allocation_delay_minutes`, `affirmation_delay_minutes` | Numeric | Synthetic allocation and affirmation delays. |
| `matching_attempts`, `trade_amendments` | Numeric/count | Matching attempts and amendments observed before the scoring cutoff. |
| `hours_until_cutoff` | Numeric | Time remaining until the operational settlement cutoff. |
| `prior_fail_rate_30d`, `prior_fail_rate_90d` | Numeric/rate | Synthetic historical counterparty failure rates over prior windows. |
| `prior_trade_count_30d` | Numeric/count | Recent counterparty activity. |
| `security_liquidity_score` | Numeric | Synthetic liquidity score from 0 to 1. |
| `market_volatility_index` | Numeric | Synthetic market-volatility level. |
| `inventory_shortfall_indicator` | Binary | Possible security inventory constraint. |
| `counterparty_avg_trade_amount` | Numeric | Synthetic rolling/reference counterparty average trade size. |
| `trade_size_ratio` | Numeric | Current trade amount divided by the synthetic counterparty average. |
| `pre_settlement_note` | Text | Synthetic pre-settlement note; reserved for NLP experiments by default. |
| `settlement_failed` | Binary target | `1` = failed; `0` = settled. Never use as a classifier input. |
| `settlement_delay_hours` | Numeric outcome | Realized settlement delay; separate regression target, excluded from pre-settlement classifier inputs. |
| `failure_reason` | Categorical outcome | Final failure reason; excluded from predictive inputs and used only for post-event analysis. |
| `failure_reason_description` | Text outcome | Final failure description; reserved for a separate post-event NLP task. |

**Leakage boundary:** Realized outcomes and final failure descriptions cannot be used to predict the same trade's pre-settlement risk. Identifiers and date fields also require deliberate handling rather than automatic inclusion.

## 5. Data Preparation and Visualization

### Preparation workflow

1. Load the CSV and check column names against the data dictionary.
2. Parse dates and numeric values; inspect invalid conversions and inconsistent categories.
3. Assess missingness, duplicate identifiers, target balance, and date coverage.
4. Separate targets, identifiers, post-outcome fields, and eligible predictors.
5. Partition trades chronologically into training, validation, and final test periods.
6. Fit imputation, encoding, scaling, and other learned transformations on training data only.
7. Apply the fitted transformations to validation and test data without refitting.

### Exploratory visualizations

| Visualization | Learning objective |
|---|---|
| Settlement outcome count/proportion chart | Show the rare-event classification challenge. |
| Failure rate by SSI status | Explore association between instruction status and outcomes. |
| Failure rate by asset class and market | Identify segments with differing observed rates. |
| Trade amount histogram with log-scale view | Inspect skew and extreme transaction sizes. |
| Operational delay distributions by outcome | Compare processing patterns for failed and settled trades. |
| Numeric correlation heatmap | Inspect related features and potential redundancy. |
| Trade volume and failure rate over time | Assess temporal variation and evaluation-period differences. |
| PCA scree plot and cluster profiles | Explore dimensionality and operational segmentation. |

Report group sample sizes alongside failure rates so that small groups do not appear more reliable than they are. Exploratory associations should not be presented as causal findings. The existing README lists saved EDA charts for outcome distribution, asset class, SSI status, market, and confirmation delay; other views above are recommended additions where supported by the notebook.

## 6. Data Cleansing, Processing, and Modeling

### Data cleansing and quality controls

- Investigate duplicate `trade_id` values before removing records; repeated operational snapshots require a different treatment from duplicate rows.
- Standardize categorical labels and distinguish a meaningful `MISSING` SSI status from an absent value.
- Validate binary values, score/rate ranges, nonnegative counts, and date consistency against field definitions.
- Inspect extreme trade values and delays rather than automatically deleting legitimate high-risk observations.
- Handle missing numeric values with training-derived imputation and categorical values with an explicit strategy appropriate to their meaning.
- Check denominator validity when calculating ratios and prevent undefined values.
- Verify that historical aggregates and operational fields respect the prediction timestamp.

These controls describe the intended preparation standard. The raw CSV and execution outputs should be checked before claiming that every control passed or that a specific cleansing operation was necessary.

### Feature processing

Candidate engineered features documented in the existing project include log-transformed trade amounts, transformed processing delays, aggregate delay indicators, operational touch counts, and counterparty risk indicators. Combining processing durations requires checking whether activities overlap; a sum is a constructed indicator rather than automatically an elapsed end-to-end duration.

Categorical encoding and numeric scaling should be model-appropriate. Scaling is especially relevant for Logistic Regression, KNN, SVM, PCA, and K-Means. Tree-based models generally do not require numeric scaling. Any feature selection, dimensionality reduction, imbalance handling, or tuning must remain inside the training/validation workflow.

### Modeling tasks

| Task | Methods documented in the existing project | Purpose |
|---|---|---|
| Failure classification | Logistic Regression, KNN, Decision Tree, Random Forest, SVM, XGBoost, CatBoost | Predict `settlement_failed`. |
| Delay regression | Linear Regression, Ridge, Random Forest regression | Estimate `settlement_delay_hours` as a separate outcome. |
| Dimensionality reduction | PCA/SVD | Explore compact representations and feature variation. |
| Segmentation | K-Means | Describe groups with similar operational characteristics. |
| Anomaly detection | Isolation Forest | Identify unusual trades for investigation; anomaly scores are not failure probabilities. |
| Post-event text classification | TF-IDF-based NLP | Classify final failure descriptions after the event. |
| Explainability | SHAP | Inspect global feature influence and local prediction contributions. |
| Risk narratives | Grounded rules/templates | Translate available scores and risk drivers into readable analyst summaries. |

### Evaluation and model selection

Use chronological validation, and time-aware cross-validation where supported, to avoid learning from future observations. Tune hyperparameters and alert thresholds on training/validation data; reserve the final test period for the final assessment. When training examples have outcomes that mature after a split boundary, account for that label-availability delay.

Classification evaluation should include precision, recall, F1, ROC-AUC, PR-AUC, confusion matrices, and threshold sensitivity. State the PR-AUC calculation used so comparisons remain consistent. Evaluate regression using MAE and RMSE, with other measures where appropriate. Probability calibration should be assessed before interpreting scores as reliable failure probabilities; the existing README identifies calibration as a next step rather than a verified completed capability.

The saved results in the existing README show Logistic Regression with the highest recorded PR-AUC and recall among the compared classifiers, while CatBoost has the highest recorded precision and F1 at the reported operating points. Several classifiers record zero recall at their evaluated thresholds. These observations support evaluating thresholds and review workload rather than selecting by accuracy alone. They are recorded results, not a fresh rerun.

The cost demonstration uses hypothetical values of **$5,000 per missed failure** and **$50 per false-positive investigation**. These are scenario assumptions, not measured institutional costs or proven savings.

### Explainable risk management

An illustrative review queue combines a trade identifier, risk score, review priority, relevant trade context, and leading model-based risk drivers. Priority thresholds should reflect validation results and available investigation capacity. A probability of failure alone does not measure monetary exposure or guarantee an effective intervention.

SHAP explanations can help analysts identify fields to inspect, but they do not prove that changing a feature will prevent failure. Proposed interventions require operational judgment. Monitor data quality, feature drift, alert volume, eventual outcome performance, and segment-level reliability. PSI provides a distribution-shift signal; it does not by itself establish model deterioration.

## 7. Process Summary

| Stage | Key activity | Deliverable or decision |
|---|---|---|
| Business understanding | Define failure prediction, scoring time, and review objectives | Clear analytical scope and success criteria. |
| Data understanding | Inspect synthetic trades, schema, target prevalence, and availability | Data profile and attribute dictionary. |
| Preparation and EDA | Validate records, explore distributions, and create temporal partitions | Prepared inputs and explanatory visualizations. |
| Feature processing | Build eligible predictors and fit training-only transformations | Reusable preprocessing workflow. |
| Modeling | Compare classifiers; analyze delay, clusters, and anomalies separately | Model outputs and comparison tables. |
| Evaluation | Assess rare-event detection, thresholds, and hypothetical costs | Evidence for model and operating-point selection. |
| Explainability | Inspect SHAP contributions and generate grounded narratives | Understandable prediction explanations. |
| Decision support | Rank trades for analyst review | Illustrative operations risk queue. |
| Monitoring | Track distribution shifts and eventual performance | Monitoring outputs and proposed review triggers. |

The capstone demonstrates how prediction, explanation, and operational prioritization can be connected in a reproducible research workflow. Production adoption would require validation on authorized real-world data, verified point-in-time feature availability, calibrated scores, approved review thresholds, and monitored operational outcomes.

**Documentation basis:** These sections were aligned with the supplied project README and `data_dictionary.csv`. Dataset statistics, model coverage, and comparative results reflect that README's recorded evidence. Recommended controls and additional analyses are distinguished from verified execution; the dataset, scripts, and notebook were not independently rerun for this documentation update.


## Data sources and scope

The notebooks use a **synthetic** trade-settlement CSV named `trade_settlement_synthetic.csv`, with features related to trade characteristics, counterparties, SSI, confirmation/allocation/affirmation delays, operational touches, and settlement outcomes. The recorded final notebook output reports **25,000 rows and 37 columns**, trade dates **2023-01-02 through 2025-11-28**, **zero duplicate trade IDs**, and a **1.96% settlement failure rate**. These are notebook-recorded results, not a fresh independent rerun.

The main classification target is `settlement_failed`. `settlement_delay_hours` is analyzed as a **separate regression target**. Outcome-derived information, including final failure reasons and realized settlement delay, must not leak into the pre-settlement classifier. Post-event failure-description classification is intentionally a separate NLP use case.

**Data availability:** Your repository listing includes `data/trade_settlement_synthetic.csv` and `data/data_dictionary.csv`. These data files were listed but not attached individually for inspection. If missing in a fresh checkout, use `src/00_generate_dataset.py` or the data-generation workflow, then confirm the CSV path resolves.

## Methodology

- **Data validation and EDA:** schema, duplicates, missing values, base rates, and outcome relationships across SSI status, asset class, inventory indicators, trade size, and operational delays.
- **Statistical tests:** chi-square association test for SSI status versus settlement outcome; Mann–Whitney U test for confirmation-delay distributions.
- **Feature engineering:** log trade amount, transformed delays, aggregate processing delays, operational touch counts, and counterparty risk indicators.
- **Temporal evaluation and leakage prevention:** chronological data partitioning; preprocessing fit on training data; pre-outcome predictors only.
- **Classification:** Logistic Regression, K-Nearest Neighbors, Decision Tree, Random Forest, Support Vector Machine, XGBoost, and CatBoost.
- **Regression:** Linear Regression, Ridge, and Random Forest regression for settlement delay, assessed with regression error metrics.
- **Unsupervised analysis:** PCA/SVD, K-Means operational-risk segments, and Isolation Forest anomaly detection.
- **Rare-event evaluation:** precision, recall, F1, ROC-AUC, PR-AUC, threshold sensitivity, and confusion matrices; overall accuracy alone is insufficient.
- **Operationalization:** illustrative false-negative versus investigation-cost tradeoffs, risk thresholds, explainability via SHAP, and ranked operations queues.
- **NLP and GenAI:** TF-IDF-based classification of *post-event* failure descriptions; a grounded, rule/template-based risk narrative. The narrative demonstration does **not** train or call an external large language model.
- **Monitoring and governance:** population stability index (PSI), drift monitoring, human oversight, lineage, auditability, and model validation.

## Results and interpretation

The synthetic data is **highly imbalanced** (1.96% failures). The final notebook's saved outputs report the following **test-set metrics at the notebook's evaluation settings**:

| Classifier | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.0494 | 0.6301 | 0.0916 | 0.7781 | 0.1308 |
| K-Nearest Neighbors | 0.0000 | 0.0000 | 0.0000 | 0.5449 | 0.0311 |
| Decision Tree | 0.0333 | 0.6027 | 0.0631 | 0.6389 | 0.0648 |
| Random Forest | 0.0000 | 0.0000 | 0.0000 | 0.7166 | 0.0802 |
| Support Vector Machine | 0.0000 | 0.0000 | 0.0000 | 0.5265 | 0.0256 |
| XGBoost | 0.1192 | 0.2466 | 0.1607 | 0.6950 | 0.0918 |
| CatBoost | 0.1298 | 0.2329 | 0.1667 | 0.7065 | 0.1169 |

**Interpretation:** Logistic Regression has the highest recorded PR-AUC and recall among these saved final-notebook outputs, but low precision means substantial review workload. CatBoost has the highest recorded F1 and precision at the reported operating point. The appropriate model and alert threshold depend on the organization's tolerance for missed failures versus false alarms. Models with high accuracy but zero failure recall are not useful for detecting rare failures at that threshold. Results may differ across notebook versions or reruns.

The notebooks also demonstrate cost-sensitive threshold selection using **hypothetical** values of **$5,000 per missed failure** and **$50 per false-positive investigation**; these are **not** actual institutional costs. Do not interpret synthetic model results or illustrative cost assumptions as realized financial savings.

## Source code and project directory hierarchy

The following tree reflects the repository listing provided for `UCBerkeleyTradeFailureAIMLCapstoneFinal`. File names and capitalization are preserved exactly as listed. The hierarchy is documentation of the supplied repository inventory; the `src/`, `outputs/`, and dataset files were not individually uploaded for independent inspection.

```text
UCBerkeleyTradeFailureAIMLCapstoneFinal/
├── README.md
├── requirements.txt
├── data/
│   ├── data_dictionary.csv
│   └── trade_settlement_synthetic.csv
├── notebooks/
│   ├── UC_Berkeley_Capstone_AI_Trade_Failure_Draft.ipynb
│   ├── UC_Berkeley_Capstone_AI_Trade_Failure_PreFinal.ipynb
│   ├── UC_Berkeley_Capstone_AI_Trade_Failure_FINAL.ipynb
│   └── catboost_info/
│       └── learn/
│           └── events.out.tfevents
├── src/
│   ├── common.py
│   ├── 00_generate_dataset.py
│   ├── 01_eda.py
│   ├── 02_feature_engineering.py
│   ├── 03_regression_settlement_delay.py
│   ├── 04_logistic_regression.py
│   ├── 05_knn.py
│   ├── 06_decision_tree.py
│   ├── 07_random_forest.py
│   ├── 08_svm.py
│   ├── 09_pca_svd.py
│   ├── 10_kmeans_clustering.py
│   ├── 11_isolation_forest.py
│   ├── 12_xgboost.py
│   ├── 13_catboost.py
│   ├── 14_model_comparison.py
│   ├── 15_threshold_cost_optimization.py
│   ├── 16_shap_explainability.py
│   ├── 17_nlp_failure_reason.py
│   ├── 18_genai_explanation_template.py
│   ├── 19_end_to_end_pipeline.py
│   └── 20_model_monitoring.py
├── outputs/
│   ├── 01_target_distribution.png
│   ├── 01_top_counterparty_risk.csv
│   ├── 02_failure_by_asset_class.png
│   ├── 02_engineered_feature_sample.csv
│   ├── 03_failure_by_ssi.png
│   ├── 03_regression_results.csv
│   ├── 04_failure_by_market.png
│   ├── 04_logistic_metrics.csv
│   ├── 05_failure_by_confirmation_delay.png
│   ├── 05_knn_metrics.csv
│   ├── 06_decision_tree_metrics.csv
│   ├── 07_random_forest_metrics.csv
│   ├── 08_svm_metrics.csv
│   ├── 09_pca_variance.csv
│   ├── 09_pca_scree.png
│   ├── 10_kmeans_selection.csv
│   ├── 10_kmeans_elbow.png
│   ├── 10_cluster_profiles.csv
│   ├── 11_top_anomalies.csv
│   ├── 12_xgboost_metrics.csv
│   ├── 13_catboost_metrics.csv
│   ├── 14_model_scorecard.csv
│   ├── 15_threshold_cost_curve.csv
│   ├── 16_local_shap_examples.csv
│   ├── 16_global_shap_importance.csv
│   ├── 17_nlp_classification_report.csv
│   ├── 19_operations_risk_queue.csv
│   └── 20_data_drift_psi.csv
└── catboost_info/
    ├── learn/
    ├── tmp/
    ├── time_left.tsv
    ├── learn_error.tsv
    └── catboost_training.json
```

### Directory responsibilities

| Path | Description |
|---|---|
| [`README.md`](README.md) | Project overview, setup, evaluation, and repository guide |
| [`requirements.txt`](requirements.txt) | Python dependency manifest; install with `pip install -r requirements.txt` |
| [`data/`](data/) | Synthetic settlement dataset and accompanying data dictionary |
| [`notebooks/`](notebooks/) | Draft, pre-final, and final Jupyter notebooks documenting the capstone progression |
| [`src/`](src/) | Modular scripts for data generation, feature engineering, modeling, interpretation, and monitoring |
| [`outputs/`](outputs/) | Generated charts, CSV metrics, model comparisons, explanations, and operations-risk queue |
| `catboost_info/` | CatBoost training logs and diagnostic artifacts (generated during training) |
| `notebooks/catboost_info/` | Additional CatBoost event logs created from notebook execution |

### Source module map

| Stage | Source modules | Intended function (inferred from filenames) |
|---|---|---|
| Data preparation | `00_generate_dataset.py`, `01_eda.py`, `02_feature_engineering.py`, `common.py` | Generate data, inspect quality and distributions, engineer predictors, share utilities |
| Regression | `03_regression_settlement_delay.py` | Estimate settlement delay |
| Baseline classification | `04_logistic_regression.py` through `08_svm.py` | Compare logistic regression, KNN, tree, forest, and SVM classifiers |
| Unsupervised learning | `09_pca_svd.py`, `10_kmeans_clustering.py`, `11_isolation_forest.py` | Dimensionality reduction, segmentation, anomaly detection |
| Advanced classification | `12_xgboost.py`, `13_catboost.py` | Gradient-boosted classification experiments |
| Model selection | `14_model_comparison.py`, `15_threshold_cost_optimization.py` | Consolidate model metrics and explore operating thresholds and hypothetical costs |
| Explainability and text | `16_shap_explainability.py`, `17_nlp_failure_reason.py`, `18_genai_explanation_template.py` | Explain predictions, classify post-event failure text, and generate grounded explanations |
| Workflow and monitoring | `19_end_to_end_pipeline.py`, `20_model_monitoring.py` | Risk-queue workflow and drift monitoring |

### Notebooks

- **[Draft notebook](notebooks/UC_Berkeley_Capstone_AI_Trade_Failure_Draft.ipynb)** — initial experiments and development.
- **[Pre-final notebook](notebooks/UC_Berkeley_Capstone_AI_Trade_Failure_PreFinal.ipynb)** — integrated modeling and evaluation workflow.
- **[Final notebook](notebooks/UC_Berkeley_Capstone_AI_Trade_Failure_FINAL.ipynb)** — final capstone notebook with comprehensive scenario coverage.

**Naming note:** Your latest repository listing uses `UC_Berkeley_Capstone_AI_Trade_Failure_FINAL.ipynb` (uppercase `FINAL`). On case-sensitive systems, this is different from the previously requested `UC_Berkeley_Capstone_AI_Trade_Failure_Final.ipynb`. The links above match the latest listing.

### Generated output guide

- **EDA:** `01_target_distribution.png`, `02_failure_by_asset_class.png`, `03_failure_by_ssi.png`, `04_failure_by_market.png`, and `05_failure_by_confirmation_delay.png`.
- **Model results:** `03_regression_results.csv`, `04_logistic_metrics.csv` through `08_svm_metrics.csv`, `12_xgboost_metrics.csv`, `13_catboost_metrics.csv`, and `14_model_scorecard.csv`.
- **Feature/unsupervised artifacts:** `02_engineered_feature_sample.csv`, `09_pca_variance.csv`, `09_pca_scree.png`, `10_kmeans_selection.csv`, `10_kmeans_elbow.png`, `10_cluster_profiles.csv`, and `11_top_anomalies.csv`.
- **Decision support:** `15_threshold_cost_curve.csv`, `16_local_shap_examples.csv`, `16_global_shap_importance.csv`, `17_nlp_classification_report.csv`, `19_operations_risk_queue.csv`, and `20_data_drift_psi.csv`.

The `outputs/` files are generated artifacts rather than manually maintained source data. `catboost_info/` contains generated training diagnostics; consider excluding transient logs from version control if they are not needed for reproducibility.

## Setup and execution

1. Use Python 3 with JupyterLab or Jupyter Notebook.
2. From the repository root, install the declared dependencies: `pip install -r requirements.txt` (prefer a dedicated virtual environment).
3. If the environment reports missing advanced packages, verify `xgboost`, `catboost`, and `shap` are installed and compatible. The final notebook also includes environment/dependency checks.
4. Confirm `data/trade_settlement_synthetic.csv` exists; if not, use the synthetic-data generator and verify paths relative to the repository root.
5. Run the **pre-final** notebook in sequence from top to bottom, or run the **final** notebook's scenario sections as described within that notebook.
6. Inspect actual metrics, visualizations, and output tables after execution; do not assume saved outputs will exactly reproduce on another environment.

No Groq or OpenAI API key is required for these uploaded capstone notebooks' grounded narrative demonstration.

## Limitations and responsible use

The data is synthetic and may encode assumptions not representative of live securities settlement. The experiment does not establish performance on real trading infrastructure. The notebooks do not demonstrate production deployment, live market integration, or validated real-world cost reduction. Thresholds, monitoring parameters, and recommendations are illustrative. Any real deployment would require point-in-time data integrity, independent validation, privacy and security review, explainability, drift controls, audit trails, and human decision-making.

## Next steps

Validate on anonymized real trade data with permission; compare models on out-of-time cohorts; calibrate probabilities; tune alert volumes and thresholds against approved business costs; test generalization by asset class, market, and counterparty; strengthen local explanations; and explore controlled intraday rescoring, event-stream ingestion, and operational monitoring.

## Contact and further information

**Author:** Rama Nuti  
**GitHub profile:** https://github.com/ramakrishna580/  
**Contact:** Please reach out through the GitHub profile. No public email address was provided.

## Notebook progression and deliverables

The **draft** provides a reproducible starting point and synthetic data generator. The **pre-final** organizes the business question, exploration, experiments, model comparison, interpretability, and recommendations into a continuous report. The **final** reformats the work into independent scenarios for data validation, leakage control, EDA, regression, classification, clustering, anomaly detection, NLP, grounded risk narratives, a high-risk trade queue, and PSI monitoring.

### Submission verification checklist

- [✅] Input CSV is available and the data path resolves.
- [✅] Notebook kernel is restarted and all intended sections run successfully.
- [✅] Data leakage exclusions are confirmed.
- [✅] Model comparison metrics reflect the latest run.
- [✅] Results and charts are retained and interpreted accurately.
- [✅] Synthetic-data and hypothetical-cost limitations are explicit.
- [✅] Author name and GitHub profile are included; add a public email address only if desired.
