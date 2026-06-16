# 🎯 Interview Q&A Guide — US Housing Price Prediction System

> **How to use this document:** Read through the questions and answers before your interview. Each answer is written in a natural, conversational tone — the way you would actually speak in an interview. Practice saying them out loud.

---

## 📌 Project Introduction Questions

---

### Q1: Can you tell me about your project?

**Simple Answer:**
I built a production-grade machine learning system that predicts US housing prices. It's not just a model — it's a complete pipeline that goes from raw CSV data all the way to a live web service deployed on AWS. It includes data cleaning, feature engineering, automated hyperparameter tuning with Optuna, a FastAPI backend for predictions, a Streamlit dashboard for visualization, and a CI/CD pipeline with GitHub Actions. The model achieves a 0.96 R² score.

**Files Related:**
- Entire `src/` directory (core logic)
- `app.py` (dashboard)
- `.github/workflows/ci-cd.yml` (CI/CD)
- `Dockerfile` and `Dockerfile.streamlit` (containerization)

**Possible Follow-up Questions:**
- What kind of data does it use?
- What is R² and what does 0.96 mean?
- Why did you choose these specific technologies?
- How long did it take to build?

---

### Q2: What problem does this project solve?

**Simple Answer:**
It solves the problem of estimating house prices in the US. Real estate agents, buyers, and analysts need quick price estimates. This system takes property details like location, listing price, and time of year, and returns an estimated sale price. More importantly, it demonstrates how to take a machine learning model from a Jupyter notebook to a production-grade deployed system.

**Files Related:**
- `README.md` (project overview)
- `notebooks/` (research phase)
- `src/` (production code)

**Possible Follow-up Questions:**
- Who is the target user?
- How is this different from Zillow's Zestimate?
- What's the business value?

---

### Q3: What makes this project different from a typical Kaggle project?

**Simple Answer:**
A Kaggle project usually ends with a model and a score. This project goes much further — it handles deployment, infrastructure, CI/CD, schema alignment, error handling, and production reliability. I focused on things like preventing training-serving skew, handling edge cases in production (like the object dtype bug), and automating the full deployment lifecycle. The model accuracy matters, but the system reliability matters more.

**Files Related:**
- `src/inference_pipeline/inference.py` (schema alignment, numeric enforcement)
- `.github/workflows/ci-cd.yml` (CI/CD)
- `scripts/` (infrastructure)

**Possible Follow-up Questions:**
- What is training-serving skew?
- What's the object dtype bug you mentioned?
- How do you ensure reliability?

---

## 🏗️ Architecture Questions

---

### Q4: Can you explain the architecture of your system?

**Simple Answer:**
The system has four main layers:

1. **Data Pipeline** — Cleans raw data, creates features, and saves encoders. This runs offline.
2. **Training Pipeline** — Trains the XGBoost model, tunes hyperparameters with Optuna, and logs experiments to MLflow. Also runs offline.
3. **Serving Layer** — A FastAPI web server that loads the model and encoders from S3 and serves predictions in real-time. There's also a Streamlit dashboard for visualization.
4. **Infrastructure** — Docker containers deployed on AWS ECS Fargate, with an Application Load Balancer routing traffic and S3 as the artifact store.

All pipelines are decoupled — you can retrain the model without touching the API, or update the dashboard without retraining.

**Files Related:**
- `src/feature_pipeline/` (data pipeline)
- `src/training_pipeline/` (training pipeline)
- `src/api/main.py` and `src/inference_pipeline/inference.py` (serving)
- `scripts/setup_infra.py` (infrastructure)

**Possible Follow-up Questions:**
- Why did you decouple the pipelines?
- How do the pipelines communicate?
- What happens if S3 goes down?

---

### Q5: Why did you choose a modular/pipeline architecture?

**Simple Answer:**
Modularity gives three big benefits:

1. **Independent scaling** — I can scale the API containers without touching the data pipeline.
2. **Easier testing** — Each pipeline can be tested in isolation with unit tests.
3. **Team-friendly** — A data scientist can work on the feature pipeline while a backend engineer works on the API without conflicts.

Also, if I need to swap XGBoost for a different model, I only change the training pipeline. The API and inference pipeline don't care what model is used — they just load a `.pkl` file.

**Files Related:**
- `src/` directory structure
- `tests/` (independent test files for each module)

**Possible Follow-up Questions:**
- Have you had to swap components before?
- How do you coordinate changes across pipelines?
- What if a feature pipeline change breaks inference?

---

### Q6: How does data flow through your system?

**Simple Answer:**
1. Raw CSV → `load.py` splits it by time into train/eval/holdout
2. Split CSVs → `preprocess.py` cleans city names, merges GPS, removes outliers
3. Cleaned CSVs → `feature_engineering.py` creates numeric features and saves encoders
4. Processed CSVs → `train.py` or `tune.py` trains the model and saves it
5. Model + encoders → `provision_aws.py` uploads to S3
6. At runtime → `main.py` (API) downloads from S3, loads model
7. User sends data → `inference.py` transforms it → model predicts → response returned

**Files Related:**
- `src/feature_pipeline/load.py` → `preprocess.py` → `feature_engineering.py`
- `src/training_pipeline/train.py`
- `scripts/provision_aws.py`
- `src/api/main.py` → `src/inference_pipeline/inference.py`

**Possible Follow-up Questions:**
- What if you add a new feature — what changes?
- How do you version the data?
- Where are the bottlenecks?

---

## 🔧 Feature-Based Questions

---

### Q7: How did you handle the high number of cities (high-cardinality categorical data)?

**Simple Answer:**
I used two techniques instead of one-hot encoding:

1. **Target Encoding** — Replace each city name with the average house price in that city. So "Boston" might become 750,000. This creates one column instead of 1000+ dummy columns.
2. **GPS Coordinate Merging** — I merged latitude and longitude for each city, so the model can learn that nearby cities have similar prices.

Both are much more efficient than one-hot encoding and preserve geographic information.

**Files Related:**
- `src/feature_pipeline/feature_engineering.py` (lines for `target_encode()` and `frequency_encode()`)
- `src/feature_pipeline/preprocess.py` (lines for `clean_and_merge()`)

**Possible Follow-up Questions:**
- What is the risk of target encoding?
- How did you prevent target leakage?
- Why not use embeddings?

---

### Q8: What feature engineering did you do?

**Simple Answer:**
I created several types of features:

1. **Temporal features**: Extracted year, quarter, and month from the date. Housing markets are seasonal — prices tend to be higher in spring.
2. **Frequency encoding**: Counted how often each zipcode appears. High-frequency zipcodes are usually urban areas with different price dynamics.
3. **Target encoding**: Mapped city names to their average house price from the training data.
4. **Spatial features**: Merged GPS coordinates (latitude/longitude) for each city so the model can learn geographic price patterns.

I also dropped columns that could cause data leakage, like `median_sale_price`.

**Files Related:**
- `src/feature_pipeline/feature_engineering.py` (all feature creation functions)
- `src/feature_pipeline/preprocess.py` (`clean_and_merge()` for GPS)

**Possible Follow-up Questions:**
- Why did you drop `median_sale_price`?
- What other features did you consider?
- How did you decide which features to keep?

---

### Q9: What is the "Schema Alignment Layer" and why do you need it?

**Simple Answer:**
When the model was trained, it saw a specific set of columns in a specific order. But in production, users might send data with missing columns, extra columns, or columns in a different order. The schema alignment layer uses Pandas' `reindex()` to force the incoming data to match the training schema exactly — missing columns get filled with zeros, extra columns get dropped.

Without this, the model would crash or give wrong predictions because XGBoost is strict about input format.

**Files Related:**
- `src/inference_pipeline/inference.py` (line: `df = df.reindex(columns=TRAIN_FEATURE_COLUMNS, fill_value=0)`)

**Possible Follow-up Questions:**
- What happens if a critical feature is missing?
- Have you seen schema mismatches in production?
- How would you handle this in a more robust way?

---

### Q10: What is the "Numeric Enforcement Layer"?

**Simple Answer:**
This is a safety net in the inference pipeline. When FastAPI receives JSON data and converts it to a Pandas DataFrame, sometimes numeric columns get read as text ("object" type) instead of numbers — especially if there are null values or very precise decimal strings. XGBoost crashes immediately when it sees non-numeric data.

The fix is one line: `df = df.apply(pd.to_numeric, errors="coerce").fillna(0)`. It forces every column to be a number, and any value that can't be converted becomes zero.

**Files Related:**
- `src/inference_pipeline/inference.py` (line 105)

**Possible Follow-up Questions:**
- When did you discover this bug?
- Could filling with 0 cause incorrect predictions?
- Is there a better default than 0?

---

## 🌐 API Questions

---

### Q11: What API endpoints does your system expose?

**Simple Answer:**
Five endpoints:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Simple "I'm alive" check |
| `/health` | GET | Returns model status and expected feature count |
| `/predict` | POST | Accepts JSON housing data, returns price predictions |
| `/run_batch` | POST | Triggers a monthly batch prediction job |
| `/latest_predictions` | GET | Returns the most recent batch prediction results |

The `/predict` endpoint is the main one. It accepts a list of dictionaries (one per house), runs the inference pipeline, and returns predicted prices plus actual prices if available.

**Files Related:**
- `src/api/main.py` (all endpoint definitions)

**Possible Follow-up Questions:**
- What's the response format?
- How do you handle errors?
- What's the latency?

---

### Q12: Why did you choose FastAPI over Flask or Django?

**Simple Answer:**
Three reasons:

1. **Speed** — FastAPI is one of the fastest Python web frameworks. It's built on Starlette and uses async where possible.
2. **Automatic docs** — It generates interactive API documentation (Swagger UI) automatically from your code.
3. **Type validation** — It uses Pydantic for automatic request/response validation, which catches bad data before it reaches the model.

For a prediction API where performance and data validation matter, FastAPI is the best choice.

**Files Related:**
- `src/api/main.py`

**Possible Follow-up Questions:**
- Have you used Flask or Django?
- How do you handle request validation?
- What about async support?

---

## 🗄️ Database / Storage Questions

---

### Q13: How do you manage data storage?

**Simple Answer:**
I use **AWS S3** as the primary storage system. It stores three types of artifacts:

1. **Model weights** (`xgb_best_model.pkl`)
2. **Encoders** (`freq_encoder.pkl`, `target_encoder.pkl`)
3. **Processed datasets** (feature-engineered CSVs)

S3 is the single source of truth. The production containers download these files on startup. Locally, files are cached so they don't re-download every time.

I don't use a traditional database because the data is batch-processed (CSVs), not transactional. S3 is simpler, cheaper, and perfectly suited for this use case.

**Files Related:**
- `scripts/provision_aws.py` (upload logic)
- `src/api/main.py` (`load_from_s3()` function)
- `app.py` (`load_from_s3()` function)

**Possible Follow-up Questions:**
- Why not use a database?
- How do you version the artifacts?
- What if you need to rollback to a previous model?

---

### Q14: How do you handle model versioning?

**Simple Answer:**
The model is versioned at the storage layer. S3 stores the current production model. MLflow tracks every experiment and trial during development. But MLflow is not the authoritative source for production — only what's in S3 matters.

To roll back to a previous model, I can revert the S3 objects to a previous version, and the containers will pick up the old model on their next restart. No code changes needed.

The model, frequency encoder, and target encoder are treated as a **triplet** — they must always be from the same training run. If any one is missing or mismatched, the system fails fast on startup.

**Files Related:**
- `scripts/provision_aws.py` (artifact upload)
- `src/api/main.py` (artifact download + integrity check)
- `src/training_pipeline/tune.py` (MLflow logging)

**Possible Follow-up Questions:**
- How do you know which model version is in production?
- What if someone accidentally uploads a bad model?
- How would you add a model registry?

---

## 🔐 Authentication / Security Questions

---

### Q15: How do you handle security and secrets?

**Simple Answer:**
I follow the principle of least privilege:

1. **No hardcoded keys** — AWS credentials are never in the code. Locally, they come from a `.env` file. In production, they're injected via IAM Task Roles.
2. **IAM Task Roles** — ECS containers get a "role" that only allows them to access the specific S3 bucket they need. No broad permissions.
3. **GitHub Secrets** — CI/CD credentials are stored in GitHub Secrets, not in the repository.
4. **Network isolation** — The API runs behind an Application Load Balancer. The ALB is the only public endpoint; the containers themselves are in a private network.

**Files Related:**
- `scripts/setup_iam.py` (IAM role creation)
- `.github/workflows/ci-cd.yml` (secrets usage: `${{ secrets.AWS_ACCESS_KEY_ID }}`)
- `.gitignore` (excludes `.env`)

**Possible Follow-up Questions:**
- What is the principle of least privilege?
- How would you add user authentication to the API?
- What if a secret is accidentally committed?

---

## ⚙️ State Management Questions

---

### Q16: How do you ensure the model in production matches the one from training?

**Simple Answer:**
This is called "training-serving parity" and I ensure it in three ways:

1. **Shared code** — The inference pipeline imports the exact same functions (`clean_and_merge`, `add_date_features`, etc.) from the feature pipeline that training uses. There's no duplicated logic.
2. **Artifact triplet** — The model, frequency encoder, and target encoder are always uploaded together. They're treated as a unit.
3. **Schema alignment** — The inference pipeline reindexes incoming data to match the training columns exactly, so the model always sees the same feature set.

**Files Related:**
- `src/inference_pipeline/inference.py` (imports from `feature_pipeline`)
- `src/feature_pipeline/preprocess.py` and `feature_engineering.py` (shared functions)
- `scripts/provision_aws.py` (uploads triplet together)

**Possible Follow-up Questions:**
- What is training-serving skew?
- Have you experienced skew in practice?
- How would you detect skew automatically?

---

## 🚀 Performance Questions

---

### Q17: What's the model's performance?

**Simple Answer:**

| Metric | Eval Set (2020–2021) | Holdout (2022–2023) |
|--------|---------------------|---------------------|
| MAE | ~$32,435 | ~$67,000 |
| RMSE | ~$70,750 | ~$138,000 |
| R² | 0.961 | N/A (live validation) |
| Avg % Error | N/A | ~8.04% |

The model is very accurate on the eval set (0.96 R²). Performance drops on the holdout because the 2022–2023 market saw unusual shifts (post-COVID boom, interest rate hikes). An 8% average error is still practical for real estate estimation.

**Files Related:**
- `src/training_pipeline/train.py` (metric calculation)
- `src/training_pipeline/eval.py` (evaluation)
- `app.py` (live error computation)

**Possible Follow-up Questions:**
- Why does performance drop on the holdout?
- Is 8% error acceptable?
- How would you improve the model?

---

### Q18: How would you improve the model's accuracy?

**Simple Answer:**
Several approaches:

1. **Add external data** — Mortgage interest rates, GDP growth, inflation rates. These economic factors heavily influence housing prices but aren't in the current dataset.
2. **More granular features** — Property-level features like square footage, number of bedrooms, lot size.
3. **Ensemble methods** — Combine XGBoost with LightGBM or a neural network.
4. **Feature interactions** — Explicitly model interactions between location and time (e.g., some cities appreciated faster than others).
5. **More recent data** — Retrain regularly with fresh market data.

**Files Related:**
- `src/feature_pipeline/feature_engineering.py` (where features are created)
- `src/training_pipeline/tune.py` (model configuration)

**Possible Follow-up Questions:**
- Why haven't you done these yet?
- What's the biggest bang-for-buck improvement?
- How would you add new features without breaking production?

---

## 🐛 Debugging Questions

---

### Q19: What was the hardest bug you encountered?

**Simple Answer:**
The "Object Dtype Mismatch" bug. After deploying to AWS, the API started crash-looping when receiving valid JSON requests. XGBoost was rejecting the input, saying it received `object` types instead of `float32`.

**Root cause:** FastAPI's JSON parser and Pandas' `from_dict()` sometimes interpret numeric columns as text, especially when the first few rows contain null values or high-precision decimal strings.

**Fix:** I added a numeric enforcement layer in the inference pipeline — one line: `df = df.apply(pd.to_numeric, errors="coerce").fillna(0)`. This forces all columns to numeric types.

**Prevention:** I added an integration test that specifically sends "problematic" JSON strings to ensure they're handled correctly.

**Files Related:**
- `src/inference_pipeline/inference.py` (the fix — line 105)
- `tests/test_inference.py` (the prevention test)

**Possible Follow-up Questions:**
- How did you debug this?
- How long did it take to find the root cause?
- Could this happen again?

---

### Q20: How do you handle errors in the API?

**Simple Answer:**
Multiple layers of defense:

1. **Input validation** — Check if the DataFrame is empty before processing.
2. **Model check** — Verify the model file exists before attempting prediction.
3. **Schema alignment** — Reindex columns to prevent feature mismatch errors.
4. **Type enforcement** — Force all values to numeric to prevent dtype crashes.
5. **Health endpoint** — The ALB uses `/health` to check if the service is ready. If the model is missing, it returns "unhealthy."

If something still goes wrong, the API returns an error response instead of crashing, and logs help trace the exact issue.

**Files Related:**
- `src/api/main.py` (error handling in endpoints)
- `src/inference_pipeline/inference.py` (defensive processing)

**Possible Follow-up Questions:**
- What happens if S3 is down?
- Do you have alerting?
- How do you monitor errors in production?

---

## 🤔 "Why Did You Build It This Way?" Questions

---

### Q21: Why did you use a chronological data split instead of random?

**Simple Answer:**
Housing prices are heavily time-dependent. If I randomly split the data, training data would contain samples from 2022 (a boom year) and test data would contain samples from 2018. The model would "cheat" by learning future market trends, giving a misleadingly high accuracy that wouldn't hold up in the real world.

By splitting chronologically — train on pre-2020, test on 2022+ — I simulate the real scenario: "Can the model predict future prices using only past data?"

**Files Related:**
- `src/feature_pipeline/load.py` (time-based split logic)

**Possible Follow-up Questions:**
- What if you had more data from different time periods?
- Is there a risk of the model being outdated?
- How often should you retrain?

---

### Q22: Why two separate Docker containers instead of one?

**Simple Answer:**
Separation of concerns. The API container (FastAPI) handles prediction logic — it needs to be fast and lightweight. The dashboard container (Streamlit) handles visualization — it needs different libraries and has different scaling needs.

If the API is slow, I can add more API containers without duplicating the heavier dashboard. If I need to update the UI, I can redeploy just the dashboard without touching the API. They communicate over HTTP through the load balancer.

**Files Related:**
- `Dockerfile` (API container)
- `Dockerfile.streamlit` (dashboard container)

**Possible Follow-up Questions:**
- How do they communicate?
- What if you added a third service?
- What about shared dependencies?

---

### Q23: Why AWS ECS Fargate over EC2 or Lambda?

**Simple Answer:**
- **EC2** requires managing servers (patching, scaling, monitoring). Fargate is serverless — AWS handles all of that.
- **Lambda** has a 15-minute execution limit and cold start latency. Our model needs to stay warm for real-time predictions, and batch jobs can take longer.
- **Fargate** gives us always-on containers that scale based on demand, without server management overhead. It's the best of both worlds for this workload.

**Files Related:**
- `scripts/setup_infra.py` (Fargate configuration)
- `housing-api-task-def.json` and `streamlit-task-def.json` (task definitions)

**Possible Follow-up Questions:**
- How much does it cost?
- How does auto-scaling work with Fargate?
- Would you use Kubernetes instead?

---

### Q24: Why did you use `uv` instead of pip or conda?

**Simple Answer:**
`uv` is a modern Python package manager that's about 10x faster than pip. It uses a lockfile (`uv.lock`) for deterministic builds — meaning everyone gets the exact same package versions. This is important for reproducibility. When I build Docker images, `uv sync --frozen` ensures the container has the exact same dependencies as my local environment.

**Files Related:**
- `pyproject.toml` (dependency definitions)
- `uv.lock` (locked dependency versions)
- `Dockerfile` (uses `uv sync`)

**Possible Follow-up Questions:**
- What's a lockfile?
- Why does determinism matter?
- Have you used Poetry?

---

## 🚢 Deployment / Build Questions

---

### Q25: How does your CI/CD pipeline work?

**Simple Answer:**
1. I push code to the `main` branch on GitHub.
2. GitHub Actions is triggered automatically.
3. It configures AWS credentials from GitHub Secrets.
4. It builds two Docker images — one for the API, one for the dashboard.
5. Images are tagged with the commit hash and pushed to AWS ECR.
6. It tells ECS to force a new deployment, which pulls the latest images.

The whole process is automated — from code commit to live deployment. If tests fail (not currently in the pipeline but should be), deployment would stop.

**Files Related:**
- `.github/workflows/ci-cd.yml` (full pipeline definition)

**Possible Follow-up Questions:**
- How long does deployment take?
- What if deployment fails?
- Do you have rollback capability?

---

### Q26: How would you add tests to the CI/CD pipeline?

**Simple Answer:**
I would add a step before the Docker build that runs `pytest`. If any test fails, the workflow stops and no image is pushed. Here's what the step would look like:

```yaml
- name: Run tests
  run: |
    pip install uv
    uv sync --frozen
    uv run pytest tests/ -v
```

I'd also add a test for the inference pipeline with mock data to ensure the full prediction flow works.

**Files Related:**
- `.github/workflows/ci-cd.yml` (where to add the step)
- `tests/` (existing test suite)

**Possible Follow-up Questions:**
- What if tests are slow?
- How do you test with production data?
- Do you use staging environments?

---

### Q27: How do you handle environment variables across environments?

**Simple Answer:**
Three layers:

| Environment | How secrets are managed |
|-------------|----------------------|
| **Local development** | `.env` file (git-ignored) with AWS keys |
| **CI/CD (GitHub Actions)** | GitHub Secrets, injected as env vars |
| **Production (ECS)** | IAM Task Roles (no keys needed) + environment variables in task definitions |

The key principle: secrets never go in code or get committed to Git. The `.env` file is in `.gitignore`, and production uses IAM roles so there are no keys at all.

**Files Related:**
- `.gitignore` (excludes `.env`)
- `.github/workflows/ci-cd.yml` (uses `secrets.*`)
- `housing-api-task-def.json` and `streamlit-task-def.json` (env vars)
- `scripts/setup_iam.py` (IAM roles)

**Possible Follow-up Questions:**
- What if you need to add a new environment variable?
- How do you handle different configs for staging vs production?
- What happens if a secret is rotated?

---

## 📝 Resume / Project Defense Questions

---

### Q28: What's the most technically challenging part of this project?

**Simple Answer:**
Ensuring **training-serving parity** while handling real-world production issues. The model expects data in a very specific format, and any difference between how data is processed during training vs inference causes failures. I had to:

1. Share the exact same transformation functions between training and inference
2. Add a schema alignment layer for column mismatches
3. Add numeric enforcement for type mismatches
4. Make sure all encoders are from the same training run

This was harder than building the model itself, because the bugs only appear in production, not in notebooks.

**Files Related:**
- `src/inference_pipeline/inference.py` (the solution)
- `src/feature_pipeline/` (shared transformation logic)

**Possible Follow-up Questions:**
- How did you test for parity?
- What other production issues did you face?
- How would you prevent this in future projects?

---

### Q29: What did you learn from this project?

**Simple Answer:**
Three big lessons:

1. **Model accuracy is only 20% of the work** — The remaining 80% is data pipelines, deployment, testing, and monitoring. A 0.96 R² model is useless if it crashes in production.
2. **Production bugs are different from dev bugs** — The object dtype mismatch never appeared in Jupyter notebooks. It only showed up when real JSON payloads hit the API. Testing in realistic conditions is essential.
3. **Infrastructure as code matters** — Having scripts to set up the entire AWS infrastructure means I can tear it down and rebuild it in minutes. No clicking through the console.

**Files Related:**
- Entire project (this is a lesson from the overall experience)

**Possible Follow-up Questions:**
- What would you do differently next time?
- What's your next project?
- How did this change your approach to ML?

---

### Q30: If you had to start this project again, what would you change?

**Simple Answer:**
1. **Add a proper test gate in CI/CD from day one** — I would block deployments if tests fail, not add tests after the fact.
2. **Use a feature store** — Instead of CSV files and manual encoder management, I'd use something like Feast to manage features and ensure training-serving consistency automatically.
3. **Add monitoring earlier** — Set up Evidently for data drift detection and Prometheus/Grafana for API monitoring from the start.
4. **Use a model registry** — Instead of just S3, use MLflow's model registry with staging/production stages to formalize model promotion.

**Files Related:**
- `.github/workflows/ci-cd.yml` (CI/CD improvements)
- `src/feature_pipeline/` (feature store replacement)
- `src/training_pipeline/tune.py` (MLflow registry)

**Possible Follow-up Questions:**
- Why didn't you do these things?
- How much effort would each change take?
- What's your priority order?

---

### Q31: How would you scale this system to handle 100x more data?

**Simple Answer:**
Several changes at different layers:

| Layer | Current | Scaled |
|-------|---------|--------|
| Data processing | Pandas (in-memory) | Spark or Polars (distributed) |
| Feature storage | CSV files on S3 | Feature store (Feast) |
| Pipeline orchestration | Manual scripts | Airflow or Prefect (scheduled) |
| Model serving | Single ECS task | Auto-scaling ECS + caching layer |
| Data storage | Single S3 bucket | Partitioned by date, with Athena for queries |

The architecture wouldn't change dramatically — I'd still have separate pipelines. The main change is switching from single-machine tools (Pandas, local files) to distributed ones (Spark, feature stores).

**Files Related:**
- `src/feature_pipeline/` (Pandas → Spark/Polars)
- `scripts/setup_infra.py` (auto-scaling config)
- `src/api/main.py` (add caching)

**Possible Follow-up Questions:**
- At what data volume would each change be needed?
- How would you measure when scaling is necessary?
- What's the cost implication?

---

### Q32: Walk me through what happens when a user clicks "Show Predictions" on the dashboard.

**Simple Answer:**
1. User selects Year, Month, Region filters and clicks the button.
2. Streamlit applies the filters to the holdout dataset (already loaded from S3).
3. The filtered data is "sanitized" — NaN and Infinity values are converted to `null` for valid JSON.
4. A POST request is sent to the FastAPI `/predict` endpoint with the data as JSON.
5. FastAPI converts the JSON to a DataFrame.
6. The inference pipeline kicks in:
   - Cleans the data (city normalization, GPS merge)
   - Adds date features (year, quarter, month)
   - Applies saved encoders (frequency and target)
   - Aligns columns with training schema
   - Forces all values to numeric
   - Model predicts prices
7. API returns predictions (and actual prices if available).
8. Dashboard calculates MAE, RMSE, and Average % Error.
9. Dashboard displays a data table and an interactive Plotly chart.

**Files Related:**
- `app.py` (steps 1–4, 8–9)
- `src/api/main.py` (step 5)
- `src/inference_pipeline/inference.py` (steps 6–7)

**Possible Follow-up Questions:**
- How long does this take?
- What if the API is down?
- Can multiple users use it simultaneously?

---

### Q33: What testing strategies did you use?

**Simple Answer:**
Four types of tests:

1. **Unit tests** (`test_features.py`) — Test individual functions like `add_date_features()`, `frequency_encode()`, `remove_outliers()`. Each test creates a tiny DataFrame, runs the function, and checks the output.

2. **Integration tests** (`test_inference.py`) — Test the full inference pipeline end-to-end. Loads a sample from the holdout set, runs prediction, and checks that the output has a `predicted_price` column with numeric values.

3. **Training tests** (`test_training.py`) — Train a quick model (2% data sample, 20 trees), verify it saves correctly and produces valid metrics. Also tests the tuning pipeline with 2 Optuna trials.

4. **Data quality tests** (`data_quality.py`) — Uses Great Expectations to validate data ranges (prices between $1K–$12M, valid dates, 5-digit zipcodes, non-null cities).

**Files Related:**
- `tests/test_features.py` (9 unit tests)
- `tests/test_inference.py` (1 integration test)
- `tests/test_training.py` (3 training tests)
- `tests/data_quality.py` (data validation)

**Possible Follow-up Questions:**
- What's your test coverage?
- How do you test the API specifically?
- Do you use mocking?

---

### Q34: How do you handle the case where S3 is unavailable?

**Simple Answer:**
The system has local caching as a fallback. Both the API (`main.py`) and the dashboard (`app.py`) use a `load_from_s3()` function that first checks if the file already exists locally. If it does, it skips the download. So if S3 goes down after the first successful download, the system continues working with cached files.

However, if a container starts fresh (new deployment) and S3 is down, the container won't be able to load the model and will start in an "unhealthy" state. The health check endpoint reports this, and the ALB won't route traffic to it.

**Files Related:**
- `src/api/main.py` (`load_from_s3()` function)
- `app.py` (`load_from_s3()` function)

**Possible Follow-up Questions:**
- How long are files cached?
- What if the cached model is outdated?
- How would you add a more robust caching layer?

---

### Q35: Explain the difference between online and batch inference in your system.

**Simple Answer:**
- **Online inference** (`/predict` endpoint): Real-time, single or small batch predictions. A user or application sends data via HTTP and gets predictions back in seconds. Used by the Streamlit dashboard for interactive exploration.

- **Batch inference** (`run_monthly.py`): High-throughput processing of large datasets. Loads an entire CSV, runs predictions on all rows, and saves results to a timestamped file. Can be triggered via the `/run_batch` endpoint or run directly as a script.

Both use the **exact same `predict()` function** from `inference.py`, so there's no logic duplication. The only difference is how data enters and exits the pipeline.

**Files Related:**
- `src/api/main.py` (`/predict` endpoint for online)
- `src/batch/run_monthly.py` (batch processing)
- `src/inference_pipeline/inference.py` (shared `predict()` function)

**Possible Follow-up Questions:**
- How often does the batch job run?
- What triggers it?
- How do you handle batch failures?

---

## 🎓 Advanced / Tricky Questions

---

### Q36: What is data leakage and how did you prevent it?

**Simple Answer:**
Data leakage is when information from outside the training dataset leaks into the model, giving it an unfair advantage that won't exist in the real world.

I prevented it in three ways:
1. **Chronological split** — No future data in the training set.
2. **Encoders fit on train only** — Target and frequency encoders are calculated only from training data, then applied to eval/holdout.
3. **Dropped leaky columns** — Removed `median_sale_price` (too close to the target `price`) and raw categorical columns.

**Files Related:**
- `src/feature_pipeline/load.py` (chronological split)
- `src/feature_pipeline/feature_engineering.py` (`target_encode()` fits on train only)
- `src/feature_pipeline/feature_engineering.py` (`drop_unused_columns()`)

**Possible Follow-up Questions:**
- Can you give an example of subtle leakage?
- How would you detect leakage after the fact?
- What if new data has different distributions?

---

### Q37: What is idempotency and where do you use it?

**Simple Answer:**
Idempotency means you can run something multiple times and get the same result every time without side effects. In this project:

1. **Feature pipeline** — Running `preprocess.py` twice on the same data produces the same output. It doesn't accumulate changes.
2. **AWS scripts** — `provision_aws.py` and `setup_infra.py` check if resources already exist before creating them. Running them twice doesn't create duplicates.
3. **S3 downloads** — `load_from_s3()` checks if the file exists locally before downloading. Multiple calls don't cause redundant downloads.

This is important because scripts sometimes fail midway and need to be rerun.

**Files Related:**
- `scripts/provision_aws.py` (checks for existing buckets/repos)
- `scripts/setup_infra.py` (checks for existing security groups, ALBs)
- `src/api/main.py` (`load_from_s3()` checks for existing files)

**Possible Follow-up Questions:**
- Why is idempotency important in distributed systems?
- Can you give an example where it's NOT idempotent?
- How do you test for idempotency?

---

### Q38: How would you add data drift detection?

**Simple Answer:**
I would use **Evidently AI** (which is already in the project dependencies) to compare incoming production data against the training data distribution. Specifically:

1. Save a "reference" dataset (the training data statistics).
2. For each batch of predictions, compare the input features against the reference.
3. If significant drift is detected (e.g., median home values shift by 20%), trigger an alert.
4. If drift is severe enough, automatically trigger a model retraining pipeline.

This would catch scenarios like the post-COVID housing boom, where prices suddenly jumped beyond the model's training range.

**Files Related:**
- `pyproject.toml` (Evidently is already a dependency)
- `src/inference_pipeline/inference.py` (where drift checks would be added)

**Possible Follow-up Questions:**
- What metrics would you use to measure drift?
- How much drift is acceptable?
- How would you retrain automatically?

---

### Q39: What is the "training-serving contract" in your system?

**Simple Answer:**
It's an informal agreement between the training and inference pipelines:

1. **Same transformations** — Any logic in the feature pipeline must be compatible with the inference pipeline.
2. **Encoder coupling** — Model + frequency encoder + target encoder are a triplet. They must come from the same training run.
3. **Determinism** — Same input always produces the same output, regardless of environment.

If someone changes the feature pipeline without updating the inference pipeline or retraining the model, the contract is broken and predictions become unreliable. The integration tests in `test_inference.py` help enforce this.

**Files Related:**
- `src/inference_pipeline/inference.py` (imports shared functions)
- `src/feature_pipeline/` (shared transformation logic)
- `tests/test_inference.py` (contract verification)

**Possible Follow-up Questions:**
- How do you enforce this contract?
- What happens when the contract is broken?
- How would you formalize it?

---

### Q40: If the interviewer says "Tell me about a time you solved a difficult production issue"

**Simple Answer:**
After deploying the API to ECS, it started crash-looping on valid requests. The logs showed XGBoost rejecting inputs due to "object dtype."

**What I did:**
1. Reproduced the issue locally by sending the same JSON payload.
2. Discovered that Pandas was inferring some numeric columns as `object` type when parsing JSON.
3. Traced it to how FastAPI's JSON parser handles nulls and high-precision decimals.
4. Implemented a one-line fix: `df.apply(pd.to_numeric, errors="coerce").fillna(0)`.
5. Added an integration test with "problematic" payloads to prevent regression.

**What I learned:** Production bugs are different from development bugs. Testing with real-world data formats (JSON from HTTP requests) is essential, not just CSV files.

**Files Related:**
- `src/inference_pipeline/inference.py` (the fix)
- `tests/test_inference.py` (regression test)

**Possible Follow-up Questions:**
- How long did this take to diagnose?
- Could you have caught this in testing?
- What monitoring would have helped?

---

## 💡 Quick Tips for the Interview

1. **Start with the big picture** — Always begin with "I built a production ML system for housing price prediction" before diving into details.

2. **Use numbers** — "0.96 R² score," "8% average error," "1000+ cities," "10+ hyperparameters." Numbers make your answers concrete.

3. **Show depth** — When asked about a feature, mention the trade-offs. "I used target encoding instead of one-hot encoding because 1000+ cities would create a sparse matrix."

4. **Acknowledge limitations** — "The model doesn't account for interest rate changes" shows maturity and self-awareness.

5. **Connect to real-world impact** — "An 8% error on a $500K house is $40K — acceptable for initial estimation but not for final pricing."

6. **Be honest** — If you don't know something, say "I haven't implemented that yet, but here's how I would approach it."
