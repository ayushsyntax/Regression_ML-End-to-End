# 🏠 Project Walkthrough — US Housing Price Prediction System

> **Purpose of this document:** A simple, beginner-friendly guide to understand the entire project. Read this before an interview to confidently explain what you built, how it works, and why you made certain choices.

---

## 1. Project Overview

### What does this project do?

This project **predicts US housing prices** using machine learning. You give it details about a house (like city, zipcode, date of listing, median listing price, etc.), and it tells you the **estimated price** of that house.

### Main purpose

- Build a **complete machine learning system** — not just a model, but the entire pipeline from raw data to a live web application anyone can use.
- The system is deployed on **AWS (Amazon Web Services)** so it can be accessed over the internet.
- It includes a **dashboard** (a visual web page) where users can explore predictions and see how accurate the model is.

### In simple words

> "I built a system that takes raw housing data, cleans it, trains a machine learning model to predict house prices, and serves those predictions through a web API and a dashboard — all deployed on AWS cloud."

---

## 2. Project Structure

Here's what each folder and important file does:

```
Regression_ML-End-to-End/
│
├── src/                          ← 🧠 The brain — all core logic lives here
│   ├── feature_pipeline/         ← Cleans data and creates useful features
│   │   ├── load.py               ← Splits raw data into train/eval/holdout sets
│   │   ├── preprocess.py         ← Cleans city names, removes duplicates/outliers
│   │   └── feature_engineering.py← Creates new columns the model can learn from
│   │
│   ├── training_pipeline/        ← Trains the ML model
│   │   ├── train.py              ← Trains a basic XGBoost model
│   │   ├── tune.py               ← Finds the best settings using Optuna + MLflow
│   │   └── eval.py               ← Tests how good the model is
│   │
│   ├── inference_pipeline/       ← Makes predictions on new data
│   │   └── inference.py          ← Takes raw data → processes → predicts price
│   │
│   ├── api/                      ← The web API (how other apps talk to our model)
│   │   └── main.py               ← FastAPI endpoints (/predict, /health, etc.)
│   │
│   └── batch/                    ← Runs predictions on large batches of data
│       └── run_monthly.py        ← Simulates monthly batch prediction job
│
├── app.py                        ← 📊 Streamlit dashboard (the visual frontend)
├── scripts/                      ← 🛠️ AWS setup and deployment scripts
│   ├── provision_aws.py          ← Creates S3 bucket and ECR repos
│   ├── setup_infra.py            ← Sets up networking, load balancer, ECS
│   ├── setup_iam.py              ← Creates security roles for AWS
│   ├── update_services.py        ← Refreshes running containers
│   └── clean_holdout.py          ← Cleans the holdout dataset specifically
│
├── tests/                        ← ✅ Automated tests
│   ├── test_features.py          ← Tests for data cleaning and feature creation
│   ├── test_inference.py         ← Tests that predictions work end-to-end
│   ├── test_training.py          ← Tests that training and tuning work
│   └── data_quality.py           ← Data validation with Great Expectations
│
├── notebooks/                    ← 📓 Jupyter notebooks (research/experimentation)
│   ├── 00_data_split.ipynb       ← Splitting raw data
│   ├── 01_EDA_cleaning.ipynb     ← Exploring and cleaning data
│   ├── 02_feature_eng_encoding.ipynb ← Feature engineering experiments
│   ├── 03_baseline.ipynb         ← First model attempt
│   ├── 04_linear_regression_regularization.ipynb ← Linear models
│   ├── 05_XGBoost.ipynb          ← XGBoost experiments
│   ├── 06_hyperparameter_tuning_MFlow.ipynb ← Tuning with MLflow
│   └── 07_S3_push_datasets_AWS.ipynb ← Pushing data to AWS
│
├── configs/                      ← Configuration files (app, MLflow)
├── .github/workflows/ci-cd.yml   ← 🔄 CI/CD pipeline (auto-deploy on push)
├── Dockerfile                    ← Docker image for the API
├── Dockerfile.streamlit          ← Docker image for the dashboard
├── housing-api-task-def.json     ← ECS task config for API container
├── streamlit-task-def.json       ← ECS task config for dashboard container
├── pyproject.toml                ← Project dependencies and settings
└── pytest.ini                    ← Test configuration
```

### Quick summary of each folder

| Folder | What it does |
|--------|-------------|
| `src/feature_pipeline/` | Takes messy raw data and makes it clean and useful for the model |
| `src/training_pipeline/` | Trains the model and finds the best settings |
| `src/inference_pipeline/` | Uses the trained model to make predictions on new data |
| `src/api/` | A web server that lets anyone send data and get predictions back |
| `src/batch/` | Runs predictions on large amounts of data at once |
| `scripts/` | Sets up everything on AWS (cloud servers, storage, etc.) |
| `tests/` | Makes sure everything works correctly before deploying |
| `notebooks/` | Research notebooks where experiments were done first |

---

## 3. How the Project Works (Step-by-Step Flow)

### The Big Picture

```
Raw CSV Data → Clean & Process → Create Features → Train Model → Save to AWS S3
                                                                        ↓
User opens Dashboard → Selects filters → Dashboard sends data to API → API loads model from S3
                                                                        ↓
                                                API processes data → Model predicts price → Returns result
                                                                        ↓
                                                Dashboard shows predictions, charts, and error metrics
```

### Step-by-Step Breakdown

#### Step 1: Load and Split Data (`load.py`)

- Reads the raw CSV file containing housing data.
- Splits it by **time** (not randomly — this is important!):
  - **Train set**: Data before 2020 (model learns from this)
  - **Eval set**: 2020–2021 (used to tune the model's settings)
  - **Holdout set**: 2022–2023 (final "exam" to see if the model works on future data)
- **Why time-based split?** Because housing prices change over time. If we mix future data with past data, the model would "cheat" by seeing future trends.

#### Step 2: Clean the Data (`preprocess.py`)

- **Normalizes city names**: Different data sources spell city names differently (e.g., "Las Vegas-Henderson-Paradise" vs "Las Vegas-Henderson-North Las Vegas"). This step makes them consistent.
- **Merges GPS coordinates**: Adds latitude and longitude for each city so the model can learn that "nearby cities have similar prices."
- **Removes duplicates**: Gets rid of repeated rows.
- **Removes outliers**: Drops houses with prices over $19 million (extreme values that could confuse the model).

#### Step 3: Create Features (`feature_engineering.py`)

"Features" are the columns/numbers the model uses to learn patterns. This step creates new useful columns:

- **Date features**: Extracts year, quarter, and month from the date (housing prices are seasonal — spring is usually busier).
- **Frequency encoding**: Counts how often each zipcode appears. Popular zipcodes (many sales) behave differently from quiet ones.
- **Target encoding**: Converts city names into numbers representing the average house price in that city. This is smarter than one-hot encoding (which would create 1000+ columns for 1000+ cities).
- **Drops leaky columns**: Removes columns that could "leak" the answer (like `median_sale_price`) or are no longer needed (like raw `city_full`).

> **Important**: Encoders are only "fitted" (calculated) on the **training data** to prevent data leakage. The same encoder is then applied to eval and holdout sets.

#### Step 4: Train the Model (`train.py` and `tune.py`)

- Uses **XGBoost** (a very popular and powerful machine learning algorithm for tabular data).
- `train.py`: Trains a baseline model with default settings.
- `tune.py`: Uses **Optuna** (an automatic tuning tool) to try different combinations of settings and find the best one. It tries things like:
  - How many trees to use (200–800)
  - How deep each tree can grow (3–10)
  - Learning rate (how fast to learn)
  - And many more...
- Every experiment is logged to **MLflow** (a tracking tool) so you can compare results.
- The best model is saved as a `.pkl` file.

#### Step 5: Save Everything to AWS S3 (`provision_aws.py`)

Three files are uploaded to an S3 bucket (cloud storage):
1. `xgb_best_model.pkl` — the trained model
2. `freq_encoder.pkl` — the frequency encoding mapping
3. `target_encoder.pkl` — the target encoding mapping

**All three must match** because they were created together during training. Using the wrong encoder with a model would produce wrong predictions.

#### Step 6: Serve Predictions via API (`main.py`)

- A **FastAPI** web server starts up.
- On startup, it downloads the model and encoders from S3 (if not already cached locally).
- It exposes these endpoints:
  - `GET /` — "Is the API alive?" check
  - `GET /health` — More detailed health status (is the model loaded?)
  - `POST /predict` — Send housing data, get price predictions back
  - `POST /run_batch` — Trigger a monthly batch prediction job
  - `GET /latest_predictions` — See the most recent batch results

#### Step 7: Show Results on Dashboard (`app.py`)

- A **Streamlit** web application that non-technical users can interact with.
- Users select **Year**, **Month**, and **Region** filters.
- The dashboard sends the filtered data to the API and displays:
  - A table of actual vs predicted prices
  - Error metrics (MAE, RMSE, Average % Error)
  - A line chart showing price trends over the year

---

## 4. Important Features

### Feature 1: Time-Series Data Splitting

**What it does:** Splits data by date instead of randomly.

**How it works internally:** In `load.py`, the data is sorted by date. Everything before 2020 goes to training, 2020–2021 goes to eval, and 2022–2023 goes to holdout.

**Important files:** `src/feature_pipeline/load.py`

**How to explain in an interview:**
> "I used chronological splitting because housing data is time-dependent. A random split would leak future market trends into the training set, giving falsely high accuracy. By training only on past data and testing on future data, I get a realistic measure of how the model would perform in the real world."

---

### Feature 2: Feature Engineering Pipeline

**What it does:** Transforms raw data into numbers the model can learn from.

**How it works internally:**
- Extracts year/quarter/month from dates
- Creates frequency encoding for zipcodes (how popular an area is)
- Creates target encoding for cities (average price per city)
- Merges GPS coordinates so the model learns spatial patterns

**Important files:** `src/feature_pipeline/feature_engineering.py`, `src/feature_pipeline/preprocess.py`

**How to explain in an interview:**
> "Instead of one-hot encoding 1000+ cities (which would explode the feature space), I used target encoding — replacing each city name with the average house price in that city. I also added GPS coordinates so the model can learn that nearby cities have similar prices. All encoders are fitted only on training data to prevent leakage."

---

### Feature 3: Hyperparameter Tuning with Optuna

**What it does:** Automatically finds the best settings for the XGBoost model.

**How it works internally:** Optuna runs multiple "trials" — each trial tries a different combination of hyperparameters (like number of trees, tree depth, learning rate). It uses Bayesian optimization (smart guessing) instead of trying every combination randomly.

**Important files:** `src/training_pipeline/tune.py`

**How to explain in an interview:**
> "I used Optuna for Bayesian hyperparameter optimization. It searched through 10+ XGBoost parameters over multiple trials, minimizing RMSE on the eval set. Every trial was logged to MLflow for comparison. This gave me a 0.96 R² score on the evaluation data."

---

### Feature 4: Inference Pipeline with Schema Alignment

**What it does:** Takes raw input data and produces predictions, making sure the data matches exactly what the model expects.

**How it works internally:**
1. Cleans and preprocesses the input (same steps as training)
2. Applies saved encoders
3. **Reindexes columns** to match the training schema (adds missing columns as zeros, drops extra columns)
4. **Forces all values to numeric** to prevent XGBoost crashes
5. Runs the model prediction

**Important files:** `src/inference_pipeline/inference.py`

**How to explain in an interview:**
> "The inference pipeline guarantees training-serving parity — it applies the exact same transformations as training. It also has a schema alignment layer that reindexes incoming data to match the model's expected features, and a numeric enforcement layer that prevents the 'object dtype' crash that commonly happens when parsing JSON."

---

### Feature 5: FastAPI Web Service

**What it does:** Provides HTTP endpoints so any application can send data and get predictions.

**How it works internally:**
- On startup, downloads model + encoders from AWS S3
- `/predict` endpoint accepts a list of dictionaries (JSON), converts to DataFrame, runs the inference pipeline, returns predictions
- `/health` endpoint checks if the model file exists

**Important files:** `src/api/main.py`

**How to explain in an interview:**
> "I built a FastAPI service that serves the model over HTTP. On startup, it syncs artifacts from S3. The predict endpoint accepts JSON payloads, runs them through the full inference pipeline, and returns predictions. It also has a health check endpoint that ECS and the load balancer use to monitor service status."

---

### Feature 6: Streamlit Dashboard

**What it does:** A visual web interface for exploring predictions.

**How it works internally:**
- Downloads holdout data from S3
- Lets users filter by Year, Month, and Region
- Sends filtered data to the FastAPI backend
- Displays predictions, error metrics (MAE, RMSE, % Error), and trend charts

**Important files:** `app.py`

**How to explain in an interview:**
> "The Streamlit dashboard is the user-facing interface. It syncs holdout data from S3, lets users filter by time and region, and calls the FastAPI backend for predictions. It computes live error metrics and shows interactive Plotly charts comparing actual vs predicted prices."

---

### Feature 7: CI/CD Pipeline

**What it does:** Automatically builds and deploys the application when code is pushed to GitHub.

**How it works internally:**
1. Push to `main` branch triggers GitHub Actions
2. Builds two Docker images (API + Dashboard)
3. Pushes images to AWS ECR (a container image registry)
4. Forces ECS services to restart with the new images

**Important files:** `.github/workflows/ci-cd.yml`

**How to explain in an interview:**
> "I set up a CI/CD pipeline with GitHub Actions. Every push to main builds Docker images for both the API and dashboard, pushes them to ECR, and triggers a rolling deployment on ECS Fargate. This means code changes go from commit to production automatically."

---

### Feature 8: AWS Infrastructure

**What it does:** Runs the entire system in the cloud.

**How it works internally:**
- **S3**: Stores model files and processed data
- **ECR**: Stores Docker images
- **ECS Fargate**: Runs containers without managing servers
- **ALB (Application Load Balancer)**: Routes traffic — `/predict` goes to the API, `/dashboard` goes to Streamlit

**Important files:** `scripts/setup_infra.py`, `scripts/provision_aws.py`, `scripts/setup_iam.py`

**How to explain in an interview:**
> "The system runs on AWS using ECS Fargate for serverless container execution. An Application Load Balancer routes traffic between the API and dashboard using path-based routing. S3 is the single source of truth for all model artifacts, and IAM Task Roles provide secure access without hardcoded keys."

---

### Feature 9: Batch Predictions

**What it does:** Processes large datasets all at once instead of one-by-one.

**How it works internally:**
- Loads a large CSV of housing data
- Runs the same `predict()` function used by the API
- Saves results to `data/predictions/` with timestamps
- Can be triggered via the `/run_batch` API endpoint

**Important files:** `src/batch/run_monthly.py`

**How to explain in an interview:**
> "The system supports both real-time and batch inference. The batch pipeline uses the same predict function as the API, ensuring consistency. It saves timestamped results and can be triggered monthly for bulk processing."

---

### Feature 10: Automated Testing

**What it does:** Verifies that every part of the system works correctly.

**How it works internally:**
- **Unit tests** (`test_features.py`): Tests individual functions like date extraction, encoding, outlier removal
- **Integration tests** (`test_inference.py`): Tests the full flow from raw data to prediction
- **Training tests** (`test_training.py`): Tests that model training, evaluation, and tuning produce valid results
- **Data quality** (`data_quality.py`): Uses Great Expectations to validate data ranges and types

**Important files:** `tests/test_features.py`, `tests/test_inference.py`, `tests/test_training.py`, `tests/data_quality.py`

**How to explain in an interview:**
> "I wrote a comprehensive test suite covering unit tests for feature engineering, integration tests for the full inference pipeline, and training validation tests. These run in CI/CD — no code deploys to production unless all tests pass."

---

## 5. Important Concepts Used

### Machine Learning (XGBoost)

**Simple explanation:** XGBoost is a popular algorithm that builds many small decision trees, where each new tree tries to fix the mistakes of the previous ones. It's especially good with tabular data (spreadsheets).

**Where it's used:** `src/training_pipeline/train.py` and `tune.py`

---

### APIs (FastAPI)

**Simple explanation:** An API is like a waiter in a restaurant — you (the client) send a request ("I want predictions for this data"), and the API sends back the response (the predictions). FastAPI is a modern Python framework that makes building APIs fast and easy.

**Where it's used:** `src/api/main.py`

---

### State Management (Artifact Synchronization)

**Simple explanation:** The system needs three files to work: the model, the frequency encoder, and the target encoder. These three are always kept together (like a matching set). They're stored in AWS S3, and every container downloads them on startup. This ensures everyone is using the same version.

**Where it's used:** `src/api/main.py` (download on startup), `scripts/provision_aws.py` (upload after training)

---

### Database / Storage (AWS S3)

**Simple explanation:** Instead of a traditional database, this project uses **AWS S3** (Simple Storage Service) — think of it as a cloud hard drive. It stores the model files, processed data, and encoders. S3 is the "single source of truth" — whatever is in S3 is what production uses.

**Where it's used:** Throughout `scripts/` and `src/api/main.py`

---

### Routing (AWS ALB Path-Based Routing)

**Simple explanation:** The Application Load Balancer looks at the URL path and sends requests to the right service:
- `/predict` and `/health` → go to the **FastAPI API** container
- `/dashboard` → goes to the **Streamlit** container

This way, both services share one public URL.

**Where it's used:** `scripts/setup_infra.py`

---

### Components / Modules (Pipeline Architecture)

**Simple explanation:** The project is built as **separate, independent modules** (pipelines):
- Feature Pipeline: Only handles data cleaning and feature creation
- Training Pipeline: Only handles model training
- Inference Pipeline: Only handles making predictions
- API: Only handles HTTP requests

Each module does one thing well and doesn't know about the internals of other modules. This makes the code easier to maintain and test.

---

### Caching

**Simple explanation:** Both the API and the dashboard **cache** (save locally) files downloaded from S3. If the file already exists on disk, it doesn't download again. This saves time and network bandwidth.

**Where it's used:** `load_from_s3()` in `src/api/main.py` and `app.py`. Also, `@st.cache_data` in `app.py` caches loaded data in memory.

---

### Security

**Simple explanation:**
- **No hardcoded passwords**: AWS credentials are never written in code. They come from environment variables or IAM Roles.
- **IAM Task Roles**: Instead of giving containers a username/password for AWS, they get a "role" that only lets them access specific resources (like one S3 bucket).
- **Private networking**: The API runs in a private network. Only the Load Balancer is exposed to the internet.
- **Secrets injection**: Production secrets are stored in GitHub Secrets and injected during deployment.

**Where it's used:** `scripts/setup_iam.py`, `.github/workflows/ci-cd.yml`, `Dockerfile`, `Dockerfile.streamlit`

---

## 6. How I Explain This Project in an Interview

### Short Version (30 seconds)

> "I built a production-grade housing price prediction system using XGBoost. It's not just a model — it's a complete ML pipeline with data cleaning, feature engineering, automated hyperparameter tuning, and a serving layer deployed on AWS. The system achieves a 0.96 R² score and includes a Streamlit dashboard for exploring predictions. Everything is containerized with Docker and deployed through a CI/CD pipeline using GitHub Actions and AWS ECS Fargate."

### Medium Version (1–2 minutes)

> "This project predicts US housing prices end-to-end. I started with raw housing data and built a feature pipeline that handles city name normalization, GPS coordinate merging, and statistical encodings like target encoding and frequency encoding.
>
> For the model, I used XGBoost and optimized it with Optuna — a Bayesian hyperparameter tuning framework — which gave me a 0.96 R² on the evaluation set. All experiments are tracked in MLflow.
>
> The key engineering challenge was preventing training-serving skew. I built a shared inference pipeline that applies the exact same transformations as training. It also has a schema alignment layer that handles missing or extra features, and a numeric enforcement layer that prevents the 'object dtype' crash that XGBoost is known for.
>
> The system is deployed on AWS using ECS Fargate with two separate containers — one for the FastAPI backend and one for the Streamlit dashboard. An Application Load Balancer handles path-based routing between them. The whole deployment is automated through GitHub Actions CI/CD.
>
> I also implemented a time-series-aware data split to avoid leaking future price trends into training, which is critical for financial data."

### Detailed Version (3+ minutes)

> *(Use the medium version, then add any of these deeper details depending on the interviewer's interest):*
>
> **On data leakage:** "I used strict chronological splitting — train on pre-2020, eval on 2020–2021, holdout on 2022–2023. Housing prices are time-dependent; a random split would leak future market booms into past data, giving misleading accuracy."
>
> **On feature engineering:** "Instead of one-hot encoding 1000+ cities, I used target encoding — mapping each city to its average price, fitted only on training data. I also merged GPS coordinates so the model can learn spatial price gradients, and added frequency encoding for zipcodes as a proxy for market density."
>
> **On production resilience:** "During deployment, I hit a real-world bug where FastAPI's JSON parser would sometimes infer numeric columns as 'object' type, causing XGBoost to crash. I fixed this with a numeric enforcement layer that coerces all features to floats before prediction."
>
> **On infrastructure:** "I chose Fargate over EC2 because it's serverless — no server management, and it scales on demand. The ALB provides a single entry point with path-based routing, and IAM Task Roles eliminate the need for hardcoded AWS keys."

---

## 7. Common Interview Questions About This Project

### Q: What does your project do?
**A:** It predicts US housing prices using machine learning. It's a full pipeline — from raw data to a deployed web service with a dashboard.

### Q: Why did you choose XGBoost?
**A:** XGBoost is excellent for tabular/structured data. It handles non-linear relationships well, is fast to train, and has many tunable parameters. It consistently outperformed linear models in my experiments.

### Q: How did you prevent data leakage?
**A:** Two ways: (1) I used chronological data splitting instead of random — train on past data, test on future data. (2) I fit all encoders (target encoding, frequency encoding) only on the training set and applied them to eval/holdout.

### Q: What is target encoding and why did you use it?
**A:** Target encoding replaces a category (like a city name) with the average of the target variable (price) for that category. I used it because one-hot encoding 1000+ cities would create a very sparse, high-dimensional dataset that's hard for any model to learn from.

### Q: What was the hardest bug you faced?
**A:** During deployment, XGBoost started crashing on valid JSON payloads. The issue was that FastAPI's JSON parser sometimes inferred numeric columns as 'object' type. I fixed it by adding a numeric enforcement layer (`pd.to_numeric`) in the inference pipeline.

### Q: How do you ensure the model in production uses the same logic as training?
**A:** The inference pipeline imports the exact same functions from the feature pipeline that training uses. Both paths go through `clean_and_merge()`, `add_date_features()`, and the same encoders. I also added an integration test that runs the full raw-to-prediction flow.

### Q: Why two Docker containers?
**A:** Separation of concerns. The API container handles prediction logic (backend), and the Streamlit container handles the UI (frontend). They can be scaled independently — if predictions are slow, I can add more API containers without touching the dashboard.

### Q: How does your CI/CD work?
**A:** Push to `main` triggers GitHub Actions, which builds Docker images, pushes to ECR, and updates ECS services. If tests fail, deployment stops. This ensures broken code never reaches production.

### Q: What are the model's limitations?
**A:** It lacks exogenous data like mortgage interest rates or GDP. It operates on monthly aggregated data, so it can't capture daily price changes. At extreme scale (10M+ rows), it would need distributed processing instead of in-memory Pandas.

### Q: What metrics do you use?
**A:** MAE (average dollar error), RMSE (penalizes big errors more), R² (overall model accuracy — 0.96 on eval), and Average % Error (8.04% on holdout — most practical for business use).

---

## 8. Project Improvements

### What could be improved later

| Improvement | Why |
|-------------|-----|
| Add exogenous features (interest rates, GDP, inflation) | The model doesn't account for economic shifts outside the housing market |
| Add data drift detection (Evidently AI) | Automatically detect when incoming data looks different from training data |
| Add a proper monitoring dashboard | Track API latency, prediction distributions, and error rates in real-time |
| Add A/B testing for models | Deploy new models to a small percentage of traffic before rolling out fully |
| Add user authentication | Currently anyone can access the dashboard and API |

### Scaling ideas

| Current State | Scaled Version |
|---------------|---------------|
| Single CSV files | Distributed feature store (Feast) |
| Manual pipeline execution | Airflow orchestration (scheduled runs) |
| Single ECS task per service | Auto-scaling based on request volume |
| In-memory Pandas processing | Spark or Polars for large datasets |
| Rolling deployment | Canary deployments (5% traffic → 100%) |

### Performance improvements

| Area | Improvement |
|------|-------------|
| API response time | Pre-compute predictions for common queries and cache them |
| Model loading | Use a model registry with versioned artifacts instead of downloading on every restart |
| Data processing | Use Polars (already in dependencies) instead of Pandas for faster data processing |
| Container startup | Pre-bake model artifacts into the Docker image to avoid S3 download delay |
| Dashboard | Add Redis caching for API responses to avoid redundant prediction calls |

---

## Quick Reference: Key Technologies Used

| Technology | What it does | Where it's used |
|-----------|-------------|----------------|
| **Python 3.11** | Programming language | Everywhere |
| **XGBoost** | ML model for predictions | `src/training_pipeline/` |
| **FastAPI** | Web API framework | `src/api/main.py` |
| **Streamlit** | Dashboard/frontend | `app.py` |
| **Optuna** | Automatic hyperparameter tuning | `src/training_pipeline/tune.py` |
| **MLflow** | Experiment tracking | `src/training_pipeline/tune.py` |
| **Pandas** | Data processing | Throughout `src/` |
| **Scikit-learn** | ML utilities (metrics, encoders) | Training and evaluation |
| **Boto3** | AWS SDK for Python | API, dashboard, scripts |
| **Docker** | Containerization | `Dockerfile`, `Dockerfile.streamlit` |
| **GitHub Actions** | CI/CD automation | `.github/workflows/ci-cd.yml` |
| **AWS ECS Fargate** | Serverless container hosting | `scripts/setup_infra.py` |
| **AWS S3** | Cloud storage for artifacts | Throughout |
| **AWS ECR** | Container image registry | CI/CD pipeline |
| **AWS ALB** | Load balancing & routing | `scripts/setup_infra.py` |
| **Great Expectations** | Data quality validation | `tests/data_quality.py` |
| **Pytest** | Testing framework | `tests/` |
| **uv** | Fast Python package manager | `pyproject.toml` |
