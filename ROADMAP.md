# 🗺️ Deployment Roadmap: From Code to Cloud

You asked to "make ready for everything". While the code logic is now fully implemented, some steps (Data Download, AWS Account Setup) require your manual action. This roadmap guides you through them.

## Phase 1: Local Data Setup (Manual Action Required)
The project requires external datasets that cannot be committed to GitHub.
1.  **Download Data**:
    *   **Housing TS**: [Kaggle Link](https://www.kaggle.com/datasets/zillow/zecon) (Download `House_TS.csv`)
    *   **US Metros**: [SimpleMaps Link](https://simplemaps.com/data/us-metros) (Download `us-en-metros.csv`)
2.  **Place Files**:
    Place them exactly here:
    ```text
    d:/housing-regression-ml/data/raw/House_TS.csv
    d:/housing-regression-ml/data/raw/us-en-metros.csv
    ```

## Phase 2: Run Pipelines Locally
Once data is in place, run these commands in your `d:/housing-regression-ml` terminal to build the models.

```powershell
# 1. Install Dependencies
pip install -r pyproject.toml
# OR if using uv
uv sync

# 2. Run Feature Pipeline (Clean & Split Data)
python -m src.feature_pipeline.load
python -m src.feature_pipeline.preprocess
python -m src.feature_pipeline.feature_engineering

# 3. Train the Model
python -m src.training_pipeline.train

# 4. (Optional) Run Hyperparameter Tuning
python -m src.training_pipeline.tune
```
*Wait for `models/xgb_best_model.pkl` to be generated.*

## Phase 3: Verify Application
Check if the App and API work on your machine.
1.  **Start API**:
    ```powershell
    uvicorn src.api.main:app --reload
    ```
    *Open `http://127.0.0.1:8000/docs` to test.*
2.  **Start UI**:
    ```powershell
    streamlit run app.py
    ```
    *Open `http://localhost:8501` to view the dashboard.*

## Phase 4: AWS Cloud Setup (One-Time Manual Setup)
To deploy "for real", you need AWS resources.
1.  **Create AWS User**: Go to IAM -> Create User -> Attach `AdministratorAccess` (or specific permissions listed in README).
2.  **Get Credentials**: Generate Access Key & Secret Key.
3.  **Configure Local CLI**:
    ```powershell
    aws configure
    # Input your keys
    ```
4.  **Create Resources**:
    ```powershell
    # Create S3 Bucket
    aws s3 mb s3://housing-ml-app-your-unique-name
    # Create ECR Repos
    aws ecr create-repository --repository-name housing-regression-api
    aws ecr create-repository --repository-name housing-regression-ui
    ```
5.  **Upload Artifacts**:
    Upload your locally trained model/data to S3 so the cloud app can find it.
    ```powershell
    aws s3 cp models/xgb_best_model.pkl s3://housing-ml-app-your-unique-name/models/
    aws s3 cp data/processed/feature_engineered_train.csv s3://housing-ml-app-your-unique-name/processed/
    ```

## Phase 5: CI/CD Deployment
Pushing code to GitHub will trigger deployment *if* secrets are set.
1.  **GitHub Secrets**: Add `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, etc. to your Repo Settings.
2.  **Push**:
    ```powershell
    git add .
    git commit -m "Deploy to AWS"
    git push origin main
    ```

## Success! 🚀
Your application will be live on the AWS Load Balancer URL provided by the ECS Service.
