import os
import boto3
from pathlib import Path
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Load env vars
load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1").strip()
# Use a distinctive name based on user identity or random if needed,
# but here we use the one from .env or a default
S3_BUCKET_NAME = os.getenv("S3_BUCKET", "housing-ml-artifacts")

# Clients
s3_client = boto3.client(
    "s3",
    region_name=REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)
ecr_client = boto3.client(
    "ecr",
    region_name=REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

def create_bucket():
    """
    Initialize the primary S3 bucket for project artifacts.

    Responsibility:
        - Creates an regional S3 bucket if not already present.
        - Handles regional configuration constraints during creation.
    """
    print(f"Creating S3 Bucket: {S3_BUCKET_NAME} in {REGION}...")
    try:
        if REGION == "us-east-1":
            s3_client.create_bucket(Bucket=S3_BUCKET_NAME)
        else:
            s3_client.create_bucket(
                Bucket=S3_BUCKET_NAME,
                CreateBucketConfiguration={"LocationConstraint": REGION}
            )
        print("[INFO] Bucket initialized.")
    except ClientError as e:
        if e.response['Error']['Code'] == 'BucketAlreadyOwnedByYou':
            print("[INFO] Bucket already exists and is owned by current account.")
        else:
            print(f"[ERROR] Failed to create bucket: {e}")


def create_ecr_repos():
    """
    Create Amazon ECR repositories for container image storage.

    Responsibility:
        - Provisions 'housing-api' and 'housing-dashboard' repositories.
        - Gracefully handles existing repositories.
    """
    repos = ["housing-api", "housing-dashboard"]
    for repo in repos:
        print(f"Creating ECR Repository: '{repo}'...")
        try:
            ecr_client.create_repository(repositoryName=repo)
            print(f"[INFO] Repository {repo} created.")
        except Exception as e:
            print(f"[INFO] Repository {repo} already exists or error occurred: {e}")


def upload_artifacts():
    """
    Synchronize local data and model artifacts to the primary S3 bucket.

    Responsibility:
        - Uploads trained model weights and processed CSV datasets.
        - Preserves directory structure within the S3 namespace.
    """
    print("Uploading artifacts to S3...")
    artifacts = {
        "models/xgb_best_model.pkl": "models/xgb_best_model.pkl",
        "data/processed/feature_engineered_train.csv": "processed/feature_engineered_train.csv",
        "data/processed/feature_engineered_eval.csv": "processed/feature_engineered_eval.csv",
        "data/processed/feature_engineered_holdout.csv": "processed/feature_engineered_holdout.csv",
        "data/processed/cleaning_holdout.csv": "processed/cleaning_holdout.csv"
    }

    for local_path, s3_key in artifacts.items():
        if os.path.exists(local_path):
            print(f"[INFO] Uploading {local_path} -> s3://{S3_BUCKET_NAME}/{s3_key}")
            s3_client.upload_file(local_path, S3_BUCKET_NAME, s3_key)
        else:
            print(f"[WARNING] Skipping missing local file: {local_path}")

if __name__ == "__main__":
    create_bucket()
    create_ecr_repos()
    upload_artifacts()
