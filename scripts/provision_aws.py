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
    print(f"Creating S3 Bucket: {S3_BUCKET_NAME} in {REGION}...")
    print(f"DEBUG: REGION='{REGION}'")
    try:
        if REGION == "us-east-1":
            s3_client.create_bucket(Bucket=S3_BUCKET_NAME)
        else:
            s3_client.create_bucket(
                Bucket=S3_BUCKET_NAME,
                CreateBucketConfiguration={"LocationConstraint": REGION}
            )
        print("✅ Bucket created (or already exists).")
    except ClientError as e:
        if e.response['Error']['Code'] == 'BucketAlreadyOwnedByYou':
            print("✅ Bucket already exists and is owned by you.")
        else:
            print(f"❌ Failed to create bucket: {e}")

def create_ecr_repos():
    repos = ["housing-api", "housing-dashboard"]
    for repo in repos:
        print(f"Creating ECR Repo: '{repo}'...")
        try:
            ecr_client.create_repository(repositoryName=repo)
            print(f"✅ Repository {repo} created.")
        except Exception as e:
            print(f"❌ Failed to create repo {repo}: {e}")
            import traceback
            traceback.print_exc()

def upload_artifacts():
    print("Uploading artifacts to S3...")

    # Define files to upload (Local Path -> S3 Key)
    artifacts = {
        "models/xgb_best_model.pkl": "models/xgb_best_model.pkl",
        # We need to make sure these exist!
        "data/processed/feature_engineered_train.csv": "processed/feature_engineered_train.csv",
        "data/processed/feature_engineered_eval.csv": "processed/feature_engineered_eval.csv",
        "data/processed/feature_engineered_holdout.csv": "processed/feature_engineered_holdout.csv",
        "data/processed/cleaning_holdout.csv": "processed/cleaning_holdout.csv"
    }

    for local_path, s3_key in artifacts.items():
        if os.path.exists(local_path):
            print(f"⬆️ Uploading {local_path} -> s3://{S3_BUCKET_NAME}/{s3_key}")
            s3_client.upload_file(local_path, S3_BUCKET_NAME, s3_key)
        else:
            print(f"⚠️ File not found (skipping): {local_path}")

if __name__ == "__main__":
    create_bucket()
    create_ecr_repos()
    upload_artifacts()
