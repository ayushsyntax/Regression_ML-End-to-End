import boto3
import json
import os
from botocore.exceptions import ClientError

AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1").strip()
iam = boto3.client("iam", region_name=AWS_REGION)

def create_execution_role():
    role_name = "ecsTaskExecutionRole"
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "ecs-tasks.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }
        ]
    }

    try:
        iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="ECS Task Execution Role"
        )
        print(f"[INFO] Created role: {role_name}")
    except ClientError as e:
        if e.response['Error']['Code'] == 'EntityAlreadyExists':
            print(f"[INFO] Role {role_name} already exists.")
        else:
            print(f"[ERROR] Failed to create role {role_name}: {e}")
            return None

    # Attach policy
    try:
        policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
        iam.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)
        print(f"[INFO] Attached {policy_arn} to {role_name}")
    except Exception as e:
        print(f"[ERROR] Failed to attach policy: {e}")

    return iam.get_role(RoleName=role_name)['Role']['Arn']

def create_task_role():
    role_name = "ecs_s3_access"
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "ecs-tasks.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }
        ]
    }

    try:
        iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="ECS Task Role for S3 Access"
        )
        print(f"[INFO] Created role: {role_name}")
    except ClientError as e:
        if e.response['Error']['Code'] == 'EntityAlreadyExists':
            print(f"[INFO] Role {role_name} already exists.")
        else:
            print(f"[ERROR] Failed to create role {role_name}: {e}")
            return None

    # Attach S3 Full Access
    try:
        policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
        iam.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)
        print(f"[INFO] Attached {policy_arn} to {role_name}")
    except Exception as e:
        print(f"[ERROR] Failed to attach policy: {e}")

    return iam.get_role(RoleName=role_name)['Role']['Arn']

if __name__ == "__main__":
    exec_arn = create_execution_role()
    task_arn = create_task_role()
    print(f"\nExecution Role ARN: {exec_arn}")
    print(f"Task Role ARN: {task_arn}")
