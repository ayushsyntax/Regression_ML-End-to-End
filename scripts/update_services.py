import boto3
import os

REGION = "us-east-1"
ecs = boto3.client("ecs", region_name=REGION)

def update_service(service_name: str, task_family: str):
    """
    Trigger a rolling update for an ECS service to pick up the latest task definition revision.

    Responsibility:
        - Updates the specified service with the latest ACTIVE revision of the task family.
        - Forces a new deployment to ensure environment changes are applied immediately.
    """
    print(f"Updating service {service_name} to latest task definition {task_family}...")
    try:
        response = ecs.update_service(
            cluster="housing-cluster",
            service=service_name,
            taskDefinition=task_family,
            forceNewDeployment=True
        )
        print(f"[INFO] Successfully triggered deployment for {service_name}")
        print(f"       Active Task Definition: {response['service']['taskDefinition']}")
    except Exception as e:
        print(f"[ERROR] Failed to update {service_name}: {e}")


if __name__ == "__main__":
    update_service("housing-api", "housing-api-task")
    update_service("housing-dashboard-v2", "housing-streamlit")
