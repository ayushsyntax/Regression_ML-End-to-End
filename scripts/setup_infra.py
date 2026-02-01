import boto3
import json
import os
import time

AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1").strip()
ec2 = boto3.client("ec2", region_name=AWS_REGION)
elbv2 = boto3.client("elbv2", region_name=AWS_REGION)
ecs = boto3.client("ecs", region_name=AWS_REGION)
logs = boto3.client("logs", region_name=AWS_REGION)

def get_default_vpc():
    vpcs = ec2.describe_vpcs(Filters=[{'Name': 'isDefault', 'Values': ['true']}])['Vpcs']
    if not vpcs:
        raise Exception("No default VPC found")
    return vpcs[0]['VpcId']

def get_subnets(vpc_id):
    subnets = ec2.describe_subnets(Filters=[{'Name': 'vpc-id', 'Values': [vpc_id]}])['Subnets']
    return [s['SubnetId'] for s in subnets]

def create_security_group(vpc_id):
    sg_name = "housing-sg"
    try:
        sgs = ec2.describe_security_groups(Filters=[{'Name': 'group-name', 'Values': [sg_name]}])['SecurityGroups']
        if sgs:
            print(f"[INFO] Security Group {sg_name} exists: {sgs[0]['GroupId']}")
            return sgs[0]['GroupId']
    except:
        pass

    sg = ec2.create_security_group(GroupName=sg_name, Description="Housing App SG", VpcId=vpc_id)
    sg_id = sg['GroupId']

    # Inbound rules
    ports = [80, 8000, 8501]
    ip_perms = []
    for p in ports:
        ip_perms.append({
            'IpProtocol': 'tcp',
            'FromPort': p,
            'ToPort': p,
            'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
        })

    ec2.authorize_security_group_ingress(GroupId=sg_id, IpPermissions=ip_perms)
    print(f"[INFO] Created Security Group {sg_name}: {sg_id}")
    return sg_id

def create_alb(subnets, sg_id):
    alb_name = "housing-alb"
    try:
        albs = elbv2.describe_load_balancers(Names=[alb_name])['LoadBalancers']
        print(f"[INFO] ALB {alb_name} exists: {albs[0]['DNSName']}")
        return albs[0]['LoadBalancerArn'], albs[0]['DNSName']
    except:
        pass

    # ALB requires at least two subnets in different AZs. Ensure that.
    # If subnets are in same AZ, this might fail. Assuming default subnets cover multiple AZs.

    alb = elbv2.create_load_balancer(
        Name=alb_name,
        Subnets=subnets,
        SecurityGroups=[sg_id],
        Scheme='internet-facing',
        Type='application',
        IpAddressType='ipv4'
    )
    arn = alb['LoadBalancers'][0]['LoadBalancerArn']
    dns = alb['LoadBalancers'][0]['DNSName']
    print(f"[INFO] Created ALB {alb_name}: {dns}")
    return arn, dns

def create_target_group(vpc_id, name, port, health_path):
    try:
        tgs = elbv2.describe_target_groups(Names=[name])['TargetGroups']
        print(f"[INFO] Target Group {name} exists")
        return tgs[0]['TargetGroupArn']
    except:
        pass

    tg = elbv2.create_target_group(
        Name=name,
        Protocol='HTTP',
        Port=port,
        VpcId=vpc_id,
        HealthCheckProtocol='HTTP',
        HealthCheckPath=health_path,
        TargetType='ip'
    )
    return tg['TargetGroups'][0]['TargetGroupArn']

def create_listener(alb_arn, api_tg_arn, dash_tg_arn):
    # Check if listener exists
    try:
        listeners = elbv2.describe_listeners(LoadBalancerArn=alb_arn)
        for l in listeners['Listeners']:
            if l['Port'] == 80:
                print("[INFO] Listener on port 80 exists")
                return l['ListenerArn']
    except:
        pass

    # Create Listener with default action -> Dashboard
    listener = elbv2.create_listener(
        LoadBalancerArn=alb_arn,
        Protocol='HTTP',
        Port=80,
        DefaultActions=[{'Type': 'forward', 'TargetGroupArn': dash_tg_arn}] # Default to dashboard
    )
    listener_arn = listener['Listeners'][0]['ListenerArn']

    # Rules
    # /health -> API
    elbv2.create_rule(
        ListenerArn=listener_arn,
        Conditions=[{'Field': 'path-pattern', 'Values': ['/health*']}],
        Priority=10,
        Actions=[{'Type': 'forward', 'TargetGroupArn': api_tg_arn}]
    )
    # /predict -> API
    elbv2.create_rule(
        ListenerArn=listener_arn,
        Conditions=[{'Field': 'path-pattern', 'Values': ['/predict*']}],
        Priority=20,
        Actions=[{'Type': 'forward', 'TargetGroupArn': api_tg_arn}]
    )
    # /dashboard -> Dashboard
    elbv2.create_rule(
        ListenerArn=listener_arn,
        Conditions=[{'Field': 'path-pattern', 'Values': ['/dashboard*']}],
        Priority=30,
        Actions=[{'Type': 'forward', 'TargetGroupArn': dash_tg_arn}]
    )
    print("[INFO] Created Listeners and Rules")

def create_log_group(name):
    try:
        logs.create_log_group(LogGroupName=name)
        print(f"[INFO] Created Log Group {name}")
    except:
        pass

def register_task_def(filename, alb_dns=None):
    with open(filename, 'r') as f:
        data = json.load(f)

    # Dynamic Account ID replacement
    sts = boto3.client("sts")
    account_id = sts.get_caller_identity()["Account"]

    # Replace Account ID in Roles
    if "executionRoleArn" in data:
        data["executionRoleArn"] = data["executionRoleArn"].replace("005905649662", account_id)
    if "taskRoleArn" in data:
        data["taskRoleArn"] = data["taskRoleArn"].replace("005905649662", account_id)

    # Replace Account ID in Image URI
    for container in data['containerDefinitions']:
        if "image" in container:
             container["image"] = container["image"].replace("005905649662", account_id)

    # If explicit subsitution needed
    if alb_dns and "housing-streamlit" in filename:
        for container in data['containerDefinitions']:
            for env in container.get('environment', []):
                if env['name'] == 'API_URL':
                    env['value'] = f"http://{alb_dns}/predict"
                    print(f"[INFO] Updated API_URL to http://{alb_dns}/predict")

    # Remove unsupported keys if any (like tags if they are strict)
    # Register
    response = ecs.register_task_definition(
        family=data['family'],
        networkMode=data['networkMode'],
        executionRoleArn=data['executionRoleArn'],
        taskRoleArn=data['taskRoleArn'],
        requiresCompatibilities=data['requiresCompatibilities'],
        cpu=data['cpu'],
        memory=data['memory'],
        containerDefinitions=data['containerDefinitions']
    )
    print(f"[INFO] Registered Task Definition: {data['family']}")
    return data['family']

def create_service(cluster, service_name, task_def, tg_arn, subnets, sg_id, port):
    try:
        ecs.create_service(
            cluster=cluster,
            serviceName=service_name,
            taskDefinition=task_def,
            desiredCount=1,
            launchType='FARGATE',
            networkConfiguration={
                'awsvpcConfiguration': {
                    'subnets': subnets,
                    'securityGroups': [sg_id],
                    'assignPublicIp': 'ENABLED'
                }
            },
            loadBalancers=[
                {
                    'targetGroupArn': tg_arn,
                    'containerName': task_def.replace("-task", ""), # Assuming container name matches
                    'containerPort': port
                }
            ]
        )
        print(f"[INFO] Created Service {service_name}")
    except Exception as e:
        if "Creation of service was not idempotent" in str(e):
             print(f"[INFO] Service {service_name} already exists (idempotency check)")
        else:
            print(f"[ERROR] Failed/Skipped Service {service_name}: {e}")

if __name__ == "__main__":
    vpc_id = get_default_vpc()
    print(f"VPC: {vpc_id}")
    subnets = get_subnets(vpc_id)
    print(f"Subnets: {len(subnets)}")

    sg_id = create_security_group(vpc_id)

    alb_arn, alb_dns = create_alb(subnets, sg_id)

    api_tg = create_target_group(vpc_id, "housing-api-tg", 8000, "/health")
    dash_tg = create_target_group(vpc_id, "housing-dashboard-tg", 8501, "/dashboard")

    create_listener(alb_arn, api_tg, dash_tg)

    try:
        ecs.create_cluster(clusterName="housing-cluster")
        print("[INFO] Cluster 'housing-cluster' created/exists")
    except:
        pass

    create_log_group("/ecs/housing-api")
    create_log_group("/ecs/housing-dashboard") # Should match what is in task def

    # Register Tasks
    print("Registering Task Definitions...")
    api_task_family = register_task_def("housing-api-task-def.json")
    dash_task_family = register_task_def("streamlit-task-def.json", alb_dns=alb_dns)

    # Create Services
    print("Creating Services...")
    # API Service
    create_service(
        cluster="housing-cluster",
        service_name="housing-api",
        task_def=api_task_family,
        tg_arn=api_tg,
        subnets=subnets,
        sg_id=sg_id,
        port=8000
    )

    # Dashboard Service
    create_service(
        cluster="housing-cluster",
        service_name="housing-dashboard",
        task_def=dash_task_family,
        tg_arn=dash_tg,
        subnets=subnets,
        sg_id=sg_id,
        port=8501
    )

    print("\n[INFO] Infrastructure Setup Complete!")
    print(f"Application Load Balancer URL: http://{alb_dns}")
    print(f"   - API Health: http://{alb_dns}/health")
    print(f"   - Dashboard: http://{alb_dns}/dashboard")
