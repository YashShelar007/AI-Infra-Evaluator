import argparse

def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark ResNet50 inference on AWS EC2"
    )
    p.add_argument(
        "--instances", nargs="+",
        default=["t3.medium", "g4dn.xlarge"],
        help="EC2 instance types to test"
    )
    p.add_argument(
        "--runs", type=int, default=100,
        help="Number of inference runs per instance"
    )
    return p.parse_args()

def main():
    args = parse_args()
    print(f"Instances: {args.instances}")
    print(f"Runs per instance: {args.runs}")

    # For each instance: launch, wait, benchmark, then terminate
    results = []
    for itype in args.instances:
        iid = launch_instance(itype)
        try:
            wait_for_instance(iid)
            host = get_instance_ip(iid)
            wait_for_service(host)
            # Measure latency
            avg_lat = run_inference(host, args.runs)
            # Compute cost
            cost = cost_per_inference(itype, avg_lat, args.runs)

            results.append({"instance": itype, "latency": avg_lat, "cost": cost})
        finally:
            terminate_instance(iid)
    # TODO: plotting

def launch_instance(instance_type):
    import boto3
    ec2 = boto3.client("ec2")
    resp = ec2.run_instances(
        ImageId="ami-0b3ceb28d1a07fa60",  
        InstanceType=instance_type,
        MinCount=1, MaxCount=1,
        UserData=open("user_data.sh").read(),
        TagSpecifications=[{
            "ResourceType": "instance",
            "Tags": [{"Key": "Name", "Value": "ai-evaluator"}]
        }]
    )
    return resp["Instances"][0]["InstanceId"]

def terminate_instance(iid):
    import boto3
    boto3.client("ec2").terminate_instances(InstanceIds=[iid])

def wait_for_instance(iid, region="us-east-1"):
    """
    Wait until the EC2 instance status is 'running' and both system and instance checks pass.
    """
    import boto3
    ec2 = boto3.client("ec2", region_name=region)
    print(f"Waiting for instance {iid} to enter 'running' state...")
    waiter = ec2.get_waiter("instance_running")
    waiter.wait(InstanceIds=[iid])
    print("Instance is running.")

def get_instance_ip(iid, region="us-east-1"):
    """
    Fetch the public DNS name (or IP) of the launched instance.
    """
    import boto3
    ec2 = boto3.resource("ec2", region_name=region)
    instance = ec2.Instance(iid)
    instance.load()
    return instance.public_dns_name or instance.public_ip_address

def wait_for_service(host, port=8080, timeout=300):
    """
    Poll the TorchServe ping endpoint until HTTP 200 or timeout.
    """
    import time, requests
    url = f"http://{host}:{port}/ping"
    start = time.time()
    print(f"Waiting for TorchServe at {url} ...")
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                print("TorchServe is ready.")
                return
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError(f"Service did not respond within {timeout}s")

# On-demand hourly pricing (USD/hr). Update as needed from AWS pricing pages.
PRICES = {
    "t3.medium": 0.0416,
    "g4dn.xlarge": 0.526,
    # add more if you’ll test additional instance types
}

def cost_per_inference(instance_type: str, avg_latency_s: float, runs: int) -> float:
    """
    Compute cost per inference:
      (hourly_rate USD/hour) * (total_seconds / 3600) / runs
    """
    rate = PRICES.get(instance_type)
    if rate is None:
        raise KeyError(f"No pricing info for {instance_type}")
    total_time = avg_latency_s * runs
    return rate * (total_time / 3600.0) / runs

def run_inference(host: str, runs: int = 100) -> float:
    """
    Fire `runs` HTTP POSTs to the TorchServe ResNet50 endpoint,
    return the average latency in seconds.
    """
    import requests, time
    url = f"http://{host}:8080/predictions/resnet50"
    img_path = "sample.png"
    with open(img_path, "rb") as f:
        data = f.read()

    latencies = []
    for i in range(runs):
        start = time.time()
        r = requests.post(url, files={"data": data})
        r.raise_for_status()
        latencies.append(time.time() - start)

    avg_latency = sum(latencies) / len(latencies)
    print(f"Avg latency on {host}: {avg_latency:.4f}s over {runs} runs")
    return avg_latency


if __name__ == "__main__":
    main()