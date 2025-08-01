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
    # TODO: launch, benchmark, terminate per instance

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

if __name__ == "__main__":
    main()