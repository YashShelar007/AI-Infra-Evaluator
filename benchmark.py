import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--instances", nargs="+", default=["t3.medium","g4dn.xlarge"])
    p.add_argument("--runs", type=int, default=100)
    return p.parse_args()

if __name__=="__main__":
    args = parse_args()
    print("Instances:", args.instances, "Runs:", args.runs)
