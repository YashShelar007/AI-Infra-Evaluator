# AI Infrastructure Evaluator

**Benchmark ResNet50 inference latency & cost on AWS EC2**  
This tool will:

1. Launch CPU & GPU EC2 instances
2. Install and start TorchServe with ResNet50
3. Run N inference requests and measure average latency
4. Compute cost per inference using on-demand pricing
5. Plot latency vs. cost trade-off in `results.png`

---

## 🚀 Features

- **Automated EC2 lifecycle**: launches, waits, benchmarks, and terminates instances
- **Flexible CLI**: choose any instance types & run counts
- **Cost analysis**: translates performance gains into USD/inference
- **Visualization**: bar + line chart of latency & cost

---

## 🛠 Prerequisites

- **Python** 3.9+
- **AWS credentials** with EC2 full-access (via `~/.aws/credentials` or env vars)
- **Docker** (required by AWS Deep Learning AMI)
- A **key pair** if you want to SSH (optional)

---

## 📦 Installation

```bash
# 1. Clone & enter
git clone https://github.com/YashShelar007/AI-Infra-Evaluator.git
cd AI-Infra-Evaluator

# 2. Create & activate venv
python3 -m venv venv
source venv/bin/activate

# 3. Install deps
pip install -r requirements.txt
```

````

---

## ⚙️ Usage

```bash
# Syntax:
python benchmark.py \
  --instances t3.medium g4dn.xlarge \
  --runs 100

# Example:
python benchmark.py --instances t3.medium g4dn.xlarge --runs 50
```

- `--instances`: one or more EC2 instance types
- `--runs`: number of inference requests per instance

**Output**:

- `results.png` in root: chart of **Avg Latency (s)** vs **Cost per Inference (USD)**

---

## 🔧 AWS Setup

1. In the EC2 **Quick Start** tab, select **Deep Learning AMI (Ubuntu 20.04)** or **Amazon Linux 2**.
2. Copy its **AMI ID**, and paste into `benchmark.py` under `launch_instance()`.
3. Ensure your IAM user can `RunInstances`, `TerminateInstances`, and `DescribeInstances`.

_No manual EC2 steps required beyond credentials & correct AMI ID._

---

## 📊 Sample Output

![results.png](./results.png)
_(Bars = latency; Line = cost/inference)_

---

## 🗑 Cleanup

All instances are terminated automatically. To double-check:

```bash
aws ec2 describe-instances \
  --filters "Name=tag:Project,Values=AI-Infra-Evaluator"
```

---

## 🐞 Troubleshooting

- **Timeout waiting for TorchServe**

  - Verify your AMI has Docker; check `user_data.sh` logs via SSH

- **Permission errors**

  - Ensure AWS creds (env or `~/.aws/credentials`) have EC2 rights

- **Missing `sample.jpg`**

  - Place any small JPEG in project root named `sample.jpg`

---

## 📝 License

MIT © Yash Ramesh Shelar
````
