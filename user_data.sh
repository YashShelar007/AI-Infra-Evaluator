#!/bin/bash
# Install TorchServe dependencies
sudo yum update -y
sudo amazon-linux-extras install docker -y
sudo service docker start
sudo usermod -a -G docker ec2-user
# Pull and run TorchServe container
docker pull pytorch/torchserve:latest-cpu
docker run -d -p 8080:8080 \
  --name torchserve \
  pytorch/torchserve:latest-cpu \
  torchserve --start --model-store model-store --models resnet50=mar-resnet50.mar
