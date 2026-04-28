#!/bin/bash
# AWS ECS deployment script
# Usage: bash scripts/aws_deploy.sh

set -e

AWS_REGION="us-east-1"
ECR_REPOSITORY="mlops-predictive-maintenance"
ECS_CLUSTER="mlops-cluster"
ECS_SERVICE="predictive-maintenance-service"
TASK_FAMILY="predictive-maintenance-task"

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}"
IMAGE_TAG=$(git rev-parse --short HEAD)

echo "==> Logging in to ECR..."
aws ecr get-login-password --region $AWS_REGION | \
  docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

echo "==> Creating ECR repository (if not exists)..."
aws ecr describe-repositories --repository-names $ECR_REPOSITORY --region $AWS_REGION 2>/dev/null || \
  aws ecr create-repository --repository-name $ECR_REPOSITORY --region $AWS_REGION

echo "==> Building Docker image..."
docker build -f docker/Dockerfile.api -t "${ECR_URI}:${IMAGE_TAG}" -t "${ECR_URI}:latest" .

echo "==> Pushing to ECR..."
docker push "${ECR_URI}:${IMAGE_TAG}"
docker push "${ECR_URI}:latest"

echo "==> Registering ECS task definition..."
TASK_DEF=$(cat scripts/ecs_task_definition.json | \
  sed "s|IMAGE_URI|${ECR_URI}:${IMAGE_TAG}|g" | \
  sed "s|AWS_REGION|${AWS_REGION}|g" | \
  sed "s|ACCOUNT_ID|${ACCOUNT_ID}|g")

aws ecs register-task-definition \
  --family $TASK_FAMILY \
  --cli-input-json "$TASK_DEF" \
  --region $AWS_REGION

echo "==> Creating ECS cluster (if not exists)..."
aws ecs describe-clusters --clusters $ECS_CLUSTER --region $AWS_REGION 2>/dev/null || \
  aws ecs create-cluster --cluster-name $ECS_CLUSTER --region $AWS_REGION

echo "==> Deploying to ECS service..."
aws ecs update-service \
  --cluster $ECS_CLUSTER \
  --service $ECS_SERVICE \
  --task-definition $TASK_FAMILY \
  --force-new-deployment \
  --region $AWS_REGION

echo "==> Waiting for service stability..."
aws ecs wait services-stable \
  --cluster $ECS_CLUSTER \
  --services $ECS_SERVICE \
  --region $AWS_REGION

echo "==> Deployment complete! Image: ${ECR_URI}:${IMAGE_TAG}"
