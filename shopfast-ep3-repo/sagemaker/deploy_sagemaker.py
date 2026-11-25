"""
Script de despliegue automatizado del modelo ShopFast en AWS SageMaker (IE4).
1. Empaqueta models/shopfast_recommender.joblib + sagemaker/inference.py en model.tar.gz.
2. Sube el artefacto a Amazon S3.
3. Despliega un Endpoint SageMaker SKLearn en instancia CPU (ml.t2.medium / ml.m5.large).
4. Registra política de Application Auto Scaling (1 a 4 instancias según SageMakerVariantInvocationsPerInstance).
"""

from __future__ import annotations

import argparse
import os
import tarfile
from pathlib import Path

import boto3

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_JOBLIB = BASE_DIR / "models" / "shopfast_recommender.joblib"
INFERENCE_PY = BASE_DIR / "sagemaker" / "inference.py"
ARCHIVE_PATH = BASE_DIR / "models" / "model.tar.gz"


def package_model() -> Path:
    """Crea model.tar.gz compatible con el contenedor oficial SKLearn de SageMaker."""
    if not MODEL_JOBLIB.exists():
        raise FileNotFoundError(f"Ejecute primero python -m ml.train_model para generar {MODEL_JOBLIB}")

    with tarfile.open(ARCHIVE_PATH, "w:gz") as tar:
        tar.add(MODEL_JOBLIB, arcname="shopfast_recommender.joblib")
        tar.add(INFERENCE_PY, arcname="code/inference.py")

    print(f"Artefacto SageMaker empaquetado: {ARCHIVE_PATH}")
    return ARCHIVE_PATH


def deploy(
    bucket: str,
    role_arn: str,
    endpoint_name: str = "shopfast-recommender-ep3",
    instance_type: str = "ml.t2.medium",
    region: str = "us-east-1",
    dry_run: bool = False,
) -> None:
    """Sube model.tar.gz a S3, crea Endpoint SageMaker y configura Auto Scaling."""
    archive = package_model()
    s3_key = f"shopfast-ep3/models/{archive.name}"
    model_data_url = f"s3://{bucket}/{s3_key}"

    if dry_run:
        print(f"[DRY-RUN] Subiría {archive} -> {model_data_url}")
        print(f"[DRY-RUN] Desplegaría endpoint '{endpoint_name}' ({instance_type}) con Auto Scaling 1-4 en {region}")
        return

    s3 = boto3.client("s3", region_name=region)
    s3.upload_file(str(archive), bucket, s3_key)
    print(f"Modelo subido a {model_data_url}")

    # Imagen SKLearn oficial de AWS SageMaker en us-east-1
    image_uri = f"683313688378.dkr.ecr.{region}.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"
    sm = boto3.client("sagemaker", region_name=region)

    model_name = f"{endpoint_name}-model"
    config_name = f"{endpoint_name}-cfg"

    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": image_uri,
            "ModelDataUrl": model_data_url,
            "Environment": {
                "SAGEMAKER_PROGRAM": "inference.py",
                "SAGEMAKER_SUBMIT_DIRECTORY": model_data_url,
            },
        },
        ExecutionRoleArn=role_arn,
    )
    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": 1,
                "InstanceType": instance_type,
                "InitialVariantWeight": 1.0,
            }
        ],
    )
    sm.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)
    print(f"Endpoint '{endpoint_name}' en creación. Esperando estado InService...")
    sm.get_waiter("endpoint_in_service").wait(EndpointName=endpoint_name)
    print(f"Endpoint '{endpoint_name}' activo (InService).")

    # Política de Application Auto Scaling (IE4)
    aas = boto3.client("application-autoscaling", region_name=region)
    resource_id = f"endpoint/{endpoint_name}/variant/AllTraffic"
    aas.register_scalable_target(
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        MinCapacity=1,
        MaxCapacity=4,
    )
    aas.put_scaling_policy(
        PolicyName=f"{endpoint_name}-target-tracking",
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        PolicyType="TargetTrackingScaling",
        TargetTrackingScalingPolicyConfiguration={
            "TargetValue": 70.0,
            "PredefinedMetricSpecification": {
                "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance"
            },
            "ScaleInCooldown": 300,
            "ScaleOutCooldown": 60,
        },
    )
    print("Auto Scaling (1-4 instancias) configurado correctamente.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Despliegue de ShopFast en AWS SageMaker")
    parser.add_argument("--bucket", default=os.getenv("SAGEMAKER_BUCKET", "shopfast-ml-artifacts"))
    parser.add_argument("--role-arn", default=os.getenv("SAGEMAKER_ROLE_ARN", "arn:aws:iam::123456789012:role/LabRole"))
    parser.add_argument("--endpoint-name", default="shopfast-recommender-ep3")
    parser.add_argument("--instance-type", default="ml.t2.medium")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    deploy(
        bucket=args.bucket,
        role_arn=args.role_arn,
        endpoint_name=args.endpoint_name,
        instance_type=args.instance_type,
        region=args.region,
        dry_run=args.dry_run,
    )
