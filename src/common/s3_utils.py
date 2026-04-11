import boto3
from pathlib import Path

def upload_to_s3(local_path, bucket, key):
    s3 = boto3.client("s3")
    s3.upload_file(str(local_path), bucket, key)
    print(f"Uploaded {local_path} → s3://{bucket}/{key}")


def download_from_s3(local_path, bucket, key):
    s3 = boto3.client("s3")
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(bucket, key, str(local_path))
    print(f"Downloaded s3://{bucket}/{key} → {local_path}")