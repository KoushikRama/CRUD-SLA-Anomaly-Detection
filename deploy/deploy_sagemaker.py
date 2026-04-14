import boto3
import time
from deploy_settings import REGION, MODEL_NAME, ENDPOINT_CONFIG_NAME, ENDPOINT_NAME, IMAGE_URI, MODEL_DATA_URL, ROLE_ARN

# ================================
# CLIENT
# ================================
sm = boto3.client("sagemaker", region_name=REGION)

# ================================
# CLEANUP 
# ================================
def safe_delete():
    try:
        sm.delete_endpoint(EndpointName=ENDPOINT_NAME)
    except:
        pass

    try:
        sm.delete_endpoint_config(EndpointConfigName=ENDPOINT_CONFIG_NAME)
    except:
        pass

    try:
        sm.delete_model(ModelName=MODEL_NAME)
    except:
        pass

safe_delete()

# ================================
# 1. CREATE MODEL
# ================================
print("Creating model...")

sm.create_model(
    ModelName=MODEL_NAME,
    ExecutionRoleArn=ROLE_ARN,
    PrimaryContainer={
        "Image": IMAGE_URI,
        "ModelDataUrl": MODEL_DATA_URL,
        "Environment": {
            "SAGEMAKER_PROGRAM": "inference.py",
            "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
            "SAGEMAKER_REQUIREMENTS": "requirements.txt"
        }
    }
)

print("Model created")

# ================================
# 2. CREATE ENDPOINT CONFIG
# ================================
print("Creating endpoint config...")

sm.create_endpoint_config(
    EndpointConfigName=ENDPOINT_CONFIG_NAME,
    ProductionVariants=[
        {
            "VariantName": "AllTraffic",
            "ModelName": MODEL_NAME,
            "ServerlessConfig": {
                "MemorySizeInMB": 2048,
                "MaxConcurrency": 5
            }
        }
    ]
)

print("Endpoint config created")

# ================================
# 3. CREATE ENDPOINT
# ================================
print("Creating endpoint...")

sm.create_endpoint(
    EndpointName=ENDPOINT_NAME,
    EndpointConfigName=ENDPOINT_CONFIG_NAME
)

print("Endpoint creation started")

# ================================
# 4. WAIT
# ================================
print("Waiting for endpoint...")

while True:
    response = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
    status = response["EndpointStatus"]

    print("Status:", status)

    if status == "InService":
        print("Endpoint is ready 🚀")
        break
    elif status == "Failed":
        raise Exception("Endpoint creation failed ❌")

    time.sleep(30)