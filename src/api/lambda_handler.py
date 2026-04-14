import json
import boto3
import os

runtime = boto3.client("sagemaker-runtime")

ENDPOINT_NAME = os.environ["ENDPOINT_NAME"]

def lambda_handler(event, context):

    try:
        # Parse request body
        body = json.loads(event.get("body", "{}"))

        # Forward to SageMaker
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=json.dumps(body)
        )

        result = response["Body"].read().decode()

        return {
            "statusCode": 200,
            "body": result
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }