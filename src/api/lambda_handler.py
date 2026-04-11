import json
import pandas as pd

from src.xgboost.inference.infer import run_inference

def lambda_handler(event, context):

    try:
        # Parse input
        body = json.loads(event["body"])

        # Support both formats
        if isinstance(body, dict) and "data" in body:
            data = body["data"]
        else:
            data = body

        df = pd.DataFrame(data)

        # Run inference
        print("running inference")
        results = run_inference(df)

        print(pd.DataFrame(results.head()))

        return {
            "statusCode": 200,
            "body": results.to_json(orient="records")
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }