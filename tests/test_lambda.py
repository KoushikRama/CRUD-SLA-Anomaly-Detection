from src.api.lambda_handler import lambda_handler
import json

event = {
    "body": json.dumps([
        {
            "timestamp": "2025-01-01 10:00:00",
            "operation": "browse_products",  # ✅ FIXED
            "success_vol": 8000,
            "fail_vol": 300,
            "success_rt_avg": 150,
            "fail_rt_avg": 120
        }
    ])
}

response = lambda_handler(event, None)

print(response)