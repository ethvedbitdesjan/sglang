import json
import requests

url = "http://127.0.0.1:30000"
json_data = {
    "text": [
        """### Instruction:
List all Canadian provinces and territories in alphabetical order.
### Response:""",
        """### Instruction:
List all Canadian provinces and territories in alphabetical order.
### Response:""",
"""### Instruction:
List all Canadian provinces and territories in alphabetical order.
### Response:""",
        """### Instruction:
List all Canadian provinces and territories in alphabetical order.
### Response:"""
    ],
    "sampling_params": {"max_new_tokens": 128},
    "lora_path": [
        "lora1",
        "lora2",
        "lora1",
        "lora2"
    ],
}
response = requests.post(
    url + "/generate",
    json=json_data,
)
print(json.dumps(response.json()))