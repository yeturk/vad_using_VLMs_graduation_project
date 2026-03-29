import dashscope
from dashscope import MultiModalConversation
import os
import json
from dotenv import load_dotenv

load_dotenv(override=True)


def _normalize_api_key(raw_key: str | None) -> str:
    if not raw_key:
        return ""
    # Handle accidental quotes/spaces in .env and ensure expected prefix.
    key = raw_key.strip().strip('"').strip("'")
    if key and not key.startswith("sk-"):
        key = f"sk-{key}"
    return key


api_key = _normalize_api_key(os.getenv("DASHSCOPE_API_KEY"))
if not api_key:
    raise RuntimeError("DASHSCOPE_API_KEY is missing. Check your .env file.")

dashscope.api_key = api_key
dashscope.base_http_api_url = os.getenv(
    "DASHSCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com/api/v1"
).strip()
print("Using DashScope endpoint:", dashscope.base_http_api_url)
print("API key suffix:", f"***{api_key[-4:]}")


response = MultiModalConversation.call(
    model='qwen-vl-plus',
    messages=[
        {
            "role": "user",
            "content": [
                {"image": "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"},
                {"text": "What do you see in this image?"}
            ]
        }
    ]
)

print("\n" + "=" * 80)
print("RESPONSE STATUS")
print("=" * 80)
print(f"Status Code: {response.get('status_code')}")
print(f"Request ID: {response.get('request_id')}")
print(f"Message: {response.get('message', 'Success')}")

print("\n" + "=" * 80)
print("MODEL RESPONSE")
print("=" * 80)

if response.get("status_code") == 200:
    output = response.get("output", {})
    choices = output.get("choices", [])
    if choices:
        content = choices[0].get("message", {}).get("content", [])
        for item in content:
            if isinstance(item, dict) and "text" in item:
                print(f"\n{item['text']}")
    
    usage = response.get("usage", {})
    print("\n" + "-" * 80)
    print("USAGE STATISTICS")
    print("-" * 80)
    print(f"Input tokens:  {usage.get('input_tokens')}")
    print(f"Output tokens: {usage.get('output_tokens')}")
    print(f"Total tokens:  {usage.get('total_tokens')}")
else:
    print(f"Error: {response.get('code')}")
    print(json.dumps(response, indent=2, ensure_ascii=False))