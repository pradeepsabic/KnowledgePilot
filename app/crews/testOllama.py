import litellm
import os

os.environ["LITELLM_LOG"] = "DEBUG"

try:
    response = litellm.completion(
        model="ollama/gemma:2b",
        messages=[{"role": "user", "content": "Say hello"}],
        api_base="http://localhost:11434",
        stream=False,
    )
    print("SUCCESS:", response.choices[0].message.content)
except Exception as e:
    print("ERROR:", e)
