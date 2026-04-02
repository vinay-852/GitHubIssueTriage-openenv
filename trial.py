from openai import OpenAI
import os
import json
from dotenv import load_dotenv


load_dotenv()


def main() -> None:
    base_url = os.getenv("API_BASE_URL")
    model = os.getenv("MODEL_NAME")
    api_key = os.getenv("OPENAI_API_KEY")

    print(f"API_BASE_URL={base_url!r}")
    print(f"MODEL_NAME={model!r}")
    print(f"API key present? {'yes' if api_key else 'no'}")

    if not (base_url and model and api_key):
        print("Missing configuration; aborting.")
        return

    client = OpenAI(api_key=api_key, base_url=base_url)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a connectivity test bot."},
                {"role": "user", "content": "Reply with a short confirmation that the API call worked."},
            ],
            stream=True,
            temperature=0.5,
            max_tokens=50,
        )
        print("Call succeeded. Raw response:")
        print(response)
    except Exception as exc:
        print(f"Call failed: {exc}")

if __name__ == "__main__":
    main()
