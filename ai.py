import os
import base64
import filetype
import json
from dotenv import load_dotenv
from anthropic import Anthropic


       
def process_image_with_claude(image_bytes: bytes) -> str:
    """Encodes raw image bytes and sends them to Claude 4.5 Haiku."""

    # Load anthropic api key from env
    load_dotenv()
    anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Guess the image encoding
    image_filetype = filetype.guess(image_bytes)
    if image_filetype is None or image_filetype.mime not in {"image/jpeg", "image/png", "image/webp"}:
        raise ValueError("Could not detect a supported image format after encoding.")

    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    # Load prompt from file
    prompt = ""
    with open("./claude_prompt.txt", "r") as f:
        prompt = f.read()
        if prompt == "":
            raise ValueError("Could not read claude prompt")

    # Get response from claude
    response = anthropic_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2000,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image_filetype.mime,
                            "data": base64_image
                        }
                    },
                    {"type": "text", "text": prompt}
                ],
            }
        ],
    )
    return response.content[0].text
