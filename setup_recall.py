"""
One-time setup script for Recall.ai.
Run this ONCE after deploying main.py to register your webhook URL.

Usage:
    python setup_recall.py --webhook-url https://your-deployed-app.com/webhook/recall
"""

import httpx
import os
import argparse
from dotenv import load_dotenv

load_dotenv()

RECALL_API_KEY = os.getenv("RECALL_API_KEY")
RECALL_BASE_URL = "https://ap-northeast-1.recall.ai/api/v1"
RECALL_WEBHOOK_URL = "https://ap-northeast-1.recall.ai/api/v2"

HEADERS = {
    "Authorization": f"Token {RECALL_API_KEY}",
    "Content-Type": "application/json"
}


def register_webhook(webhook_url: str):
    """Registers your server's webhook URL with Recall.ai."""
    print(f"Registering webhook: {webhook_url}")

    payload = {
        "url": webhook_url,
        "events": [
            "bot.status_change",
            "recording.done"
        ]
    }

    response = httpx.post(
        f"{RECALL_WEBHOOK_URL}/webhook/",
        json=payload,
        headers=HEADERS
    )

    if response.status_code in (200, 201):
        print("Webhook registered successfully.")
        print(response.json())
    else:
        print(f"Failed to register webhook: {response.status_code}")
        print(response.text)


def verify_connection():
    """Checks that the Recall.ai API key is valid."""
    response = httpx.get(
        f"{RECALL_BASE_URL}/bot/",
        headers=HEADERS
    )

    if response.status_code == 200:
        print("Recall.ai connection verified.")
    else:
        print(f"Connection failed: {response.status_code} — check your RECALL_API_KEY in .env")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--webhook-url", required=True, help="Your deployed server URL e.g. https://yourapp.up.railway.app/webhook/recall")
    args = parser.parse_args()

    verify_connection()
    register_webhook(args.webhook_url)
