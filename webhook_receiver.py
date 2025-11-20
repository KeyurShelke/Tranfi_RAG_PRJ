# webhook_receiver.py
"""
Simple webhook receiver for Part 2.
Accepts ANY JSON payload and prints it.
"""

from fastapi import FastAPI, Request
import uvicorn
from datetime import datetime

app = FastAPI()

@app.post("/webhook")
async def webhook(request: Request):
    payload = await request.json()
    print("\n=== WEBHOOK RECEIVED ===")
    print("Timestamp:", datetime.now())
    print("Payload:", payload)
    print("========================\n")
    return {"status": "ok"}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    print(f"\n🚀 Webhook Receiver running on http://localhost:{args.port}/webhook\n")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
