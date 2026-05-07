"""
Run this once to generate a secure access token and save it to .env.

    python secret.py

The token protects your backend — share it only with people you want
to allow connecting to your server. To rotate it, just run this again.
"""
import secrets
import re
from pathlib import Path

token = secrets.token_urlsafe(48)

env_path = Path(__file__).parent / ".env"

if env_path.exists():
    content = env_path.read_text()
    if re.search(r"^ACCESS_TOKEN=", content, re.MULTILINE):
        # Replace existing (commented-out or active) ACCESS_TOKEN line
        content = re.sub(r"^#?\s*ACCESS_TOKEN=.*$", f"ACCESS_TOKEN={token}", content, flags=re.MULTILINE)
    else:
        content = content.rstrip("\n") + f"\nACCESS_TOKEN={token}\n"
    env_path.write_text(content)
    print(f"ACCESS_TOKEN updated in .env")
else:
    env_path.write_text(f"ACCESS_TOKEN={token}\n")
    print(f".env created with ACCESS_TOKEN")

print(f"\nYour access token:\n\n    {token}\n")
print("Keep this secret. Enter it in the frontend settings panel to connect.")
