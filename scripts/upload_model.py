#!/usr/bin/env python3
"""
upload_model.py — PadelLabs ML Toolchain
==========================================
Uploads a trained .mlmodel to a PRIVATE Supabase Storage bucket and generates
a signed URL (1-year expiry) for the app to download it securely.

Steps:
  1. Verifies the .mlmodel exists
  2. Uploads it to a private Supabase Storage bucket (ml-models/)
  3. Generates a signed URL valid for 1 year
  4. Prints the two Remote Config values to set in Firebase console:
       ml_stroke_model_version = <version>
       ml_stroke_model_url     = <signed-url>

Security model:
  - Bucket is PRIVATE — direct access without a valid signature returns 400
  - The signed URL embeds a time-limited token in the query string
  - No credentials are stored in the app binary
  - The signed URL is only discoverable by users who can initialise the Firebase
    SDK with the app's google-services config (i.e. your own app)
  - Renew the URL automatically each time you upload a new model version

Usage:
    python3 upload_model.py --version v2
    python3 upload_model.py --version v2 --bucket ml-models --dry-run
    python3 upload_model.py --version v2 --signed-url-expiry 31536000

Requirements:
    pip install supabase python-dotenv

Environment variables (or .env file in repo root):
    SUPABASE_URL         — your Supabase project URL
    SUPABASE_SERVICE_KEY — service-role key (needs storage write + sign)
"""

import argparse
import os
import sys
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
ML_DIR     = SCRIPT_DIR.parent
MODELS_DIR = ML_DIR / "models"

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Upload CoreML model to private Supabase Storage")
    parser.add_argument("--version",           required=True,          help="Model version, e.g. v2")
    parser.add_argument("--bucket",            default="ml-models",    help="Supabase Storage bucket name (must be private)")
    parser.add_argument("--signed-url-expiry", type=int, default=31536000, help="Signed URL lifetime in seconds (default 1 year)")
    parser.add_argument("--dry-run",           action="store_true",    help="Skip upload, just print what would happen")
    args = parser.parse_args()

    version = args.version
    model_file = MODELS_DIR / version / f"PadelLabs-StrokeClassifier-{version}.mlmodel"

    if not model_file.exists():
        print(f"[ERROR] Model not found: {model_file}")
        print(f"        Expected: models/{version}/PadelLabs-StrokeClassifier-{version}.mlmodel")
        print(f"        Run retrain.py first.")
        sys.exit(1)

    file_size_mb = model_file.stat().st_size / (1024 * 1024)
    print(f"\n  Model : {model_file}")
    print(f"  Size  : {file_size_mb:.1f} MB")
    print(f"  Target: {args.bucket}/stroke-classifier/{model_file.name}")

    if args.dry_run:
        print("\n[DRY RUN] Skipping upload.")
        _print_remote_config(version, url="<signed URL will appear after upload>")
        return

    # ── Load env ───────────────────────────────────────────────────────────────
    _load_env()

    supabase_url = os.environ.get("SUPABASE_URL", "")
    service_key  = os.environ.get("SUPABASE_SERVICE_KEY", "")

    if not supabase_url or not service_key:
        print("[ERROR] SUPABASE_URL and SUPABASE_SERVICE_KEY must be set.")
        print("        Add them to a .env file in the repo root or export them.")
        sys.exit(1)

    # ── Upload ─────────────────────────────────────────────────────────────────
    try:
        from supabase import create_client, Client
    except ImportError:
        print("[ERROR] supabase package not installed.  Run: pip install supabase")
        sys.exit(1)

    client: Client = create_client(supabase_url, service_key)

    storage_path = f"stroke-classifier/{model_file.name}"

    print("\nUploading to private bucket...")
    try:
        with open(model_file, "rb") as f:
            client.storage.from_(args.bucket).upload(
                path=storage_path,
                file=f,
                file_options={"content-type": "application/octet-stream", "upsert": "true"}
            )
    except Exception as e:
        msg = str(e).lower()
        if "bucket not found" in msg or "404" in msg:
            print(f"[ERROR] Bucket '{args.bucket}' not found.")
            print(f"        Supabase Dashboard → Storage → New bucket → name: {args.bucket}, Public: OFF")
        elif "unauthorized" in msg or "403" in msg:
            print(f"[ERROR] Permission denied. Make sure SUPABASE_SERVICE_KEY is the service_role key, not the anon key.")
        else:
            print(f"[ERROR] Upload failed: {e}")
        sys.exit(1)
    print(f"  ✅ Uploaded: {args.bucket}/{storage_path}")

    # ── Generate signed URL ────────────────────────────────────────────────────
    # The signed URL embeds a time-limited token in the query string.
    # No auth header is needed at download time — the URL IS the credential.
    expiry = args.signed_url_expiry
    expiry_days = expiry // 86400
    print(f"\nGenerating signed URL (valid {expiry_days} days)...")
    result = client.storage.from_(args.bucket).create_signed_url(
        path=storage_path,
        expires_in=expiry
    )
    signed_url = result.get("signedURL") or result.get("signed_url") or result.get("signedUrl", "")

    if not signed_url:
        print(f"[ERROR] Failed to generate signed URL. Response: {result}")
        sys.exit(1)

    print(f"  ✅ Signed URL generated (expires in {expiry_days} days)")

    _print_remote_config(version, url=signed_url)


def _print_remote_config(version: str, url: str):
    print(f"""
{'='*60}
  Firebase Remote Config — set these two values:
{'='*60}

  Key  : ml_stroke_model_version
  Value: {version}

  Key  : ml_stroke_model_url
  Value: {url}

{'='*60}
  Next: open Firebase Console → Remote Config → add/update
  the two keys above, then publish.  The app will download
  and hot-swap the model on next launch (no App Store update).
{'='*60}
""")


def _load_env():
    """Load .env from repo root if it exists."""
    env_file = Path(__file__).parent.parent / ".env"
    if not env_file.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file)
    except ImportError:
        # Manual parse fallback
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


if __name__ == "__main__":
    main()
