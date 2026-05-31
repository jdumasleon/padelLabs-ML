#!/usr/bin/env python3
"""
upload_model_firebase.py — PadelLabs ML Toolchain
==================================================
Uploads a trained .mlmodel to Firebase Storage and prints the download URL +
the two Firebase Remote Config values to set. Firebase replacement for the
legacy Supabase-based upload_model.py.

The PadelLabs app (`ModelUpdateService.swift`) downloads the model from whatever
HTTPS URL is in Remote Config `ml_stroke_model_url` and compares
`ml_stroke_model_version` by STRING equality — so any stable HTTPS URL works and
the version is just a label (e.g. "v4").

Auth (pick one):
  1. Service account JSON (recommended, reusable):
       Firebase Console → Project Settings → Service accounts → Generate new private key
       export GOOGLE_APPLICATION_CREDENTIALS=/path/to/serviceAccount.json
  2. Application Default Credentials:
       gcloud auth application-default login

Requirements:
    pip install firebase-admin

Usage:
    python3 upload_model_firebase.py --version v4
    python3 upload_model_firebase.py --version v4 --bucket padellabs-f40f7.firebasestorage.app
    python3 upload_model_firebase.py --version v4 --signed-url-days 3650   # 10y signed URL
    python3 upload_model_firebase.py --version v4 --public                 # public URL (no token)
    python3 upload_model_firebase.py --version v4 --set-remote-config       # also push Remote Config
"""

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
ML_DIR = SCRIPT_DIR.parent
MODELS_DIR = ML_DIR / "models"

DEFAULT_BUCKET = "padellabs-f40f7.firebasestorage.app"
PROJECT_ID = "padellabs-f40f7"
STORAGE_PREFIX = "ml-models"  # object path prefix inside the bucket


def main():
    ap = argparse.ArgumentParser(description="Upload CoreML model to Firebase Storage")
    ap.add_argument("--version", required=True, help="Model version, e.g. v4")
    ap.add_argument("--bucket", default=DEFAULT_BUCKET, help="Firebase Storage bucket")
    ap.add_argument("--public", action="store_true",
                    help="Make the object public (stable tokenless URL).")
    ap.add_argument("--signed-url-days", type=int, default=0,
                    help="If >0, emit a V4 signed URL of this many days (MAX 7 — Google cap). "
                         "Default 0 = Firebase download-token URL (stable, never expires, token-gated).")
    ap.add_argument("--set-remote-config", action="store_true",
                    help="Also push ml_stroke_model_version/url to Remote Config via the firebase CLI.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    version = args.version
    model_file = MODELS_DIR / version / f"PadelLabs-StrokeClassifier-{version}.mlmodel"
    if not model_file.exists():
        sys.exit(f"[ERROR] Model not found: {model_file}")

    object_path = f"{STORAGE_PREFIX}/PadelLabs-StrokeClassifier-{version}.mlmodel"
    print(f"Model:   {model_file}  ({model_file.stat().st_size/1e6:.1f} MB)")
    print(f"Bucket:  gs://{args.bucket}")
    print(f"Object:  {object_path}")

    if args.dry_run:
        print("\n(dry-run — no upload)")
        return

    try:
        import firebase_admin
        from firebase_admin import credentials, storage
    except ImportError:
        sys.exit("[ERROR] firebase-admin not installed.  pip install firebase-admin")

    # Initialise — uses GOOGLE_APPLICATION_CREDENTIALS or ADC automatically.
    if not firebase_admin._apps:
        cred = None
        sa = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if sa and Path(sa).exists():
            cred = credentials.Certificate(sa)
        firebase_admin.initialize_app(cred, {"storageBucket": args.bucket})

    import uuid as _uuid
    from urllib.parse import quote

    bucket = storage.bucket()
    blob = bucket.blob(object_path)
    blob.cache_control = "public, max-age=31536000, immutable"

    if args.public:
        download_token = None
    else:
        # Firebase-style stable download token (never expires; URL is token-gated).
        download_token = str(_uuid.uuid4())
        blob.metadata = {"firebaseStorageDownloadTokens": download_token}

    print("\nUploading…")
    blob.upload_from_filename(str(model_file), content_type="application/octet-stream")

    if args.public:
        blob.make_public()
        url = blob.public_url
        print(f"  Public URL: {url}")
    elif args.signed_url_days > 0:
        days = min(args.signed_url_days, 7)  # Google V4 hard cap
        url = blob.generate_signed_url(expiration=dt.timedelta(days=days), method="GET", version="v4")
        print(f"  Signed URL ({days}d, expires {(dt.datetime.utcnow()+dt.timedelta(days=days)).date()}): {url}")
    else:
        encoded = quote(object_path, safe="")
        url = (f"https://firebasestorage.googleapis.com/v0/b/{args.bucket}"
               f"/o/{encoded}?alt=media&token={download_token}")
        print(f"  Firebase download URL (stable, never expires): {url}")

    print("\n" + "=" * 70)
    print("Firebase Remote Config values to set:")
    print(f"  ml_stroke_model_version = {version}")
    print(f"  ml_stroke_model_url     = {url}")
    print("=" * 70)

    if args.set_remote_config:
        push_remote_config(version, url)


def push_remote_config(version: str, url: str):
    """Update the two keys via the Remote Config REST API using the service-account creds.

    Avoids the firebase CLI (which needs a firebase.json with a remoteconfig block).
    Read-modify-write with the ETag (If-Match) the API requires.
    """
    import google.auth
    import google.auth.transport.requests
    import requests

    print("\nPushing Remote Config via REST API…")
    scopes = ["https://www.googleapis.com/auth/firebase.remoteconfig"]
    creds, _ = google.auth.default(scopes=scopes)
    creds.refresh(google.auth.transport.requests.Request())
    base = f"https://firebaseremoteconfig.googleapis.com/v1/projects/{PROJECT_ID}/remoteConfig"
    headers = {"Authorization": f"Bearer {creds.token}"}

    get = requests.get(base, headers=headers)
    get.raise_for_status()
    tmpl = get.json()
    etag = get.headers.get("ETag", "*")

    params = tmpl.setdefault("parameters", {})
    for key, val in (("ml_stroke_model_version", version), ("ml_stroke_model_url", url)):
        params.setdefault(key, {})["defaultValue"] = {"value": val}

    put = requests.put(
        base, headers={**headers, "Content-Type": "application/json; UTF-8", "If-Match": etag},
        data=json.dumps(tmpl),
    )
    put.raise_for_status()
    print(f"✅ Remote Config updated (new version {put.json().get('version', {}).get('versionNumber', '?')}).")


if __name__ == "__main__":
    main()
