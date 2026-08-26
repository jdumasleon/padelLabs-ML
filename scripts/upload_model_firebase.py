#!/usr/bin/env python3
"""
upload_model_firebase.py — PadelLabs ML Toolchain
==================================================
Uploads a trained .mlmodel to Firebase Storage and prints the `ml_stroke_model`
Remote Config descriptor to publish. Firebase replacement for the legacy
Supabase-based upload_model.py.

The object is uploaded WITHOUT a Firebase download token. That is deliberate: a
download-token URL bypasses Storage security rules by design — that is what tokens
are for — so the model it points at is readable by anyone, with no credentials, for
as long as the token exists. The app now fetches through the Storage SDK instead,
which carries Firebase Auth and App Check tokens, and `storage.rules` gates
`ml-models/**` on `request.auth != null`.

The app (`ModelUpdateService.swift`) reads a single JSON key, `ml_stroke_model`:

    {"version": "v5",
     "storagePath": "ml-models/PadelLabs-StrokeClassifier-v5.mlmodel",
     "sha256": "<hex>",
     "format": "mlmodel-v1"}

One object so version, path and hash can never be observed out of step mid-rollout.
The client verifies the SHA-256 before installing, and skips any `format` it does
not recognise. Install state is keyed on version AND format, so re-publishing the
same model in a new artifact shape still triggers a reinstall.

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
    ap.add_argument("--set-remote-config", action="store_true",
                    help="Also publish the ml_stroke_model descriptor to Remote Config.")
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

    from urllib.parse import quote  # noqa: F401  (kept for callers that build console links)

    bucket = storage.bucket()
    blob = bucket.blob(object_path)
    blob.cache_control = "public, max-age=31536000, immutable"

    # No `firebaseStorageDownloadTokens` metadata. A download token produces a URL that
    # bypasses Storage rules entirely, which is what made the classifier world-readable.
    # Access is now decided by storage.rules + App Check, per request.

    digest = _sha256(model_file)

    print("\nUploading…")
    blob.upload_from_filename(str(model_file), content_type="application/octet-stream")

    # Clear any token left over from a previous upload of this object; re-uploading does
    # not remove existing custom metadata on its own, and a stale token keeps working.
    if blob.metadata and "firebaseStorageDownloadTokens" in blob.metadata:
        metadata = dict(blob.metadata)
        metadata.pop("firebaseStorageDownloadTokens", None)
        blob.metadata = metadata
        blob.patch()
        print("  Revoked a pre-existing download token on this object.")

    descriptor = {
        "version": version,
        "storagePath": object_path,
        "sha256": digest,
        "format": "mlmodel-v1",
    }

    print(f"  gs://{args.bucket}/{object_path}")
    print(f"  sha256: {digest}")
    print("\n" + "=" * 70)
    print("Remote Config — set ml_stroke_model to:")
    print(json.dumps(descriptor, indent=2))
    print("=" * 70)

    if args.set_remote_config:
        push_remote_config(descriptor)


def _sha256(path: Path) -> str:
    """Streamed so a ~24 MB model is not held in memory twice."""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def push_remote_config(descriptor: dict):
    """Publish the ml_stroke_model descriptor via the Remote Config REST API.

    Avoids the firebase CLI (which needs a firebase.json with a remoteconfig block).
    Read-modify-write with the ETag (If-Match) the API requires.

    Writes one key. The legacy ml_stroke_model_version / ml_stroke_model_url scalars are
    deliberately left untouched: they published a download-token URL, and rewriting them
    would republish that access path.
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
    entry = params.setdefault("ml_stroke_model", {})
    entry["defaultValue"] = {"value": json.dumps(descriptor, separators=(",", ":"))}
    entry.setdefault("valueType", "JSON")
    entry.setdefault(
        "description",
        "Stroke classifier to install: version, storagePath, sha256, format.",
    )

    put = requests.put(
        base, headers={**headers, "Content-Type": "application/json; UTF-8", "If-Match": etag},
        data=json.dumps(tmpl),
    )
    put.raise_for_status()
    print(f"✅ Remote Config updated (new version {put.json().get('version', {}).get('versionNumber', '?')}).")


if __name__ == "__main__":
    main()
