#!/usr/bin/env python3
"""
revoke_download_tokens.py — PadelLabs ML Toolchain
===================================================
Removes Firebase Storage download tokens from the published model objects.

WHY THIS EXISTS
---------------
A Firebase Storage download token produces a URL of the form

    https://firebasestorage.googleapis.com/v0/b/<bucket>/o/<path>?alt=media&token=<uuid>

That URL **bypasses Storage security rules entirely** — that is what download tokens are
for; they exist to publish assets openly. So `allow read, write: if false` in storage.rules
never applied to an object carrying one. The token also never expires and is identical for
every caller, and ours was shipped to every install through client Remote Config, which is
readable using only the API key embedded in the app binary. The classifier was therefore
downloadable by anyone, with no credentials, in a single request.

Deleting the token is what actually closes that. Rules and App Check govern the SDK path;
they do not govern a token URL.

WHAT BREAKS
-----------
App builds that fetch the model over the token URL stop being able to download it. They do
not crash: `ModelUpdateService` catches the failure and keeps the model already installed,
falling back to the one compiled into the Watch binary if there is none. Those installs are
frozen at their current model until they update.

Run this AFTER the TestFlight build that fetches through the Storage SDK is released.

IRREVERSIBLE
------------
The old URL cannot be brought back. Firebase can mint a *new* token, which produces a
*different* URL; anything holding the old one is permanently locked out. That is the point.

Auth: same as upload_model_firebase.py —
    export GOOGLE_APPLICATION_CREDENTIALS=/path/to/serviceAccount.json

Usage:
    python3 scripts/revoke_download_tokens.py                 # dry run, lists what would change
    python3 scripts/revoke_download_tokens.py --apply         # actually revoke
    python3 scripts/revoke_download_tokens.py --apply --verify # revoke, then prove the URL 403s
"""

import argparse
import sys
from urllib.parse import quote

DEFAULT_BUCKET = "padellabs-f40f7.firebasestorage.app"
STORAGE_PREFIX = "ml-models"
TOKEN_KEY = "firebaseStorageDownloadTokens"


def main():
    ap = argparse.ArgumentParser(description="Revoke Firebase Storage download tokens on model objects")
    ap.add_argument("--bucket", default=DEFAULT_BUCKET)
    ap.add_argument("--prefix", default=STORAGE_PREFIX,
                    help="Object prefix to sweep (default: ml-models)")
    ap.add_argument("--apply", action="store_true",
                    help="Actually revoke. Without this the script only reports.")
    ap.add_argument("--verify", action="store_true",
                    help="After revoking, re-request each old URL and assert it is refused.")
    args = ap.parse_args()

    try:
        import firebase_admin
        from firebase_admin import credentials, storage
    except ImportError:
        sys.exit("[ERROR] firebase-admin not installed.  pip install firebase-admin")

    import os
    from pathlib import Path

    if not firebase_admin._apps:
        cred = None
        sa = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if sa and Path(sa).exists():
            cred = credentials.Certificate(sa)
        firebase_admin.initialize_app(cred, {"storageBucket": args.bucket})

    try:
        bucket = storage.bucket()
        blobs = list(bucket.list_blobs(prefix=f"{args.prefix}/"))
    except Exception as exc:  # noqa: BLE001 — surface any auth/permission failure plainly
        sys.exit(
            f"[ERROR] Could not read gs://{args.bucket}: {exc}\n\n"
            "This needs admin credentials — the same ones upload_model_firebase.py uses:\n"
            "  export GOOGLE_APPLICATION_CREDENTIALS=/path/to/serviceAccount.json\n"
            "(Firebase Console → Project Settings → Service accounts → Generate new private key)"
        )

    if not blobs:
        print(f"No objects under {args.prefix}/ in gs://{args.bucket}")
        return

    print(f"Bucket: gs://{args.bucket}")
    print(f"Prefix: {args.prefix}/\n")

    tokened = []
    for blob in blobs:
        blob.reload()
        metadata = blob.metadata or {}
        token = metadata.get(TOKEN_KEY)
        state = "TOKEN PRESENT" if token else "clean"
        print(f"  [{state:13}] {blob.name}  ({blob.size/1e6:.1f} MB)")
        if token:
            tokened.append((blob, token))

    if not tokened:
        print("\nNothing to revoke — no object carries a download token.")
        return

    print(f"\n{len(tokened)} object(s) currently reachable without credentials.")

    if not args.apply:
        print("\nDry run. Re-run with --apply to revoke.")
        print("Do this only AFTER the TestFlight build that downloads via the Storage SDK is out:")
        print("older builds fetch over the token URL and will stop receiving model updates")
        print("(they keep the model they already have — they do not break).")
        return

    print("\nRevoking…")
    revoked = []
    for blob, token in tokened:
        metadata = dict(blob.metadata or {})
        metadata.pop(TOKEN_KEY, None)
        # Explicit None tells the API to clear the key rather than leave it untouched.
        blob.metadata = {**metadata, TOKEN_KEY: None}
        blob.patch()
        print(f"  revoked: {blob.name}")
        revoked.append((blob.name, token))

    print(f"\n✅ {len(revoked)} token(s) revoked.")

    if args.verify:
        import requests
        print("\nVerifying the old URLs are refused…")
        ok = True
        for name, token in revoked:
            url = (f"https://firebasestorage.googleapis.com/v0/b/{args.bucket}"
                   f"/o/{quote(name, safe='')}?alt=media&token={token}")
            code = requests.get(url, stream=True).status_code
            verdict = "refused" if code in (401, 403, 404) else "STILL REACHABLE"
            if code not in (401, 403, 404):
                ok = False
            print(f"  [{verdict}] HTTP {code}  {name}")
        if not ok:
            sys.exit("[ERROR] At least one object is still served. Do not consider this closed.")
        print("\n✅ All old URLs refused.")


if __name__ == "__main__":
    main()
