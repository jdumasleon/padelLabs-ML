#!/usr/bin/env python3
"""
compile_model.py — PadelLabs ML Toolchain
==========================================
Compiles a .mlmodel to .mlmodelc on the build machine, optionally encrypting it.

WHY COMPILE HERE
----------------
Today the phone downloads a raw .mlmodel and runs `MLModel.compileModel(at:)` itself, then
ships the resulting .mlmodelc to the Watch. The phone never runs inference — it only
compiles and relays. Compiling here makes it a pure relay and, more importantly, is the
only way to ship an *encrypted* model: Core ML encrypts the compiled form, not the source.

Note the trade this makes. A phone-compiled .mlmodelc is by construction built by the
device's own Core ML compiler, so it always matches that OS. Compiling here freezes it to
this Xcode's compiler, which is why --platform/--deployment-target are passed: they make
the compiler check compatibility rather than leaving it to fail at load time on a watch.

ENCRYPTION
----------
`--encrypt` takes a .mlmodelkey generated in Xcode (see below). Core ML encrypts the
compiled model with AES-128; the key is held by Apple, fetched by the app on first load and
cached by the OS, and the model is decrypted only into memory — never back to disk. The key
is never in the app binary, so pulling the .mlmodelc off a device yields nothing usable.

To create the key (one time, Xcode UI — there is no CLI for this):
    1. Open any Xcode project containing a .mlmodel.
    2. Select the model → Utilities inspector → "Create Encryption Key".
    3. Choose the team. Xcode asks Apple to generate and store the key, and drops a
       .mlmodelkey next to the model.
    4. Do NOT commit the .mlmodelkey. It carries AES key material, so a copy in git hands
       the model to anyone with repo access — which is most of what encrypting it was for.
       Keep it in a password manager or secure storage and pass its path to --encrypt.
       Losing it means existing encrypted models can no longer be reproduced; you would
       generate a new key and re-encrypt.

What encryption does NOT stop: anyone who can attach a debugger to the running Watch app
can read the decrypted model out of memory. It stops file theft, not runtime instrumentation.

Usage:
    python3 scripts/compile_model.py --version v5
    python3 scripts/compile_model.py --version v5 --encrypt models/keys/padellabs.mlmodelkey
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
ML_DIR = SCRIPT_DIR.parent
MODELS_DIR = ML_DIR / "models"

# Must match the app's watchOS deployment target. The compiled model is loaded by the Watch,
# never the phone, so this is the platform that matters.
TARGET_PLATFORM = "watchOS"
TARGET_DEPLOYMENT = "10.0"


def main():
    ap = argparse.ArgumentParser(description="Compile (and optionally encrypt) a CoreML model")
    ap.add_argument("--version", required=True, help="Model version, e.g. v5")
    ap.add_argument("--encrypt", metavar="KEYFILE",
                    help="Path to a .mlmodelkey. Omit to produce a plaintext .mlmodelc.")
    ap.add_argument("--out", help="Output directory (default: models/<version>/compiled)")
    args = ap.parse_args()

    source = MODELS_DIR / args.version / f"PadelLabs-StrokeClassifier-{args.version}.mlmodel"
    if not source.exists():
        sys.exit(f"[ERROR] Model not found: {source}")

    out_dir = Path(args.out) if args.out else MODELS_DIR / args.version / "compiled"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "xcrun", "coremlcompiler", "compile", str(source), str(out_dir),
        "--platform", TARGET_PLATFORM,
        "--deployment-target", TARGET_DEPLOYMENT,
    ]

    if args.encrypt:
        key = Path(args.encrypt)
        if not key.exists():
            sys.exit(
                f"[ERROR] Encryption key not found: {key}\n"
                "Generate one in Xcode: select a .mlmodel → Utilities inspector →\n"
                "'Create Encryption Key'. There is no command-line equivalent."
            )
        cmd += ["--encrypt", str(key)]

    print(f"Source:   {source}  ({source.stat().st_size/1e6:.1f} MB)")
    print(f"Target:   {TARGET_PLATFORM} {TARGET_DEPLOYMENT}")
    print(f"Encrypt:  {'yes — ' + str(args.encrypt) if args.encrypt else 'no (plaintext)'}")
    print(f"Output:   {out_dir}\n")
    print("$ " + " ".join(cmd) + "\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"[ERROR] coremlcompiler failed ({result.returncode})")

    compiled = out_dir / f"PadelLabs-StrokeClassifier-{args.version}.mlmodelc"
    if not compiled.exists():
        found = list(out_dir.glob("*.mlmodelc"))
        if not found:
            sys.exit("[ERROR] No .mlmodelc produced.")
        compiled = found[0]

    total = sum(f.stat().st_size for f in compiled.rglob("*") if f.is_file())
    files = sum(1 for f in compiled.rglob("*") if f.is_file())
    print(f"\n✅ {compiled.name} — {files} files, {total/1e6:.1f} MB")

    if args.encrypt:
        print("\nEncrypted. Before publishing this, prove on a PHYSICAL Apple Watch that")
        print("MLModel.load(contentsOf:configuration:) can fetch the key under the WATCH")
        print("app's bundle id, and that it still loads in airplane mode afterwards.")
        print("The simulator does not exercise the real key-fetch path.")


if __name__ == "__main__":
    main()
