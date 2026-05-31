# Dynamic ML Model Update — Setup Guide

End-to-end guide for shipping new stroke classifier versions to the PadelLabs app without an App Store update.

---

## How it works

```
Retrain locally
     ↓
upload_model.py → Supabase Storage (stores .mlmodel file)
     ↓
Firebase Remote Config (set version + download URL)
     ↓
App launch → iPhone downloads + compiles model → notifies Watch via WatchConnectivity
     ↓
Watch downloads + compiles model → CoreMLStrokeClassifier hot-reloads
```

No App Store update. No restart required. The classifier swaps silently in the background.

---

## Part 1 — One-time infrastructure setup

### 1.1 Create the Supabase Storage bucket

1. Open [Supabase Dashboard](https://supabase.com) → your PadelLabs project
2. Go to **Storage** → **New bucket**
3. Name: `ml-models`
4. Set to **Private** (uncheck "Public bucket")
5. Click **Create bucket**

> **Why private?** A public bucket means anyone with the URL can download your model.
> With a private bucket, the upload script generates a **signed URL** — a time-limited
> token embedded in the URL itself. The app uses this URL to download without any
> credentials in the binary. When you ship a new model, the old signed URL is
> automatically replaced.

### 1.2 Get your Supabase service-role key

1. Go to **Settings** → **API**
2. Copy the **service_role** key (not the anon key — the upload script needs storage write + sign access)
3. Add to a `.env` file in the repo root (**never commit this file**):

```
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_SERVICE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

Make sure `.env` is in your `.gitignore`:
```bash
echo ".env" >> .gitignore
```

> The service-role key is only used by the upload script on your Mac — it never
> touches the app binary or the Watch.

### 1.3 Set up Firebase Remote Config keys

1. Open [Firebase Console](https://console.firebase.google.com) → your PadelLabs project
2. Go to **Remote Config** → **Add parameter**
3. Add these two parameters:

| Parameter key | Default value | Description |
|---|---|---|
| `ml_stroke_model_version` | *(empty)* | Current model version, e.g. `v1` |
| `ml_stroke_model_url` | *(empty)* | Supabase Storage public URL of the .mlmodel file |

4. Leave both empty for now — you will fill them after the first upload
5. Click **Save** (do not publish yet)

### 1.4 Install upload script dependencies

```bash
cd PadelLabs-ML
pip install supabase python-dotenv
```

---

## Part 2 — Retraining and uploading a new model

Do this every time you have enough new labeled data to retrain.

### 2.1 Retrain the classifier

```bash
cd PadelLabs-ML
python3 scripts/retrain.py
```

This reads `labeled-strokes/train/`, trains a new Random Forest, and writes:
- `models/PadelLabs-StrokeClassifier-v<N>.mlmodel`
- `models/PadelLabs-StrokeClassifier-v<N>.json` (metadata)

The version auto-increments (v1 → v2 → v3, …).

To force a specific version:
```bash
python3 scripts/retrain.py --output-version v3
```

### 2.2 Review the training report

Before uploading, check the output:
- Cross-validation accuracy should be **≥ 70%**
- No class should have **0 samples** in training
- Classes with low F1 (< 0.5) need more labeled data before shipping

### 2.3 Upload to Supabase Storage

```bash
python3 scripts/upload_model.py --version v2
```

The script will:
1. Upload `models/PadelLabs-StrokeClassifier-v2.mlmodel` to the private `ml-models` bucket
2. Generate a **signed URL** valid for 1 year (token is embedded in the URL query string)
3. Print the two Remote Config values to set

Example output:
```
  Uploading to private bucket...
  ✅ Uploaded: ml-models/stroke-classifier/PadelLabs-StrokeClassifier-v2.mlmodel

  Generating signed URL (valid 365 days)...
  ✅ Signed URL generated

============================================================
  Firebase Remote Config — set these two values:
============================================================

  Key  : ml_stroke_model_version
  Value: v2

  Key  : ml_stroke_model_url
  Value: https://xyz.supabase.co/storage/v1/object/sign/ml-models/stroke-classifier/...?token=eyJ...
============================================================
```

Dry-run (no upload, just validate paths):
```bash
python3 scripts/upload_model.py --version v2 --dry-run
```

Custom signed URL expiry (e.g. 180 days):
```bash
python3 scripts/upload_model.py --version v2 --signed-url-expiry 15552000
```

### 2.4 Update Firebase Remote Config

1. Open Firebase Console → **Remote Config**
2. Update `ml_stroke_model_version` → `v2`
3. Update `ml_stroke_model_url` → paste the URL from step 2.3
4. Click **Publish changes**

The update is now live. The app will pick it up on next launch.

---

## Part 3 — Security model

| Layer | What it does | What it does NOT do |
|---|---|---|
| Private Supabase bucket | Blocks unauthenticated direct access | — |
| Signed URL | Embeds a time-limited token in the URL | Expose credentials in the app binary |
| Firebase Remote Config | Delivers the signed URL to authenticated app instances | Allow arbitrary clients to fetch it |
| Service-role key | Used only by the upload script on your Mac | Never leave your machine |
| App binary | Contains no secrets — just fetches from the URL it receives | — |

**The signed URL is the only "credential" the app uses.** It is not a password or API key — it is a pre-authorised, time-limited URL. If it leaks (e.g. via a proxy log), the worst case is that someone can download your model file until the URL expires. It cannot be used to access any other Supabase resource.

**Firebase Remote Config** itself is initialised with your `GoogleService-Info.plist`, which is already in the app. This means only your own app can fetch the Remote Config values — a random HTTP client cannot query your Remote Config.

---

## Part 4 — What happens on the device

### iPhone
1. App launches → `ModelUpdateService.checkAndUpdate()` runs
2. Fetches Remote Config → detects new version
3. Downloads `.mlmodel` from Supabase Storage
4. Compiles on-device via `MLModel.compileModel()` → stored in `Application Support/MLModels/`
5. Notifies Watch via `WCSession.transferUserInfo` (works even if Watch is unreachable — queued for delivery)

### Apple Watch
1. Receives the model version + URL from iPhone via WatchConnectivity
2. `WatchModelUpdateService` downloads `.mlmodel` from Supabase Storage directly
3. Compiles on-device → stored in `Documents/MLModels/`
4. Posts `Notification.Name.watchModelUpdated`
5. `CoreMLStrokeClassifier` observes the notification and hot-reloads — **no restart needed**

### Fallback behaviour
- If Remote Config is unreachable → app uses the currently installed model
- If download fails → app uses the currently installed model
- If no downloaded model exists → app uses the model compiled into the bundle (`v1`)

---

## Part 5 — Verifying the update

### Check Remote Config is serving the new values
In Firebase Console → Remote Config → **Conditions** tab, you can verify the published values.

### Check the iPhone received the new model
Look for these log lines in the Xcode console or device logs:

```
ModelUpdate: Downloading v2 (was: v1)
ModelUpdate: v2 installed successfully
ModelUpdate: Watch notified of model v2
```

### Check the Watch received the new model
```
WatchConnectivity: ML model update received — v2
WatchModelUpdate: v2 installed — classifier will reload
StrokeClassifier: reloaded with new downloaded model
```

### Force-check without waiting for app relaunch
The Remote Config minimum fetch interval is set to **60 seconds** in debug builds.
Kill and relaunch the app to trigger an immediate check.

---

## Part 6 — Rollback

If a new model performs worse, roll back instantly without touching the app:

1. Firebase Console → Remote Config
2. Set `ml_stroke_model_version` back to the previous version (e.g. `v1`)
3. Set `ml_stroke_model_url` back to the **previous model's signed URL**
4. Publish changes

The app will install the previous version on next launch.

> Keep the signed URLs from all previous uploads in a safe place (e.g. a private note or password manager). Do not delete old model files from Supabase Storage.
>
> If a previous signed URL has expired, re-generate it:
> ```bash
> python3 scripts/upload_model.py --version v1
> # The script uses upsert=true so it re-uploads (or you can run it after uploading manually)
> ```
> Then update Remote Config with the new signed URL.

---

## Part 7 — Recommended training cadence

| Trigger | Action |
|---|---|
| After labeling **≥ 200 new strokes** | Retrain + upload |
| After adding a **new stroke type** | Retrain + upload + update `CoreMLStrokeClassifier.labels` array |
| After a **player complaint** about misclassification | Label more data for that stroke type, retrain |
| **Monthly** minimum | Review class distribution in `models/vN.json` |

---

## Quick reference — full release checklist

```
[ ] Label new session(s) using the web labeling tool
[ ] Run extract_windows.py to generate training windows
[ ] Run retrain.py — check accuracy ≥ 70%
[ ] Run upload_model.py --version vN
[ ] Update Firebase Remote Config (version + URL) → Publish
[ ] Relaunch app on test device → check logs for "installed successfully"
[ ] Relaunch watch → check logs for "classifier will reload"
[ ] Play a session → verify stroke types in session detail
```
