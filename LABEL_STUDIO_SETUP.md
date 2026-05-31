# Label Studio Setup — PadelLabs ML

Label Studio is used for QC review of extracted stroke windows — visually inspect the
accel/gyro plots, discard bad reps, and fix any mislabeled strokes.

## 1. Prerequisites

- Docker Desktop installed and running
- Port 8080 free on localhost

## 2. Start Label Studio

```bash
cd /Users/jldumas/Jo/PadelLabs/PadelLabs-ML

docker pull heartexlabs/label-studio:latest

docker run -d \
  --name padellabs-label-studio \
  -p 8080:8081 \
  -v $(pwd)/label-studio-data:/label-studio/data \
  heartexlabs/label-studio:latest

# Open in browser:
open http://localhost:8081
```

## 3. First-time account setup

1. Open http://localhost:8080
2. Create a local account (username/password — not shared anywhere)
3. Sign in

## 4. Create a Time Series project

1. Click **Create Project** → name it "PadelLabs Stroke QC"
2. Go to **Settings → Labeling Interface → Code**
3. Paste this config:

```xml
<View>
  <TimeSeries name="ts" value="$csv" sep="," timeColumn="timestamp">
    <Channel column="accelX" legend="Accel X" strokeColor="#e74c3c"/>
    <Channel column="accelY" legend="Accel Y" strokeColor="#27ae60"/>
    <Channel column="accelZ" legend="Accel Z" strokeColor="#2980b9"/>
    <Channel column="gyroX"  legend="Gyro X"  strokeColor="#f39c12" displayFormat=",.2f"/>
    <Channel column="gyroY"  legend="Gyro Y"  strokeColor="#8e44ad" displayFormat=",.2f"/>
    <Channel column="gyroZ"  legend="Gyro Z"  strokeColor="#16a085" displayFormat=",.2f"/>
  </TimeSeries>
  <Choices name="strokeType" toName="ts" choice="single">
    <Choice value="smash"/>
    <Choice value="vibora"/>
    <Choice value="bandeja"/>
    <Choice value="rulo"/>
    <Choice value="forehand"/>
    <Choice value="backhand"/>
    <Choice value="forehandLob"/>
    <Choice value="backhandLob"/>
    <Choice value="forehandVolley"/>
    <Choice value="backhandVolley"/>
    <Choice value="serve"/>
    <Choice value="unknown"/>
  </Choices>
  <Choices name="quality" toName="ts" choice="single">
    <Choice value="good"/>
    <Choice value="mishit"/>
    <Choice value="double_spike"/>
    <Choice value="discard"/>
  </Choices>
</View>
```

4. Click **Save**

## 5. Import stroke windows

After running `extract_windows.py`, import a class folder:

```bash
# Label Studio expects a JSON list of tasks. Use this helper:
python3 scripts/ls_import.py \
    --windows labeled-strokes/train/smash \
    --output  label-studio-import/smash_tasks.json
```

Then in Label Studio: **Import → Upload** the JSON file.

> **Tip:** Review the hardest class pairs first: smash↔vibora, bandeja↔smash, forehandVolley↔forehand

## 6. QC workflow per window

For each window in Label Studio:
- ✅ `good` — clean spike, correct label, use for training
- ⚠️ `mishit` — player missed or topped the ball — discard
- ⚠️ `double_spike` — two strokes collapsed into one window — discard
- ❌ `discard` — NaN, noise, unclear motion — discard

Target rejection rate: **< 10%** per session.

## 7. Stop / restart Label Studio

```bash
docker stop padellabs-label-studio
docker start padellabs-label-studio
```

## 8. Data lives in

```
PadelLabs-ML/label-studio-data/   ← all Label Studio state (bind-mounted)
```

Back this folder up before upgrading Label Studio.
