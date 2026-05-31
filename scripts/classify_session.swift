#!/usr/bin/env swift
//
// classify_session.swift — PadelLabs ML Toolchain
// =================================================
// Classifies every spike in a DataCollection session using the on-device CoreML model.
// Replicates the exact feature extraction and inference pipeline from the Watch app.
//
// Usage:
//   swift classify_session.swift \
//       --session  ../../DataCollection/Jose\ full\ sessions\ strokes/19\ April/4AF757C9-...csv \
//       [--model   ../../PadelLabs-StrokeClassifier-v1.mlmodel] \
//       [--output  results.csv] \
//       [--threshold 0.55]
//       [--no-mirror]  # for right-handed / Watch on right wrist (matches training data)
//
// Output CSV columns:
//   timestamp_s, predicted_stroke, confidence, samples_in_window, raw_top3
//

import Foundation
import CoreML

// ── Label order (alphabetical — must match Python training) ─────────────────
let LABELS = [
    "backhand",         // 0
    "backhand_lob",     // 1
    "backhand_volley",  // 2
    "bandeja",          // 3
    "forehand",         // 4
    "forehand_lob",     // 5
    "forehand_volley",  // 6
    "smash",            // 7
    "vibora"            // 8
]

let CONFIDENCE_THRESHOLD = 0.55
let WINDOW_SIZE = 100
let PRE_PEAK = 30    // 300ms
let POST_PEAK = 70   // 700ms

// ── Argument parsing ─────────────────────────────────────────────────────────

struct Args {
    var sessionCSV: String = ""
    var modelPath: String = ""
    var outputPath: String = ""
    var threshold: Double = CONFIDENCE_THRESHOLD
    var mirror: Double = 1.0   // 1.0 = no flip (right/right), -1.0 = flip (right/left)
}

func parseArgs() -> Args {
    var args = Args()
    // Default model path relative to script location
    let scriptDir = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent()
    args.modelPath = scriptDir.appendingPathComponent("../PadelLabs-StrokeClassifier-v1.mlmodel").path

    let argv = Array(CommandLine.arguments.dropFirst())
    var i = 0
    while i < argv.count {
        switch argv[i] {
        case "--session":
            i += 1; args.sessionCSV = argv[i]
        case "--model":
            i += 1; args.modelPath = argv[i]
        case "--output":
            i += 1; args.outputPath = argv[i]
        case "--threshold":
            i += 1; args.threshold = Double(argv[i]) ?? CONFIDENCE_THRESHOLD
        case "--no-mirror":
            args.mirror = 1.0  // right/right — no X-axis flip (default)
        case "--mirror":
            args.mirror = -1.0  // right/left — flip X axis
        default:
            print("Unknown argument: \(argv[i])")
            exit(1)
        }
        i += 1
    }

    if args.sessionCSV.isEmpty {
        print("ERROR: --session <path-to-csv> is required")
        print("Usage: swift classify_session.swift --session path/to/session.csv [--output results.csv]")
        exit(1)
    }

    // Default output = session name with _classified.csv suffix
    if args.outputPath.isEmpty {
        let base = URL(fileURLWithPath: args.sessionCSV)
            .deletingPathExtension()
            .lastPathComponent
        args.outputPath = URL(fileURLWithPath: args.sessionCSV)
            .deletingLastPathComponent()
            .appendingPathComponent("\(base)_classified.csv")
            .path
    }

    return args
}

// ── IMU sample ────────────────────────────────────────────────────────────────

struct IMUSample {
    let ts: Double
    let ax, ay, az: Double
    let gx, gy, gz: Double
    let roll, pitch, yaw: Double
}

// ── CSV loading ───────────────────────────────────────────────────────────────

func loadCSV(path: String) -> [IMUSample] {
    guard let content = try? String(contentsOfFile: path, encoding: .utf8) else {
        print("ERROR: Cannot read CSV: \(path)"); exit(1)
    }
    var lines = content.components(separatedBy: "\n")
    guard !lines.isEmpty else { return [] }
    lines.removeFirst()  // header

    var samples: [IMUSample] = []
    for line in lines {
        let parts = line.components(separatedBy: ",")
        guard parts.count >= 10,
              let ts   = Double(parts[0]),
              let ax   = Double(parts[1]),
              let ay   = Double(parts[2]),
              let az   = Double(parts[3]),
              let gx   = Double(parts[4]),
              let gy   = Double(parts[5]),
              let gz   = Double(parts[6]),
              let roll = Double(parts[7]),
              let pitch = Double(parts[8]),
              let yaw  = Double(parts[9])
        else { continue }

        samples.append(IMUSample(ts: ts, ax: ax, ay: ay, az: az,
                                  gx: gx, gy: gy, gz: gz,
                                  roll: roll, pitch: pitch, yaw: yaw))
    }
    return samples
}

// ── JSON loading ──────────────────────────────────────────────────────────────

func loadMarkers(jsonPath: String) -> [(ts: Double, originalLabel: String)] {
    guard let data = FileManager.default.contents(atPath: jsonPath),
          let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
          let markers = json["markers"] as? [[String: Any]]
    else {
        print("WARNING: Cannot load JSON sidecar — will classify whole session uniformly")
        return []
    }

    return markers.compactMap { m in
        guard let ts = m["timestamp"] as? Double ?? (m["timestamp"] as? String).flatMap(Double.init),
              let label = m["strokeType"] as? String
        else { return nil }
        return (ts: ts, originalLabel: label)
    }
}

// ── Window extraction ─────────────────────────────────────────────────────────

func extractWindow(samples: [IMUSample], peakIdx: Int) -> [IMUSample]? {
    let start = peakIdx - PRE_PEAK
    let end   = peakIdx + POST_PEAK
    guard start >= 0, end <= samples.count else { return nil }
    let window = Array(samples[start..<end])
    guard window.count == WINDOW_SIZE else { return nil }
    return window
}

func findClosestIndex(samples: [IMUSample], ts: Double) -> Int? {
    guard !samples.isEmpty else { return nil }
    var bestIdx = 0
    var bestDiff = abs(samples[0].ts - ts)
    for (i, s) in samples.enumerated() {
        let diff = abs(s.ts - ts)
        if diff < bestDiff { bestDiff = diff; bestIdx = i }
    }
    return bestIdx
}

// ── 89-Feature extraction (must match Swift CoreMLStrokeClassifier) ───────────

/// 9 stats per channel: mean, std, min, max, range, rms, zcr, p25, p75
func perChannelStats(_ col: [Double]) -> [Double] {
    let n = Double(col.count)
    let mean = col.reduce(0, +) / n
    let variance = col.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / n
    let std = variance.squareRoot()
    let minVal = col.min() ?? 0
    let maxVal = col.max() ?? 0
    let rms = (col.map { $0 * $0 }.reduce(0, +) / n).squareRoot()

    var zcr = 0.0
    for i in 1..<col.count {
        if col[i] * col[i-1] < 0 { zcr += 1 }
    }

    let sorted = col.sorted()
    let p25 = sorted[max(0, Int(n * 0.25))]
    let p75 = sorted[min(col.count - 1, Int(n * 0.75))]

    return [mean, std, minVal, maxVal, maxVal - minVal, rms, zcr, p25, p75]
}

/// 4 stats for magnitude channels: mean, std, max, rms
func magnitudeStats(_ mag: [Double]) -> [Double] {
    let n = Double(mag.count)
    let mean = mag.reduce(0, +) / n
    let variance = mag.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / n
    let std = variance.squareRoot()
    let maxVal = mag.max() ?? 0
    let rms = (mag.map { $0 * $0 }.reduce(0, +) / n).squareRoot()
    return [mean, std, maxVal, rms]
}

func extract89Features(window: [IMUSample], mirror: Double) -> [Double] {
    // 9 channels with optional mirror on lateral axes
    let channels: [[Double]] = [
        window.map { $0.ax * mirror },  // 0: accelX
        window.map { $0.ay },           // 1: accelY
        window.map { $0.az },           // 2: accelZ
        window.map { $0.gx * mirror },  // 3: gyroX
        window.map { $0.gy },           // 4: gyroY
        window.map { $0.gz },           // 5: gyroZ
        window.map { $0.roll * mirror },// 6: roll
        window.map { $0.pitch },        // 7: pitch
        window.map { $0.yaw },          // 8: yaw
    ]

    var features: [Double] = []

    // 9 channels × 9 stats = 81
    for col in channels {
        features += perChannelStats(col)
    }

    // Accel magnitude stats (4)
    let accelMag = window.map { s -> Double in
        let ax = s.ax * mirror, ay = s.ay, az = s.az
        return (ax*ax + ay*ay + az*az).squareRoot()
    }
    features += magnitudeStats(accelMag)

    // Gyro magnitude stats (4)
    let gyroMag = window.map { s -> Double in
        let gx = s.gx * mirror, gy = s.gy, gz = s.gz
        return (gx*gx + gy*gy + gz*gz).squareRoot()
    }
    features += magnitudeStats(gyroMag)

    return features  // 81 + 8 = 89
}

// ── CoreML inference ──────────────────────────────────────────────────────────

struct Prediction {
    let label: String
    let confidence: Double
    let aboveThreshold: Bool
    let top3: [(label: String, conf: Double)]
}

func runModel(model: MLModel, features: [Double], threshold: Double) throws -> Prediction {
    var featureDict: [String: MLFeatureValue] = [:]
    for (i, v) in features.enumerated() {
        featureDict["f\(i)"] = MLFeatureValue(double: v)
    }
    let provider = try MLDictionaryFeatureProvider(dictionary: featureDict)
    let output = try model.prediction(from: provider)

    // strokeType is int64 index into LABELS
    guard let labelVal = output.featureValue(for: "strokeType") else {
        throw NSError(domain: "classify", code: 1, userInfo: [NSLocalizedDescriptionKey: "No strokeType output"])
    }
    let labelIndex = Int(labelVal.int64Value)
    guard labelIndex >= 0, labelIndex < LABELS.count else {
        throw NSError(domain: "classify", code: 2, userInfo: [NSLocalizedDescriptionKey: "Label index out of range: \(labelIndex)"])
    }
    let label = LABELS[labelIndex]

    // classProbability has int64 keys.
    // Model outputs raw tree-vote sums (0–300 range), NOT normalized probabilities.
    // Normalize by dividing each class score by the total across all classes.
    var confidence = 0.0
    var allProbs: [(label: String, conf: Double)] = []
    if let probsVal = output.featureValue(for: "classProbability"),
       let probs = probsVal.dictionaryValue as? [NSNumber: Double] {
        let total = probs.values.reduce(0, +)
        let rawScore = probs[NSNumber(value: labelIndex)] ?? 0.0
        confidence = total > 0 ? rawScore / total : 0.0
        allProbs = probs.compactMap { (key, val) -> (String, Double)? in
            let idx = key.intValue
            guard idx >= 0, idx < LABELS.count else { return nil }
            return (LABELS[idx], total > 0 ? val / total : 0.0)
        }.sorted { $0.1 > $1.1 }
    }

    let top3 = Array(allProbs.prefix(3))
    let aboveThreshold = confidence >= threshold

    return Prediction(
        label: aboveThreshold ? label : "unknown",
        confidence: confidence,
        aboveThreshold: aboveThreshold,
        top3: top3
    )
}

// ── Main ──────────────────────────────────────────────────────────────────────

func main() {
    let args = parseArgs()

    // Load model — compile .mlmodel → .mlmodelc if needed
    print("Loading model: \(args.modelPath)")
    guard FileManager.default.fileExists(atPath: args.modelPath) else {
        print("ERROR: Model not found at \(args.modelPath)"); exit(1)
    }
    let modelURL = URL(fileURLWithPath: args.modelPath)
    let model: MLModel
    do {
        let config = MLModelConfiguration()
        config.computeUnits = .cpuOnly
        // Compile .mlmodel → temporary .mlmodelc (required before loading)
        print("  Compiling model...")
        let compiledURL = try MLModel.compileModel(at: modelURL)
        model = try MLModel(contentsOf: compiledURL, configuration: config)
    } catch {
        print("ERROR: Failed to load model: \(error)"); exit(1)
    }
    print("Model loaded ✓")

    // Load session CSV
    print("Loading session: \(args.sessionCSV)")
    let samples = loadCSV(path: args.sessionCSV)
    print("  \(samples.count) IMU samples loaded")
    guard !samples.isEmpty else { print("ERROR: No samples in CSV"); exit(1) }

    // Load JSON sidecar for markers
    let jsonPath = URL(fileURLWithPath: args.sessionCSV).deletingPathExtension().appendingPathExtension("json").path
    let markers = loadMarkers(jsonPath: jsonPath)
    print("  \(markers.count) spike markers found in JSON")

    let mirrorLabel = args.mirror == 1.0 ? "none (right/right)" : "X-axis flip (right/left)"
    print("  Axis normalization: \(mirrorLabel)")
    print("  Confidence threshold: \(args.threshold)")
    print("")

    // Process each spike marker
    var outputLines: [String] = ["timestamp_s,original_label,predicted_stroke,confidence,above_threshold,samples_in_window,top1,conf1,top2,conf2,top3,conf3"]
    var stats = [String: Int]()
    var belowThreshold = 0
    var noWindow = 0
    var errors = 0

    for (i, marker) in markers.enumerated() {
        if i % 100 == 0 {
            print("Processing spike \(i)/\(markers.count)...")
        }

        guard let peakIdx = findClosestIndex(samples: samples, ts: marker.ts) else {
            noWindow += 1; continue
        }

        guard let window = extractWindow(samples: samples, peakIdx: peakIdx) else {
            noWindow += 1; continue
        }

        let features = extract89Features(window: window, mirror: args.mirror)
        assert(features.count == 89, "Expected 89 features, got \(features.count)")

        do {
            let pred = try runModel(model: model, features: features, threshold: args.threshold)

            stats[pred.label, default: 0] += 1
            if !pred.aboveThreshold { belowThreshold += 1 }

            let top3Str = pred.top3.prefix(3).map { "\($0.label),\(String(format: "%.3f", $0.conf))" }.joined(separator: ",")
            outputLines.append(
                "\(String(format: "%.4f", marker.ts))," +
                "\(marker.originalLabel)," +
                "\(pred.label)," +
                "\(String(format: "%.4f", pred.confidence))," +
                "\(pred.aboveThreshold ? "yes" : "no")," +
                "\(window.count)," +
                "\(top3Str)"
            )
        } catch {
            errors += 1
            outputLines.append("\(String(format: "%.4f", marker.ts)),\(marker.originalLabel),ERROR,0.0,no,\(window.count),,,,,")
        }
    }

    // Write output
    let output = outputLines.joined(separator: "\n") + "\n"
    do {
        try output.write(toFile: args.outputPath, atomically: true, encoding: .utf8)
        print("\nResults written to: \(args.outputPath)")
    } catch {
        print("ERROR writing output: \(error)"); exit(1)
    }

    // Print summary
    print("")
    print("=== Classification Summary ===")
    print("Total spikes:       \(markers.count)")
    print("No window (edge):   \(noWindow)")
    print("Model errors:       \(errors)")
    print("Below threshold:    \(belowThreshold) (\(String(format: "%.1f", Double(belowThreshold) / Double(max(1, markers.count)) * 100))%)")
    print("")
    print("Predictions:")
    for (label, count) in stats.sorted(by: { $0.value > $1.value }) {
        let pct = String(format: "%.1f", Double(count) / Double(max(1, markers.count)) * 100)
        print("  \(label.padding(toLength: 22, withPad: " ", startingAt: 0)) \(count) (\(pct)%)")
    }
}

main()
