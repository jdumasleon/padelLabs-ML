#!/usr/bin/env swift
//
// train_createml.swift — PadelLabs ML Toolchain
// ===============================================
// macOS command-line script to train a Create ML Activity Classifier from
// labeled stroke-window folders — no Xcode GUI required.
//
// Requirements:
//   macOS 12+ (Monterey or later)
//   Xcode Command Line Tools installed
//
// Usage:
//   swift train_createml.swift \
//       --train      ../labeled-strokes/train \
//       --validation ../labeled-strokes/validation \
//       --output     ../models/v1/PadelStrokeClassifier_v1.mlmodel \
//       [--iterations 30] \
//       [--window-size 100]
//
// The training folder must follow Create ML's expected structure:
//   train/
//     smash/       smash_00001.csv, smash_00002.csv …
//     forehand/    …
//     unknown/     …
//
// Each CSV must have exactly 100 rows and these columns:
//   timestamp, accelX, accelY, accelZ, gyroX, gyroY, gyroZ, roll, pitch, yaw
//
// Output:
//   - .mlmodel file at --output path
//   - metrics printed to stdout (training accuracy, validation accuracy)
//   - metrics JSON saved to same directory as the model
//

import Foundation
import CreateML

// ── Argument parsing ──────────────────────────────────────────────────────────

struct Args {
    var trainPath: String = ""
    var validationPath: String = ""
    var outputPath: String = ""
    var testPath: String = ""
    var iterations: Int = 30
    var windowSize: Int = 100
}

func parseArgs() -> Args {
    var args = Args()
    let argv = CommandLine.arguments.dropFirst()
    var i = argv.startIndex

    while i < argv.endIndex {
        switch argv[i] {
        case "--train":
            i = argv.index(after: i)
            args.trainPath = argv[i]
        case "--validation":
            i = argv.index(after: i)
            args.validationPath = argv[i]
        case "--output":
            i = argv.index(after: i)
            args.outputPath = argv[i]
        case "--test":
            i = argv.index(after: i)
            args.testPath = argv[i]
        case "--iterations":
            i = argv.index(after: i)
            args.iterations = Int(argv[i]) ?? 30
        case "--window-size":
            i = argv.index(after: i)
            args.windowSize = Int(argv[i]) ?? 100
        case "--help", "-h":
            printHelp()
            exit(0)
        default:
            print("Unknown argument: \(argv[i])", to: &standardError)
            exit(1)
        }
        i = argv.index(after: i)
    }

    if args.trainPath.isEmpty || args.validationPath.isEmpty || args.outputPath.isEmpty {
        print("ERROR: --train, --validation, and --output are required.", to: &standardError)
        printHelp()
        exit(1)
    }

    return args
}

func printHelp() {
    print("""
    Usage: swift train_createml.swift \\
        --train      <path to labeled-strokes/train> \\
        --validation <path to labeled-strokes/validation> \\
        --output     <path to output .mlmodel> \\
        [--iterations 30] \\
        [--window-size 100]
    """)
}

// ── stderr writer ─────────────────────────────────────────────────────────────

var standardError = FileHandle.standardError

extension FileHandle: @retroactive TextOutputStream {
    public func write(_ string: String) {
        let data = Data(string.utf8)
        self.write(data)
    }
}

// ── Training ──────────────────────────────────────────────────────────────────

func run() throws {
    let args = parseArgs()

    let trainURL      = URL(fileURLWithPath: args.trainPath).standardizedFileURL
    let validationURL = URL(fileURLWithPath: args.validationPath).standardizedFileURL
    let outputURL     = URL(fileURLWithPath: args.outputPath).standardizedFileURL

    // Validate paths
    var isDir: ObjCBool = false
    guard FileManager.default.fileExists(atPath: trainURL.path, isDirectory: &isDir), isDir.boolValue else {
        print("ERROR: training folder not found: \(trainURL.path)", to: &standardError)
        exit(1)
    }
    guard FileManager.default.fileExists(atPath: validationURL.path, isDirectory: &isDir), isDir.boolValue else {
        print("ERROR: validation folder not found: \(validationURL.path)", to: &standardError)
        exit(1)
    }

    // Count training samples
    let fm = FileManager.default
    let classDirs = try fm.contentsOfDirectory(at: trainURL, includingPropertiesForKeys: [.isDirectoryKey])
        .filter { (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true }

    print("Training data: \(trainURL.path)")
    print("Classes found:")
    var totalSamples = 0
    for classDir in classDirs.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
        let csvFiles = (try? fm.contentsOfDirectory(at: classDir, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "csv" }) ?? []
        print("  \(classDir.lastPathComponent): \(csvFiles.count) windows")
        totalSamples += csvFiles.count
    }
    print("Total training windows: \(totalSamples)")
    print("")

    // Feature columns (no timestamp — Create ML uses row order, not absolute time)
    let featureColumns = ["accelX", "accelY", "accelZ", "gyroX", "gyroY", "gyroZ", "roll", "pitch", "yaw"]

    print("Configuring Create ML Activity Classifier...")
    print("  Feature columns : \(featureColumns.joined(separator: ", "))")
    print("  Window size     : \(args.windowSize) samples (1.0s @ 100 Hz)")
    print("  Max iterations  : \(args.iterations)")
    print("")

    let parameters = MLActivityClassifier.ModelParameters(
        validation: .dataSource(.labeledFiles(at: validationURL)),
        maximumIterations: args.iterations,
        predictionWindowSize: args.windowSize
    )

    print("Training… (this may take several minutes)")
    let startTime = Date()

    let classifier = try MLActivityClassifier(
        trainingData: .labeledFiles(at: trainURL),
        featureColumns: featureColumns,
        labelColumn: "label",         // label comes from the folder name in Create ML
        recordingFileColumn: nil,
        parameters: parameters
    )

    let elapsed = Date().timeIntervalSince(startTime)
    print("\nTraining complete in \(String(format: "%.1f", elapsed))s")

    // ── Metrics ───────────────────────────────────────────────────────────────

    let trainMetrics = classifier.trainingMetrics
    let valMetrics   = classifier.validationMetrics

    print("")
    print("=" * 50)
    print("  Training accuracy  : \(String(format: "%.1f%%", (1 - trainMetrics.classificationError) * 100))")
    print("  Validation accuracy: \(String(format: "%.1f%%", (1 - valMetrics.classificationError) * 100))")
    print("=" * 50)

    let targetAcc = 0.80
    let validationAcc = 1 - valMetrics.classificationError
    if validationAcc >= 0.88 {
        print("  ✅ PASS — meets Sprint 9 target (≥ 88%)")
    } else if validationAcc >= targetAcc {
        print("  ⚠️  PARTIAL — meets Sprint 8 baseline (≥ 80%) but not production target")
    } else {
        print("  ❌ FAIL — below \(Int(targetAcc * 100))% minimum. Collect more data or review labels.")
    }
    print("")

    // ── Held-out test evaluation (apples-to-apples vs the RF) ──────────────────

    if !args.testPath.isEmpty {
        let testURL = URL(fileURLWithPath: args.testPath).standardizedFileURL
        var tIsDir: ObjCBool = false
        if fm.fileExists(atPath: testURL.path, isDirectory: &tIsDir), tIsDir.boolValue {
            print("\nEvaluating on held-out test: \(testURL.path)")
            let testMetrics = classifier.evaluation(on: .labeledFiles(at: testURL))
            let testAcc = 1 - testMetrics.classificationError
            print("=" * 50)
            print("  HELD-OUT TEST accuracy: \(String(format: "%.1f%%", testAcc * 100))")
            print("=" * 50)
            print(testMetrics.description)
        } else {
            print("WARN: --test folder not found, skipping: \(testURL.path)", to: &standardError)
        }
    }

    // ── Save model ────────────────────────────────────────────────────────────

    try fm.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)

    let metadata = MLModelMetadata(
        author: "PadelLabs",
        shortDescription: "Padel stroke classifier — 12 classes (100Hz, 1.0s window)",
        license: "Proprietary — PadelLabs",
        version: "1.0",
        additional: [
            "featureColumns": featureColumns.joined(separator: ","),
            "windowSizeSamples": "\(args.windowSize)",
            "samplingHz": "100",
            "classes": "smash,vibora,bandeja,rulo,forehand,backhand,forehandLob,backhandLob,forehandVolley,backhandVolley,serve,unknown",
            "trainingSamples": "\(totalSamples)",
            "validationAccuracy": String(format: "%.4f", validationAcc),
        ]
    )

    try classifier.write(to: outputURL, metadata: metadata)
    print("Model saved to: \(outputURL.path)")

    // ── Save metrics JSON ─────────────────────────────────────────────────────

    let metricsDict: [String: Any] = [
        "training_accuracy": 1 - trainMetrics.classificationError,
        "validation_accuracy": validationAcc,
        "training_samples": totalSamples,
        "iterations": args.iterations,
        "window_size": args.windowSize,
        "feature_columns": featureColumns,
        "model_path": outputURL.path,
        "created_at": ISO8601DateFormatter().string(from: Date()),
    ]

    let metricsURL = outputURL.deletingLastPathComponent()
        .appendingPathComponent("metrics_\(outputURL.deletingPathExtension().lastPathComponent).json")
    let metricsData = try JSONSerialization.data(withJSONObject: metricsDict, options: [.prettyPrinted, .sortedKeys])
    try metricsData.write(to: metricsURL)
    print("Metrics saved to: \(metricsURL.path)")
    print("")
    print("Next step: run evaluate.py against the test split to get the final held-out score.")
    print("  python3 evaluate.py \\")
    print("      --test   ../labeled-strokes/test \\")
    print("      --model  <path to .pkl> (or use --baseline for sanity check)")
}

// ── String repeat helper ──────────────────────────────────────────────────────

extension String {
    static func * (lhs: String, rhs: Int) -> String {
        String(repeating: lhs, count: rhs)
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

do {
    try run()
} catch {
    print("ERROR: \(error.localizedDescription)", to: &standardError)
    if let mlError = error as? MLCreateError {
        print("  CreateML detail: \(mlError)", to: &standardError)
    }
    exit(1)
}
