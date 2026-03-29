// VWW iOS Benchmarking App
//
// Supporting repository for the MSc dissertation project: "Building Lightweight Neural Networks" by Thomas Morris. 
// This repository contains the iOS benchmarking harness for the deployment and evaluation component of the project.
//
// Usage:
//
// 1. Model Preparation
//    - Export Core ML models (`baseline`, `student`, `pruned`, `quantized`) from the main pipeline repository as `.mlpackage` files.
//    - Add these models to the app bundle in Xcode.
//
// 2. Running the App
//    - Build and install the app on your iPhone.
//    - Tap "Run Benchmarks" to execute the protocol: the app will display p90 latency for each model variant after each run.
//
// Project Structure
// - CoreMLBenchmarker.swift: Core benchmarking logic and model runner
// - ContentView.swift: SwiftUI interface for running and displaying benchmarks
// - (Other supporting files: app entry point, assets, etc.)
//
// Relation to Main Repository
// Training, compression, and Core ML export scripts are in a separate repository referenced in the dissertation. This app is for deployment and benchmarking only.
//
