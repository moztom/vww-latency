import SwiftUI
import CoreML

struct ModelResult: Identifiable {
    let id = UUID()
    let name: String
    let p90Ms: Double
}

struct ContentView: View {
    @State private var results: [ModelResult] = []
    @State private var isRunning = false
    @State private var status = "Ready"

    var body: some View {
        NavigationView {
            VStack(spacing: 16) {
                Button(action: runBenchmarks) {
                    Text(isRunning ? "Running..." : "Run Benchmarks")
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(isRunning ? Color.gray : Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(12)
                }
                .disabled(isRunning)

                Text(status)
                    .font(.footnote)
                    .foregroundColor(.secondary)

                List(results) { r in
                    HStack {
                        Text(r.name)
                        Spacer()
                        Text(String(format: "%.2f ms", r.p90Ms))
                            .monospacedDigit()
                    }
                }
                Text("Latency values are p90 (90th percentile)")
                    .font(.footnote)
                    .foregroundColor(.secondary)
            }
            .padding()
            .navigationTitle("VWW Latency Bench")
        }
    }

    func runBenchmarks() {
        guard !isRunning else { return }
        isRunning = true
        status = "Running on-device benchmarks..."

        DispatchQueue.global(qos: .userInitiated).async {
            let bench = CoreMLBenchmarker()

            var newResults: [ModelResult] = []

            let iters = 1024

            if let m = bench.benchmarkBaseline(iterations: iters) {
                newResults.append(ModelResult(name: "Baseline", p90Ms: m.p90Ms))
            }
            if let m = bench.benchmarkStudent(iterations: iters) {
                newResults.append(ModelResult(name: "Student", p90Ms: m.p90Ms))
            }
            if let m = bench.benchmarkPruned(iterations: iters) {
                newResults.append(ModelResult(name: "Pruned", p90Ms: m.p90Ms))
            }
            if let m = bench.benchmarkQuantized(iterations: iters) {
                newResults.append(ModelResult(name: "Quantized", p90Ms: m.p90Ms))
            }

            DispatchQueue.main.async {
                self.results = newResults
                self.isRunning = false
                self.status = "Done. Measured \(iters) iterations per model."
            }
        }
    }
}
