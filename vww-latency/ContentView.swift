import SwiftUI
import CoreML

struct ModelResult: Identifiable {
    let id = UUID()
    let name: String
    let avgMs: Double
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
                        Text(String(format: "%.2f ms", r.avgMs))
                            .monospacedDigit()
                    }
                }
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

            // 100 iterations, you can tweak
            let iters = 500

            if let ms = bench.benchmarkBaseline(iterations: iters) {
                newResults.append(ModelResult(name: "Baseline", avgMs: ms))
            }
            if let ms = bench.benchmarkStudent(iterations: iters) {
                newResults.append(ModelResult(name: "Student", avgMs: ms))
            }
            if let ms = bench.benchmarkPruned(iterations: iters) {
                newResults.append(ModelResult(name: "Pruned", avgMs: ms))
            }
            if let ms = bench.benchmarkPrunedQuant(iterations: iters) {
                newResults.append(ModelResult(name: "Pruned (CoreML int8)", avgMs: ms))
            }

            DispatchQueue.main.async {
                self.results = newResults
                self.isRunning = false
                self.status = "Done. Measured \(iters) iterations per model."
            }
        }
    }
}
