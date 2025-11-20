import Foundation
import CoreML

final class CoreMLBenchmarker {

    // keep models in memory
    private let baselineModel: baseline_fp32
    private let studentModel: student_fp32
    private let prunedModel: pruned_fp32
    private let prunedQuantModel: pruned_int8

    init() {
        baselineModel = try! baseline_fp32(configuration: MLModelConfiguration())
        studentModel  = try! student_fp32(configuration: MLModelConfiguration())
        prunedModel   = try! pruned_fp32(configuration: MLModelConfiguration())
        prunedQuantModel = try! pruned_int8(configuration: MLModelConfiguration())
    }

    private func makeRandomInput() -> MLMultiArray {
        // dummy input tensor of shape (nchw): 1x3x96x96
        let shape: [NSNumber] = [1, 3, 96, 96]
        let arr = try! MLMultiArray(shape: shape, dataType: .float32)

        // fill it with zeros - doesn't make a difference for latency test
        let count = arr.count
        for i in 0..<count {
            arr[i] = 0.0
        }
        return arr
    }

    private func baselineInput(_ arr: MLMultiArray) -> baseline_fp32Input {
        baseline_fp32Input(input: arr)
    }

    private func studentInput(_ arr: MLMultiArray) -> student_fp32Input {
        student_fp32Input(input: arr)
    }

    private func prunedInput(_ arr: MLMultiArray) -> pruned_fp32Input {
        pruned_fp32Input(input: arr)
    }
    
    private func prunedQuantInput(_ arr: MLMultiArray) -> pruned_int8Input {
            pruned_int8Input(input: arr)
    }

    private func timeModel<Input: MLFeatureProvider, Output>(
        _ call: (Input) throws -> Output,
        makeInput: () -> Input,
        iterations: Int,
        warmup: Int = 5
    ) -> Double? {
        var times: [Double] = []
        times.reserveCapacity(iterations)

        // warm up run
        for _ in 0..<warmup {
            let warmInput = makeInput()
            _ = try? call(warmInput)
        }
        
        // timed runs
        for _ in 0..<iterations {
            let inp = makeInput()
            let start = CFAbsoluteTimeGetCurrent()
            _ = try? call(inp)
            let end = CFAbsoluteTimeGetCurrent()
            times.append(end - start)
        }
        guard !times.isEmpty else { return nil }
        let avg = times.reduce(0, +) / Double(times.count)
        return avg * 1000.0  // ms
    }

    func benchmarkBaseline(iterations: Int) -> Double? {
        let arr = makeRandomInput()
        return timeModel(
            { try baselineModel.prediction(input: $0) },
            makeInput: { baselineInput(arr) },
            iterations: iterations
        )
    }

    func benchmarkStudent(iterations: Int) -> Double? {
        let arr = makeRandomInput()
        return timeModel(
            { try studentModel.prediction(input: $0) },
            makeInput: { studentInput(arr) },
            iterations: iterations
        )
    }

    func benchmarkPruned(iterations: Int) -> Double? {
        let arr = makeRandomInput()
        return timeModel(
            { try prunedModel.prediction(input: $0) },
            makeInput: { prunedInput(arr) },
            iterations: iterations
        )
    }
    
    func benchmarkPrunedQuant(iterations: Int) -> Double? {
            let arr = makeRandomInput()
            return timeModel(
                { try prunedQuantModel.prediction(input: $0) },
                makeInput: { prunedQuantInput(arr) },
                iterations: iterations
            )
    }
}
