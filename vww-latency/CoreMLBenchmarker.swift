import Foundation
import CoreML

final class CoreMLBenchmarker {

    struct Metrics {
        let p90Ms: Double
    }

    // keep models in memory
    private let baselineModel: baseline
    private let studentModel: student
    private let prunedModel: pruned
    private let quantizedModel: quantized

    init() {
        baselineModel = try! baseline(configuration: MLModelConfiguration())
        studentModel  = try! student(configuration: MLModelConfiguration())
        prunedModel   = try! pruned(configuration: MLModelConfiguration())
        quantizedModel = try! quantized(configuration: MLModelConfiguration())
    }

    private func makeRandomInput() -> MLMultiArray {
        // random input tensor of shape (nchw): 1x3x96x96
        let shape: [NSNumber] = [1, 3, 96, 96]
        let arr = try! MLMultiArray(shape: shape, dataType: .float32)

        // fill with random float32 values in [0, 1)
        let count = arr.count
        for i in 0..<count {
            arr[i] = NSNumber(value: Float.random(in: 0..<1))
        }
        return arr
    }

    private func makeRandomInputPool(count: Int) -> [MLMultiArray] {
        precondition(count > 0, "Pool size must be > 0")
        var pool: [MLMultiArray] = []
        pool.reserveCapacity(count)
        for _ in 0..<count {
            pool.append(makeRandomInput())
        }
        return pool
    }

    private func makeBaselineInput(_ arr: MLMultiArray) -> baselineInput {
        baselineInput(input: arr)
    }
    private func makeStudentInput(_ arr: MLMultiArray) -> studentInput {
        studentInput(input: arr)
    }
    private func makePrunedInput(_ arr: MLMultiArray) -> prunedInput {
        prunedInput(input: arr)
    }
    private func makeQuantizedInput(_ arr: MLMultiArray) -> quantizedInput {
        quantizedInput(input: arr)
    }

    private func timeModel<Input: MLFeatureProvider, Output>(
        _ call: (Input) throws -> Output,
        makeInput: (MLMultiArray) -> Input,
        iterations: Int,
        warmup: Int = 5,
        poolSize: Int = 128
    ) -> Metrics? {
        var times: [Double] = []
        times.reserveCapacity(iterations)

        // pre-generate a pool of random inputs and wrap them into Input providers (outside timed path)
        let inputPool: [Input] = makeRandomInputPool(count: poolSize).map { makeInput($0) }
        var idx = 0

        // warm up run (not timed)
        for _ in 0..<warmup {
            let warmInput = inputPool[idx % inputPool.count]
            idx += 1
            _ = try? call(warmInput)
        }
        
        // timed runs
        for _ in 0..<iterations {
            let inp = inputPool[idx % inputPool.count]
            idx += 1
            let start = CFAbsoluteTimeGetCurrent()
            _ = try? call(inp)
            let end = CFAbsoluteTimeGetCurrent()
            times.append(end - start)
        }
        guard !times.isEmpty else { return nil }
        // Compute p90 (ms)
        let sorted = times.sorted()
        let rank = Int(ceil(0.90 * Double(sorted.count)))
        let index = max(0, min(sorted.count - 1, rank - 1))
        let p90 = sorted[index] * 1000.0
        return Metrics(p90Ms: p90)
    }

    func benchmarkBaseline(iterations: Int) -> Metrics? {
        return timeModel(
            { try baselineModel.prediction(input: $0) },
            makeInput: { self.makeBaselineInput($0) },
            iterations: iterations
        )
    }

    func benchmarkStudent(iterations: Int) -> Metrics? {
        return timeModel(
            { try studentModel.prediction(input: $0) },
            makeInput: { self.makeStudentInput($0) },
            iterations: iterations
        )
    }

    func benchmarkPruned(iterations: Int) -> Metrics? {
        return timeModel(
            { try prunedModel.prediction(input: $0) },
            makeInput: { self.makePrunedInput($0) },
            iterations: iterations
        )
    }
    
    func benchmarkQuantized(iterations: Int) -> Metrics? {
        return timeModel(
            { try quantizedModel.prediction(input: $0) },
            makeInput: { self.makeQuantizedInput($0) },
            iterations: iterations
        )
    }
}
