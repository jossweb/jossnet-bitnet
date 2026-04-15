//
//  MetalBridge.swift
//  jossnet-bitnet
//
//  Created by FIGUEIRAS Jossua on 19/02/2026.
//
import Metal


enum MissingBuffer: Error {
    case missingData
}

let urlW = path.appendingPathComponent("weights_W_q.bin")
let urlWk = path.appendingPathComponent("weights_W_k.bin")
let urlWv = path.appendingPathComponent("weights_W_v.bin")
let urlWo = path.appendingPathComponent("weights_W_o.bin")
let urlRMS = path.appendingPathComponent("weights_RMS_mlp.bin")
let urlDown = path.appendingPathComponent("weights_W_down.bin")
let urlUp = path.appendingPathComponent("weights_W_up.bin")
let urlGate = path.appendingPathComponent("weights_W_gate.bin")
let urlEmbeddings = path.appendingPathComponent("embeddings.bin")
let urlLMHead = path.appendingPathComponent("weights_lm_head.bin")
let urlRMSFinal = path.appendingPathComponent("weights_RMS_final.bin")


func QuantizeActivations(entry: MTLBuffer?, nbCols: Int, nbTokens: Int, outBufferQuantized: MTLBuffer, outBufferScales: MTLBuffer, commandBuffer: MTLCommandBuffer, pipeline: MTLComputePipelineState) {
    
    guard let entry = entry else { return }
    
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    
    var cols32 = UInt32(nbCols)
    
    encoder.setBuffer(entry, offset: 0, index: 0)
    
    encoder.setBuffer(outBufferQuantized, offset: 0, index: 1)
    encoder.setBuffer(outBufferScales, offset: 0, index: 2)
    
    encoder.setBytes(&cols32, length: MemoryLayout<UInt32>.size, index: 3)
    
    let gridSize = MTLSize(width: nbTokens, height: 1, depth: 1)
    let threadGroupSize = MTLSize(width: 32, height: 1, depth: 1)
    
    encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
    encoder.endEncoding()
}
func ComputeLogits(finalVectors: MTLBuffer?, embeddings: MTLBuffer?, nbTokens: Int, dim: Int, commandBuffer: MTLCommandBuffer, metal: MTLDevice, pipeline: MTLComputePipelineState, answer : MTLBuffer) throws{
    guard let finalVectors = finalVectors, let embeddings = embeddings else { throw MissingBuffer.missingData }
    
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    
    let vocabSize = 128256
    
    var dim32 = UInt32(dim)
    var lastTokenIdx = UInt32(nbTokens - 1)
    
    encoder.setBuffer(finalVectors, offset: 0, index: 0)
    encoder.setBuffer(embeddings, offset: 0, index: 1)
    encoder.setBuffer(answer, offset: 0, index: 2)
    encoder.setBytes(&dim32, length: MemoryLayout<UInt32>.size, index: 3)
    encoder.setBytes(&lastTokenIdx, length: MemoryLayout<UInt32>.size, index: 4)
    
    let gridSize = MTLSize(width: vocabSize, height: 1, depth: 1)
    let threadGroupSize = MTLSize(width: 32, height: 1, depth: 1)

    encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
    encoder.endEncoding()
}
func Embedding(tokens: [UInt32], colsEmbeddings: Int, embeddingsBuffer: MTLBuffer?, commandBuffer: MTLCommandBuffer, metal: MTLDevice, pipeline: MTLComputePipelineState, answer: MTLBuffer) {
    
    guard let embeddingsBuffer = embeddingsBuffer else { return }
    
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    
    let tokensInput = metal.makeBuffer(bytes: tokens, length: tokens.count * MemoryLayout<UInt32>.size, options: .storageModeShared)!
    
    var colsEmbeddings32 = UInt32(colsEmbeddings)
    
    encoder.setBuffer(embeddingsBuffer, offset: 0, index: 7)
    encoder.setBuffer(answer, offset: 0, index: 8)
    encoder.setBuffer(tokensInput, offset: 0, index: 9)
    encoder.setBytes(&colsEmbeddings32, length: MemoryLayout<UInt32>.size, index: 10)
    
    let embeddingsGridSize = MTLSize(width: colsEmbeddings, height: tokens.count, depth: 1)
    let embeddingsThreadGroupSize = MTLSize(width: 32, height: 1, depth: 1)

    encoder.dispatchThreads(embeddingsGridSize, threadsPerThreadgroup: embeddingsThreadGroupSize)
    encoder.endEncoding()
}
func ApplySiLUandMul(gate: MTLBuffer?, up : MTLBuffer?, size : Int, commandBuffer: MTLCommandBuffer, metal : MTLDevice, pipeline: MTLComputePipelineState) -> Void {
    guard let gate = gate, let up = up else { return }
    
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    
    encoder.setBuffer(gate, offset: 0, index: 0)
    encoder.setBuffer(up, offset: 0, index: 1)
    
    let gridSize = MTLSize(width: size, height: 1, depth: 1)
    let threadGroup = MTLSize(width: 32, height: 1, depth: 1)
    
    encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroup)
    encoder.endEncoding()
}
func ApplyRmsNorm(entry: MTLBuffer?, sizeEntry : Int, weightRMS : MTLBuffer?, tokenCount : Int, commandBuffer: MTLCommandBuffer, metal : MTLDevice, pipeline: MTLComputePipelineState, answer : MTLBuffer) throws{
    guard let entry, let weightRMS else {
        throw MissingBuffer.missingData
    }
    let encoderRms = commandBuffer.makeComputeCommandEncoder()!
    encoderRms.setComputePipelineState(pipeline);
    
    var cols = UInt32(sizeEntry)
    
    encoderRms.setBuffer(entry, offset: 0, index: 1);
    encoderRms.setBytes(&cols, length: MemoryLayout<UInt32>.size, index: 2)
    encoderRms.setBuffer(weightRMS, offset: 0, index: 3)
    encoderRms.setBuffer(answer, offset: 0, index: 4)

    let RmsGridSize = MTLSize(width: tokenCount, height: 1, depth: 1)
    let RmsThreadGroupSize = MTLSize(width: 1, height: 1, depth: 1)

    encoderRms.dispatchThreads(RmsGridSize, threadsPerThreadgroup: RmsThreadGroupSize)
    encoderRms.endEncoding()
}
func AddArray(mat1: MTLBuffer?, mat2: MTLBuffer?, size: Int, commandBuffer: MTLCommandBuffer, metal: MTLDevice, pipeline: MTLComputePipelineState, answer : MTLBuffer) throws{
    guard let mat1, let mat2 else { throw MissingBuffer.missingData }

    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    
    encoder.setBuffer(mat1, offset: 0, index: 0)
    encoder.setBuffer(mat2, offset: 0, index: 1)
    encoder.setBuffer(answer, offset: 0, index: 2) 
    
    let gridSize = MTLSize(width: size, height: 1, depth: 1)
    let threadGroup = MTLSize(width: 32, height: 1, depth: 1)
    
    encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroup)
    encoder.endEncoding()
}
func ComputeWeightedSum(mat: MTLBuffer?, v: MTLBuffer?, nbTokens: Int, commandBuffer: MTLCommandBuffer, metal : MTLDevice, pipeline: MTLComputePipelineState, answer : MTLBuffer) throws {
    guard let mat, let v else {
        throw MissingBuffer.missingData
    }
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    
    let nbHeads = 20
    let headDim = 128
    let ratio = 4
        
    var tokens32 = Int32(nbTokens)
    var heads32 = Int32(nbHeads)
    var ratio32 = Int32(ratio)
    
    encoder.setBuffer(mat, offset: 0, index: 0)
    encoder.setBuffer(v, offset: 0, index: 1)
    encoder.setBuffer(answer, offset: 0, index: 2)
    
    encoder.setBytes(&tokens32, length: 4, index: 3)
    encoder.setBytes(&heads32, length: 4, index: 4)
    encoder.setBytes(&ratio32, length: 4, index: 5)

    let gridSize = MTLSize(width: headDim, height: nbTokens, depth: nbHeads)
    
    let threadGroup = MTLSize(width: 32, height: 1, depth: 1)
    
    encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroup)
    encoder.endEncoding()

}
func DivideBySum(mat: MTLBuffer?, nbcols : Int, nbToken : Int, commandBuffer : MTLCommandBuffer, metal : MTLDevice, pipeline: MTLComputePipelineState, pipelineSumValue: MTLComputePipelineState, sumBuffer: MTLBuffer) throws{
    guard let mat else {
        throw MissingBuffer.missingData
    }
    
    try! GetSumByVector(mat: mat, nbcols: nbcols, nbToken: nbToken, commandBuffer: commandBuffer, metal: metal, pipeline: pipelineSumValue, answer : sumBuffer)
    
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline);
    
    var cols : UInt32 = UInt32(nbcols)
    
    encoder.setBuffer(mat, offset: 0, index: 0)
    encoder.setBytes(&cols, length: MemoryLayout<UInt32>.size, index: 1)
    encoder.setBuffer(sumBuffer, offset: 0, index: 2)
    
    let gridSize = MTLSize(width: nbcols, height: nbToken, depth: 1)
    let threadGroupSize = MTLSize(width: 1, height: 1, depth: 1)

    encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
    encoder.endEncoding()
    
}
func SubstractMax(mat: MTLBuffer?, nbcols : Int, nbToken : Int, commandBuffer : MTLCommandBuffer, metal : MTLDevice, pipeline: MTLComputePipelineState, pipelineGetMax: MTLComputePipelineState, maxBuffer : MTLBuffer) throws{
    guard let mat else {
        throw MissingBuffer.missingData
    }
    
    try! GetMaxByVector(mat: mat, nbcols: nbcols, nbToken: nbToken, commandBuffer: commandBuffer, metal: metal, pipeline: pipelineGetMax, answer : maxBuffer)

    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline);
    
    var cols : UInt32 = UInt32(nbcols)
    
    encoder.setBuffer(mat, offset: 0, index: 0)
    encoder.setBytes(&cols, length: MemoryLayout<UInt32>.size, index: 1)
    encoder.setBuffer(maxBuffer, offset: 0, index: 2)
    
    let gridSize = MTLSize(width: nbcols, height: nbToken, depth: 1)
    let threadGroupSize = MTLSize(width: 1, height: 1, depth: 1)

    encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
    encoder.endEncoding()
    
}
func GetSumByVector(mat: MTLBuffer?, nbcols : Int, nbToken : Int, commandBuffer : MTLCommandBuffer, metal : MTLDevice, pipeline: MTLComputePipelineState, answer : MTLBuffer) throws{
    guard let mat else {
        throw MissingBuffer.missingData
    }
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline);
    
    var cols : UInt32 = UInt32(nbcols)
    
    encoder.setBuffer(mat, offset: 0, index: 0)
    encoder.setBytes(&cols, length: MemoryLayout<Float>.size, index: 1)
    encoder.setBuffer(answer, offset: 0, index: 2)
    
    let gridSize = MTLSize(width: nbToken, height: 1, depth: 1)
    let threadGroupSize = MTLSize(width: 32, height: 1, depth: 1)

    encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
    encoder.endEncoding()
}
func GetMaxByVector(mat: MTLBuffer?, nbcols : Int, nbToken : Int, commandBuffer : MTLCommandBuffer, metal : MTLDevice, pipeline: MTLComputePipelineState, answer : MTLBuffer) throws{
    guard let mat else {
        throw MissingBuffer.missingData
    }
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline);
    
    var cols : UInt32 = UInt32(nbcols)
    
    encoder.setBuffer(mat, offset: 0, index: 0)
    encoder.setBytes(&cols, length: MemoryLayout<UInt32>.size, index: 1)
    encoder.setBuffer(answer, offset: 0, index: 2)
    
    let gridSize = MTLSize(width: nbToken, height: 1, depth: 1)
    let threadGroupSize = MTLSize(width: 32, height: 1, depth: 1)

    encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
    encoder.endEncoding()
}
func AttentionScore(buffer1 : MTLBuffer?, buffer2 : MTLBuffer?, nbToken : Int, commandBuffer : MTLCommandBuffer, metal : MTLDevice, pipeline: MTLComputePipelineState, answer : MTLBuffer) throws{
    guard let buffer1, let buffer2 else {
        throw MissingBuffer.missingData
    }
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline);
    
    let heads = 20
    var headsInt = Int32(heads)
    var nbTokenInt = Int32(nbToken)
    var ratioInt = Int32(4.0)
    
    encoder.setBuffer(buffer1, offset: 0, index: 0)
    encoder.setBuffer(buffer2, offset: 0, index: 1)
    encoder.setBuffer(answer, offset: 0, index: 2)
    encoder.setBytes(&headsInt, length: MemoryLayout<UInt32>.size, index: 3)
    encoder.setBytes(&nbTokenInt, length: MemoryLayout<UInt32>.size, index: 4)
    encoder.setBytes(&ratioInt, length: MemoryLayout<UInt32>.size, index: 5)
    
    let gridSize = MTLSize(width: nbToken, height: nbToken, depth: heads)
    let threadGroupSize = MTLSize(width: 32, height: 1, depth: 1)

    encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
    encoder.endEncoding()
}
func ApplyRoPE(buffer : MTLBuffer?, nbTokens : Int, dim : Int, commandBuffer : MTLCommandBuffer, metal : MTLDevice, pipeline: MTLComputePipelineState) throws {
    guard let buffer = buffer else {
        throw MissingBuffer.missingData
    }
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    
    var dim32 = UInt32(dim)
    
    encoder.setBuffer(buffer, offset: 0, index: 0)
    encoder.setBytes(&dim32, length: MemoryLayout<UInt32>.size, index: 1)
    
    let gridSize = MTLSize(width: nbTokens, height: dim / 2, depth: 1)
    let threadGroupSize = MTLSize(width: 32, height: 32, depth: 1)

    encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
    encoder.endEncoding()
}
func GetFromBuffer(buffer : MTLBuffer?, size: Int)->[Float]{
    guard let buffer = buffer else {
        return []
    }
    let rawPointer = buffer.contents()

    let floatPointer = rawPointer.bindMemory(to: Float.self, capacity: size)

    let bufferPointer = UnsafeBufferPointer(start: floatPointer, count: size)

    return Array(bufferPointer)
}

func MultByW(WSize: matrixInfos, inputSize: matrixInfos, bufferW: MTLBuffer, scaleW: MTLBuffer?, inputQuant: MTLBuffer?, scaleX: MTLBuffer?, commandBuffer: MTLCommandBuffer, metal: MTLDevice, pipeline: MTLComputePipelineState, answer: MTLBuffer) throws{
    
    guard let inputQuant = inputQuant, let scaleW = scaleW, let scaleX = scaleX else { throw MissingBuffer.missingData }
    
    let encoderMult = commandBuffer.makeComputeCommandEncoder()!
    encoderMult.setComputePipelineState(pipeline)
    
    let dims: [UInt32] = [UInt32(inputSize.nbColumn), UInt32(inputSize.nbLine), UInt32(WSize.nbLine)]
    
    encoderMult.setBuffer(inputQuant, offset: 0, index: 0)
    encoderMult.setBuffer(bufferW, offset: 0, index: 1)
    encoderMult.setBytes(dims, length: 12, index: 2)
    encoderMult.setBuffer(answer, offset: 0, index: 6)
    encoderMult.setBuffer(scaleW, offset: 0, index: 7)
    encoderMult.setBuffer(scaleX, offset: 0, index: 8)
    
    var scaleWSize = UInt32(scaleW.length / MemoryLayout<Float>.size)
    encoderMult.setBytes(&scaleWSize, length: MemoryLayout<UInt32>.size, index: 9)

    let MultGridSize = MTLSize(width: WSize.nbLine, height: inputSize.nbLine, depth: 1)
    let groupWidth = min(pipeline.threadExecutionWidth, WSize.nbLine)
    let MultThreadGroupSize = MTLSize(width: groupWidth, height: 1, depth: 1)

    encoderMult.dispatchThreads(MultGridSize, threadsPerThreadgroup: MultThreadGroupSize)
    encoderMult.endEncoding()
}
