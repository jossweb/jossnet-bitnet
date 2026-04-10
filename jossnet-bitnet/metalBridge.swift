//
//  MetalBridge.swift
//  jossnet-bitnet
//
//  Created by FIGUEIRAS Jossua on 19/02/2026.
//
import Metal



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


func QuantizeActivations(entry: MTLBuffer?, nbCols: Int, nbTokens: Int, commandBuffer: MTLCommandBuffer, metal: MTLDevice, pipeline: MTLComputePipelineState) -> (quantized: MTLBuffer, scales: MTLBuffer)? {
    
    guard let entry = entry else { return nil }
    
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    
    let bufferXQuant = metal.makeBuffer(length: nbTokens * nbCols * MemoryLayout<Int8>.size, options: .storageModeShared)!
    
    let bufferScalesX = metal.makeBuffer(length: nbTokens * MemoryLayout<Float>.size, options: .storageModeShared)!
    
    var cols32 = UInt32(nbCols)
    
    encoder.setBuffer(entry, offset: 0, index: 0)
    encoder.setBuffer(bufferXQuant, offset: 0, index: 1)
    encoder.setBuffer(bufferScalesX, offset: 0, index: 2)
    encoder.setBytes(&cols32, length: MemoryLayout<UInt32>.size, index: 3)
    
    let gridSize = MTLSize(width: nbTokens, height: 1, depth: 1)
    let threadGroupSize = MTLSize(width: 32, height: 1, depth: 1)
    
    encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
    encoder.endEncoding()
    
    return (bufferXQuant, bufferScalesX)
}
func ComputeLogits(finalVectors: MTLBuffer?, embeddings: MTLBuffer?, nbTokens: Int, dim: Int, commandBuffer: MTLCommandBuffer, metal: MTLDevice, pipeline: MTLComputePipelineState) -> MTLBuffer? {
    guard let finalVectors = finalVectors, let embeddings = embeddings else { return nil }
    
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    
    let vocabSize = 128256
    let logitsBuffer = metal.makeBuffer(length: vocabSize * MemoryLayout<Float>.size, options: .storageModeShared)!
    
    var dim32 = UInt32(dim)
    var lastTokenIdx = UInt32(nbTokens - 1)
    
    encoder.setBuffer(finalVectors, offset: 0, index: 0)
    encoder.setBuffer(embeddings, offset: 0, index: 1)
    encoder.setBuffer(logitsBuffer, offset: 0, index: 2)
    encoder.setBytes(&dim32, length: MemoryLayout<UInt32>.size, index: 3)
    encoder.setBytes(&lastTokenIdx, length: MemoryLayout<UInt32>.size, index: 4)
    
    let gridSize = MTLSize(width: vocabSize, height: 1, depth: 1)
    let threadGroupSize = MTLSize(width: 32, height: 1, depth: 1)

    encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
    encoder.endEncoding()
    
    return logitsBuffer
}
func Embedding(tokens: [UInt32], colsEmbeddings : Int, embeddingsBuffer: MTLBuffer?, commandBuffer: MTLCommandBuffer, metal: MTLDevice, pipeline: MTLComputePipelineState )->MTLBuffer?{
    
    guard let embeddingsBuffer else { return nil}
    
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline);
    
    let embeddingsResponseBuffer = metal.makeBuffer(length: (colsEmbeddings * MemoryLayout<Float>.size) * tokens.count, options: .storageModeShared)!
    let tokensInput = metal.makeBuffer(bytes: tokens, length: tokens.count * MemoryLayout<UInt32>.size, options: .storageModeShared)!

    var colsEmbeddings = colsEmbeddings
    let bufferNbColsEmbeddings = metal.makeBuffer(bytes: &colsEmbeddings, length: MemoryLayout<UInt32>.size, options: .storageModeShared)!
    
    encoder.setBuffer(embeddingsBuffer, offset: 0, index: 7)
    encoder.setBuffer(embeddingsResponseBuffer, offset: 0, index: 8)
    encoder.setBuffer(tokensInput, offset: 0, index: 9)
    encoder.setBuffer(bufferNbColsEmbeddings, offset: 0, index: 10)

    let embeddingsGridSize = MTLSize(width: colsEmbeddings, height: tokens.count, depth: 1)
    let embeddingsThreadGroupSize = MTLSize(width: 32, height: 1, depth: 1)

    encoder.dispatchThreads(embeddingsGridSize, threadsPerThreadgroup: embeddingsThreadGroupSize)
    encoder.endEncoding()
    
    return embeddingsResponseBuffer
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
func ApplyRmsNorm(entry: MTLBuffer?, sizeEntry : Int, weightRMS : MTLBuffer?, tokenCount : Int, commandBuffer: MTLCommandBuffer, metal : MTLDevice, pipeline: MTLComputePipelineState)->MTLBuffer?{
    guard let entry, let weightRMS else {
        return nil
    }
    let encoderRms = commandBuffer.makeComputeCommandEncoder()!
    encoderRms.setComputePipelineState(pipeline);
    
    let bufferAnswer = metal.makeBuffer(length: tokenCount * sizeEntry * MemoryLayout<Float>.size, options: .storageModeShared)!
    
    
    var cols = UInt32(sizeEntry)
    
    encoderRms.setBuffer(entry, offset: 0, index: 1);
    encoderRms.setBytes(&cols, length: MemoryLayout<UInt32>.size, index: 2)
    encoderRms.setBuffer(weightRMS, offset: 0, index: 3)
    encoderRms.setBuffer(bufferAnswer, offset: 0, index: 4)

    let RmsGridSize = MTLSize(width: tokenCount, height: 1, depth: 1)
    let RmsThreadGroupSize = MTLSize(width: 1, height: 1, depth: 1)

    encoderRms.dispatchThreads(RmsGridSize, threadsPerThreadgroup: RmsThreadGroupSize)
    encoderRms.endEncoding()
    
    return bufferAnswer
}
func AddArray(mat1: MTLBuffer?, mat2: MTLBuffer?, size: Int, commandBuffer: MTLCommandBuffer, metal: MTLDevice, pipeline: MTLComputePipelineState) -> MTLBuffer? {
    guard let mat1, let mat2 else { return nil }

    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    
    let mat3 = metal.makeBuffer(length: size * MemoryLayout<Float>.size, options: .storageModeShared)!
    
    encoder.setBuffer(mat1, offset: 0, index: 0)
    encoder.setBuffer(mat2, offset: 0, index: 1)
    encoder.setBuffer(mat3, offset: 0, index: 2) 
    
    let gridSize = MTLSize(width: size, height: 1, depth: 1)
    let threadGroup = MTLSize(width: 32, height: 1, depth: 1)
    
    encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroup)
    encoder.endEncoding()
    
    return mat3
}
func ComputeWeightedSum(mat: MTLBuffer?, v: MTLBuffer?, nbTokens: Int, commandBuffer: MTLCommandBuffer, metal : MTLDevice, pipeline: MTLComputePipelineState) -> MTLBuffer? {
    guard let mat, let v else {
        return nil
    }
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    
    let nbHeads = 20
    let headDim = 128
    let ratio = 4
    
    let embedDim = nbHeads * headDim
    let outputSize = nbTokens * embedDim * MemoryLayout<Float>.size
    
    let bufferOutput = metal.makeBuffer(length: outputSize, options: .storageModeShared)!
    
    var tokens32 = Int32(nbTokens)
    var heads32 = Int32(nbHeads)
    var ratio32 = Int32(ratio)
    
    encoder.setBuffer(mat, offset: 0, index: 0)
    encoder.setBuffer(v, offset: 0, index: 1)
    encoder.setBuffer(bufferOutput, offset: 0, index: 2)
    
    encoder.setBytes(&tokens32, length: 4, index: 3)
    encoder.setBytes(&heads32, length: 4, index: 4)
    encoder.setBytes(&ratio32, length: 4, index: 5)

    let gridSize = MTLSize(width: headDim, height: nbTokens, depth: nbHeads)
    
    let threadGroup = MTLSize(width: 32, height: 1, depth: 1)
    
    encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroup)
    encoder.endEncoding()
    
    return bufferOutput
}
func DivideBySum(mat: MTLBuffer?, nbcols : Int, nbToken : Int, commandBuffer : MTLCommandBuffer, metal : MTLDevice, pipeline: MTLComputePipelineState, pipelineSumValue: MTLComputePipelineState)->MTLBuffer?{
    guard let mat else {
        return nil
    }
    
    let bufferSumValues = GetSumByVector(mat: mat, nbcols: nbcols, nbToken: nbToken, commandBuffer: commandBuffer, metal: metal, pipeline: pipelineSumValue)
    
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline);
    
    var cols : UInt32 = UInt32(nbcols)
    
    let buffernbCols = metal.makeBuffer(bytes: &cols, length: MemoryLayout<UInt32>.size, options: .storageModeShared)!
    
    encoder.setBuffer(mat, offset: 0, index: 0)
    encoder.setBuffer(buffernbCols, offset: 0, index: 1)
    encoder.setBuffer(bufferSumValues, offset: 0, index: 2)
    
    let gridSize = MTLSize(width: nbcols, height: nbToken, depth: 1)
    let threadGroupSize = MTLSize(width: 1, height: 1, depth: 1)

    encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
    encoder.endEncoding()
    
    return mat;
    
}
func SubstractMax(mat: MTLBuffer?, nbcols : Int, nbToken : Int, commandBuffer : MTLCommandBuffer, metal : MTLDevice, pipeline: MTLComputePipelineState, pipelineGetMax: MTLComputePipelineState)->MTLBuffer?{
    guard let mat else {
        return nil
    }
    
    
    let bufferMaxValues = GetMaxByVector(mat: mat, nbcols: nbcols, nbToken: nbToken, commandBuffer: commandBuffer, metal: metal, pipeline: pipelineGetMax)

    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline);
    
    var cols : UInt32 = UInt32(nbcols)
    
    let buffernbCols = metal.makeBuffer(bytes: &cols, length: MemoryLayout<UInt32>.size, options: .storageModeShared)!
    
    encoder.setBuffer(mat, offset: 0, index: 0)
    encoder.setBuffer(buffernbCols, offset: 0, index: 1)
    encoder.setBuffer(bufferMaxValues, offset: 0, index: 2)
    
    let gridSize = MTLSize(width: nbcols, height: nbToken, depth: 1)
    let threadGroupSize = MTLSize(width: 1, height: 1, depth: 1)

    encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
    encoder.endEncoding()
    
    return mat;
    
}
func GetSumByVector(mat: MTLBuffer?, nbcols : Int, nbToken : Int, commandBuffer : MTLCommandBuffer, metal : MTLDevice, pipeline: MTLComputePipelineState)->MTLBuffer?{
    guard let mat else {
        return nil
    }
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline);
    
    var cols : UInt32 = UInt32(nbcols)
    
    
    let buffernbCols = metal.makeBuffer(bytes: &cols, length: MemoryLayout<UInt32>.size, options: .storageModeShared)!
    let bufferSumValues = metal.makeBuffer(length: MemoryLayout<Float>.size * nbToken, options : .storageModeShared)!
    
    encoder.setBuffer(mat, offset: 0, index: 0)
    encoder.setBuffer(buffernbCols, offset: 0, index: 1)
    encoder.setBuffer(bufferSumValues, offset: 0, index: 2)
    
    let gridSize = MTLSize(width: nbToken, height: 1, depth: 1)
    let threadGroupSize = MTLSize(width: 32, height: 1, depth: 1)

    encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
    encoder.endEncoding()
    
    return bufferSumValues;
}
func GetMaxByVector(mat: MTLBuffer?, nbcols : Int, nbToken : Int, commandBuffer : MTLCommandBuffer, metal : MTLDevice, pipeline: MTLComputePipelineState)->MTLBuffer?{
    guard let mat else {
        return nil
    }
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline);
    
    var cols : UInt32 = UInt32(nbcols)
    
    
    let buffernbCols = metal.makeBuffer(bytes: &cols, length: MemoryLayout<UInt32>.size, options: .storageModeShared)!
    let bufferMaxValues = metal.makeBuffer(length: MemoryLayout<Float>.size * nbToken, options : .storageModeShared)!
    
    encoder.setBuffer(mat, offset: 0, index: 0)
    encoder.setBuffer(buffernbCols, offset: 0, index: 1)
    encoder.setBuffer(bufferMaxValues, offset: 0, index: 2)
    
    let gridSize = MTLSize(width: nbToken, height: 1, depth: 1)
    let threadGroupSize = MTLSize(width: 32, height: 1, depth: 1)

    encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
    encoder.endEncoding()
    
    return bufferMaxValues;
}
func AttentionScore(buffer1 : MTLBuffer?, buffer2 : MTLBuffer?, nbToken : Int, commandBuffer : MTLCommandBuffer, metal : MTLDevice, pipeline: MTLComputePipelineState)->MTLBuffer?{
    guard let buffer1, let buffer2 else {
        return nil
    }
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline);

    let bufferAns = metal.makeBuffer(length: MemoryLayout<Float>.size * 20 * nbToken * nbToken, options: .storageModeShared)!
    
    let heads = 20
    var headsInt = Int32(heads)
    var nbTokenInt = Int32(nbToken)
    var ratioInt = Int32(4.0)
    
    let bufferHeadQ = metal.makeBuffer(bytes: &headsInt, length: MemoryLayout<Int32>.size, options: .storageModeShared)!
    let buffernbToken = metal.makeBuffer(bytes: &nbTokenInt, length: MemoryLayout<Int32>.size, options: .storageModeShared)!
    let bufferRatioInt = metal.makeBuffer(bytes : &ratioInt, length: MemoryLayout<Int32>.size, options: .storageModeShared)!
    
    encoder.setBuffer(buffer1, offset: 0, index: 0)
    encoder.setBuffer(buffer2, offset: 0, index: 1)
    encoder.setBuffer(bufferAns, offset: 0, index: 2)
    encoder.setBuffer(bufferHeadQ, offset: 0, index: 3)
    encoder.setBuffer(buffernbToken, offset: 0, index: 4)
    encoder.setBuffer(bufferRatioInt, offset: 0, index: 5)
    
    let gridSize = MTLSize(width: nbToken, height: nbToken, depth: heads)
    let threadGroupSize = MTLSize(width: 32, height: 1, depth: 1)

    encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
    encoder.endEncoding()
    
    return bufferAns;
}
func ApplyRoPE(buffer : MTLBuffer?, nbTokens : Int, dim : Int, commandBuffer : MTLCommandBuffer, metal : MTLDevice, pipeline: MTLComputePipelineState) -> MTLBuffer? {
    guard let buffer = buffer else {
        return nil
    }
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    
    var dim32 = UInt32(dim)
    let bufferDim = metal.makeBuffer(bytes: &dim32, length: MemoryLayout<UInt32>.size, options: .storageModeShared)!
    
    encoder.setBuffer(buffer, offset: 0, index: 0)
    encoder.setBuffer(bufferDim, offset: 0, index: 1)
    
    let gridSize = MTLSize(width: nbTokens, height: dim / 2, depth: 1)
    let threadGroupSize = MTLSize(width: 32, height: 32, depth: 1)

    encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
    encoder.endEncoding()
    
    return buffer
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

func MultByW(WSize: matrixInfos, inputSize: matrixInfos, bufferW: MTLBuffer, scaleW: MTLBuffer?, inputQuant: MTLBuffer?, scaleX: MTLBuffer?, commandBuffer: MTLCommandBuffer, metal: MTLDevice, pipeline: MTLComputePipelineState) -> MTLBuffer? {
    
    guard let inputQuant = inputQuant, let scaleW = scaleW, let scaleX = scaleX else { return nil }
    
    let encoderMult = commandBuffer.makeComputeCommandEncoder()!
    encoderMult.setComputePipelineState(pipeline)
    
    let dims: [UInt32] = [UInt32(inputSize.nbColumn), UInt32(inputSize.nbLine), UInt32(WSize.nbLine)]
    let bufferDims = metal.makeBuffer(bytes: dims, length: dims.count * 4, options: .storageModeShared)!
    
    let bufferA = metal.makeBuffer(length: inputSize.nbLine * WSize.nbLine * 4, options: .storageModeShared)!
    
    encoderMult.setBuffer(inputQuant, offset: 0, index: 0)
    encoderMult.setBuffer(bufferW, offset: 0, index: 1)
    encoderMult.setBuffer(bufferDims, offset: 0, index: 2)
    encoderMult.setBuffer(bufferA, offset: 0, index: 6)
    encoderMult.setBuffer(scaleW, offset: 0, index: 7)
    encoderMult.setBuffer(scaleX, offset: 0, index: 8)
    
    var scaleWSize = UInt32(scaleW.length / MemoryLayout<Float>.size)
    encoderMult.setBytes(&scaleWSize, length: MemoryLayout<UInt32>.size, index: 9)

    let MultGridSize = MTLSize(width: WSize.nbLine, height: inputSize.nbLine, depth: 1)
    let groupWidth = min(pipeline.threadExecutionWidth, WSize.nbLine)
    let MultThreadGroupSize = MTLSize(width: groupWidth, height: 1, depth: 1)

    encoderMult.dispatchThreads(MultGridSize, threadsPerThreadgroup: MultThreadGroupSize)
    encoderMult.endEncoding()
    
    return bufferA
}
