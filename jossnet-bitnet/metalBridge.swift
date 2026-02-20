//
//  MetalBridge.swift
//  jossnet-bitnet
//
//  Created by FIGUEIRAS Jossua on 19/02/2026.
//
import Metal

let urlW = URL(fileURLWithPath: "/Users/jossua/Documents/jossnet-bitnet/py/weights_W_q.bin")
let urlWk = URL(fileURLWithPath: "/Users/jossua/Documents/jossnet-bitnet/py/weights_W_k.bin")
let urlWv = URL(fileURLWithPath: "/Users/jossua/Documents/jossnet-bitnet/py/weights_W_v.bin")
let urlWo = URL(fileURLWithPath: "/Users/jossua/Documents/jossnet-bitnet/py/weights_W_o.bin")
let urlRMS = URL(fileURLWithPath: "/Users/jossua/Documents/jossnet-bitnet/py/weights_RMS_mlp.bin")
let urlDown = URL(fileURLWithPath: "/Users/jossua/Documents/jossnet-bitnet/py/weights_W_down.bin")
let urlUp = URL(fileURLWithPath: "/Users/jossua/Documents/jossnet-bitnet/py/weights_W_up.bin")
let urlGate = URL(fileURLWithPath: "/Users/jossua/Documents/jossnet-bitnet/py/weights_W_gate.bin")
let urlEmbeddings = URL(fileURLWithPath: "/Users/jossua/Documents/jossnet-bitnet/py/embeddings.bin")

func Embedding(tokens: [UInt32], colsEmbeddings : Int, embeddingsBuffer: MTLBuffer?, commandBuffer: MTLCommandBuffer, metal: MTLDevice, metalFunction : MTLFunction )->MTLBuffer?{
    
    guard let embeddingsBuffer else { return nil}
    
    let pipeline = try! metal.makeComputePipelineState(function: metalFunction)
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
func ApplySiLUandMul(gate: MTLBuffer?, up : MTLBuffer?, size : Int, commandBuffer: MTLCommandBuffer, metal : MTLDevice, metalFunction : MTLFunction) -> Void {
    guard let gate = gate, let up = up else { return }
    
    let pipeline = try! metal.makeComputePipelineState(function: metalFunction)
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    
    encoder.setBuffer(gate, offset: 0, index: 0)
    encoder.setBuffer(up, offset: 0, index: 1)
    
    let gridSize = MTLSize(width: size, height: 1, depth: 1)
    let threadGroup = MTLSize(width: 32, height: 1, depth: 1)
    
    encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroup)
    encoder.endEncoding()
}
func ApplyRmsNorm(entry: MTLBuffer?, sizeEntry : Int, weightRMS : MTLBuffer?, tokenCount : Int, commandBuffer: MTLCommandBuffer, metal : MTLDevice, metalFunction : MTLFunction)->MTLBuffer?{
    guard let entry, let weightRMS else {
        return nil
    }
    // rmsNorm
    let pipelineRms = try! metal.makeComputePipelineState(function: metalFunction)
    let encoderRms = commandBuffer.makeComputeCommandEncoder()!
    encoderRms.setComputePipelineState(pipelineRms);
    
    let bufferAnswer = metal.makeBuffer(length: tokenCount * sizeEntry * MemoryLayout<Float>.size, options: .storageModeShared)!
    
    var sizeEntry : UInt32 = UInt32(sizeEntry)
    
    let bufferEntrySize = metal.makeBuffer(bytes: &sizeEntry,length: MemoryLayout<UInt32>.size, options: .storageModeShared)!

    
    encoderRms.setBuffer(entry, offset: 0, index: 1);
    encoderRms.setBuffer(bufferEntrySize, offset: 0, index: 2)
    encoderRms.setBuffer(weightRMS, offset: 0, index: 3)
    encoderRms.setBuffer(bufferAnswer, offset: 0, index: 4)

    let RmsGridSize = MTLSize(width: tokenCount, height: 1, depth: 1)
    let RmsThreadGroupSize = MTLSize(width: 1, height: 1, depth: 1)

    encoderRms.dispatchThreads(RmsGridSize, threadsPerThreadgroup: RmsThreadGroupSize)
    encoderRms.endEncoding()
    
    return bufferAnswer
}
func AddArray(mat1: MTLBuffer?, mat2: MTLBuffer?, size: Int, commandBuffer: MTLCommandBuffer, metal : MTLDevice, metalFunction : MTLFunction)->MTLBuffer?{
    guard let mat1, let mat2 else {
        return nil
    }
    let pipeline = try! metal.makeComputePipelineState(function: metalFunction)
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    
    
    encoder.setBuffer(mat1, offset: 0, index: 0)
    encoder.setBuffer(mat2, offset: 0, index: 1)
    
    let gridSize = MTLSize(width: size, height: 1, depth: 1)
    
    let threadGroup = MTLSize(width: 32, height: 1, depth: 1)
    
    encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroup)
    encoder.endEncoding()
    
    return mat1
}
func ComputeWeightedSum(mat: MTLBuffer?, v: MTLBuffer?, nbTokens: Int, commandBuffer: MTLCommandBuffer, metal : MTLDevice, metalFunction : MTLFunction) -> MTLBuffer? {
    guard let mat, let v else {
        return nil
    }
    let pipeline = try! metal.makeComputePipelineState(function: metalFunction)
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
func DivideBySum(mat: MTLBuffer?, nbcols : Int, nbToken : Int, commandBuffer : MTLCommandBuffer, metal : MTLDevice, metalFunction : MTLFunction, getSumFunction: MTLFunction)->MTLBuffer?{
    guard let mat else {
        return nil
    }
    
    let bufferSumValues = GetSumByVector(mat: mat, nbcols: nbcols, nbToken: nbToken, commandBuffer: commandBuffer, metal: metal, metalFunction: getSumFunction)
    
    let pipeline = try! metal.makeComputePipelineState(function: metalFunction)
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline);
    
    var cols : UInt32 = UInt32(nbcols)
    
    let buffernbCols = metal.makeBuffer(bytes: &cols, length: MemoryLayout<UInt32>.size, options: .storageModeShared)!
    
    encoder.setBuffer(mat, offset: 0, index: 0)
    encoder.setBuffer(buffernbCols, offset: 0, index: 1)
    encoder.setBuffer(bufferSumValues, offset: 0, index: 2)
    
    let gridSize = MTLSize(width: nbcols, height: nbToken, depth: 1)
    let threadGroupSize = MTLSize(width: 32, height: 32, depth: 1)

    encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
    encoder.endEncoding()
    
    return mat;
    
}
func SubstractMax(mat: MTLBuffer?, nbcols : Int, nbToken : Int, commandBuffer : MTLCommandBuffer, metal : MTLDevice, metalFunction : MTLFunction, getMaxFunction : MTLFunction)->MTLBuffer?{
    guard let mat else {
        return nil
    }
    
    
    let bufferMaxValues = GetMaxByVector(mat: mat, nbcols: nbcols, nbToken: nbToken, commandBuffer: commandBuffer, metal: metal, metalFunction: getMaxFunction)

    let pipeline = try! metal.makeComputePipelineState(function: metalFunction)
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline);
    
    var cols : UInt32 = UInt32(nbcols)
    
    let buffernbCols = metal.makeBuffer(bytes: &cols, length: MemoryLayout<UInt32>.size, options: .storageModeShared)!
    
    encoder.setBuffer(mat, offset: 0, index: 0)
    encoder.setBuffer(buffernbCols, offset: 0, index: 1)
    encoder.setBuffer(bufferMaxValues, offset: 0, index: 2)
    
    let gridSize = MTLSize(width: nbcols, height: nbToken, depth: 1)
    let threadGroupSize = MTLSize(width: 32, height: 32, depth: 1)

    encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
    encoder.endEncoding()
    
    return mat;
    
}
func GetSumByVector(mat: MTLBuffer?, nbcols : Int, nbToken : Int, commandBuffer : MTLCommandBuffer, metal : MTLDevice, metalFunction : MTLFunction)->MTLBuffer?{
    guard let mat else {
        return nil
    }
    let pipeline = try! metal.makeComputePipelineState(function: metalFunction)
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
func GetMaxByVector(mat: MTLBuffer?, nbcols : Int, nbToken : Int, commandBuffer : MTLCommandBuffer, metal : MTLDevice, metalFunction : MTLFunction)->MTLBuffer?{
    guard let mat else {
        return nil
    }
    let pipeline = try! metal.makeComputePipelineState(function: metalFunction)
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
func AttentionScore(buffer1 : MTLBuffer?, buffer2 : MTLBuffer?, nbToken : Int, commandBuffer : MTLCommandBuffer, metal : MTLDevice, metalFunction : MTLFunction)->MTLBuffer?{
    guard let buffer1, let buffer2 else {
        return nil
    }
    let pipeline = try! metal.makeComputePipelineState(function: metalFunction)
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
func ApplyRoPE(buffer : MTLBuffer?, nbTokens : Int, dim : Int, commandBuffer : MTLCommandBuffer, metal : MTLDevice, metalFunction : MTLFunction) -> MTLBuffer? {
    guard let buffer = buffer else {
        return nil
    }
    let pipeline = try! metal.makeComputePipelineState(function: metalFunction)
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    
    var dim32 = UInt32(dim)
    let bufferDim = metal.makeBuffer(bytes: &dim32, length: MemoryLayout<UInt32>.size, options: .storageModeShared)!
    
    encoder.setBuffer(buffer, offset: 0, index: 0)
    encoder.setBuffer(bufferDim, offset: 0, index: 1)
    
    let gridSize = MTLSize(width: nbTokens, height: dim / 2, depth: 1)
    let threadGroupSize = MTLSize(width: 1, height: 32, depth: 1)

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

func MultByW(WSize: matrixInfos, inputSize: matrixInfos, weight : [Int8], input : MTLBuffer?, commandBuffer: MTLCommandBuffer, metal : MTLDevice, metalFunction : MTLFunction) -> MTLBuffer?{
    
    // TODO :
    // weight [Int8] -> MTLBuffer?
    
    guard let input = input else {
        return nil
    }
    var nbLineW = UInt32(WSize.nbLine)
    var nbColW = UInt32(WSize.nbColumn)
    var nbLineInput = UInt32(inputSize.nbLine)
    var nbColInput = UInt32(inputSize.nbColumn)
    
    let pipelineMult = try! metal.makeComputePipelineState(function: metalFunction)
    let encoderMult = commandBuffer.makeComputeCommandEncoder()!
    encoderMult.setComputePipelineState(pipelineMult);
    
    // mult mat part
    let bufferW = metal.makeBuffer(bytes: weight, length: weight.count * MemoryLayout<Int8>.size, options: .storageModeShared)!

    let bufferA = metal.makeBuffer(length: Int(nbLineInput) * Int(nbLineW) * MemoryLayout<Float>.size, options: .storageModeShared)!
    
    memset(bufferA.contents(), 0, Int(nbLineInput) * Int(nbLineW) * MemoryLayout<Float>.size)

    let bufferNbColX = metal.makeBuffer(bytes: &nbColInput, length: MemoryLayout<UInt32>.size, options: .storageModeShared)!
    let bufferNbLineX = metal.makeBuffer(bytes: &nbLineInput, length: MemoryLayout<UInt32>.size, options: .storageModeShared)!
    let bufferNbColW = metal.makeBuffer(bytes: &nbColW, length: MemoryLayout<UInt32>.size, options: .storageModeShared)!
    let bufferNbLineW = metal.makeBuffer(bytes: &nbLineW, length: MemoryLayout<UInt32>.size, options: .storageModeShared)!

    encoderMult.setBuffer(input, offset: 0, index: 0);
    encoderMult.setBuffer(bufferW, offset: 0, index: 1);
    encoderMult.setBuffer(bufferA, offset: 0, index: 6);

    encoderMult.setBuffer(bufferNbColX, offset: 0, index: 2)
    encoderMult.setBuffer(bufferNbLineX, offset: 0, index: 3)
    encoderMult.setBuffer(bufferNbColW, offset: 0, index: 4)
    encoderMult.setBuffer(bufferNbLineW, offset: 0, index: 5)

    let MultGridSize = MTLSize(width: WSize.nbLine, height: inputSize.nbLine, depth: 1)
    let MultThreadGroupSize = MTLSize(width: 32, height: 1, depth: 1)

    encoderMult.dispatchThreads(MultGridSize, threadsPerThreadgroup: MultThreadGroupSize)
    encoderMult.endEncoding()
    
    return bufferA
}
