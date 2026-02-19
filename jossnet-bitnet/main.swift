//
//  main.swift
//  jossnet-bitnet
//
//  Created by FIGUEIRAS Jossua on 31/01/2026.
//

import Foundation
import Metal

struct matrixInfos{
    let nbLine : Int
    let nbColumn : Int
}

let urlW = URL(fileURLWithPath: "/Users/jossua/Documents/jossnet-bitnet/py/weights_W_q.bin")
let urlWk = URL(fileURLWithPath: "/Users/jossua/Documents/jossnet-bitnet/py/weights_W_k.bin")
let urlWv = URL(fileURLWithPath: "/Users/jossua/Documents/jossnet-bitnet/py/weights_W_v.bin")
let urlWo = URL(fileURLWithPath: "/Users/jossua/Documents/jossnet-bitnet/py/weights_W_o.bin")
let urlRMS = URL(fileURLWithPath: "/Users/jossua/Documents/jossnet-bitnet/py/weights_RMS_mlp.bin")
let urlDown = URL(fileURLWithPath: "/Users/jossua/Documents/jossnet-bitnet/py/weights_W_down.bin")
let urlUp = URL(fileURLWithPath: "/Users/jossua/Documents/jossnet-bitnet/py/weights_W_up.bin")
let urlGate = URL(fileURLWithPath: "/Users/jossua/Documents/jossnet-bitnet/py/weights_W_gate.bin")
let urlEmbeddings = URL(fileURLWithPath: "/Users/jossua/Documents/jossnet-bitnet/py/embeddings.bin")

guard let metal = MTLCreateSystemDefaultDevice() else
{
    fatalError("Error : Metal is not available")
}

let queue = metal.makeCommandQueue()!

let library = metal.makeDefaultLibrary()!

guard let kernelFunctionMult = library.makeFunction(name: "mult") else {
    fatalError("Error : Function mult is not available")
}
guard let kernelFunctionGetFromEmbedding = library.makeFunction(name: "getFromEmbedding") else {
    fatalError("Error : Function getFromEmbedding is not available")
}
guard let kernelFunctionRmsNorm = library.makeFunction(name: "rmsNorm") else {
    fatalError("Error : Function rmsNorm is not available")
}
guard let kernelFunctionRoPE = library.makeFunction(name: "RoPE") else {
    fatalError("Error : Function RoPE is not available")
}
guard let kernelFunctionAttentionScore = library.makeFunction(name: "AttentionScore") else {
    fatalError("Error : Function RoPE is not available")
}
guard let kernelFunctionSearchMaxVector = library.makeFunction(name: "SearchMaxVector") else {
    fatalError("Error : Function SearchMaxVector is not available")
}
guard let kernelFunctionSubstractMax = library.makeFunction(name: "SubstractMax") else {
    fatalError("Error : Function substractMax is not available")
}
guard let kernelFunctionSumVector = library.makeFunction(name: "SumVector") else {
    fatalError("Error : Function SumVector is not available")
}
guard let kernelFunctionDivideBySum = library.makeFunction(name: "DivideBySum") else {
    fatalError("Error : Function divideBySum is not available")
}
guard let kernelFunctionWeightedSum = library.makeFunction(name: "weightedSum") else {
    fatalError("Error : Function divideBySum is not available")
}
guard let kernelFunctionAddArrays = library.makeFunction(name: "addArrays") else {
    fatalError("Error : Function divideBySum is not available")
}
guard let kernelFunctionSiluAndMul = library.makeFunction(name: "siluAndMul") else {
    fatalError("Error : Function siluAndMul is not available")
}
print(Main()!)

func Main()->String?{
    var dataW : Data
    var dataWk : Data
    var dataWv : Data
    var dataWo : Data
    var dataWRMS : Data
    var dataWUp : Data
    var dataWDown : Data
    var dataWGate : Data
    var dateEmbeddings : Data
    do{
        dataW = try Data(contentsOf: urlW)
        dataWk = try Data(contentsOf: urlWk)
        dataWv = try Data(contentsOf: urlWv)
        dataWo = try Data(contentsOf: urlWo)
        dataWRMS = try Data(contentsOf: urlRMS)
        dataWUp = try Data(contentsOf: urlUp)
        dataWDown = try Data(contentsOf: urlDown)
        dataWGate = try Data(contentsOf: urlGate)
        dateEmbeddings = try Data(contentsOf: urlEmbeddings)
    }catch{
        return "Error : can't load the weights"
    }
    let weight = dataW.withUnsafeBytes { ptr -> [Int8] in
        let count = dataW.count / MemoryLayout<Int8>.size
        
        let buffer = UnsafeBufferPointer(start: ptr.bindMemory(to: Int8.self).baseAddress, count: count)
        return Array(buffer)
    }
    let weightK = dataWk.withUnsafeBytes { ptr -> [Int8] in
        let count = dataWk.count / MemoryLayout<Int8>.size
        
        let buffer = UnsafeBufferPointer(start: ptr.bindMemory(to: Int8.self).baseAddress, count: count)
        return Array(buffer)
    }
    let weightV = dataWv.withUnsafeBytes { ptr -> [Int8] in
        let count = dataWv.count / MemoryLayout<Int8>.size
        
        let buffer = UnsafeBufferPointer(start: ptr.bindMemory(to: Int8.self).baseAddress, count: count)
        return Array(buffer)
    }
    let weightO = dataWo.withUnsafeBytes { ptr -> [Int8] in
        let count = dataWo.count / MemoryLayout<Int8>.size
        
        let buffer = UnsafeBufferPointer(start: ptr.bindMemory(to: Int8.self).baseAddress, count: count)
        return Array(buffer)
    }
    let weightUp = dataWUp.withUnsafeBytes { ptr -> [Int8] in
        let count = dataWUp.count / MemoryLayout<Int8>.size
        
        let buffer = UnsafeBufferPointer(start: ptr.bindMemory(to: Int8.self).baseAddress, count: count)
        return Array(buffer)
    }
    //let up = metal.makeBuffer(bytes: weightUp, length: weightUp.count * MemoryLayout<Float>.size)!
    let weightDown = dataWDown.withUnsafeBytes { ptr -> [Int8] in
        let count = dataWDown.count / MemoryLayout<Int8>.size
        
        let buffer = UnsafeBufferPointer(start: ptr.bindMemory(to: Int8.self).baseAddress, count: count)
        return Array(buffer)
    }
    //let down = metal.makeBuffer(bytes: weightDown, length: weightDown.count * MemoryLayout<Float>.size)!
    let weightGate = dataWGate.withUnsafeBytes { ptr -> [Int8] in
        let count = dataWGate.count / MemoryLayout<Int8>.size
        
        let buffer = UnsafeBufferPointer(start: ptr.bindMemory(to: Int8.self).baseAddress, count: count)
        return Array(buffer)
    }
    //let gate = metal.makeBuffer(bytes: weightGate, length: weightGate.count * MemoryLayout<Float>.size)!
    let weightRMS = dataWRMS.withUnsafeBytes { ptr -> [Int8] in
        let count = dataWRMS.count / MemoryLayout<Int8>.size
        
        let buffer = UnsafeBufferPointer(start: ptr.bindMemory(to: Int8.self).baseAddress, count: count)
        return Array(buffer)
    }
    let RMS = metal.makeBuffer(bytes: weightRMS, length: weightRMS.count * MemoryLayout<Float>.size)!
    
    // embeddings hardcode infos
    let linesEmbeddings = 128256
    var colsEmbeddings = 2560

    let embeddingSize = linesEmbeddings * colsEmbeddings

    let embeddings = dateEmbeddings.withUnsafeBytes {
        ptr in
        Array(UnsafeBufferPointer(start: ptr.bindMemory(to: Float.self).baseAddress, count: embeddingSize))
    }

    // prompt
    let prompt = "Hi, my name is Jossua!"
    let tokens = formatString(str : prompt)
    print("IDs : \(tokens)")

    //embeddings
    let pipelineEmbeddings = try! metal.makeComputePipelineState(function: kernelFunctionGetFromEmbedding)
    let commandBuffer = queue.makeCommandBuffer()!
    let encoderEmbeddings = commandBuffer.makeComputeCommandEncoder()!
    encoderEmbeddings.setComputePipelineState(pipelineEmbeddings);

    // buffer creation

    var embeddingsBuffer = metal.makeBuffer(bytes: embeddings, length: embeddings.count * MemoryLayout<Float>.size)!

    let embeddingsResponseBuffer = metal.makeBuffer(length: (colsEmbeddings * MemoryLayout<Float>.size) * tokens.count, options: .storageModeShared)!
    let tokensInput = metal.makeBuffer(bytes: tokens, length: tokens.count * MemoryLayout<UInt32>.size, options: .storageModeShared)!

    let bufferNbColsEmbeddings = metal.makeBuffer(bytes: &colsEmbeddings, length: MemoryLayout<UInt32>.size, options: .storageModeShared)!

    //embeddings buffer
    encoderEmbeddings.setBuffer(embeddingsBuffer, offset: 0, index: 7)
    encoderEmbeddings.setBuffer(embeddingsResponseBuffer, offset: 0, index: 8)
    encoderEmbeddings.setBuffer(tokensInput, offset: 0, index: 9)
    encoderEmbeddings.setBuffer(bufferNbColsEmbeddings, offset: 0, index: 10)

    let embeddingsGridSize = MTLSize(width: colsEmbeddings, height: tokens.count, depth: 1)
    let embeddingsThreadGroupSize = MTLSize(width: 1, height: 1, depth: 1)

    encoderEmbeddings.dispatchThreads(embeddingsGridSize, threadsPerThreadgroup: embeddingsThreadGroupSize)
    encoderEmbeddings.endEncoding()

    //apply rms norm
    
    embeddingsBuffer = ApplyRmsNorm(entry: embeddingsBuffer, sizeEntry: colsEmbeddings, weightRMS: RMS, tokenCount: tokens.count, commandBuffer: commandBuffer)!
    
    //mult
    let InputSize = matrixInfos(nbLine : tokens.count, nbColumn : 2560)
    //q
    let wSize = matrixInfos(nbLine : 2560, nbColumn : 2560)
    var bufferQ = MultByW(WSize: wSize, inputSize: InputSize, weight: weight, input: embeddingsResponseBuffer, commandBuffer: commandBuffer)
    
    //k
    let wKSize = matrixInfos(nbLine : 640, nbColumn : 640)
    var bufferK = MultByW(WSize: wKSize, inputSize: InputSize, weight: weightK, input: embeddingsResponseBuffer, commandBuffer: commandBuffer)
    
    //v
    let wVSize = matrixInfos(nbLine : 640, nbColumn : 640)
    let bufferV = MultByW(WSize: wVSize, inputSize: InputSize, weight: weightV, input: embeddingsResponseBuffer, commandBuffer: commandBuffer)
    
    // RoPE
    bufferQ = ApplyRoPE(buffer : bufferQ, nbCols : InputSize.nbLine, nbLines : wSize.nbColumn, commandBuffer: commandBuffer)
    
    bufferK = ApplyRoPE(buffer : bufferK, nbCols : InputSize.nbLine, nbLines : wKSize.nbColumn, commandBuffer: commandBuffer)
    
    // Attention score
    
    var resultAttention = AttentionScore(buffer1: bufferQ, buffer2: bufferK, nbToken: tokens.count, commandBuffer: commandBuffer)
    

    let totalAttentionRows = 20 * tokens.count
    let attentionWidth = tokens.count
        
    resultAttention = SubstractMax(mat: resultAttention, nbcols: attentionWidth, nbToken: totalAttentionRows, commandBuffer: commandBuffer)
        
    resultAttention = DivideBySum(mat: resultAttention, nbcols: attentionWidth, nbToken: totalAttentionRows, commandBuffer: commandBuffer)
    
    //context
    let contextBuffer = ComputeWeightedSum(mat: resultAttention, v: bufferV, nbTokens: tokens.count, commandBuffer: commandBuffer)
    
    // mult by o
    let wOSize = matrixInfos(nbLine: 2560, nbColumn: 2560)
    let contextSize = matrixInfos(nbLine: tokens.count, nbColumn: 2560)

    let bufferAttentionOutput = MultByW(WSize: wOSize, inputSize: contextSize, weight: weightO, input: contextBuffer, commandBuffer: commandBuffer)
    
    //add
    let addResult = AddArray(mat1: embeddingsResponseBuffer, mat2: bufferAttentionOutput, size: tokens.count * 2560, commandBuffer: commandBuffer)
    
    // apply rms norm
    let rmsNormResult = ApplyRmsNorm(entry: addResult, sizeEntry: colsEmbeddings, weightRMS: RMS, tokenCount: tokens.count, commandBuffer: commandBuffer)!
    
    //mult by Gate & Up
    let resultSize = matrixInfos(nbLine: tokens.count, nbColumn: 2560)
    
    let weightsSize = matrixInfos(nbLine : 2560, nbColumn : 6912)
    
    let gateResult = MultByW(WSize: weightsSize, inputSize: resultSize, weight: weightGate, input: rmsNormResult, commandBuffer: commandBuffer)
    
    let upResult = MultByW(WSize: weightsSize, inputSize: resultSize, weight: weightUp, input: rmsNormResult, commandBuffer: commandBuffer)
    
    let sizeIntermediaire = tokens.count * 6912
    ApplySiLUandMul(gate: gateResult, up: upResult, size: sizeIntermediaire, commandBuffer: commandBuffer)
    
    let inputSizeDown = matrixInfos(nbLine: tokens.count, nbColumn: 6912)
    let weightsSizeDown = matrixInfos(nbLine: 6912, nbColumn: 2560)
    
    let downResult = MultByW(WSize: weightsSizeDown, inputSize: inputSizeDown, weight: weightDown, input: gateResult, commandBuffer: commandBuffer)
    _ = AddArray(mat1: addResult, mat2: downResult, size: tokens.count * 2560, commandBuffer: commandBuffer)


    // Start

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    let rawPointer = embeddingsResponseBuffer.contents()

    let floatPointer = rawPointer.bindMemory(to: Float.self, capacity: Int(colsEmbeddings) * tokens.count)

    let bufferPointer = UnsafeBufferPointer(start: floatPointer, count: Int(colsEmbeddings) * tokens.count)

    let resultArray = Array(bufferPointer)


    print("Nb tokens : \(resultArray.count/2560)")

    let stride = Int(colsEmbeddings)
    if tokens.count > 1 {
        print("Valeur Token 0 [0] : \(resultArray[0])")
        print("Valeur Token 1 [0] : \(resultArray[stride])")
    }

    let resultArrayMultQ = GetFromBuffer(buffer: bufferQ, size: (Int(colsEmbeddings) * tokens.count))
    
    let resultArrayMultV = GetFromBuffer(buffer: bufferV, size: (640 * tokens.count))
    
    let resultArrayMultK = GetFromBuffer(buffer: bufferK, size: (640 * tokens.count))
    
    let resultAttentionTest = GetFromBuffer(buffer: resultAttention, size: 20 * tokens.count * tokens.count )
    
    let context = GetFromBuffer(buffer: contextBuffer, size:  tokens.count * 20 * 128)
    
    let Attention = GetFromBuffer(buffer: bufferAttentionOutput, size:  tokens.count * 20 * 128)
    
    let add = GetFromBuffer(buffer: addResult, size:  tokens.count * 20 * 128)
    
    let uptab = GetFromBuffer(buffer: upResult, size:  tokens.count * 20 * 128)
    
    let gatetab = GetFromBuffer(buffer: gateResult, size:  tokens.count * 20 * 128)

    let layer0Output = GetFromBuffer(buffer: addResult, size: 10)
    
    
    print("Here result mult Q :\n \(resultArrayMultQ[0..<10])")
    
    print("Here result mult V :\n \(resultArrayMultV[0..<10])")
    
    print("Here result mult K :\n \(resultArrayMultK[0..<10])")
    
    print("Here result Attention :\n \(resultAttentionTest[0..<10])")
    
    print("Here result context :\n \(context[0..<10])")
    
    print("Here result attention :\n \(Attention[0..<10])")
    
    print("Here result add :\n \(add[0..<10])")
    
    print("Here result up :\n \(uptab[0..<10])")
    
    print("Here result tab :\n \(gatetab[0..<10])")
    
    print("Exit : \n \(layer0Output[0..<10])")
    
    return("ok")
}
func ApplySiLUandMul(gate: MTLBuffer?, up : MTLBuffer?, size : Int, commandBuffer: MTLCommandBuffer) -> Void {
    guard let gate = gate, let up = up else { return }
    
    let pipeline = try! metal.makeComputePipelineState(function: kernelFunctionSiluAndMul)
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    
    encoder.setBuffer(gate, offset: 0, index: 0)
    encoder.setBuffer(up, offset: 0, index: 1)
    
    let gridSize = MTLSize(width: size, height: 1, depth: 1)
    let threadGroup = MTLSize(width: 32, height: 1, depth: 1)
    
    encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroup)
    encoder.endEncoding()
}
func ApplyRmsNorm(entry: MTLBuffer?, sizeEntry : Int, weightRMS : MTLBuffer?, tokenCount : Int, commandBuffer: MTLCommandBuffer)->MTLBuffer?{
    guard let entry, let weightRMS else {
        return nil
    }
    // rmsNorm
    let pipelineRms = try! metal.makeComputePipelineState(function: kernelFunctionRmsNorm)
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
    
    return entry
}
func AddArray(mat1: MTLBuffer?, mat2: MTLBuffer?, size: Int, commandBuffer: MTLCommandBuffer)->MTLBuffer?{
    guard let mat1, let mat2 else {
        return nil
    }
    let pipeline = try! metal.makeComputePipelineState(function: kernelFunctionAddArrays)
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
func ComputeWeightedSum(mat: MTLBuffer?, v: MTLBuffer?, nbTokens: Int, commandBuffer: MTLCommandBuffer) -> MTLBuffer? {
    guard let mat, let v else {
        return nil
    }
    let pipeline = try! metal.makeComputePipelineState(function: kernelFunctionWeightedSum)
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
func DivideBySum(mat: MTLBuffer?, nbcols : Int, nbToken : Int, commandBuffer : MTLCommandBuffer)->MTLBuffer?{
    guard let mat else {
        return nil
    }
    
    let bufferSumValues = GetSumByVector(mat: mat, nbcols: nbcols, nbToken: nbToken, commandBuffer: commandBuffer)
    
    let pipeline = try! metal.makeComputePipelineState(function: kernelFunctionDivideBySum)
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
func SubstractMax(mat: MTLBuffer?, nbcols : Int, nbToken : Int, commandBuffer : MTLCommandBuffer)->MTLBuffer?{
    guard let mat else {
        return nil
    }
    
    let bufferMaxValues = GetMaxByVector(mat: mat, nbcols: nbcols, nbToken: nbToken, commandBuffer: commandBuffer)
    
    
    let pipeline = try! metal.makeComputePipelineState(function: kernelFunctionSubstractMax)
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
func GetSumByVector(mat: MTLBuffer?, nbcols : Int, nbToken : Int, commandBuffer : MTLCommandBuffer)->MTLBuffer?{
    guard let mat else {
        return nil
    }
    let pipeline = try! metal.makeComputePipelineState(function: kernelFunctionSumVector)
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
func GetMaxByVector(mat: MTLBuffer?, nbcols : Int, nbToken : Int, commandBuffer : MTLCommandBuffer)->MTLBuffer?{
    guard let mat else {
        return nil
    }
    let pipeline = try! metal.makeComputePipelineState(function: kernelFunctionSearchMaxVector)
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
func AttentionScore(buffer1 : MTLBuffer?, buffer2 : MTLBuffer?, nbToken : Int, commandBuffer : MTLCommandBuffer)->MTLBuffer?{
    guard let buffer1, let buffer2 else {
        return nil
    }
    let pipeline = try! metal.makeComputePipelineState(function: kernelFunctionAttentionScore)
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
func ApplyRoPE(buffer : MTLBuffer?, nbCols : Int, nbLines : Int,  commandBuffer : MTLCommandBuffer)->MTLBuffer?{
    guard let buffer = buffer else {
        return nil
    }
    let pipeline = try! metal.makeComputePipelineState(function: kernelFunctionRoPE)
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline);
    
    var nbcols = nbCols
    
    let bufferNbColX = metal.makeBuffer(bytes: &nbcols, length: MemoryLayout<Int32>.size, options: .storageModeShared)!
    
    encoder.setBuffer(buffer, offset: 0, index: 0)
    encoder.setBuffer(bufferNbColX, offset: 0, index: 1)
    
    let gridSize = MTLSize(width: nbCols, height: nbLines, depth: 1)
    let threadGroupSize = MTLSize(width: 32, height: 1, depth: 1)

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

func MultByW(WSize: matrixInfos, inputSize: matrixInfos, weight : [Int8], input : MTLBuffer?, commandBuffer: MTLCommandBuffer) -> MTLBuffer?{
    
    // TODO :
    // weight [Int8] -> MTLBuffer?
    
    guard let input = input else {
        return nil
    }
    var nbLineW = UInt32(WSize.nbLine)
    var nbColW = UInt32(WSize.nbColumn)
    var nbLineInput = UInt32(inputSize.nbLine)
    var nbColInput = UInt32(inputSize.nbColumn)
    
    let pipelineMult = try! metal.makeComputePipelineState(function: kernelFunctionMult)
    let encoderMult = commandBuffer.makeComputeCommandEncoder()!
    encoderMult.setComputePipelineState(pipelineMult);
    
    // mult mat part
    let bufferW = metal.makeBuffer(bytes: weight, length: weight.count * MemoryLayout<Int8>.size, options: .storageModeShared)!

    let bufferA = metal.makeBuffer(length: Int(nbLineInput) * Int(nbLineW) * MemoryLayout<Float>.size, options: .storageModeShared)!
    
    memset(bufferA.contents(), 0, Int(nbLineInput) * Int(nbLineW))

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
