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

func loadInt8Matrix(path: String) -> [Int8] {
    guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)) else { fatalError("File not found!!") }
    return data.withUnsafeBytes { ptr in
        Array(UnsafeBufferPointer(start: ptr.bindMemory(to: Int8.self).baseAddress, count: data.count))
    }
}

func loadFloatMatrix(path: String) -> [Float] {
    guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)) else { fatalError("File not found!!") }
    return data.withUnsafeBytes { ptr in
        Array(UnsafeBufferPointer(start: ptr.bindMemory(to: Float.self).baseAddress, count: data.count / MemoryLayout<Float>.size))
    }
}


guard let metal = MTLCreateSystemDefaultDevice() else
{
    fatalError("Error : Metal is not available")
}

let queue = metal.makeCommandQueue()!

let library = metal.makeDefaultLibrary()!

//let commandBuffer = queue.makeCommandBuffer()!
//let finalCommandBuffer = queue.makeCommandBuffer()!

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
guard let kernelFunctionRelu2 = library.makeFunction(name: "relu2") else {
    fatalError("Error : Function siluAndMul is not available")
}
guard let kernelFunctionComputeLogits = library.makeFunction(name: "computeLogits") else {
    fatalError("Error : Function computeLogits is not available")
}
guard let kernelFunctionQuantize = library.makeFunction(name: "quantizeActivations") else {
    fatalError("Error : Function quantizeActivations is not available")
}


print(Main()!)

func Main()->String?{
    var dataEmbeddings : Data
    var dataRMSFinal : Data
    do{
        dataEmbeddings = try Data(contentsOf: urlEmbeddings)
        dataRMSFinal = try Data(contentsOf: urlRMSFinal)
    }catch{
        return "Error : can't load the weights"
    }
    
    let dataLMHead = try! Data(contentsOf: urlLMHead)
    let lmHead = dataLMHead.withUnsafeBytes { ptr in
        Array(UnsafeBufferPointer(start: ptr.bindMemory(to: Float.self).baseAddress, count: 128256 * 2560))
    }
    let lmHeadBuffer = metal.makeBuffer(bytes: lmHead, length: lmHead.count * MemoryLayout<Float>.size)!
    
    // embeddings hardcode infos
    let linesEmbeddings = 128256
    let colsEmbeddings = 2560
    
    let rmsFinal = dataRMSFinal.withUnsafeBytes {ptr in
        let count = dataRMSFinal.count / MemoryLayout<Float>.size
        let buffer = UnsafeBufferPointer(start: ptr.bindMemory(to: Float.self).baseAddress, count: count)
        return Array(buffer)
    }
    let rmsFinalBuffer = metal.makeBuffer(bytes: rmsFinal, length: rmsFinal.count * MemoryLayout<Float>.size)!

    // prompt
    let prompt = "The capital of France is"
    //var tokens = formatString(str : prompt)
    var tokens: [UInt32] = [128000, 25, 578, 6864, 315, 9822, 374, 128009, 72803, 25, 220]
    print("IDs : \(tokens)")
    let embeddings = dataEmbeddings.withUnsafeBytes {
        ptr in
        Array(UnsafeBufferPointer(start: ptr.bindMemory(to: Float.self).baseAddress, count: linesEmbeddings * colsEmbeddings))
    }
    let embeddingsBuffer = metal.makeBuffer(bytes: embeddings, length: embeddings.count * MemoryLayout<Float>.size)!
    
    //var kCaches: [MTLBuffer?] = Array(repeating: nil, count: 26)
    //var vCaches: [MTLBuffer?] = Array(repeating: nil, count: 26)
    
    print("Generating ...")
    
    for no_prediction in 0..<10{
        //embeddings

        let embCommandBuffer = queue.makeCommandBuffer()!
        guard let embeddingsResponseBuffer = Embedding(tokens: tokens, colsEmbeddings: colsEmbeddings, embeddingsBuffer: embeddingsBuffer, commandBuffer: embCommandBuffer, metal: metal, metalFunction: kernelFunctionGetFromEmbedding) else{
            return nil
        }
        
        var currentHiddenState: MTLBuffer? = embeddingsResponseBuffer
        
        embCommandBuffer.commit()
        embCommandBuffer.waitUntilCompleted()
        
        print("prediction \(no_prediction)")
     
        for layerIndex in 0..<30{
            
            autoreleasepool {
                let layerCommandBuffer = queue.makeCommandBuffer()!
                let layerDir = "/Users/jossua/Documents/jossnet-bitnet/py/layers/layer_\(layerIndex)/"
                
                let weightQ = loadInt8Matrix(path: layerDir + "W_q.bin")
                let weightK = loadInt8Matrix(path: layerDir + "W_k.bin")
                let weightV = loadInt8Matrix(path: layerDir + "W_v.bin")
                let weightO = loadInt8Matrix(path: layerDir + "W_o.bin")
                let weightGate = loadInt8Matrix(path: layerDir + "W_gate.bin")
                let weightUp = loadInt8Matrix(path: layerDir + "W_up.bin")
                let weightDown = loadInt8Matrix(path: layerDir + "W_down.bin")
                let rmsInput = loadFloatMatrix(path: layerDir + "RMS_input.bin")
                let rmsAttnSub = loadFloatMatrix(path: layerDir + "RMS_attn.bin")
                let rmsPostAttn = loadFloatMatrix(path: layerDir + "RMS_post_attn.bin")
                let rmsMlpSub = loadFloatMatrix(path: layerDir + "RMS_mlp_sub.bin")
                let scaleQ = loadFloatMatrix(path: layerDir + "Scale_q.bin")

                let scaleK = loadFloatMatrix(path: layerDir + "Scale_k.bin")
                let scaleO = loadFloatMatrix(path: layerDir + "Scale_o.bin")
                let scaleV = loadFloatMatrix(path: layerDir + "Scale_v.bin")
                let scaleUp = loadFloatMatrix(path: layerDir + "Scale_up.bin")
                let scaleDown = loadFloatMatrix(path: layerDir + "Scale_down.bin")
                let scaleGate = loadFloatMatrix(path: layerDir + "Scale_gate.bin")
                
                // create buffers
                
                let bufferRMSInput = metal.makeBuffer(bytes: rmsInput, length: rmsInput.count * 4, options: .storageModeShared)!
                let bufferRMSAttnSub = metal.makeBuffer(bytes: rmsAttnSub, length: rmsAttnSub.count * 4, options: .storageModeShared)!
                let bufferRMSPostAttn = metal.makeBuffer(bytes: rmsPostAttn, length: rmsPostAttn.count * 4, options: .storageModeShared)!
                let bufferRMSMlpSub = metal.makeBuffer(bytes: rmsMlpSub, length: rmsMlpSub.count * 4, options: .storageModeShared)!
                let bufferScaleQ = metal.makeBuffer(bytes: scaleQ, length: scaleQ.count * MemoryLayout<Float>.size, options: .storageModeShared)!
                let bufferScaleK = metal.makeBuffer(bytes: scaleK, length: scaleK.count * MemoryLayout<Float>.size, options: .storageModeShared)!
                let bufferScaleO = metal.makeBuffer(bytes: scaleO, length: scaleO.count * MemoryLayout<Float>.size, options: .storageModeShared)!
                let bufferScaleV = metal.makeBuffer(bytes: scaleV, length: scaleV.count * MemoryLayout<Float>.size, options: .storageModeShared)!
                let bufferScaleUp = metal.makeBuffer(bytes: scaleUp, length: scaleUp.count * MemoryLayout<Float>.size, options: .storageModeShared)!
                let bufferScaleDown = metal.makeBuffer(bytes: scaleDown, length: scaleDown.count * MemoryLayout<Float>.size, options: .storageModeShared)!
                let bufferScaleGate = metal.makeBuffer(bytes: scaleGate, length: scaleGate.count * MemoryLayout<Float>.size, options: .storageModeShared)!
                
                let normalizedInput = ApplyRmsNorm(entry: currentHiddenState, sizeEntry: colsEmbeddings, weightRMS: bufferRMSInput, tokenCount: tokens.count, commandBuffer: layerCommandBuffer, metal: metal, metalFunction: kernelFunctionRmsNorm)!
                
                
                let quantizedData = QuantizeActivations(entry: normalizedInput, nbCols: colsEmbeddings, nbTokens: tokens.count, commandBuffer: layerCommandBuffer, metal: metal, metalFunction: kernelFunctionQuantize)!
                
                let inputQuant = quantizedData.quantized
                let scaleX = quantizedData.scales
                
                //mult
                let InputSize = matrixInfos(nbLine : tokens.count, nbColumn : 2560)
                let wSize = matrixInfos(nbLine : 2560, nbColumn : 2560)
                let wVSize = matrixInfos(nbLine : 640, nbColumn : 2560)
                let wKSize = matrixInfos(nbLine : 640, nbColumn : 2560)
                
                var bufferQ = MultByW(WSize: wSize, inputSize: InputSize, weight: weightQ, scaleW: bufferScaleQ, inputQuant: inputQuant, scaleX: scaleX, commandBuffer: layerCommandBuffer, metal: metal, metalFunction: kernelFunctionMult)

                var bufferK = MultByW(WSize: wKSize, inputSize: InputSize, weight: weightK, scaleW: bufferScaleK, inputQuant: inputQuant, scaleX: scaleX, commandBuffer: layerCommandBuffer, metal: metal, metalFunction: kernelFunctionMult)

                let bufferV = MultByW(WSize: wVSize, inputSize: InputSize, weight: weightV, scaleW: bufferScaleV, inputQuant: inputQuant, scaleX: scaleX, commandBuffer: layerCommandBuffer, metal: metal, metalFunction: kernelFunctionMult)
            
                bufferQ = ApplyRoPE(buffer: bufferQ, nbTokens: tokens.count, dim: wSize.nbLine, commandBuffer: layerCommandBuffer, metal: metal, metalFunction: kernelFunctionRoPE)
                            
                bufferK = ApplyRoPE(buffer: bufferK, nbTokens: tokens.count, dim: wKSize.nbLine, commandBuffer: layerCommandBuffer, metal: metal, metalFunction: kernelFunctionRoPE)
                
                var resultAttention = AttentionScore(buffer1: bufferQ, buffer2: bufferK, nbToken: tokens.count, commandBuffer: layerCommandBuffer, metal: metal, metalFunction: kernelFunctionAttentionScore)
                
                
                let totalAttentionRows = 20 * tokens.count
                let attentionWidth = tokens.count
                
                resultAttention = SubstractMax(mat: resultAttention, nbcols: attentionWidth, nbToken: totalAttentionRows, commandBuffer: layerCommandBuffer, metal: metal, metalFunction: kernelFunctionSubstractMax, getMaxFunction: kernelFunctionSearchMaxVector)
                
                resultAttention = DivideBySum(mat: resultAttention, nbcols: attentionWidth, nbToken: totalAttentionRows, commandBuffer: layerCommandBuffer, metal: metal, metalFunction: kernelFunctionDivideBySum, getSumFunction: kernelFunctionSumVector)
                
                //context
                let contextBuffer = ComputeWeightedSum(mat: resultAttention, v: bufferV, nbTokens: tokens.count, commandBuffer: layerCommandBuffer, metal: metal, metalFunction: kernelFunctionWeightedSum)
                
                let normContextBuffer = ApplyRmsNorm(entry: contextBuffer, sizeEntry: 2560, weightRMS: bufferRMSAttnSub, tokenCount: tokens.count, commandBuffer: layerCommandBuffer, metal: metal, metalFunction: kernelFunctionRmsNorm)!
                
                // mult by o
                let wOSize = matrixInfos(nbLine: 2560, nbColumn: 2560)
                let contextSize = matrixInfos(nbLine: tokens.count, nbColumn: 2560)
                
                
                let quantizedContextData = QuantizeActivations(entry: normContextBuffer, nbCols: 2560, nbTokens: tokens.count, commandBuffer: layerCommandBuffer, metal: metal, metalFunction: kernelFunctionQuantize)!

                let bufferAttentionOutput = MultByW(WSize: wOSize, inputSize: contextSize, weight: weightO, scaleW: bufferScaleO, inputQuant: quantizedContextData.quantized, scaleX: quantizedContextData.scales, commandBuffer: layerCommandBuffer, metal: metal, metalFunction: kernelFunctionMult)
                
                //add
                let afterAttn = AddArray(mat1: currentHiddenState!, mat2: bufferAttentionOutput, size: tokens.count * 2560, commandBuffer: layerCommandBuffer, metal: metal, metalFunction: kernelFunctionAddArrays)
                
                
                // apply rms norm
                let rmsNormResult = ApplyRmsNorm(entry: afterAttn, sizeEntry: colsEmbeddings, weightRMS: bufferRMSPostAttn, tokenCount: tokens.count, commandBuffer: layerCommandBuffer, metal: metal, metalFunction: kernelFunctionRmsNorm)!
                
                //mult by Gate & Up
                let resultSize = matrixInfos(nbLine: tokens.count, nbColumn: 2560)
                
                let weightsSize = matrixInfos(nbLine : 6912, nbColumn : 2560)
                
                let quantizedMlpData = QuantizeActivations(entry: rmsNormResult, nbCols: 2560, nbTokens: tokens.count, commandBuffer: layerCommandBuffer, metal: metal, metalFunction: kernelFunctionQuantize)!

                let gateResult = MultByW(WSize: weightsSize, inputSize: resultSize, weight: weightGate, scaleW: bufferScaleGate, inputQuant: quantizedMlpData.quantized, scaleX: quantizedMlpData.scales, commandBuffer: layerCommandBuffer, metal: metal, metalFunction: kernelFunctionMult)

                let upResult = MultByW(WSize: weightsSize, inputSize: resultSize, weight: weightUp, scaleW: bufferScaleUp, inputQuant: quantizedMlpData.quantized, scaleX: quantizedMlpData.scales, commandBuffer: layerCommandBuffer, metal: metal, metalFunction: kernelFunctionMult)

                let sizeIntermediaire = tokens.count * 6912
                ApplySiLUandMul(gate: gateResult, up: upResult, size: sizeIntermediaire, commandBuffer: layerCommandBuffer, metal: metal, metalFunction: kernelFunctionRelu2)
                
                let normDownInput = ApplyRmsNorm(entry: gateResult, sizeEntry: 6912, weightRMS: bufferRMSMlpSub, tokenCount: tokens.count, commandBuffer: layerCommandBuffer, metal: metal, metalFunction: kernelFunctionRmsNorm)!
                
                let quantizedDownData = QuantizeActivations(entry: normDownInput, nbCols: 6912, nbTokens: tokens.count, commandBuffer: layerCommandBuffer, metal: metal, metalFunction: kernelFunctionQuantize)!
                
                let inputSizeDown = matrixInfos(nbLine: tokens.count, nbColumn: 6912)
                let weightsSizeDown = matrixInfos(nbLine: 2560, nbColumn: 6912)
                
                let downResult = MultByW(WSize: weightsSizeDown, inputSize: inputSizeDown, weight: weightDown, scaleW: bufferScaleDown, inputQuant: quantizedDownData.quantized, scaleX: quantizedDownData.scales, commandBuffer: layerCommandBuffer, metal: metal, metalFunction: kernelFunctionMult)
                
                currentHiddenState = AddArray(mat1: afterAttn, mat2: downResult, size: tokens.count * 2560, commandBuffer: layerCommandBuffer, metal: metal, metalFunction: kernelFunctionAddArrays)
                
                layerCommandBuffer.commit()
                layerCommandBuffer.waitUntilCompleted()
                
            }
        }
        
        let finalCmdBuffer = queue.makeCommandBuffer()!
        let resultFinalRms = ApplyRmsNorm(entry: currentHiddenState, sizeEntry: colsEmbeddings, weightRMS: rmsFinalBuffer, tokenCount: tokens.count, commandBuffer: finalCmdBuffer, metal: metal, metalFunction: kernelFunctionRmsNorm)

        let logitsBuffer = ComputeLogits(finalVectors: resultFinalRms, embeddings: lmHeadBuffer, nbTokens: tokens.count, dim: colsEmbeddings, commandBuffer: finalCmdBuffer, metal: metal, metalFunction: kernelFunctionComputeLogits)
        
        // Start

        finalCmdBuffer.commit()
        finalCmdBuffer.waitUntilCompleted()

        let logitsArray = GetFromBuffer(buffer: logitsBuffer, size: 128256)
        
        var bestScore: Float = -Float.greatestFiniteMagnitude
        var bestTokenID = 0
        
        for i in 0..<logitsArray.count {
            if logitsArray[i] > bestScore {
                bestScore = logitsArray[i]
                bestTokenID = i
            }
        }
        tokens.append(UInt32(bestTokenID))
        
    }
    print("exit : \(tokens)")
    return("ok")
}
