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

let commandBuffer = queue.makeCommandBuffer()!
let finalCommandBuffer = queue.makeCommandBuffer()!

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
guard let kernelFunctionComputeLogits = library.makeFunction(name: "computeLogits") else {
    fatalError("Error : Function computeLogits is not available")
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
    let prompt = "Hi, my name is Jossua!"
    let tokens = formatString(str : prompt)
    print("IDs : \(tokens)")
    let embeddings = dataEmbeddings.withUnsafeBytes {
        ptr in
        Array(UnsafeBufferPointer(start: ptr.bindMemory(to: Float.self).baseAddress, count: linesEmbeddings * colsEmbeddings))
    }
    let embeddingsBuffer = metal.makeBuffer(bytes: embeddings, length: embeddings.count * MemoryLayout<Float>.size)!

    //embeddings

    guard let embeddingsResponseBuffer = Embedding(tokens: tokens, colsEmbeddings: colsEmbeddings, embeddingsBuffer: embeddingsBuffer, commandBuffer: commandBuffer, metal: metal, metalFunction: kernelFunctionGetFromEmbedding) else{
        return nil
    }

    var currentHiddenState : MTLBuffer? = embeddingsResponseBuffer
    
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()


    for layerIndex in 0..<26{
        
        autoreleasepool {
            let layerCommandBuffer = queue.makeCommandBuffer()!
            //import .bin
            let layerDir = "/Users/jossua/Documents/jossnet-bitnet/py/layers/layer_\(layerIndex)/"
                        
            let weightQ = loadInt8Matrix(path: layerDir + "W_q.bin")
            let weightK = loadInt8Matrix(path: layerDir + "W_k.bin")
            let weightV = loadInt8Matrix(path: layerDir + "W_v.bin")
            let weightO = loadInt8Matrix(path: layerDir + "W_o.bin")
            let weightGate = loadInt8Matrix(path: layerDir + "W_gate.bin")
            let weightUp = loadInt8Matrix(path: layerDir + "W_up.bin")
            let weightDown = loadInt8Matrix(path: layerDir + "W_down.bin")
            let rmsAttn = loadFloatMatrix(path: layerDir + "RMS_attn.bin")
            let rmsMlp = loadFloatMatrix(path: layerDir + "RMS_mlp.bin")
            let scaleQ = loadFloatMatrix(path: layerDir + "Scale_q.bin")
            let scaleK = loadFloatMatrix(path: layerDir + "Scale_k.bin")
            let scaleO = loadFloatMatrix(path: layerDir + "Scale_o.bin")
            let scaleV = loadFloatMatrix(path: layerDir + "Scale_v.bin")
            let scaleUp = loadFloatMatrix(path: layerDir + "Scale_up.bin")
            let scaleDown = loadFloatMatrix(path: layerDir + "Scale_down.bin")
            let scaleGate = loadFloatMatrix(path: layerDir + "Scale_gate.bin")
            
            // create buffers
            
            let bufferRMSAttn = metal.makeBuffer(bytes: rmsAttn, length: rmsAttn.count * MemoryLayout<Float>.size, options: .storageModeShared)!
            let bufferRMSMlp = metal.makeBuffer(bytes: rmsMlp, length: rmsMlp.count * MemoryLayout<Float>.size, options: .storageModeShared)!
            let bufferScaleQ = metal.makeBuffer(bytes: scaleQ, length: scaleQ.count * MemoryLayout<Float>.size, options: .storageModeShared)!
            let bufferScaleK = metal.makeBuffer(bytes: scaleK, length: scaleK.count * MemoryLayout<Float>.size, options: .storageModeShared)!
            let bufferScaleO = metal.makeBuffer(bytes: scaleO, length: scaleO.count * MemoryLayout<Float>.size, options: .storageModeShared)!
            let bufferScaleV = metal.makeBuffer(bytes: scaleV, length: scaleV.count * MemoryLayout<Float>.size, options: .storageModeShared)!
            let bufferScaleUp = metal.makeBuffer(bytes: scaleUp, length: scaleUp.count * MemoryLayout<Float>.size, options: .storageModeShared)!
            let bufferScaleDown = metal.makeBuffer(bytes: scaleDown, length: scaleDown.count * MemoryLayout<Float>.size, options: .storageModeShared)!
            let bufferScaleGate = metal.makeBuffer(bytes: scaleGate, length: scaleGate.count * MemoryLayout<Float>.size, options: .storageModeShared)!
            print("count : \(scaleQ.count)")
            print("count : \(scaleGate.count)")
            
            
            
            let normalizedAttn = ApplyRmsNorm(entry: currentHiddenState, sizeEntry: colsEmbeddings, weightRMS: bufferRMSAttn, tokenCount: tokens.count, commandBuffer: layerCommandBuffer, metal: metal, metalFunction: kernelFunctionRmsNorm)!
            
            //mult
            let InputSize = matrixInfos(nbLine : tokens.count, nbColumn : 2560)
            //q
            let wSize = matrixInfos(nbLine : 2560, nbColumn : 2560)
            var bufferQ = MultByW(WSize: wSize, inputSize: InputSize, weight: weightQ, scale: bufferScaleQ, input: normalizedAttn, commandBuffer: layerCommandBuffer, metal: metal, metalFunction: kernelFunctionMult)
            
            //k
            let wKSize = matrixInfos(nbLine : 640, nbColumn : 640)
            var bufferK = MultByW(WSize: wKSize, inputSize: InputSize, weight: weightK, scale: bufferScaleK ,input: normalizedAttn, commandBuffer: layerCommandBuffer, metal: metal, metalFunction: kernelFunctionMult)
            
            //v
            let wVSize = matrixInfos(nbLine : 640, nbColumn : 640)
            let bufferV = MultByW(WSize: wVSize, inputSize: InputSize, weight: weightV, scale: bufferScaleV,input: normalizedAttn, commandBuffer: layerCommandBuffer, metal: metal, metalFunction: kernelFunctionMult)
            
            // RoPE
            bufferQ = ApplyRoPE(buffer: bufferQ, nbTokens: tokens.count, dim: wSize.nbColumn, commandBuffer: layerCommandBuffer, metal: metal, metalFunction: kernelFunctionRoPE)
            
            bufferK = ApplyRoPE(buffer: bufferK, nbTokens: tokens.count, dim: wKSize.nbColumn, commandBuffer: layerCommandBuffer, metal: metal, metalFunction: kernelFunctionRoPE)
            
            // Attention score
            
            var resultAttention = AttentionScore(buffer1: bufferQ, buffer2: bufferK, nbToken: tokens.count, commandBuffer: layerCommandBuffer, metal: metal, metalFunction: kernelFunctionAttentionScore)
            
            
            let totalAttentionRows = 20 * tokens.count
            let attentionWidth = tokens.count
            
            resultAttention = SubstractMax(mat: resultAttention, nbcols: attentionWidth, nbToken: totalAttentionRows, commandBuffer: layerCommandBuffer, metal: metal, metalFunction: kernelFunctionSubstractMax, getMaxFunction: kernelFunctionSearchMaxVector)
            
            resultAttention = DivideBySum(mat: resultAttention, nbcols: attentionWidth, nbToken: totalAttentionRows, commandBuffer: layerCommandBuffer, metal: metal, metalFunction: kernelFunctionDivideBySum, getSumFunction: kernelFunctionSumVector)
            
            //context
            let contextBuffer = ComputeWeightedSum(mat: resultAttention, v: bufferV, nbTokens: tokens.count, commandBuffer: layerCommandBuffer, metal: metal, metalFunction: kernelFunctionWeightedSum)
            
            // mult by o
            let wOSize = matrixInfos(nbLine: 2560, nbColumn: 2560)
            let contextSize = matrixInfos(nbLine: tokens.count, nbColumn: 2560)
            
            let bufferAttentionOutput = MultByW(WSize: wOSize, inputSize: contextSize, weight: weightO, scale: bufferScaleO,input: contextBuffer, commandBuffer: layerCommandBuffer, metal: metal, metalFunction: kernelFunctionMult)
            
            //add
            let addResult = AddArray(mat1: currentHiddenState, mat2: bufferAttentionOutput, size: tokens.count * 2560, commandBuffer: layerCommandBuffer, metal: metal, metalFunction: kernelFunctionAddArrays)
            
            // apply rms norm
            let rmsNormResult = ApplyRmsNorm(entry: addResult, sizeEntry: colsEmbeddings, weightRMS: bufferRMSMlp, tokenCount: tokens.count, commandBuffer: layerCommandBuffer, metal: metal, metalFunction: kernelFunctionRmsNorm)!
            
            //mult by Gate & Up
            let resultSize = matrixInfos(nbLine: tokens.count, nbColumn: 2560)
            
            let weightsSize = matrixInfos(nbLine : 2560, nbColumn : 6912)
            
            let gateResult = MultByW(WSize: weightsSize, inputSize: resultSize, weight: weightGate, scale: bufferScaleGate, input: rmsNormResult, commandBuffer: layerCommandBuffer, metal: metal, metalFunction: kernelFunctionMult)
            
            let upResult = MultByW(WSize: weightsSize, inputSize: resultSize, weight: weightUp, scale: bufferScaleUp, input: rmsNormResult, commandBuffer: layerCommandBuffer, metal: metal, metalFunction: kernelFunctionMult)
            
            let sizeIntermediaire = tokens.count * 6912
            ApplySiLUandMul(gate: gateResult, up: upResult, size: sizeIntermediaire, commandBuffer: layerCommandBuffer, metal: metal, metalFunction: kernelFunctionSiluAndMul)
            
            let inputSizeDown = matrixInfos(nbLine: tokens.count, nbColumn: 6912)
            let weightsSizeDown = matrixInfos(nbLine: 6912, nbColumn: 2560)
            
            let downResult = MultByW(WSize: weightsSizeDown, inputSize: inputSizeDown, weight: weightDown, scale: bufferScaleDown, input: gateResult, commandBuffer: layerCommandBuffer, metal: metal, metalFunction: kernelFunctionMult)
            
            currentHiddenState = AddArray(mat1: addResult, mat2: downResult, size: tokens.count * 2560, commandBuffer: layerCommandBuffer, metal: metal, metalFunction: kernelFunctionAddArrays)
            
            layerCommandBuffer.commit()
            layerCommandBuffer.waitUntilCompleted()
            
        }
    }
    
    let resultFinalRms = ApplyRmsNorm(entry: currentHiddenState, sizeEntry: colsEmbeddings, weightRMS: rmsFinalBuffer, tokenCount: tokens.count, commandBuffer: finalCommandBuffer, metal: metal, metalFunction: kernelFunctionRmsNorm)

    let logitsBuffer = ComputeLogits(finalVectors: resultFinalRms, embeddings: embeddingsBuffer, nbTokens: tokens.count, dim: colsEmbeddings, commandBuffer: finalCommandBuffer, metal: metal, metalFunction: kernelFunctionComputeLogits)
    
    // Start

    finalCommandBuffer.commit()
    finalCommandBuffer.waitUntilCompleted()

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

    /*let resultArrayMultQ = GetFromBuffer(buffer: bufferQ, size: (Int(colsEmbeddings) * tokens.count))
    
    let resultArrayMultV = GetFromBuffer(buffer: bufferV, size: (640 * tokens.count))
    
    let resultArrayMultK = GetFromBuffer(buffer: bufferK, size: (640 * tokens.count))
    
    let resultAttentionTest = GetFromBuffer(buffer: resultAttention, size: 20 * tokens.count * tokens.count )
    
    let context = GetFromBuffer(buffer: contextBuffer, size:  tokens.count * 20 * 128)
    
    let Attention = GetFromBuffer(buffer: bufferAttentionOutput, size:  tokens.count * 20 * 128)
    
    let add = GetFromBuffer(buffer: addResult, size:  tokens.count * 20 * 128)
    
    let uptab = GetFromBuffer(buffer: upResult, size:  tokens.count * 20 * 128)
    
    let gatetab = GetFromBuffer(buffer: gateResult, size:  tokens.count * 20 * 128)

    let layer0Output = GetFromBuffer(buffer: addResult, size: 10)
    
    let testresultFinalRms = GetFromBuffer(buffer: resultFinalRms, size: 10)*/
    
    let logitsArray = GetFromBuffer(buffer: logitsBuffer, size: 128256)
    
    var bestScore: Float = -Float.greatestFiniteMagnitude
    var bestTokenID = 0
    
    for i in 0..<logitsArray.count {
        if logitsArray[i] > bestScore {
            bestScore = logitsArray[i]
            bestTokenID = i
        }
    }
    
    
    /*print("Here result mult Q :\n \(resultArrayMultQ[0..<10])")
    
    print("Here result mult V :\n \(resultArrayMultV[0..<10])")
    
    print("Here result mult K :\n \(resultArrayMultK[0..<10])")
    
    print("Here result Attention :\n \(resultAttentionTest[0..<10])")
    
    print("Here result context :\n \(context[0..<10])")
    
    print("Here result attention :\n \(Attention[0..<10])")
    
    print("Here result add :\n \(add[0..<10])")
    
    print("Here result up :\n \(uptab[0..<10])")
    
    print("Here result tab :\n \(gatetab[0..<10])")
    
    print("Exit : \n \(layer0Output[0..<10])")
    
    print("FINAL RMS : \n \(testresultFinalRms[0..<10])")*/
    
    print("result")
    print("exit token : \(bestTokenID)")
    print("Score : \(bestScore)")
    return("ok")
}
