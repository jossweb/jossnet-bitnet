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



guard let metal = MTLCreateSystemDefaultDevice() else
{
    fatalError("Error : Metal is not available")
}

let queue = metal.makeCommandQueue()!

let library = metal.makeDefaultLibrary()!

let commandBuffer = queue.makeCommandBuffer()!

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
    let weightRMS = dataWRMS.withUnsafeBytes { ptr -> [Float] in
            let count = dataWRMS.count / MemoryLayout<Float>.size
            let buffer = UnsafeBufferPointer(start: ptr.bindMemory(to: Float.self).baseAddress, count: count)
            return Array(buffer)
    }
    let RMS = metal.makeBuffer(bytes: weightRMS, length: weightRMS.count * MemoryLayout<Float>.size, options: .storageModeShared)!
    
    // embeddings hardcode infos
    let linesEmbeddings = 128256
    let colsEmbeddings = 2560
    
    let embeddings = dateEmbeddings.withUnsafeBytes {
        ptr in
        Array(UnsafeBufferPointer(start: ptr.bindMemory(to: Float.self).baseAddress, count: linesEmbeddings * colsEmbeddings))
    }
    let embeddingsBuffer = metal.makeBuffer(bytes: embeddings, length: embeddings.count * MemoryLayout<Float>.size)!

    // prompt
    let prompt = "Hi, my name is Jossua!"
    let tokens = formatString(str : prompt)
    print("IDs : \(tokens)")

    //embeddings

    guard var embeddingsResponseBuffer = Embedding(tokens: tokens, colsEmbeddings: colsEmbeddings, embeddingsBuffer: embeddingsBuffer, commandBuffer: commandBuffer, metal: metal, metalFunction: kernelFunctionGetFromEmbedding) else{
        return nil
    }


    //apply rms norm
    
    embeddingsResponseBuffer = ApplyRmsNorm(entry: embeddingsResponseBuffer, sizeEntry: colsEmbeddings, weightRMS: RMS, tokenCount: tokens.count, commandBuffer: commandBuffer, metal: metal, metalFunction: kernelFunctionRmsNorm)!
    
    //mult
    let InputSize = matrixInfos(nbLine : tokens.count, nbColumn : 2560)
    //q
    let wSize = matrixInfos(nbLine : 2560, nbColumn : 2560)
    var bufferQ = MultByW(WSize: wSize, inputSize: InputSize, weight: weight, input: embeddingsResponseBuffer, commandBuffer: commandBuffer, metal: metal, metalFunction: kernelFunctionMult)
    
    //k
    let wKSize = matrixInfos(nbLine : 640, nbColumn : 640)
    var bufferK = MultByW(WSize: wKSize, inputSize: InputSize, weight: weightK, input: embeddingsResponseBuffer, commandBuffer: commandBuffer, metal: metal, metalFunction: kernelFunctionMult)
    
    //v
    let wVSize = matrixInfos(nbLine : 640, nbColumn : 640)
    let bufferV = MultByW(WSize: wVSize, inputSize: InputSize, weight: weightV, input: embeddingsResponseBuffer, commandBuffer: commandBuffer, metal: metal, metalFunction: kernelFunctionMult)
    
    // RoPE
    bufferQ = ApplyRoPE(buffer: bufferQ, nbTokens: tokens.count, dim: wSize.nbColumn, commandBuffer: commandBuffer, metal: metal, metalFunction: kernelFunctionRoPE)
        
    bufferK = ApplyRoPE(buffer: bufferK, nbTokens: tokens.count, dim: wKSize.nbColumn, commandBuffer: commandBuffer, metal: metal, metalFunction: kernelFunctionRoPE)
    
    // Attention score
    
    var resultAttention = AttentionScore(buffer1: bufferQ, buffer2: bufferK, nbToken: tokens.count, commandBuffer: commandBuffer, metal: metal, metalFunction: kernelFunctionAttentionScore)
    

    let totalAttentionRows = 20 * tokens.count
    let attentionWidth = tokens.count
        
    resultAttention = SubstractMax(mat: resultAttention, nbcols: attentionWidth, nbToken: totalAttentionRows, commandBuffer: commandBuffer, metal: metal, metalFunction: kernelFunctionSubstractMax, getMaxFunction: kernelFunctionSearchMaxVector)
        
    resultAttention = DivideBySum(mat: resultAttention, nbcols: attentionWidth, nbToken: totalAttentionRows, commandBuffer: commandBuffer, metal: metal, metalFunction: kernelFunctionDivideBySum, getSumFunction: kernelFunctionSumVector)
    
    //context
    let contextBuffer = ComputeWeightedSum(mat: resultAttention, v: bufferV, nbTokens: tokens.count, commandBuffer: commandBuffer, metal: metal, metalFunction: kernelFunctionWeightedSum)
    
    // mult by o
    let wOSize = matrixInfos(nbLine: 2560, nbColumn: 2560)
    let contextSize = matrixInfos(nbLine: tokens.count, nbColumn: 2560)

    let bufferAttentionOutput = MultByW(WSize: wOSize, inputSize: contextSize, weight: weightO, input: contextBuffer, commandBuffer: commandBuffer, metal: metal, metalFunction: kernelFunctionMult)
    
    //add
    let addResult = AddArray(mat1: embeddingsResponseBuffer, mat2: bufferAttentionOutput, size: tokens.count * 2560, commandBuffer: commandBuffer, metal: metal, metalFunction: kernelFunctionAddArrays)
    
    // apply rms norm
    let rmsNormResult = ApplyRmsNorm(entry: addResult, sizeEntry: colsEmbeddings, weightRMS: RMS, tokenCount: tokens.count, commandBuffer: commandBuffer, metal: metal, metalFunction: kernelFunctionRmsNorm)!
    
    //mult by Gate & Up
    let resultSize = matrixInfos(nbLine: tokens.count, nbColumn: 2560)
    
    let weightsSize = matrixInfos(nbLine : 2560, nbColumn : 6912)
    
    let gateResult = MultByW(WSize: weightsSize, inputSize: resultSize, weight: weightGate, input: rmsNormResult, commandBuffer: commandBuffer, metal: metal, metalFunction: kernelFunctionMult)
    
    let upResult = MultByW(WSize: weightsSize, inputSize: resultSize, weight: weightUp, input: rmsNormResult, commandBuffer: commandBuffer, metal: metal, metalFunction: kernelFunctionMult)
    
    let sizeIntermediaire = tokens.count * 6912
    ApplySiLUandMul(gate: gateResult, up: upResult, size: sizeIntermediaire, commandBuffer: commandBuffer, metal: metal, metalFunction: kernelFunctionSiluAndMul)
    
    let inputSizeDown = matrixInfos(nbLine: tokens.count, nbColumn: 6912)
    let weightsSizeDown = matrixInfos(nbLine: 6912, nbColumn: 2560)
    
    let downResult = MultByW(WSize: weightsSizeDown, inputSize: inputSizeDown, weight: weightDown, input: gateResult, commandBuffer: commandBuffer, metal: metal, metalFunction: kernelFunctionMult)
    _ = AddArray(mat1: addResult, mat2: downResult, size: tokens.count * 2560, commandBuffer: commandBuffer, metal: metal, metalFunction: kernelFunctionAddArrays)


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
