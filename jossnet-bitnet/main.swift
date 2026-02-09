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
    fatalError("Error : Function getFromEmbedding is not available")
}
print(Main()!)

func Main()->String?{
    var dataW : Data
    var dataWk : Data
    var dataWv : Data
    var dateEmbeddings : Data
    do{
        dataW = try Data(contentsOf: urlW)
        dataWk = try Data(contentsOf: urlWk)
        dataWv = try Data(contentsOf: urlWv)
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

    let embeddingsBuffer = metal.makeBuffer(bytes: embeddings, length: embeddings.count * MemoryLayout<Float>.size)!

    let embeddingsResponseBuffer = metal.makeBuffer(length: (colsEmbeddings * MemoryLayout<Float>.size) * tokens.count, options: .storageModeShared)!
    var tokensInput = metal.makeBuffer(bytes: tokens, length: tokens.count * MemoryLayout<UInt32>.size, options: .storageModeShared)!

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



    // rmsNorm
    let pipelineRms = try! metal.makeComputePipelineState(function: kernelFunctionRmsNorm)
    let encoderRms = commandBuffer.makeComputeCommandEncoder()!
    encoderRms.setComputePipelineState(pipelineRms);

    encoderRms.setBuffer(embeddingsResponseBuffer, offset: 0, index: 11);
    encoderRms.setBuffer(bufferNbColsEmbeddings, offset: 0, index: 12)

    let RmsGridSize = MTLSize(width: tokens.count, height: 1, depth: 1)
    let RmsThreadGroupSize = MTLSize(width: 1, height: 1, depth: 1)

    encoderRms.dispatchThreads(RmsGridSize, threadsPerThreadgroup: RmsThreadGroupSize)
    encoderRms.endEncoding()

    //mult
    let wSize = matrixInfos(nbLine : 2560, nbColumn : 2560)
    let InputSize = matrixInfos(nbLine : tokens.count, nbColumn : 2560)
    let bufferA = MultByW(WSize: wSize, inputSize: InputSize, weight: weight, input: embeddingsResponseBuffer, commandBuffer: commandBuffer)

    // Start

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    let rawPointer = embeddingsResponseBuffer.contents()

    let floatPointer = rawPointer.bindMemory(to: Float.self, capacity: Int(colsEmbeddings) * tokens.count)

    let bufferPointer = UnsafeBufferPointer(start: floatPointer, count: Int(colsEmbeddings) * tokens.count)

    let resultArray = Array(bufferPointer)

    let rawPointerMultAns = bufferA!.contents()

    let floatPointerMultAns = rawPointerMultAns.bindMemory(to: Float.self, capacity: Int(colsEmbeddings) * tokens.count)

    let bufferPointerMultAns = UnsafeBufferPointer(start: floatPointerMultAns, count: Int(colsEmbeddings) * tokens.count)

    let resultArrayMultAns = Array(bufferPointerMultAns)

    print("Nb tokens : \(resultArray.count/2560)")

    let stride = Int(colsEmbeddings)
    if tokens.count > 1 {
        print("Valeur Token 0 [0] : \(resultArray[0])")
        print("Valeur Token 1 [0] : \(resultArray[stride])")
    }

    print("Here result mult :\n \(resultArrayMultAns)")
    return("ok")
}



func MultByW(WSize: matrixInfos, inputSize: matrixInfos, weight : [Int8], input : MTLBuffer, commandBuffer: MTLCommandBuffer) -> MTLBuffer?{
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
