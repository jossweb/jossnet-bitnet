//
//  main.swift
//  jossnet-bitnet
//
//  Created by FIGUEIRAS Jossua on 31/01/2026.
//

import Foundation
import Metal

let urlX = URL(fileURLWithPath: "/Users/jossua/Documents/jossnet-bitnet/py/input_X.bin")
let urlW = URL(fileURLWithPath: "/Users/jossua/Documents/jossnet-bitnet/py/weights_W.bin")
let urlExpectedResponses = URL(fileURLWithPath: "/Users/jossua/Documents/jossnet-bitnet/py/output/expected_Y.bin")
let urlEmbeddings = URL(fileURLWithPath: "/Users/jossua/Documents/jossnet-bitnet/py/embeddings.bin")

let dataX = try Data(contentsOf: urlX)
let dataW = try Data(contentsOf: urlW)
let dataExpectedResponses = try Data(contentsOf: urlExpectedResponses)
let dateEmbeddings = try Data(contentsOf: urlEmbeddings)

let inputX = dataX.withUnsafeBytes {
    ptr in
    Array(UnsafeBufferPointer(start: ptr.bindMemory(to: Int8.self).baseAddress, count: 64))
}
let weight = dataW.withUnsafeBytes {
    ptr in
    Array(UnsafeBufferPointer(start: ptr.bindMemory(to: Float.self).baseAddress, count: 64))
}
let expectedResponses = dataExpectedResponses.withUnsafeBytes {
    ptr in
    Array(UnsafeBufferPointer(start: ptr.bindMemory(to: Float.self).baseAddress, count: 64))
}
// embeddings hardcode infos
let linesEmbeddings = 128256
var colsEmbeddings = 2560

let embeddingSize = linesEmbeddings * colsEmbeddings

let embeddings = dateEmbeddings.withUnsafeBytes {
    ptr in
    Array(UnsafeBufferPointer(start: ptr.bindMemory(to: Float.self).baseAddress, count: embeddingSize))
}

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

let pipelineState = try metal.makeComputePipelineState(function: kernelFunctionMult)

// prompt
let prompt = "Hi, my name is Jossua!"
let tokens = formatString(str : prompt)
print("IDs : \(tokens)")

//embeddings
let pipelineEmbeddings = try metal.makeComputePipelineState(function: kernelFunctionGetFromEmbedding)
let commandBuffer = queue.makeCommandBuffer()!
let encoderEmbeddings = commandBuffer.makeComputeCommandEncoder()!
encoderEmbeddings.setComputePipelineState(pipelineEmbeddings);

// buffer creation

let bufferX = metal.makeBuffer(bytes: inputX,
                               length: inputX.count * MemoryLayout<Float>.size,
                               options: .storageModeShared)!
let bufferW = metal.makeBuffer(bytes: weight,
                               length: weight.count * MemoryLayout<Int8>.size,
                               options: .storageModeShared)!

let bufferA = metal.makeBuffer(length: 4 * 8 * MemoryLayout<Float>.size, options: .storageModeShared)!

let embeddingsBuffer = metal.makeBuffer(bytes: embeddings, length: embeddings.count * MemoryLayout<Float>.size)!

let embeddingsResponseBuffer = metal.makeBuffer(length: (colsEmbeddings * MemoryLayout<Float>.size) * tokens.count, options: .storageModeShared)!

var tokensInput = metal.makeBuffer(bytes: tokens, length: tokens.count * MemoryLayout<UInt32>.size, options: .storageModeShared)!

let bufferNbColsEmbeddings = metal.makeBuffer(bytes: &colsEmbeddings, length: MemoryLayout<UInt32>.size, options: .storageModeShared)!

//embeddings buffer
encoderEmbeddings.setBuffer(embeddingsBuffer, offset: 0, index: 7)
encoderEmbeddings.setBuffer(embeddingsResponseBuffer, offset: 0, index: 8)
encoderEmbeddings.setBuffer(tokensInput, offset: 0, index: 9)
encoderEmbeddings.setBuffer(bufferNbColsEmbeddings, offset: 0, index: 10)


let gridSize = MTLSize(width: 8, height: 4, depth: 1)
let threadGroupSize = MTLSize(width: 1, height: 1, depth: 1)

let embeddingsGridSize = MTLSize(width: colsEmbeddings, height: tokens.count, depth: 1)
let embeddingsThreadGroupSize = MTLSize(width: 1, height: 1, depth: 1)

encoderEmbeddings.dispatchThreads(embeddingsGridSize, threadsPerThreadgroup: embeddingsThreadGroupSize)
encoderEmbeddings.endEncoding()



// rmsNorm
let pipelineRms = try metal.makeComputePipelineState(function: kernelFunctionRmsNorm)
let encoderRms = commandBuffer.makeComputeCommandEncoder()!
encoderRms.setComputePipelineState(pipelineRms);

encoderRms.setBuffer(embeddingsResponseBuffer, offset: 0, index: 11);
encoderRms.setBuffer(bufferNbColsEmbeddings, offset: 0, index: 12)

let RmsGridSize = MTLSize(width: tokens.count, height: 1, depth: 1)
let RmsThreadGroupSize = MTLSize(width: 1, height: 1, depth: 1)

encoderRms.dispatchThreads(RmsGridSize, threadsPerThreadgroup: RmsThreadGroupSize)
encoderRms.endEncoding()

commandBuffer.commit()
commandBuffer.waitUntilCompleted()

// mult mat part

/*var nbColW_val: UInt32 = 27648
var nbLineW_val: UInt32 = 640
var nbColX_val: UInt32 = 27648
var nbLineX_val: UInt32 = 4

let bufferNbColX = metal.makeBuffer(bytes: &nbColX_val, length: MemoryLayout<UInt32>.size, options: .storageModeShared)!
let bufferNbLineX = metal.makeBuffer(bytes: &nbLineX_val, length: MemoryLayout<UInt32>.size, options: .storageModeShared)!
let bufferNbColW = metal.makeBuffer(bytes: &nbColW_val, length: MemoryLayout<UInt32>.size, options: .storageModeShared)!
let bufferNbLineW = metal.makeBuffer(bytes: &nbLineW_val, length: MemoryLayout<UInt32>.size, options: .storageModeShared)!

let encoder = commandBuffer.makeComputeCommandEncoder()!
encoder.setComputePipelineState(pipelineState);

encoder.setBuffer(bufferX, offset: 0, index: 0);
encoder.setBuffer(bufferW, offset: 0, index: 1);
encoder.setBuffer(bufferA, offset: 0, index: 6);

encoder.setBuffer(bufferNbColX, offset: 0, index: 2)
encoder.setBuffer(bufferNbLineX, offset: 0, index: 3)
encoder.setBuffer(bufferNbColW, offset: 0, index: 4)
encoder.setBuffer(bufferNbLineW, offset: 0, index: 5)

encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
encoder.endEncoding()*/

// Start 



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

print("Finished")
