//
//  main.swift
//  jossnet-bitnet
//
//  Created by FIGUEIRAS Jossua on 31/01/2026.
//

import Foundation
import Metal

let urlX = URL(fileURLWithPath: "/Users/jossua/Documents/jossnet-bitnet/py/output/input_X.bin")
let urlW = URL(fileURLWithPath: "/Users/jossua/Documents/jossnet-bitnet/py/output/weights_W.bin")
let urlExpectedResponses = URL(fileURLWithPath: "/Users/jossua/Documents/jossnet-bitnet/py/output/expected_Y.bin")

let dataX = try Data(contentsOf: urlX)
let dataW = try Data(contentsOf: urlW)
let dataExpectedResponses = try Data(contentsOf: urlExpectedResponses)

let inputX = dataX.withUnsafeBytes {
    ptr in
    Array(UnsafeBufferPointer(start: ptr.bindMemory(to: Float.self).baseAddress, count: 64))
}
let weight = dataW.withUnsafeBytes {
    ptr in
    Array(UnsafeBufferPointer(start: ptr.bindMemory(to: Float.self).baseAddress, count: 64))
}
let expectedResponses = dataExpectedResponses.withUnsafeBytes {
    ptr in
    Array(UnsafeBufferPointer(start: ptr.bindMemory(to: Float.self).baseAddress, count: 64))
}

guard let metal = MTLCreateSystemDefaultDevice()else
    {
        fatalError("Error : Metal is not available")
    }

let queue = metal.makeCommandQueue()!

let library = metal.makeDefaultLibrary()!

guard let kernelFunction = library.makeFunction(name: "mult") else {
    fatalError("Error : Function mult is not available")
}

let pipelineState = try metal.makeComputePipelineState(function: kernelFunction)

let commandBuffer = queue.makeCommandBuffer()!

let encoder = commandBuffer.makeComputeCommandEncoder()!

encoder.setComputePipelineState(pipelineState);

// buffer creation

let bufferX = metal.makeBuffer(bytes: inputX,
                               length: inputX.count * MemoryLayout<Float>.size,
                               options: .storageModeShared)!
let bufferW = metal.makeBuffer(bytes: weight,
                               length: weight.count * MemoryLayout<Float>.size,
                               options: .storageModeShared)!

let bufferA = metal.makeBuffer(length: 4 * 8 * MemoryLayout<Float>.size, options: .storageModeShared)!


var nbColX_val: UInt32 = 16
var nbLineX_val: UInt32 = 4
var nbColW_val: UInt32 = 16
var nbLineW_val: UInt32 = 8

let bufferNbColX = metal.makeBuffer(bytes: &nbColX_val, length: MemoryLayout<UInt32>.size, options: .storageModeShared)!
let bufferNbLineX = metal.makeBuffer(bytes: &nbLineX_val, length: MemoryLayout<UInt32>.size, options: .storageModeShared)!
let bufferNbColW = metal.makeBuffer(bytes: &nbColW_val, length: MemoryLayout<UInt32>.size, options: .storageModeShared)!
let bufferNbLineW = metal.makeBuffer(bytes: &nbLineW_val, length: MemoryLayout<UInt32>.size, options: .storageModeShared)!


encoder.setBuffer(bufferX, offset: 0, index: 0);
encoder.setBuffer(bufferW, offset: 0, index: 1);
encoder.setBuffer(bufferA, offset: 0, index: 6);

encoder.setBuffer(bufferNbColX, offset: 0, index: 2)
encoder.setBuffer(bufferNbLineX, offset: 0, index: 3)
encoder.setBuffer(bufferNbColW, offset: 0, index: 4)
encoder.setBuffer(bufferNbLineW, offset: 0, index: 5)


let gridSize = MTLSize(width: 8, height: 4, depth: 1)
let threadGroupSize = MTLSize(width: 1, height: 1, depth: 1)

encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
encoder.endEncoding()

// Start 
commandBuffer.commit()
commandBuffer.waitUntilCompleted()

print("Finished")
