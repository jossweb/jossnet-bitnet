//
//  main.swift
//  jossnet-bitnet
//
//  Created by FIGUEIRAS Jossua on 31/01/2026.
//

import Foundation
import Metal

let url_x = URL(fileURLWithPath: "/Users/jossua/Documents/jossnet-bitnet/py/output/input_X.bin")
let url_w = URL(fileURLWithPath: "/Users/jossua/Documents/jossnet-bitnet/py/output/weights_W.bin")
let url_expected_responses = URL(fileURLWithPath: "/Users/jossua/Documents/jossnet-bitnet/py/output/expected_Y.bin")

let data_x = try Data(contentsOf: url_x)
let data_w = try Data(contentsOf: url_w)
let data_expected_responses = try Data(contentsOf: url_expected_responses)

let input_x = data_x.withUnsafeBytes {
    ptr in
    Array(UnsafeBufferPointer(start: ptr.bindMemory(to: Float.self).baseAddress, count: 64))
}
let weight = data_w.withUnsafeBytes {
    ptr in
    Array(UnsafeBufferPointer(start: ptr.bindMemory(to: Float.self).baseAddress, count: 64))
}
let expected_responses = data_expected_responses.withUnsafeBytes {
    ptr in
    Array(UnsafeBufferPointer(start: ptr.bindMemory(to: Float.self).baseAddress, count: 64))
}

guard let device = MTLCreateSystemDefaultDevice()else
    {
        fatalError("Error : Metal is not available")
    }


