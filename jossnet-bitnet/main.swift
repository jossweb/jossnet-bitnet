//
//  main.swift
//  jossnet-bitnet
//
//  Created by FIGUEIRAS Jossua on 31/01/2026.
//

import Foundation
import Metal

let path = URL(fileURLWithPath: "/Users/jossua/Desktop/jossnet-bitnet-resources/")
let NBMAXTOKEN = 1000

struct matrixInfos{
    let nbLine : Int
    let nbColumn : Int
}

func loadBufferFromInt8File(path: String, metal: MTLDevice) -> MTLBuffer {
    let url = URL(fileURLWithPath: path)
    guard let data = try? Data(contentsOf: url, options: .alwaysMapped) else { fatalError("File not found: \(path)") }
    
    return data.withUnsafeBytes { ptr in
        return metal.makeBuffer(bytes: ptr.baseAddress!, length: data.count, options: .storageModeShared)!
    }
}

func loadBufferFromFloatFile(path: String, metal: MTLDevice) -> MTLBuffer {
    let url = URL(fileURLWithPath: path)
    guard let data = try? Data(contentsOf: url, options: .alwaysMapped) else { fatalError("File not found: \(path)") }
    
    return data.withUnsafeBytes { ptr in
        return metal.makeBuffer(bytes: ptr.baseAddress!, length: data.count, options: .storageModeShared)!
    }
}

struct LayersResources{
    let bufferWeightQ : MTLBuffer;
    let bufferWeightK : MTLBuffer;
    let bufferWeightO : MTLBuffer;
    let bufferWeightV : MTLBuffer;
    let bufferWeightGate : MTLBuffer;
    let bufferWeightUp : MTLBuffer;
    let bufferWeightDown : MTLBuffer;
    let bufferRMSInput : MTLBuffer;
    let bufferRMSAttnSub : MTLBuffer;
    let bufferRMSPostAttn : MTLBuffer;
    let bufferRMSMlpSub : MTLBuffer;
    let bufferScaleQ : MTLBuffer;
    let bufferScaleK : MTLBuffer;
    let bufferScaleO : MTLBuffer;
    let bufferScaleV : MTLBuffer;
    let bufferScaleUp : MTLBuffer;
    let bufferScaleDown: MTLBuffer;
    let bufferScaleGate : MTLBuffer;
}

guard let metal = MTLCreateSystemDefaultDevice() else
{
    fatalError("Error : Metal is not available")
}

let queue = metal.makeCommandQueue()!

let library = metal.makeDefaultLibrary()!

guard let kernelFunctionMult = library.makeFunction(name: "Mult") else {
    fatalError("Error : Function mult is not available")
}
guard let kernelFunctionGetFromEmbedding = library.makeFunction(name: "GetFromEmbedding") else {
    fatalError("Error : Function getFromEmbedding is not available")
}
guard let kernelFunctionRmsNorm = library.makeFunction(name: "RmsNorm") else {
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
guard let kernelFunctionWeightedSum = library.makeFunction(name: "WeightedSum") else {
    fatalError("Error : Function divideBySum is not available")
}
guard let kernelFunctionAddArrays = library.makeFunction(name: "AddArrays") else {
    fatalError("Error : Function divideBySum is not available")
}
guard let kernelFunctionRelu2 = library.makeFunction(name: "Relu2") else {
    fatalError("Error : Function siluAndMul is not available")
}
guard let kernelFunctionComputeLogits = library.makeFunction(name: "ComputeLogits") else {
    fatalError("Error : Function computeLogits is not available")
}
guard let kernelFunctionQuantize = library.makeFunction(name: "QuantizeActivations") else {
    fatalError("Error : Function quantizeActivations is not available")
}


print(Main()!)

func Main()->String?{
    
    var genTimeList: [Double] = []
    
    print("Compiling metal ...")
    
    let startMetalCompiling = CFAbsoluteTimeGetCurrent()
    
    let pipelineMult = try! metal.makeComputePipelineState(function: kernelFunctionMult)
    let pipelineGetFromEmbedding = try! metal.makeComputePipelineState(function: kernelFunctionGetFromEmbedding)
    let pipelineRmsNorm = try! metal.makeComputePipelineState(function: kernelFunctionRmsNorm)
    let pipelineRoPE = try! metal.makeComputePipelineState(function: kernelFunctionRoPE)
    let pipelineAttentionScore = try! metal.makeComputePipelineState(function: kernelFunctionAttentionScore)
    let pipelineMaxVector = try! metal.makeComputePipelineState(function: kernelFunctionSearchMaxVector)
    let pipelineSubstractMax = try! metal.makeComputePipelineState(function: kernelFunctionSubstractMax)
    let pipelineSumVector = try! metal.makeComputePipelineState(function: kernelFunctionSumVector)
    let pipelineDivideBySum = try! metal.makeComputePipelineState(function: kernelFunctionDivideBySum)
    let pipelineWeightedSum = try! metal.makeComputePipelineState(function: kernelFunctionWeightedSum)
    let pipelineAddArrays = try! metal.makeComputePipelineState(function: kernelFunctionAddArrays)
    let pipelineRelu2 = try! metal.makeComputePipelineState(function: kernelFunctionRelu2)
    let pipelineComputeLogits = try! metal.makeComputePipelineState(function: kernelFunctionComputeLogits)
    let pipelineQuantize = try! metal.makeComputePipelineState(function: kernelFunctionQuantize)
    
    print("Compiling metal: \(CFAbsoluteTimeGetCurrent() - startMetalCompiling) secondes")
    
    print("Loading resources ...")
    let startWeightLoading = CFAbsoluteTimeGetCurrent()
    var layerResourcesList : [LayersResources] = []
    for layerIndex in 0..<30 {
            autoreleasepool {
                let layerDir = path.appendingPathComponent("/layers/layer_\(layerIndex)/")
                
                let bufferWeightQ = loadBufferFromInt8File(path: layerDir.appendingPathComponent("W_q.bin").path, metal: metal)
                let bufferWeightK = loadBufferFromInt8File(path: layerDir.appendingPathComponent("W_k.bin").path, metal: metal)
                let bufferWeightV = loadBufferFromInt8File(path: layerDir.appendingPathComponent("W_v.bin").path, metal: metal)
                let bufferWeightO = loadBufferFromInt8File(path: layerDir.appendingPathComponent("W_o.bin").path, metal: metal)
                let bufferWeightGate = loadBufferFromInt8File(path: layerDir.appendingPathComponent("W_gate.bin").path, metal: metal)
                let bufferWeightUp = loadBufferFromInt8File(path: layerDir.appendingPathComponent("W_up.bin").path, metal: metal)
                let bufferWeightDown = loadBufferFromInt8File(path: layerDir.appendingPathComponent("W_down.bin").path, metal: metal)
                
                let bufferRMSInput = loadBufferFromFloatFile(path: layerDir.appendingPathComponent("RMS_input.bin").path, metal: metal)
                let bufferRMSAttnSub = loadBufferFromFloatFile(path: layerDir.appendingPathComponent("RMS_attn.bin").path, metal: metal)
                let bufferRMSPostAttn = loadBufferFromFloatFile(path: layerDir.appendingPathComponent("RMS_post_attn.bin").path, metal: metal)
                let bufferRMSMlpSub = loadBufferFromFloatFile(path: layerDir.appendingPathComponent("RMS_mlp_sub.bin").path, metal: metal)
                
                let bufferScaleQ = loadBufferFromFloatFile(path: layerDir.appendingPathComponent("Scale_q.bin").path, metal: metal)
                let bufferScaleK = loadBufferFromFloatFile(path: layerDir.appendingPathComponent("Scale_k.bin").path, metal: metal)
                let bufferScaleO = loadBufferFromFloatFile(path: layerDir.appendingPathComponent("Scale_o.bin").path, metal: metal)
                let bufferScaleV = loadBufferFromFloatFile(path: layerDir.appendingPathComponent("Scale_v.bin").path, metal: metal)
                let bufferScaleUp = loadBufferFromFloatFile(path: layerDir.appendingPathComponent("Scale_up.bin").path, metal: metal)
                let bufferScaleDown = loadBufferFromFloatFile(path: layerDir.appendingPathComponent("Scale_down.bin").path, metal: metal)
                let bufferScaleGate = loadBufferFromFloatFile(path: layerDir.appendingPathComponent("Scale_gate.bin").path, metal: metal)
                
                let layerResources = LayersResources(
                    bufferWeightQ: bufferWeightQ,
                    bufferWeightK: bufferWeightK,
                    bufferWeightO: bufferWeightO,
                    bufferWeightV: bufferWeightV,
                    bufferWeightGate : bufferWeightGate,
                    bufferWeightUp : bufferWeightUp,
                    bufferWeightDown : bufferWeightDown,
                    bufferRMSInput : bufferRMSInput,
                    bufferRMSAttnSub : bufferRMSAttnSub,
                    bufferRMSPostAttn : bufferRMSPostAttn,
                    bufferRMSMlpSub : bufferRMSMlpSub,
                    bufferScaleQ : bufferScaleQ,
                    bufferScaleK : bufferScaleK,
                    bufferScaleO : bufferScaleO,
                    bufferScaleV : bufferScaleV,
                    bufferScaleUp : bufferScaleUp,
                    bufferScaleDown : bufferScaleDown,
                    bufferScaleGate : bufferScaleGate
                )
                layerResourcesList.append(layerResources)
            }
        }
    print("Loading weights: \(CFAbsoluteTimeGetCurrent() - startWeightLoading) secondes")
    

    let colsEmbeddings = 2560

    let lmHeadBuffer = loadBufferFromFloatFile(path: urlLMHead.path, metal: metal)
    let embeddingsBuffer = loadBufferFromFloatFile(path: urlEmbeddings.path, metal: metal)
    let rmsFinalBuffer = loadBufferFromFloatFile(path: urlRMSFinal.path, metal: metal)

    let prompt = "The capital of France is"
    var tokens: [UInt32] = [128000, 25, 578, 6864, 315, 9822, 374, 128009, 72803, 25, 220]
    print("IDs : \(tokens)")
    
    //var kCaches: [MTLBuffer?] = Array(repeating: nil, count: 26)
    //var vCaches: [MTLBuffer?] = Array(repeating: nil, count: 26)
    
    print("Creating Buffers")

    let wSize = matrixInfos(nbLine : 2560, nbColumn : 2560)
    let wKVSize = matrixInfos(nbLine : 640, nbColumn : 2560)
    
    let maxDim = 6912
    let maxHiddenDim = 2560
    let kvDim = 640
    
    let workBufferWA = metal.makeBuffer(length: NBMAXTOKEN * maxDim * 4, options: .storageModeShared)!
    let workBufferWB = metal.makeBuffer(length: NBMAXTOKEN * maxDim * 4, options: .storageModeShared)!

    let workBufferVA = metal.makeBuffer(length: NBMAXTOKEN * kvDim * 4, options: .storageModeShared)!
    let workBufferKA = metal.makeBuffer(length: NBMAXTOKEN * kvDim * 4, options: .storageModeShared)!

    let attentionBuffer = metal.makeBuffer(length: 4 * 20 * NBMAXTOKEN * NBMAXTOKEN, options: .storageModeShared)!
    let attentionMaxBuffer = metal.makeBuffer(length: 4 * 20 * NBMAXTOKEN, options: .storageModeShared)!
    let attentionSumBuffer = metal.makeBuffer(length: 4 * 20 * NBMAXTOKEN, options: .storageModeShared)!

    let workBufferQuantized = metal.makeBuffer(length: NBMAXTOKEN * maxDim * MemoryLayout<Int8>.size, options: .storageModeShared)!
    let workBufferScales = metal.makeBuffer(length: NBMAXTOKEN * 4, options: .storageModeShared)!

    let residualBufferB = metal.makeBuffer(length: NBMAXTOKEN * maxHiddenDim * 4, options: .storageModeShared)!
    let residualBufferA = metal.makeBuffer(length: NBMAXTOKEN * maxHiddenDim * 4, options: .storageModeShared)!

    let workBufferLogits = metal.makeBuffer(length: 128256 * 4, options: .storageModeShared)!
    
    print("Generating ...")
    
    for no_prediction in 0..<10{
        let start = CFAbsoluteTimeGetCurrent()
        
        //embeddings

        let embCommandBuffer = queue.makeCommandBuffer()!
        
        Embedding(tokens: tokens,colsEmbeddings: colsEmbeddings, embeddingsBuffer: embeddingsBuffer, commandBuffer: embCommandBuffer, metal: metal, pipeline: pipelineGetFromEmbedding, answer: residualBufferA)
        
        embCommandBuffer.commit()
        embCommandBuffer.waitUntilCompleted()
        
        var currentHiddenState: MTLBuffer? = residualBufferA
        
        print("prediction \(no_prediction)")
     
        let layerCommandBuffer = queue.makeCommandBuffer()!
        
        for layerIndex in 0..<30{
            
            autoreleasepool {
                let inputSize = matrixInfos(nbLine : tokens.count, nbColumn : 2560)
                
                
                try! ApplyRmsNorm(entry: currentHiddenState, sizeEntry: colsEmbeddings, weightRMS: layerResourcesList[layerIndex].bufferRMSInput, tokenCount: tokens.count, commandBuffer: layerCommandBuffer, metal: metal, pipeline: pipelineRmsNorm, answer: workBufferWA)
                
                
                try! QuantizeActivations(entry: workBufferWA, nbCols: colsEmbeddings, nbTokens: tokens.count, outBufferQuantized: workBufferQuantized, outBufferScales: workBufferScales, commandBuffer: layerCommandBuffer, pipeline: pipelineQuantize)

                try! MultByW(WSize: wSize, inputSize: inputSize, bufferW: layerResourcesList[layerIndex].bufferWeightQ, scaleW: layerResourcesList[layerIndex].bufferScaleQ, inputQuant: workBufferQuantized, scaleX: workBufferScales, commandBuffer: layerCommandBuffer, metal: metal, pipeline: pipelineMult, answer: workBufferWA)

                try! MultByW(WSize: wKVSize, inputSize: inputSize, bufferW: layerResourcesList[layerIndex].bufferWeightK, scaleW: layerResourcesList[layerIndex].bufferScaleK, inputQuant: workBufferQuantized, scaleX: workBufferScales, commandBuffer: layerCommandBuffer, metal: metal, pipeline: pipelineMult, answer: workBufferKA)

                try! MultByW(WSize: wKVSize, inputSize: inputSize, bufferW: layerResourcesList[layerIndex].bufferWeightV, scaleW: layerResourcesList[layerIndex].bufferScaleV, inputQuant: workBufferQuantized, scaleX: workBufferScales, commandBuffer: layerCommandBuffer, metal: metal, pipeline: pipelineMult, answer: workBufferVA)
            
                try! ApplyRoPE(buffer: workBufferWA, nbTokens: tokens.count, dim: wSize.nbLine, commandBuffer: layerCommandBuffer, metal: metal, pipeline: pipelineRoPE)
                            
                try! ApplyRoPE(buffer: workBufferKA, nbTokens: tokens.count, dim: wKVSize.nbLine, commandBuffer: layerCommandBuffer, metal: metal, pipeline: pipelineRoPE)
                
                try! AttentionScore(buffer1: workBufferWA, buffer2: workBufferKA, nbToken: tokens.count, commandBuffer: layerCommandBuffer, metal: metal, pipeline: pipelineAttentionScore, answer: attentionBuffer)
                
                
                let totalAttentionRows = 20 * tokens.count
                let attentionWidth = tokens.count
                
                try! SubstractMax(mat: attentionBuffer, nbcols: attentionWidth, nbToken: totalAttentionRows, commandBuffer: layerCommandBuffer, metal: metal, pipeline: pipelineSubstractMax, pipelineGetMax: pipelineMaxVector, maxBuffer : attentionMaxBuffer)
                
                try! DivideBySum(mat: attentionBuffer, nbcols: attentionWidth, nbToken: totalAttentionRows, commandBuffer: layerCommandBuffer, metal: metal, pipeline: pipelineDivideBySum, pipelineSumValue: pipelineSumVector, sumBuffer : attentionSumBuffer)
                
                //context
                try! ComputeWeightedSum(mat: attentionBuffer, v: workBufferVA, nbTokens: tokens.count, commandBuffer: layerCommandBuffer, metal: metal, pipeline: pipelineWeightedSum, answer: workBufferWB)
                
                try! ApplyRmsNorm(entry: workBufferWB, sizeEntry: 2560, weightRMS: layerResourcesList[layerIndex].bufferRMSAttnSub, tokenCount: tokens.count, commandBuffer: layerCommandBuffer, metal: metal, pipeline : pipelineRmsNorm, answer : workBufferWA)
                
                // mult by o
                let wOSize = matrixInfos(nbLine: 2560, nbColumn: 2560)
                let contextSize = matrixInfos(nbLine: tokens.count, nbColumn: 2560)
                
                
                try! QuantizeActivations(entry: workBufferWA, nbCols: 2560, nbTokens: tokens.count,outBufferQuantized: workBufferQuantized, outBufferScales: workBufferScales, commandBuffer: layerCommandBuffer, pipeline: pipelineQuantize)

                try! MultByW(WSize: wOSize, inputSize: contextSize, bufferW: layerResourcesList[layerIndex].bufferWeightO, scaleW: layerResourcesList[layerIndex].bufferScaleO, inputQuant: workBufferQuantized, scaleX: workBufferScales, commandBuffer: layerCommandBuffer, metal: metal, pipeline: pipelineMult, answer: workBufferWA)
                
                //add
                try! AddArray(mat1: currentHiddenState!, mat2: workBufferWA, size: tokens.count * 2560, commandBuffer: layerCommandBuffer, metal: metal, pipeline: pipelineAddArrays, answer: residualBufferB)
                currentHiddenState = residualBufferB
                
                // apply rms norm
                try! ApplyRmsNorm(entry: currentHiddenState, sizeEntry: colsEmbeddings, weightRMS: layerResourcesList[layerIndex].bufferRMSPostAttn, tokenCount: tokens.count, commandBuffer: layerCommandBuffer, metal: metal, pipeline: pipelineRmsNorm, answer: workBufferWB)
                
                //mult by Gate & Up
                let resultSize = matrixInfos(nbLine: tokens.count, nbColumn: 2560)
                
                let weightsSize = matrixInfos(nbLine : 6912, nbColumn : 2560)
                
                try! QuantizeActivations(entry: workBufferWB, nbCols: 2560, nbTokens: tokens.count, outBufferQuantized: workBufferQuantized, outBufferScales: workBufferScales, commandBuffer: layerCommandBuffer, pipeline: pipelineQuantize)

                try! MultByW(WSize: weightsSize, inputSize: resultSize, bufferW: layerResourcesList[layerIndex].bufferWeightGate, scaleW: layerResourcesList[layerIndex].bufferScaleGate, inputQuant: workBufferQuantized, scaleX: workBufferScales, commandBuffer: layerCommandBuffer, metal: metal, pipeline: pipelineMult, answer: workBufferWA)

                try! MultByW(WSize: weightsSize, inputSize: resultSize, bufferW: layerResourcesList[layerIndex].bufferWeightUp, scaleW: layerResourcesList[layerIndex].bufferScaleUp, inputQuant: workBufferQuantized, scaleX: workBufferScales, commandBuffer: layerCommandBuffer, metal: metal, pipeline: pipelineMult, answer: workBufferWB)

                let sizeIntermediaire = tokens.count * 6912
                ApplySiLUandMul(gate: workBufferWA, up: workBufferWB, size: sizeIntermediaire, commandBuffer: layerCommandBuffer, metal: metal, pipeline: pipelineRelu2)
                
                try! ApplyRmsNorm(entry: workBufferWA, sizeEntry: 6912, weightRMS: layerResourcesList[layerIndex].bufferRMSMlpSub, tokenCount: tokens.count, commandBuffer: layerCommandBuffer, metal: metal, pipeline: pipelineRmsNorm, answer: workBufferWB)
                
                try! QuantizeActivations(entry: workBufferWB, nbCols: 6912, nbTokens: tokens.count, outBufferQuantized: workBufferQuantized, outBufferScales: workBufferScales, commandBuffer: layerCommandBuffer, pipeline: pipelineQuantize)
                
                let inputSizeDown = matrixInfos(nbLine: tokens.count, nbColumn: 6912)
                let weightsSizeDown = matrixInfos(nbLine: 2560, nbColumn: 6912)
                
                try! MultByW(WSize: weightsSizeDown, inputSize: inputSizeDown, bufferW: layerResourcesList[layerIndex].bufferWeightDown, scaleW: layerResourcesList[layerIndex].bufferScaleDown, inputQuant: workBufferQuantized, scaleX: workBufferScales, commandBuffer: layerCommandBuffer, metal: metal, pipeline: pipelineMult, answer: workBufferWA)
                
                try! AddArray(mat1: currentHiddenState, mat2: workBufferWA, size: tokens.count * 2560, commandBuffer: layerCommandBuffer, metal: metal, pipeline: pipelineAddArrays, answer: residualBufferA)
                
                currentHiddenState = residualBufferA
                
            }
        }
        layerCommandBuffer.commit()
        layerCommandBuffer.waitUntilCompleted()
        
        let finalCmdBuffer = queue.makeCommandBuffer()!
        try! ApplyRmsNorm(entry: currentHiddenState, sizeEntry: colsEmbeddings, weightRMS: rmsFinalBuffer, tokenCount: tokens.count, commandBuffer: finalCmdBuffer, metal: metal, pipeline: pipelineRmsNorm, answer: workBufferWA)

        try! ComputeLogits(finalVectors: workBufferWA, embeddings: lmHeadBuffer, nbTokens: tokens.count, dim: colsEmbeddings, commandBuffer: finalCmdBuffer, metal: metal, pipeline: pipelineComputeLogits, answer: workBufferLogits)
        
        // Start

        finalCmdBuffer.commit()
        finalCmdBuffer.waitUntilCompleted()

        let logitsArray = GetFromBuffer(buffer: workBufferLogits, size: 128256)
        
        var bestScore: Float = -Float.greatestFiniteMagnitude
        var bestTokenID = 0
        
        for i in 0..<logitsArray.count {
            if logitsArray[i] > bestScore {
                bestScore = logitsArray[i]
                bestTokenID = i
            }
        }
        tokens.append(UInt32(bestTokenID))
        print("Token : \(bestTokenID)")
        genTimeList.append(CFAbsoluteTimeGetCurrent() - start)
        print("Time spend: \(genTimeList[genTimeList.count - 1]) secondes")

        
    }
    print("exit : \(tokens)")
    var total: Double = 0
    for i in 0..<genTimeList.count {
        total += genTimeList[i]
    }
    total /= Double(genTimeList.count)
    print("avg token gen : \(total)")
    return("ok")
}
