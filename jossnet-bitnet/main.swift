//
//  main.swift
//  jossnet-bitnet
//
//  Created by FIGUEIRAS Jossua on 31/01/2026.
//

import Foundation
import Metal

let path = URL(fileURLWithPath: "/Users/jossua/Desktop/jossnet-bitnet-resources/")

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
        let layerDir = path.appendingPathComponent("/layers/layer_\(layerIndex)/")
        
        let weightQ = loadInt8Matrix(path: layerDir.appendingPathComponent("W_q.bin").path)
        let weightK = loadInt8Matrix(path: layerDir.appendingPathComponent("W_k.bin").path)
        let weightV = loadInt8Matrix(path: layerDir.appendingPathComponent("W_v.bin").path)
        let weightO = loadInt8Matrix(path: layerDir.appendingPathComponent("W_o.bin").path)
        let weightGate = loadInt8Matrix(path: layerDir.appendingPathComponent("W_gate.bin").path)
        let weightUp = loadInt8Matrix(path: layerDir.appendingPathComponent("W_up.bin").path)
        let weightDown = loadInt8Matrix(path: layerDir.appendingPathComponent("W_down.bin").path)
        let rmsInput = loadFloatMatrix(path: layerDir.appendingPathComponent("RMS_input.bin").path)
        let rmsAttnSub = loadFloatMatrix(path: layerDir.appendingPathComponent("RMS_attn.bin").path)
        let rmsPostAttn = loadFloatMatrix(path: layerDir.appendingPathComponent("RMS_post_attn.bin").path)
        let rmsMlpSub = loadFloatMatrix(path: layerDir.appendingPathComponent("RMS_mlp_sub.bin").path)
        let scaleQ = loadFloatMatrix(path: layerDir.appendingPathComponent("Scale_q.bin").path)

        let scaleK = loadFloatMatrix(path: layerDir.appendingPathComponent("Scale_k.bin").path)
        let scaleO = loadFloatMatrix(path: layerDir.appendingPathComponent("Scale_o.bin").path)
        let scaleV = loadFloatMatrix(path: layerDir.appendingPathComponent("Scale_v.bin").path)
        let scaleUp = loadFloatMatrix(path: layerDir.appendingPathComponent("Scale_up.bin").path)
        let scaleDown = loadFloatMatrix(path: layerDir.appendingPathComponent("Scale_down.bin").path)
        let scaleGate = loadFloatMatrix(path: layerDir.appendingPathComponent("Scale_gate.bin").path)
        
        // create buffers
        
        let bufferWeightQ = metal.makeBuffer(bytes: weightQ, length: weightQ.count * MemoryLayout<Int8>.size, options: .storageModeShared)!
        let bufferWeightK = metal.makeBuffer(bytes: weightK, length: weightK.count * MemoryLayout<Int8>.size, options: .storageModeShared)!
        let bufferWeightO = metal.makeBuffer(bytes: weightO, length: weightO.count * MemoryLayout<Int8>.size, options: .storageModeShared)!
        let bufferWeightV = metal.makeBuffer(bytes: weightV, length: weightV.count * MemoryLayout<Int8>.size, options: .storageModeShared)!
        let bufferWeightGate = metal.makeBuffer(bytes: weightGate, length: weightGate.count * MemoryLayout<Int8>.size, options: .storageModeShared)!
        let bufferWeightUp = metal.makeBuffer(bytes: weightUp, length: weightUp.count * MemoryLayout<Int8>.size, options: .storageModeShared)!
        let bufferWeightDown = metal.makeBuffer(bytes: weightDown, length: weightUp.count * MemoryLayout<Int8>.size, options: .storageModeShared)!
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
        
        let layerResources : LayersResources = LayersResources(bufferWeightQ: bufferWeightQ,
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
                                                bufferScaleGate : bufferScaleGate)
        layerResourcesList.append(layerResources)
    }
    print("Loading weights: \(CFAbsoluteTimeGetCurrent() - startWeightLoading) secondes")
    
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
        let start = CFAbsoluteTimeGetCurrent()
        
        //embeddings

        let embCommandBuffer = queue.makeCommandBuffer()!
        guard let embeddingsResponseBuffer = Embedding(tokens: tokens, colsEmbeddings: colsEmbeddings, embeddingsBuffer: embeddingsBuffer, commandBuffer: embCommandBuffer, metal: metal, pipeline: pipelineGetFromEmbedding) else{
            return nil
        }
        
        var currentHiddenState: MTLBuffer? = embeddingsResponseBuffer
        
        embCommandBuffer.commit()
        embCommandBuffer.waitUntilCompleted()
        
        print("prediction \(no_prediction)")
     
        let layerCommandBuffer = queue.makeCommandBuffer()!
        for layerIndex in 0..<30{
            
            autoreleasepool {
                
                
                let normalizedInput = ApplyRmsNorm(entry: currentHiddenState, sizeEntry: colsEmbeddings, weightRMS: layerResourcesList[layerIndex].bufferRMSInput, tokenCount: tokens.count, commandBuffer: layerCommandBuffer, metal: metal, pipeline: pipelineRmsNorm)!
                
                
                let quantizedData = QuantizeActivations(entry: normalizedInput, nbCols: colsEmbeddings, nbTokens: tokens.count, commandBuffer: layerCommandBuffer, metal: metal, pipeline: pipelineQuantize)!
                
                let inputQuant = quantizedData.quantized
                let scaleX = quantizedData.scales
                
                //mult
                let InputSize = matrixInfos(nbLine : tokens.count, nbColumn : 2560)
                let wSize = matrixInfos(nbLine : 2560, nbColumn : 2560)
                let wVSize = matrixInfos(nbLine : 640, nbColumn : 2560)
                let wKSize = matrixInfos(nbLine : 640, nbColumn : 2560)
                
                var bufferQ = MultByW(WSize: wSize, inputSize: InputSize, bufferW: layerResourcesList[layerIndex].bufferWeightQ, scaleW: layerResourcesList[layerIndex].bufferScaleQ, inputQuant: inputQuant, scaleX: scaleX, commandBuffer: layerCommandBuffer, metal: metal, pipeline: pipelineMult)

                var bufferK = MultByW(WSize: wKSize, inputSize: InputSize, bufferW: layerResourcesList[layerIndex].bufferWeightK, scaleW: layerResourcesList[layerIndex].bufferScaleK, inputQuant: inputQuant, scaleX: scaleX, commandBuffer: layerCommandBuffer, metal: metal, pipeline: pipelineMult)

                let bufferV = MultByW(WSize: wVSize, inputSize: InputSize, bufferW: layerResourcesList[layerIndex].bufferWeightV, scaleW: layerResourcesList[layerIndex].bufferScaleV, inputQuant: inputQuant, scaleX: scaleX, commandBuffer: layerCommandBuffer, metal: metal, pipeline: pipelineMult)
            
                bufferQ = ApplyRoPE(buffer: bufferQ, nbTokens: tokens.count, dim: wSize.nbLine, commandBuffer: layerCommandBuffer, metal: metal, pipeline: pipelineRoPE)
                            
                bufferK = ApplyRoPE(buffer: bufferK, nbTokens: tokens.count, dim: wKSize.nbLine, commandBuffer: layerCommandBuffer, metal: metal, pipeline: pipelineRoPE)
                
                var resultAttention = AttentionScore(buffer1: bufferQ, buffer2: bufferK, nbToken: tokens.count, commandBuffer: layerCommandBuffer, metal: metal, pipeline: pipelineAttentionScore)
                
                
                let totalAttentionRows = 20 * tokens.count
                let attentionWidth = tokens.count
                
                resultAttention = SubstractMax(mat: resultAttention, nbcols: attentionWidth, nbToken: totalAttentionRows, commandBuffer: layerCommandBuffer, metal: metal, pipeline: pipelineSubstractMax, pipelineGetMax: pipelineMaxVector)
                
                resultAttention = DivideBySum(mat: resultAttention, nbcols: attentionWidth, nbToken: totalAttentionRows, commandBuffer: layerCommandBuffer, metal: metal, pipeline: pipelineDivideBySum, pipelineSumValue: pipelineSumVector)
                
                //context
                let contextBuffer = ComputeWeightedSum(mat: resultAttention, v: bufferV, nbTokens: tokens.count, commandBuffer: layerCommandBuffer, metal: metal, pipeline: pipelineWeightedSum)
                
                let normContextBuffer = ApplyRmsNorm(entry: contextBuffer, sizeEntry: 2560, weightRMS: layerResourcesList[layerIndex].bufferRMSAttnSub, tokenCount: tokens.count, commandBuffer: layerCommandBuffer, metal: metal, pipeline : pipelineRmsNorm)!
                
                // mult by o
                let wOSize = matrixInfos(nbLine: 2560, nbColumn: 2560)
                let contextSize = matrixInfos(nbLine: tokens.count, nbColumn: 2560)
                
                
                let quantizedContextData = QuantizeActivations(entry: normContextBuffer, nbCols: 2560, nbTokens: tokens.count, commandBuffer: layerCommandBuffer, metal: metal, pipeline: pipelineQuantize)!

                let bufferAttentionOutput = MultByW(WSize: wOSize, inputSize: contextSize, bufferW: layerResourcesList[layerIndex].bufferWeightO, scaleW: layerResourcesList[layerIndex].bufferScaleO, inputQuant: quantizedContextData.quantized, scaleX: quantizedContextData.scales, commandBuffer: layerCommandBuffer, metal: metal, pipeline: pipelineMult)
                
                //add
                let afterAttn = AddArray(mat1: currentHiddenState!, mat2: bufferAttentionOutput, size: tokens.count * 2560, commandBuffer: layerCommandBuffer, metal: metal, pipeline: pipelineAddArrays)
                
                
                // apply rms norm
                let rmsNormResult = ApplyRmsNorm(entry: afterAttn, sizeEntry: colsEmbeddings, weightRMS: layerResourcesList[layerIndex].bufferRMSPostAttn, tokenCount: tokens.count, commandBuffer: layerCommandBuffer, metal: metal, pipeline: pipelineRmsNorm)!
                
                //mult by Gate & Up
                let resultSize = matrixInfos(nbLine: tokens.count, nbColumn: 2560)
                
                let weightsSize = matrixInfos(nbLine : 6912, nbColumn : 2560)
                
                let quantizedMlpData = QuantizeActivations(entry: rmsNormResult, nbCols: 2560, nbTokens: tokens.count, commandBuffer: layerCommandBuffer, metal: metal, pipeline: pipelineQuantize)!

                let gateResult = MultByW(WSize: weightsSize, inputSize: resultSize, bufferW: layerResourcesList[layerIndex].bufferWeightGate, scaleW: layerResourcesList[layerIndex].bufferScaleGate, inputQuant: quantizedMlpData.quantized, scaleX: quantizedMlpData.scales, commandBuffer: layerCommandBuffer, metal: metal, pipeline: pipelineMult)

                let upResult = MultByW(WSize: weightsSize, inputSize: resultSize, bufferW: layerResourcesList[layerIndex].bufferWeightUp, scaleW: layerResourcesList[layerIndex].bufferScaleUp, inputQuant: quantizedMlpData.quantized, scaleX: quantizedMlpData.scales, commandBuffer: layerCommandBuffer, metal: metal, pipeline: pipelineMult)

                let sizeIntermediaire = tokens.count * 6912
                ApplySiLUandMul(gate: gateResult, up: upResult, size: sizeIntermediaire, commandBuffer: layerCommandBuffer, metal: metal, pipeline: pipelineRelu2)
                
                let normDownInput = ApplyRmsNorm(entry: gateResult, sizeEntry: 6912, weightRMS: layerResourcesList[layerIndex].bufferRMSMlpSub, tokenCount: tokens.count, commandBuffer: layerCommandBuffer, metal: metal, pipeline: pipelineRmsNorm)!
                
                let quantizedDownData = QuantizeActivations(entry: normDownInput, nbCols: 6912, nbTokens: tokens.count, commandBuffer: layerCommandBuffer, metal: metal, pipeline: pipelineQuantize)!
                
                let inputSizeDown = matrixInfos(nbLine: tokens.count, nbColumn: 6912)
                let weightsSizeDown = matrixInfos(nbLine: 2560, nbColumn: 6912)
                
                let downResult = MultByW(WSize: weightsSizeDown, inputSize: inputSizeDown, bufferW: layerResourcesList[layerIndex].bufferWeightDown, scaleW: layerResourcesList[layerIndex].bufferScaleDown, inputQuant: quantizedDownData.quantized, scaleX: quantizedDownData.scales, commandBuffer: layerCommandBuffer, metal: metal, pipeline: pipelineMult)
                
                currentHiddenState = AddArray(mat1: afterAttn, mat2: downResult, size: tokens.count * 2560, commandBuffer: layerCommandBuffer, metal: metal, pipeline: pipelineAddArrays)
                
            }
        }
        layerCommandBuffer.commit()
        layerCommandBuffer.waitUntilCompleted()
        
        let finalCmdBuffer = queue.makeCommandBuffer()!
        let resultFinalRms = ApplyRmsNorm(entry: currentHiddenState, sizeEntry: colsEmbeddings, weightRMS: rmsFinalBuffer, tokenCount: tokens.count, commandBuffer: finalCmdBuffer, metal: metal, pipeline: pipelineRmsNorm)

        let logitsBuffer = ComputeLogits(finalVectors: resultFinalRms, embeddings: lmHeadBuffer, nbTokens: tokens.count, dim: colsEmbeddings, commandBuffer: finalCmdBuffer, metal: metal, pipeline: pipelineComputeLogits)
        
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
