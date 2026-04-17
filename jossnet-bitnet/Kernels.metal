//
//  Kernels.metal
//  jossnet-bitnet
//
//  Created by FIGUEIRAS Jossua on 31/01/2026.
//
#include <metal_stdlib>

using namespace metal;


kernel void mult(device const char* X_quant [[ buffer(0) ]],
                 device const char* W [[ buffer(1) ]],
                 constant uint* dims [[ buffer(2) ]],
                 device float* answer [[ buffer(6) ]],
                 device const float* scale_W [[ buffer(7) ]],
                 device const float* scale_X [[ buffer(8) ]],
                 constant uint& scale_w_size [[ buffer(9) ]],
                 uint2 gid [[ thread_position_in_grid ]]) {
    
    uint K = dims[0];
    uint M = dims[1];
    uint N = dims[2];
    
    if (gid.x >= N || gid.y >= M) return;

    float sum = 0.0f;
    uint base_X = gid.y * K;
    uint base_W = gid.x * K;

    for(uint i = 0; i < K; i++) {
        sum += (float)X_quant[base_X + i] * (float)W[base_W + i];
    }
    
    answer[gid.y * N + gid.x] = sum * scale_W[gid.x % scale_w_size] * scale_X[gid.y];
}
kernel void quantizeActivations(device const float* X [[ buffer(0) ]],
                                device char* X_quant [[ buffer(1) ]],
                                device float* scales_X [[ buffer(2) ]],
                                constant uint& nbCols [[ buffer(3) ]],
                                uint gid [[ thread_position_in_grid ]]) {
    
    uint base_idx = gid * nbCols;
    
    float max_val = 1e-6f;
    for(uint i = 0; i < nbCols; i++) {
        float val = abs(X[base_idx + i]);
        if(val > max_val) {
            max_val = val;
        }
    }
    
    float scale_x = max_val / 127.0f;
    scales_X[gid] = scale_x;
    
    float inv_scale = 1.0f / scale_x;
    for(uint i = 0; i < nbCols; i++) {
        float scaled_val = X[base_idx + i] * inv_scale;
        int quantized = (int)round(scaled_val);
        quantized = quantized > 127 ? 127 : (quantized < -128 ? -128 : quantized);
        X_quant[base_idx + i] = (char)quantized;
    }
}
kernel void getFromEmbedding(device const float* embeddingTable [[ buffer(7) ]],
                             device float* answer [[ buffer(8) ]],
                             device const uint* index [[ buffer(9) ]],
                             constant uint& nbCols [[ buffer(10) ]],
                             uint2 gid [[ thread_position_in_grid ]]){
    // gid.y : actual token in tab
    // gid.x : index for actual token
    uint tokenId = index[gid.y];
    answer[gid.x + gid.y * nbCols] = embeddingTable[tokenId * nbCols + gid.x];
}

kernel void rmsNorm(device const float* vecList [[ buffer(1) ]],
                    constant uint& nbCols [[ buffer(2) ]],
                    device const float* weight [[ buffer(3) ]],
                    device float* answer [[ buffer(4) ]],
                    uint2 gid [[ thread_position_in_grid ]],
                    uint  ti  [[ thread_index_in_simdgroup ]],
                    uint  lanes [[ threads_per_simdgroup ]]){
    
    uint startingIndex = gid.y * nbCols;

    float localSum = 0.0f;

    for(uint i = ti; i < nbCols; i += lanes) {
        float val = vecList[startingIndex + i];
        localSum += val * val;
    }

    float totalSum = simd_sum(localSum);

    float meanSquare = totalSum / (float)nbCols;
    
    float invCoeff = rsqrt(meanSquare + 1e-5f);
    
    for(uint i = ti; i < nbCols; i += lanes) {
        answer[startingIndex + i] = (vecList[startingIndex + i] * invCoeff) * weight[i];
    }
}
kernel void RoPE(device float* matrix [[ buffer(0) ]],
                 constant uint& dim [[ buffer(1) ]],
                 uint2 gid [[ thread_position_in_grid ]]){
    
    if (gid.y * 2 >= dim) return;

    uint head_dim = 128;
    uint half_head_dim = 64;
    
    uint head_idx = (gid.y * 2) / head_dim;
    
    uint feature_idx_in_half = gid.y % half_head_dim;
    
    uint base_idx = (gid.x * dim) + (head_idx * head_dim);
    uint index1 = base_idx + feature_idx_in_half;
    uint index2 = base_idx + feature_idx_in_half + half_head_dim;

    float position = (float)gid.x;
    
    float exponent = (float)(feature_idx_in_half * 2) / 128.0f;
    float denominator = pow(500000.0f, exponent);
    float angle = position / denominator;

    float temp1 = matrix[index1];
    float temp2 = matrix[index2];

    matrix[index1] = temp1 * cos(angle) - temp2 * sin(angle);
    matrix[index2] = temp2 * cos(angle) + temp1 * sin(angle);
}
kernel void AttentionScore(device const float* Q [[ buffer(0) ]],
                           device const float* K [[ buffer(1) ]],
                           device float* ans [[ buffer(2) ]],
                           constant int& nbHeadsQ [[ buffer(3) ]],
                           constant int& nbTokens [[ buffer(4) ]],
                           constant int& ratio [[ buffer(5) ]],
                           uint3 gid [[ thread_position_in_grid ]]){
    
    if (gid.x >= uint(nbTokens) || gid.y >= uint(nbTokens) || gid.z >= uint(nbHeadsQ)) return;


    int head_dim = 128;
    int k_head_id = gid.z / ratio;

    int width_Q = nbHeadsQ * head_dim;
    int width_K = (nbHeadsQ / ratio) * head_dim;

    int offset_Q = (gid.y * width_Q) + (gid.z * head_dim);
    int offset_K = (gid.x * width_K) + (k_head_id * head_dim);

    float score = 0.0;
    for (int i = 0; i < head_dim; i++) {
        score += Q[offset_Q + i] * K[offset_K + i];
    }
    score = score / sqrt(128.0);
    
    if (gid.x > gid.y) {
        score = -1e9f;
    }
    
    int index_out = (gid.z * nbTokens * nbTokens) + (gid.y * nbTokens) + gid.x;
    ans[index_out] = score;
}
kernel void SearchMaxVector(device const float* m [[ buffer(0) ]],
                            constant uint& nbCols [[ buffer(1) ]],
                            device float* max [[ buffer(2) ]],
                            uint gid [[ thread_position_in_grid ]]){
    int vec = gid;
    float maxVal = m[vec * nbCols];
    
    for(uint i = 1 ; i< nbCols; i++){
        if(m[vec * nbCols + i]>maxVal){
            maxVal = m[vec * nbCols + i];
        }
    }
    max[vec] = maxVal;
}
kernel void SubstractMax(device float* m [[buffer(0)]],
                         constant uint& nbCols [[ buffer(1) ]],
                         device const float* max [[ buffer(2) ]],
                         uint2 gid [[ thread_position_in_grid ]]){
    if (gid.x >= uint(nbCols)) return;
    
    m[nbCols * gid.y + gid.x] = exp(m[nbCols * gid.y + gid.x ] - max[gid.y]);
}
kernel void SumVector(device const float* m [[ buffer(0) ]],
                            constant uint& nbCols [[ buffer(1) ]],
                            device float* sum [[ buffer(2) ]],
                            uint gid [[ thread_position_in_grid ]]){
    int vec = gid;
    float sumVal = m[vec * nbCols];
    
    for(uint i = 1 ; i<nbCols; i++){
        sumVal += m[vec * nbCols + i];
    }
    sum[vec] = sumVal;
}
kernel void DivideBySum(device float* m [[buffer(0)]],
                         constant uint& nbCols [[ buffer(1) ]],
                         device const float* sum [[ buffer(2) ]],
                         uint2 gid [[ thread_position_in_grid ]]){
    
    if (gid.x >= uint(nbCols)) return;
    
    m[nbCols * gid.y + gid.x] = m[nbCols * gid.y + gid.x ]/sum[gid.y];
}
kernel void weightedSum(device const float* Scores [[ buffer(0) ]],
                         device const float* V [[ buffer(1) ]],
                         device float* Output [[ buffer(2) ]],
                        constant int& nbTokens [[ buffer(3) ]],
                        constant int& nbHeads [[ buffer(4) ]],
                        constant int& ratio [[ buffer(5) ]],
                         uint3 gid [[ thread_position_in_grid ]]){
    
    int head_dim = 128;
    
    if (gid.x >= uint(head_dim) || gid.y >= uint(nbTokens) || gid.z >= uint(nbHeads)) return;
    
    int v_head_id = gid.z / ratio;
    
    int width_V_line = (nbHeads / ratio) * head_dim;
    
    float sum = 0;
    
    for (int k = 0; k < nbTokens; k++) {
        
        int index_score = (gid.z * nbTokens * nbTokens) + (gid.y * nbTokens) + k;
        float prob = Scores[index_score];
        
        int index_v = (k * width_V_line) + (v_head_id * head_dim) + gid.x;
        float value = V[index_v];

        sum += prob * value;
    }
    
    int width_Out_line = nbHeads * head_dim;
    
    int index_out = (gid.y * width_Out_line) + (gid.z * head_dim) + gid.x;
    
    Output[index_out] = sum;
}
kernel void addArrays(device const float* A [[ buffer(0) ]],
                      device const float* B [[ buffer(1) ]],
                      device float* C [[ buffer(2) ]],
                      uint id [[ thread_position_in_grid ]]){
    C[id] = A[id] + B[id];
}
kernel void relu2(device float* Gate [[ buffer(0) ]],
                       device const float* Up [[ buffer(1) ]],
                       uint id [[ thread_position_in_grid ]]){
    float g = Gate[id];
    float relu = (g > 0.0f) ? g : 0.0f;
    float reluSquared = relu * relu;
    Gate[id] = reluSquared * Up[id];
}
kernel void computeLogits(device const float* final_vectors [[ buffer(0) ]],
                          device const float* embeddings [[ buffer(1) ]],
                          device float* logits [[ buffer(2) ]],
                          constant uint& hidden_dim [[ buffer(3) ]],
                          constant uint& last_token_index [[ buffer(4) ]],
                          uint id [[ thread_position_in_grid ]]) {
    
    if (id >= 128256) return;
    
    float score = 0.0f;
    uint offset = last_token_index * hidden_dim;
    
    for(uint i = 0; i < hidden_dim; i++) {
        score += final_vectors[offset + i] * embeddings[id * hidden_dim + i];
    }
    logits[id] = score;
}
