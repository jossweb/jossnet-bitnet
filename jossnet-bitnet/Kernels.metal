//
//  Kernels.metal
//  jossnet-bitnet
//
//  Created by FIGUEIRAS Jossua on 31/01/2026.
//
#include <metal_stdlib>

using namespace metal;

kernel void mult(device const float* X [[ buffer(0) ]],
                 device const char* W [[ buffer(1) ]],
                 device const uint& nbColX [[ buffer(2) ]],
                 device const uint& nbLineX [[ buffer(3) ]],
                 device const uint& nbColW [[ buffer(4) ]],
                 device const uint& nbLineW [[ buffer(5) ]],
                 device float* answer [[ buffer(6) ]],
                 device const float* scale_ptr [[ buffer(7) ]],
                 uint2 gid [[ thread_position_in_grid ]]){
    
    
    if (gid.x >= nbLineW || gid.y >= nbLineX) {
        return;
    }
    float sum = 0;
    for(uint i = 0; i<nbColX; i++){
        uint indexX = gid.y * nbColX + i;
        
        uint index_W = gid.x * nbColX + i;

        sum += X[indexX] * W[index_W];
    }
    uint index_Answer = gid.y * nbLineW + gid.x;
    
    answer[index_Answer] = sum * scale_ptr[0];
}
kernel void getFromEmbedding(device const float* embeddingTable [[ buffer(7) ]],
                             device float* answer [[ buffer(8) ]],
                             device const uint* index [[ buffer(9) ]],
                             device const uint* nbCols [[ buffer(10) ]],
                             uint2 gid [[ thread_position_in_grid ]]){
    // gid.y : actual token in tab
    // gid.x : index for actual token
    uint tokenId = index[gid.y];
    int cols = nbCols[0];
    answer[gid.x + gid.y * cols] = embeddingTable[tokenId * cols + gid.x];
}

kernel void rmsNorm(device const float* vecList [[ buffer(1) ]],
                    device const uint* nbCols [[ buffer(2) ]],
                    device const float* weight [[ buffer(3) ]],
                    device float* answer [[ buffer(4) ]],
                    uint2 gid [[ thread_position_in_grid ]]){
    
    uint startingIndex = gid.x * *nbCols;

    float powSum = 0;
    for(uint i = 0; i < *nbCols; i++){
        float val = vecList[startingIndex + i];
        powSum += val * val;
    }
    
    if(powSum != 0){
        float meanSquare = powSum / (float)(*nbCols);
        float coeff = sqrt(meanSquare + 1e-5);
        for(uint i = 0; i < *nbCols; i++){
            answer[startingIndex + i] = (vecList[startingIndex + i] / coeff) * weight[i];
        }
    }
}
kernel void RoPE(device float* matrix [[ buffer(0) ]],
                 device const uint* dim_ptr [[ buffer(1) ]],
                 uint2 gid [[ thread_position_in_grid ]]){
    
    if (gid.y * 2 >= *dim_ptr) return;

    uint index = (gid.x * *dim_ptr) + (gid.y * 2);
    
    uint feature_in_head = gid.y % 64;

    float position = (float)gid.x;
    float denominator = pow(10000.0f, ((float)feature_in_head * 2.0f) / 128.0f);
    float angle = position * (1.0f / denominator);
    

    float temp = matrix[index];
    
    matrix[index]   = matrix[index] * cos(angle) - matrix[index + 1] * sin(angle);
    matrix[index+1] = temp * sin(angle) + matrix[index + 1] * cos(angle);
}
kernel void AttentionScore(device const float* Q [[ buffer(0) ]],
                           device const float* K [[ buffer(1) ]],
                           device float* ans [[ buffer(2) ]],
                           device const int& nbHeadsQ [[ buffer(3) ]],
                           device const int& nbTokens [[ buffer(4) ]],
                           device const int& ratio [[ buffer(5) ]],
                           uint3 gid [[ thread_position_in_grid ]]){
    
    if (gid.x >= uint(nbTokens) || gid.y >= uint(nbTokens) || gid.z >= uint(nbHeadsQ)) return;


    int head_dim = 128;
    int k_head_id = gid.z / ratio;

    int width_Q = nbHeadsQ * head_dim;
    int width_K = (nbHeadsQ / ratio) * head_dim;

    int offset_Q = (gid.y * width_Q) + (gid.z * head_dim);
    
    int offset_K = (gid.x* width_K) + (k_head_id * head_dim);

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
                            device const uint* nbCols [[ buffer(1) ]],
                            device float* max [[ buffer(2) ]],
                            uint gid [[ thread_position_in_grid ]]){
    int vec = gid;
    float maxVal = m[vec * *nbCols];
    
    for(uint i = 1 ; i<*nbCols; i++){
        if(m[vec * *nbCols + i]>maxVal){
            maxVal = m[vec * *nbCols + i];
        }
    }
    max[vec] = maxVal;
}
kernel void SubstractMax(device float* m [[buffer(0)]],
                         device const int* nbCols [[ buffer(1) ]],
                         device const float* max [[ buffer(2) ]],
                         uint2 gid [[ thread_position_in_grid ]]){
    if (gid.x >= uint(*nbCols)) return;
    
    m[*nbCols * gid.y + gid.x] = exp(m[*nbCols * gid.y + gid.x ]-max[gid.y]);
}
kernel void SumVector(device const float* m [[ buffer(0) ]],
                            device const uint* nbCols [[ buffer(1) ]],
                            device float* sum [[ buffer(2) ]],
                            uint gid [[ thread_position_in_grid ]]){
    int vec = gid;
    float sumVal = m[vec * *nbCols];
    
    for(uint i = 1 ; i<*nbCols; i++){
        sumVal += m[vec * *nbCols + i];
    }
    sum[vec] = sumVal;
}
kernel void DivideBySum(device float* m [[buffer(0)]],
                         device const int* nbCols [[ buffer(1) ]],
                         device const float* sum [[ buffer(2) ]],
                         uint2 gid [[ thread_position_in_grid ]]){
    
    if (gid.x >= uint(*nbCols)) return;
    
    m[*nbCols * gid.y + gid.x] = m[*nbCols * gid.y + gid.x ]/sum[gid.y];
}
kernel void weightedSum(device const float* Scores [[ buffer(0) ]],
                         device const float* V [[ buffer(1) ]],
                         device float* Output [[ buffer(2) ]],
                         device const int& nbTokens [[ buffer(3) ]],
                         device const int& nbHeads [[ buffer(4) ]],
                         device const int& ratio [[ buffer(5) ]],
                         uint3 gid [[ thread_position_in_grid ]])
                    {
    
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
kernel void addArrays(device float* A [[ buffer(0) ]],
                       device const float* B [[ buffer(1) ]],
                       uint id [[ thread_position_in_grid ]])
{
    A[id] += B[id];
}
kernel void siluAndMul(device float* Gate [[ buffer(0) ]],
                         device const float* Up [[ buffer(1) ]],
                         uint id [[ thread_position_in_grid ]])
{
    Gate[id] = (Gate[id] * 1.0f / (1.0f + exp(-Gate[id]))) * Up[id];
}
kernel void computeLogits(device const float* final_vectors [[ buffer(0) ]],
                          device const float* embeddings [[ buffer(1) ]],
                          device float* logits [[ buffer(2) ]],
                          device const uint* hidden_dim [[ buffer(3) ]],
                          device const uint* last_token_index [[ buffer(4) ]],
                          uint id [[ thread_position_in_grid ]]) {
    
    uint vocab_size = 128256;
    float score = 0;
    if (id >= vocab_size) return;
    
    uint dim = *hidden_dim;
    uint offset = (*last_token_index) * dim;
    for(uint i = 0; i < dim; i++) {
        score += final_vectors[offset + i] * embeddings[id * dim + i];
    }
    logits[id] = score;
}
