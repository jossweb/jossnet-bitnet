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
    answer[index_Answer] = sum;
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

kernel void rmsNorm(device float* vecList [[ buffer(11) ]],
                    device const uint* nbCols [[ buffer(12) ]],
                    uint2 gid [[ thread_position_in_grid ]]){
    
    uint startingIndex = gid.x * *nbCols;
    
    float powSum = 0;
    for(uint i = startingIndex; i < startingIndex + *nbCols; i++){
        powSum += vecList[i] * vecList[i];
    }
    if(powSum != 0){
        float coeff = sqrt(powSum/ *nbCols);
        for(uint i = startingIndex; i < startingIndex + *nbCols; i++){
            vecList[i] = vecList [i] / coeff;
        }
    }
}
