#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <time.h>

using namespace std;

__global__ void multiplication(int* a, int* b, int* result){
    int id = threadIdx.x;
    result[id] = a[id]*b[id];
}

__global__ void printKernel() {
    printf("threadIdx %d %d %d \n", threadIdx.x, threadIdx.y, threadIdx.z);
    printf("blockIdx %d %d %d \n", blockIdx.x, blockIdx.y, blockIdx.z);
    printf("gridDim %d %d %d \n", gridDim.x, gridDim.y, gridDim.z);
}

int main(){
    const int n = 3;
    int size = n * sizeof(n);

    int vectorA[n] = { 2, 7, 10 };
    int vectorB[n] = { 4, 0, 1 };
    int result[n] = { 0, 0, 0 };

    int* devA = 0;
    int* devB = 0;
    int* devC = 0;

    cudaMalloc((void**)&devA, size);
    cudaMalloc((void**)&devB, size);
    cudaMalloc((void**)&devC, size);

    cudaMemcpy(devA, vectorA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(devB, vectorB, size, cudaMemcpyHostToDevice);
    cudaMemcpy(devC, result, size, cudaMemcpyHostToDevice);

    multiplication << <1, n >> > (devA, devB, devC);
    cudaDeviceSynchronize();

    cudaMemcpy(result, devC, size, cudaMemcpyDeviceToHost);
    printf("Vector A: {%d, %d, %d}\n", vectorA[0], vectorA[1], vectorA[2]);
    printf("Vector B: {%d, %d, %d}\n", vectorB[0], vectorB[1], vectorB[2]);
    printf("Resultado multiplicación: {%d, %d, %d}\n", result[0], result[1], result[2]);
    cudaDeviceReset();

    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);

    dim3 block(2, 2, 2);
    dim3 grid(4 / block.x, 4 / block.y, 4 / block.z);
    printKernel << <grid, block >> > ();

    return 0;
}