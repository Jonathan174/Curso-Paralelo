//Práctica 5: Matrix Transpose Unrolling Complete   ----    Jonathan Cuevas 0225174
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <time.h>
#include <random>

using namespace std;

__global__ void unrollingTranspose(int* a, int* b, int size) {
    int gid = (threadIdx.x + threadIdx.y * blockDim.x) + (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y);
    int offset = blockDim.x / 2;

    for (int i = 0; i < (size* size + blockDim.x * blockDim.y - 1) / (blockDim.x * blockDim.y); i += 2)
    {
        if (gid + blockDim.x * blockDim.y * i < size * size) {
            b[(gid % size * size + gid / size) + offset * i] = a[gid + blockDim.x * blockDim.y * i];
        }
        if (gid + blockDim.x * blockDim.y * i + blockDim.x * blockDim.y < size * size) {
            b[(gid % size * size + gid / size) + offset * i + offset] = a[gid + blockDim.x * blockDim.y * i + blockDim.x * blockDim.y];
        }
    }
}

int main() {
    const int size = 16;
    int* host_a, * host_b;
    int* dev_a, * dev_b;
    host_a = (int*)malloc(size * size * sizeof(int));
    host_b = (int*)malloc(size * size * sizeof(int));
    cudaMalloc(&dev_a, size * size * sizeof(int));
    cudaMalloc(&dev_b, size * size * sizeof(int));

    for (int i = 0; i < size * size; i++) {
        int r1 = rand() % (256);
        host_a[i] = r1;
        host_b[i] = 0;
    }

    printf("\nMatriz original: \n");
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%d ", host_a[i * size + j]);
        }
        printf("\n");
    }

    cudaMemcpy(dev_a, host_a, size * size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, size * size * sizeof(int), cudaMemcpyHostToDevice);

    dim3 grid(1);
    dim3 block(4, 4);
    unrollingTranspose << <1, block >> > (dev_a, dev_b, size);
    cudaMemcpy(host_b, dev_b, size * size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaDeviceReset();

    printf("\nMatriz transpuesta: \n");
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%d ", host_b[i * size + j]);
        }
        printf("\n");
    }

    free(host_a);
    free(host_b);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return 0;
}