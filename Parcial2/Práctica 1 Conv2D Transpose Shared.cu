//Práctica 1: Conv2D Transpose Shared Memory 	---	Jonathan Cuevas 0225174
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <time.h>
#include <random>

using namespace std;

__global__ void matrizTranspose(int* a, int* b, int n) {
    __shared__ int s[64];
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int offsetBlock = blockIdx.x * blockDim.x * blockDim.y;
    int offsetRow = blockIdx.y * blockDim.x * blockDim.y * gridDim.x;
    int gid = tid + offsetBlock + offsetRow;

    int row = gid / n;
    int col = gid - row * n;
    if (gid < n * n) {
        s[row * n + col] = a[row * n + col];
        __syncthreads();
        b[col * n + row] = s[row * n + col];
    }
}

__global__ void convolucion(int* a, int* b, int* k, int n, int m, int kernelSize) {
    __shared__ int s[64];
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int offsetBlock = blockIdx.x * blockDim.x * blockDim.y;
    int offsetRow = blockIdx.y * blockDim.x * blockDim.y * gridDim.x;
    int gid = tid + offsetBlock + offsetRow;

    int rowActual = gid / n;
    int colActual = gid - rowActual * n;

    int suma = 0;
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            if (rowActual + i >= 0 && rowActual + i < n && colActual + j >= 0 && colActual + j < n) {
                s[(i + 1) * kernelSize + j + 1] = k[(i + 1) * kernelSize + j + 1];
                __syncthreads();
                suma += a[(rowActual + i) * m + colActual + j] * s[(i + 1) * kernelSize + j + 1];
            }
        }
    }
    b[rowActual * m + colActual] = suma;
}

int main() {
    const int kernelSize = 5, row = 8, col = 8;
    int* host_aKernel, * host_convKernel, * host_a, * host_b;
    int* dev_aKernel, * dev_convKernel, * dev_a, * dev_b;
    host_aKernel = (int*)malloc(kernelSize * kernelSize * sizeof(int));
    host_convKernel = (int*)malloc(kernelSize * kernelSize * sizeof(int));
    host_a = (int*)malloc(row * col * sizeof(int));
    host_b = (int*)malloc(row * col * sizeof(int));

    cudaMalloc(&dev_aKernel, kernelSize * kernelSize * sizeof(int));
    cudaMalloc(&dev_convKernel, kernelSize * kernelSize * sizeof(int));
    cudaMalloc(&dev_a, row * col * sizeof(int));
    cudaMalloc(&dev_b, row * col * sizeof(int));

    for (int i = 0; i < kernelSize * kernelSize; i++) {
        int r1 = rand() % (1);
        host_aKernel[i] = r1;
        host_convKernel[i] = 0;
    }

    for (int i = 0; i < (row* col); i++) {
        int r1 = rand() % (10);
        host_a[i] = r1;
        host_b[i] = 0;
    }

    host_aKernel[3] = 1;
    printf("Kernel 5x5: \n");
    for (int i = 0; i < kernelSize; i++) {
        for (int j = 0; j < kernelSize; j++) {
            printf("%d ", host_aKernel[i * kernelSize + j]);
        }
        printf("\n");
    }

    printf("\nMatriz A: \n");
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            printf("%d ", host_a[i * col + j]);
        }
        printf("\n");
    }

    cudaMemcpy(dev_aKernel, host_aKernel, kernelSize * kernelSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_convKernel, host_convKernel, kernelSize * kernelSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_a, host_a, row*col*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, row*col*sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(32, 32);
    dim3 grid(32 / (kernelSize * kernelSize), 32 / (kernelSize * kernelSize));
    matrizTranspose << <grid, block >> > (dev_aKernel, dev_convKernel, kernelSize);
    cudaMemcpy(host_convKernel, dev_convKernel, kernelSize * kernelSize * sizeof(int), cudaMemcpyDeviceToHost);

    cout << "\nRes Kernel:\n";
    for (int i = 0; i < kernelSize; i++) {
        for (int j = 0; j < kernelSize; j++) {
            cout << host_convKernel[i * kernelSize + j] << " ";
        }
        cout << "\n";
    }

    dim3 block2(32, 32);
    dim3 grid2((64 + (row*col) - 1) / (row *col), (64 + (row *col) - 1) / (row*col));
    convolucion << <grid2, block2 >> > (dev_a, dev_b, dev_convKernel, row, col, kernelSize);
    cudaMemcpy(host_b, dev_b, row*col*sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaDeviceReset();


    printf("\nMatriz B (Convolución): \n");
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            printf("%d ", host_b[i * col + j]);
        }
        printf("\n");
    }

    free(host_aKernel);
    free(host_convKernel);

    cudaFree(dev_aKernel);
    cudaFree(dev_convKernel);

    return 0;
}