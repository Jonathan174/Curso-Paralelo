//Práctica 2: Parallel Sort Algorithm   ----    Jonathan Cuevas 0225174
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <time.h>
#include <random>

using namespace std;

void bubbleSort_host(int* a, int size) {
    for (int i = 0; i < size-1; i++) {
        for (int j = 0; j < size-i - 1; j++) {
            if (a[j] > a[j + 1]) {
                int aux = a[j + 1];
                a[j + 1] = a[j];
                a[j] = aux;
            }
        }
    }
}

__global__ void bubbleSort_dev(int* a, int size) {
    int tid = threadIdx.x;
    for (int i = 0; i < size; i++) {
        int offset = i % 2;
        int leftSide = 2 * tid + offset;
        int rightSide = leftSide + 1;
        if (rightSide < size) {
            if (a[leftSide] > a[rightSide]) {
                int aux = a[leftSide];
                a[leftSide] = a[rightSide];
                a[rightSide] = aux;
            }
        }
        __syncthreads();
    }
}

int main() {
    int Array = 1024;
    int* host_config, * res;
    int* dev_config;
    host_config = (int*)malloc(Array * sizeof(int));
    res = (int*)malloc(Array * sizeof(int));
    cudaMalloc(&dev_config, Array * sizeof(Array));

    printf("Arreglo original: \n");
    for (int i = 0; i < Array; i++) {
        int r1 = (rand() % (1024));
        host_config[i] = r1;
        printf("%d ", host_config[i]);
    }
    printf("\n");

    dim3 grid(1);
    dim3 block(Array);

    clock_t gpu_start, gpu_stop;
    gpu_start = clock();
    cudaMemcpy(dev_config, host_config, Array * sizeof(int), cudaMemcpyHostToDevice);
    bubbleSort_dev << <grid, block >> > (dev_config, Array);
    gpu_stop = clock();
    double cps_gpu = (double)((double)(gpu_stop - gpu_start) / CLOCKS_PER_SEC);

    printf("\n\nBubble sort CPU: \n");
    for (int i = 0; i < Array; i++) {
        printf("%d ", host_config[i]);
    }
    printf("\nTIEMPO DE EJECUCION CPU: %4.6f \n\r", cps_gpu);

    clock_t gpu_start1, gpu_stop1;
    gpu_start1 = clock();
    cudaMemcpy(res, dev_config, Array * sizeof(int), cudaMemcpyDeviceToHost);
    bubbleSort_host(host_config, Array);
    gpu_stop1 = clock();
    cps_gpu = (double)((double)(gpu_stop1 - gpu_start1) / CLOCKS_PER_SEC);
    
    printf("\n\nBubble sort GPU\n");
    for (int i = 0; i < Array; i++) {
        printf("%d ", res[i]);
    }
    printf("\nTIEMPO DE EJECUCION GPU: %4.6f \n\r", cps_gpu);
    return 0;
}