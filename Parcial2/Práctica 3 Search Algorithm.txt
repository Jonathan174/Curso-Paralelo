//Práctica 3: Search Algorithm   ----    Jonathan Cuevas 0225174
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <time.h>
#include <random>

using namespace std;

void bubbleSort_host(int* a, int size) {
    for (int i = 0; i < size - 1; i++) {
        for (int j = 0; j < size - i - 1; j++) {
            if (a[j] > a[j + 1]) {
                int aux = a[j + 1];
                a[j + 1] = a[j];
                a[j] = aux;
            }
        }
    }
}
__global__ void search(int* a, int size, int* pos, int target) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        if (a[tid] == target) {
            *pos = tid;
        }
    }
}
int main() {
    int size = 128, target = 77;
    int* dev_a, * dev_pos;
    int* host_a = (int*)malloc(size * sizeof(int));
    int* res = (int*)malloc(size * sizeof(int));
    int* pos = (int*)malloc(sizeof(int));
    pos[0] = -1;
    
    cudaMalloc(&dev_a, size * sizeof(int));
    cudaMalloc(&dev_pos, sizeof(int));

    printf("Arreglo:\n");
    for (int i = 0; i < size; i++) {
        int r1 = rand() % (128);
        host_a[i] = r1;
        printf("%d ", host_a[i]);
    }
    printf("\n");

    cudaMemcpy(dev_a, host_a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_pos, pos, sizeof(int), cudaMemcpyHostToDevice);

    dim3 grid(1);
    dim3 block(1024);
    search << <grid, block >> > (dev_a, size, dev_pos, target);
    cudaDeviceSynchronize();

    cudaMemcpy(pos, dev_pos, sizeof(int), cudaMemcpyDeviceToHost);

    if(pos[0] == -1){
        printf("No está en la lista\n");
    }
    else{
        printf("El numero %d está en el indice %d \n",target, pos[0]);
    }

    free(host_a);
    free(pos);
    free(res);
    cudaFree(dev_a);
    cudaFree(dev_pos);

    return 0;
}