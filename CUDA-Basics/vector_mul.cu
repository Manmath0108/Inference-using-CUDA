#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define N 1024
#define BLOCK_SIZE 256

void vector_mul_cpu(float *a, float *b, float *c, int n){
    for (int i = 0; i<n; i++){
        c[i] = a[i]*b[i];
    }
}

__global__ void vector_mul_gpu(float *a, float *b, float *c, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n){
        c[i] = a[i]*b[i];
    }
}

int main(){
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu;
    float *d_a, *d_b, *d_c;

    size_t size = N * sizeof(float);

    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c_cpu = (float*)malloc(size);
    h_c_gpu = (float*)malloc(size);

    for (int i = 0; i < N; i++){
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    vector_mul_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA ERROR: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);

        vector_mul_cpu(h_a, h_b, h_c_cpu, N);

    int correct = 1;
    for (int i = 0; i < N; i++) {
        if (fabs(h_c_cpu[i] - h_c_gpu[i]) > 1e-5) {
            correct = 0;
            break;
        }
    }

    printf("Result is %s\n", correct ? "CORRECT" : "WRONG");

    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}