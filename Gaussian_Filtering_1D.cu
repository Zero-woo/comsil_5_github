#define NUMBER_OF_TESTS_ON_GPU 10
#define NUMBER_OF_ELEMENTS (1 << 26)


#include <stdio.h>
#include <cmath>
#include <random>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "measure_host_time_3.h"

const float GAUSSIAN_KERNEL_1D[36] = {
    1.0f,
    0.25f, 0.5f, 0.25f,
    0.0625f, 0.2500f, 0.3750f, 0.2500f, 0.0625f,
	0.015625f, 0.09375f, 0.234375f, 0.3125f, 0.234375f, 0.09375f, 0.015625f,
    0.0039f, 0.0312f, 0.1094f, 0.2188f, 0.2734f, 0.2188f, 0.1094f, 0.0312f, 0.0039f,
    0.000977f, 0.009766f, 0.043945f, 0.117188f, 0.205078f, 0.246094f, 0.205078f, 0.117188f, 0.043945f, 0.009766f, 0.000977f
};

void gen_uniform_distribution_int(int* array, int a, int b, int N) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(a, b);

    for (int i = 0; i < N; i++)  
        array[i] = dist(gen);
}

#define _MAX_KERNEL_SIZE_DEVICE 11
__constant__ float d_kernel_weights[_MAX_KERNEL_SIZE_DEVICE];

void apply_1D_Gaussian_host(const int* image_in, int* image_out, int N, int kernel_width) {
    int half_width = kernel_width / 2;
    const float* kernel = GAUSSIAN_KERNEL_1D + half_width * half_width;

    for (int i = 0; i < N; i++) {
        float sum = 0.0f;

        for (int j = -half_width; j <= half_width; j++) {
            int index = i + j;
            // boarder handling: mirror reflection
            if (index < 0) {
                index = -index;
            }
            else if (index >= N) {
                index = 2 * N - 2 - index;
            }
            sum += image_in[index] * kernel[j + half_width];
        }
		image_out[i] = (int)(sum + 0.5f); // looks faster than std::round(), and it is sufficient for this application  
        // image_out[i] = static_cast<int>(std::round(sum));
    }
}

// 경계 핸들링 조건이 두번쓰여서 따로 함수로 정의. 위의 host함수에도 있지만 기존 코드라 건들지 않음 
__device__ int mirror_index_1D(int index, int N) {

    if (index < 0) {
        index = -index;
    }
    else if (index >= N) {
        index = 2 * N - 2 - index;
    }
    return index;
}


__global__ void apply_1D_Gaussian_Device_basic(const int* image_in, int* image_out, int N, int kernel_width) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    // thread가 범위를 넘으면 return
    if (i >= N) return;

    // 어차피 width는 양수라 나누기 대신 시프트 연산
    int half_width = kernel_width >> 1;
    float sum = 0.0f;

    for (int j = -half_width; j <= half_width ; j++)
    {
        int index = mirror_index_1D(i + j, N);
        sum += image_in[index] * d_kernel_weights[j + half_width];
    }

    image_out[i] = (int)(sum + 0.5f);

}


__global__ void apply_1D_Gaussian_Device_stride(const int* image_in, int* image_out, int N, int kernel_width) {

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int half_width = kernel_width >> 1;

    for (int i = tid; i < N; i+=stride)
    {
        float sum = 0.0f;
        
        for (int j = -half_width; j <= half_width; j++)
        {
            int index = mirror_index_1D(i + j, N);
            sum += image_in[index] * d_kernel_weights[j + half_width];
        }

        image_out[i] = (int)(sum + 0.5f);

    }


}

// 검증 함수
bool compare_exact_result(const int* ref, const int* out, int N, const char* tag) {
    int mismatch_count = 0;
    int first_mismatch_index = -1;

    for (int i = 0; i < N; ++i) {
        if (ref[i] != out[i]) {
            if (first_mismatch_index < 0) {
                first_mismatch_index = i;
            }
            ++mismatch_count;
        }
    }

    if (mismatch_count == 0) {
        fprintf(stdout, "^^^ %s : EXACT MATCH\n", tag);
        return true;
    }

    fprintf(stdout,
        "^^^ %s : EXACT MISMATCH (count=%d, first index=%d, host=%d, device=%d)\n",
        tag,
        mismatch_count,
        first_mismatch_index,
        ref[first_mismatch_index],
        out[first_mismatch_index]);
    return false;
}


int main(void) {
	int n_elements = NUMBER_OF_ELEMENTS;


    size_t size = n_elements * sizeof(int);
   
    // Allocate the host memory
    int* h_image_in = (int*)malloc(size);
    int* h_image_out_host = (int*)malloc(size);
    int* h_image_out_device = (int*)malloc(size);
    if (h_image_in == NULL || h_image_out_host == NULL || h_image_out_device == NULL) {
        fprintf(stderr, "^^^ Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    gen_uniform_distribution_int(h_image_in, 0, 255, n_elements);
    // Allocate the device memory
    int* d_image_in = NULL;
    cudaMalloc((void**)&d_image_in, size);
    int* d_image_out = NULL;
    cudaMalloc((void**)&d_image_out, size);

    //host에서 device로 데이터 복사
    cudaMemcpy(d_image_in, h_image_in, size, cudaMemcpyHostToDevice);
   

    int block_sizes[] = { 32, 64, 128, 256, 512 };
    int num_block_sizes = sizeof(block_sizes) / sizeof(block_sizes[0]);


    //blocksize 변경하면서 반복
    for (int bs_idx = 0; bs_idx < num_block_sizes; bs_idx++) {
        int threads_per_block = block_sizes[bs_idx];

        fprintf(stdout, "\n============================================================================\n");
        fprintf(stdout, "[THREADS_PER_BLOCK = %d]\n", threads_per_block);


        // kernel width 변경하면서 반복 
        for (int kernel_width = 1; kernel_width <= _MAX_KERNEL_SIZE_DEVICE; kernel_width += 2) {
            fprintf(stdout, "\n[1D Gaussian filtering of image of %d pixels(KERNEL WIDTH = %d)]\n",
                n_elements, kernel_width);



            //host 측정
            CHECK_TIME_START(_start, _freq);
            apply_1D_Gaussian_host(h_image_in, h_image_out_host, n_elements, kernel_width);
            CHECK_TIME_END(_start, _end, _freq, _compute_time);
            fprintf(stdout, "\n^^^ Time to filter an image of %d pixels on the host = %.3f(ms)\n",
                n_elements, _compute_time);
            float host_time = _compute_time;

            cudaFree(0);
            int half_width = kernel_width / 2;
            // Copy the kernel weights to the constant memory on the device
            cudaMemcpyToSymbol(d_kernel_weights,
                (const float*)GAUSSIAN_KERNEL_1D + half_width * half_width,
                kernel_width * sizeof(float));

            cudaDeviceSynchronize(); // actually not necessary here, but it is a good practice to ensure that all preceding CUDA calls have completed before proceeding, especially when measuring time.


            //thread block 크기를 변경하기때문에 기존 코드 주석처리

            // Launch the 1D GF kernel
            //int threads_per_block = threads_per_block;
            int blocks_per_grid = (n_elements + threads_per_block - 1) / threads_per_block;
            //int blocks_per_grid_reduced = blocks_per_grid / GRID_REDUCTION_FACTOR;
            //fprintf(stdout, "\n^^^ CUDA kernel launch with %d(%d) blocks of %d threads\n", blocks_per_grid_reduced, blocks_per_grid,
            //    threads_per_block);


            fprintf(stdout, "\n[Method 1: one-thread-per-pixel]\n");
            fprintf(stdout, "^^^ CUDA kernel launch with %d blocks of %d threads\n",
                blocks_per_grid, threads_per_block);

            // 웜업
            apply_1D_Gaussian_Device_basic << < blocks_per_grid, threads_per_block >> > (d_image_in, d_image_out, n_elements, kernel_width);
            cudaDeviceSynchronize();


            //basic kernel 수행
            CHECK_TIME_START(_start, _freq);
            for (int j = 0; j < NUMBER_OF_TESTS_ON_GPU; j++)
            {
                apply_1D_Gaussian_Device_basic << < blocks_per_grid, threads_per_block >> > (d_image_in, d_image_out, n_elements, kernel_width);
                cudaDeviceSynchronize();
            }
            CHECK_TIME_END(_start, _end, _freq, _compute_time);
            fprintf(stdout, "\n^^^ Time to filter an image of %d pixels on the basic = %.3f(ms)\n", n_elements, _compute_time / NUMBER_OF_TESTS_ON_GPU);
           
            //데이터 복사 후 검증
            cudaMemcpy(h_image_out_device, d_image_out, size, cudaMemcpyDeviceToHost);
            compare_exact_result(h_image_out_host, h_image_out_device, n_elements, "basic");




            int reduction_factors[] = { 1, 2, 4, 8, 16, 32 };
            fprintf(stdout, "\n[Method 2: grid-stride loop]\n");
            //reduction factor 변경하면서 반복
            for (int rf_idx = 0; rf_idx < 6; rf_idx++) {
                int reduction_factor = reduction_factors[rf_idx];

                int blocks_per_grid_reduced = blocks_per_grid / reduction_factor;
                if (blocks_per_grid_reduced < 1) blocks_per_grid_reduced = 1;

                fprintf(stdout, "\n[GRID_REDUCTION_FACTOR = %d]\n",
                    reduction_factor);
                fprintf(stdout, "^^^ CUDA kernel launch with %d(%d) blocks of %d threads\n",
                    blocks_per_grid_reduced, blocks_per_grid, threads_per_block);

                // 웜업
                apply_1D_Gaussian_Device_stride << <blocks_per_grid_reduced, threads_per_block >> > (
                    d_image_in, d_image_out, n_elements, kernel_width);
                cudaDeviceSynchronize();


                //grid-stride kernel 수행
                CHECK_TIME_START(_start, _freq);
                for (int j = 0; j < NUMBER_OF_TESTS_ON_GPU; j++) {
                    apply_1D_Gaussian_Device_stride << <blocks_per_grid_reduced, threads_per_block >> > (
                        d_image_in, d_image_out, n_elements, kernel_width);
                    cudaDeviceSynchronize();
                }
                CHECK_TIME_END(_start, _end, _freq, _compute_time);
                fprintf(stdout, "^^^ Time on stride = %.3f(ms)\n",
                    _compute_time / NUMBER_OF_TESTS_ON_GPU);


                // 데이터 복사 후 검증
                cudaMemcpy(h_image_out_device, d_image_out, size, cudaMemcpyDeviceToHost);
                char tag[64];
                snprintf(tag, sizeof(tag), "stride (RF=%d)", reduction_factor);
                compare_exact_result(h_image_out_host, h_image_out_device, n_elements, tag);
            }

            fprintf(stdout, "\n============================================================================");
        }
    }

	// Free device memory   
	cudaFree(d_image_in);
	cudaFree(d_image_out);  

    // Free host memory
    free(h_image_in);
    free(h_image_out_host);
    free(h_image_out_device);

    fprintf(stdout, "\n^^^ Done\n");
    return 0;
}