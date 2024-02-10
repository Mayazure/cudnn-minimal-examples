// $env:CUDNN_LOGINFO_DBG=1
// $env:CUDNN_LOG_ERR=1
// $env:CUDNN_LOGWARN_DBG=1
// $env:CUDNN_LOGDEST_DBG="stdout"
// $env:CUDNN_LOGERR_DBG=1

// nvcc -I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\include\nvtx3" -I C:\Users\mayaz\Programs\cudnn-windows-x86_64-8.6.0.163_cuda11-archive\include -L C:\Users\mayaz\Programs\cudnn-windows-x86_64-8.6.0.163_cuda11-archive\lib\x64 -l cudnn -l cudart -l cublas -l cublasLt -o testcudnn_convdatabwd .\testcudnn_convdatabwd.cu
// testcudnn_convdatabwd.cu

#include <iostream>
#include <cuda_runtime.h>
#include "cudnn.h"
// #include "nvToolsExt.h"

int main(int argc, char** argv)
{    
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);
    std::cout << "numGPUs: " << numGPUs << std::endl;
    cudaSetDevice(1); // use GPU0
    int device; 
    struct cudaDeviceProp devProp;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&devProp, device);
    std::cout << "GPU0: " << devProp.name << std::endl;
    std::cout << "Total global mem: " << devProp.totalGlobalMem/1024/1024/1024 << " GB" << std::endl;
    std::cout << "Total const mem: " << devProp.totalConstMem/1024/1024/1024 << " GB" << std::endl;
    std::cout << "Compute capability:" << devProp.major << "." << devProp.minor << std::endl;

    cudnnHandle_t handle_;
    cudnnCreate(&handle_);
    std::cout << "Created cuDNN handle" << std::endl;

    cudnnStatus_t cudnnStatus;
    cudnnStatus = cudnnCnnInferVersionCheck();
    std::cout << "Cudnn Status: " << cudnnStatus << std::endl;

    float alpha[1] = {1};
    float beta[1] = {0.0};
    
    // Create filter
    cudnnFilterDescriptor_t wDesc;
    cudnnStatus_t status = cudnnCreateFilterDescriptor(&wDesc);
    status = cudnnSetFilter4dDescriptor(
        wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 
        2, 2, 3, 3
    );
    
    float *w_host = (float*)malloc(18*sizeof(float));
    for(int i=0;i<9;i++){
        w_host[i] = 1.0f;
    }
    for(int i=9;i<18;i++){
        w_host[i] = 2.0f;
    }
    float *w;
    cudaMalloc(&w, 18 * sizeof(float));
    cudaMemcpy(w, w_host, 18*sizeof(float), cudaMemcpyHostToDevice);

    // Create tensor
    cudnnTensorDescriptor_t dyDesc;
    status = cudnnCreateTensorDescriptor(&dyDesc);
    status = cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 2, 2, 2);
    float *dy_host = (float*)malloc(8*sizeof(float));
    for(int i=0;i<8;i++){
        dy_host[i] = 1.0f;
    }
    float *dy;
    cudaMalloc(&dy, 8 * sizeof(float));
    cudaMemcpy(dy, dy_host, 8*sizeof(float), cudaMemcpyHostToDevice);

    cudnnConvolutionDescriptor_t convDesc;
    cudnnCreateConvolutionDescriptor(&convDesc);
    cudnnSetConvolution2dDescriptor(convDesc, 0, 0, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT);
    cudnnSetConvolutionGroupCount(convDesc, 2);

    // Create dx
    cudnnTensorDescriptor_t dxDesc;
    cudnnCreateTensorDescriptor(&dxDesc);
    cudnnSetTensor4dDescriptor(dxDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 2, 4, 4);
    float *dx_host = (float*)malloc( 32 * sizeof(float) );
    for(int i=0;i<32;i++) {
        dx_host[i] = (i+1) * 1.0f;
    }
    float *dx;
    cudaMalloc(&dx, 32 * sizeof(float));
    cudaMemset((void**)&dx, 0, 32*sizeof(float));

    cudnnConvolutionBwdDataAlgo_t algos[] = {
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED
    };

    const char* algoNames[6] = {
        "CUDNN_CONVOLUTION_BWD_DATA_ALGO_0",
        "CUDNN_CONVOLUTION_BWD_DATA_ALGO_1",
        "CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT",
        "CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING",
        "CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD",
        "CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED"
    };

    for(int i=0;i<6;i++){
        std::cout<<"--------"<<std::endl<<algoNames[i]<<std::endl;
        cudnnConvolutionBwdDataAlgo_t algo = algos[i];

        size_t workspaceSizeInBytes;
        status = cudnnGetConvolutionBackwardDataWorkspaceSize(
                    handle_,
                    wDesc,
                    dyDesc,
                    convDesc,
                    dxDesc,
                    algo,
                    &workspaceSizeInBytes
                );
        std::cout<<"WorkspaceSize: "<<workspaceSizeInBytes<<std::endl;
        void *workspace;
        cudaMalloc(&workspace, workspaceSizeInBytes);

        cudnnStatus = cudnnConvolutionBackwardData(
            handle_,
            alpha,
            wDesc,
            w,
            dyDesc,
            dy,
            convDesc,
            algo,
            workspace,
            workspaceSizeInBytes,
            beta,
            dxDesc,
            dx
        );

        cudaMemcpy( dx_host, dx, 32 * sizeof(float), cudaMemcpyDeviceToHost);
        std::cout<<"dx:";
        for (int i=0;i<32;i++){
                if(i%4==0){
                    std::cout << std::endl;
                };
            std::cout << dx_host[i] << " ";
        };
        std::cout << std::endl;
        cudaFree(workspace);
    }

    cudaFree(w);
    cudaFree(dy);
    cudaFree(dx);
    
    free(dx_host);
    free(dy_host);
    free(w_host);

    cudnnDestroy(handle_);
    return 0;
}