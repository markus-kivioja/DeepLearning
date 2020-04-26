#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "include/helper_cuda.h"
#include "include/utils.h"

__device__ float sigmoid(float z)
{
    return 1.0f / (1.0f + expf(-z));
}

__global__ void sigmoid(const float* in, float* out)
{
    out[threadIdx.x] = 1.0f / (1.0f + expf(-in[threadIdx.x]));
}

__device__ float sigmoidDeriv(float z)
{
    float temp = sigmoid(z);
    return temp * (1.0f - temp);
}

__device__ float quadraticCostDeriv(float a, float y)
{
    return a - y;
}

__global__ void dot(const float* vec1, const float* vec2, float* out)
{
    extern __shared__ float cache[];
    cache[threadIdx.x] = vec1[threadIdx.x] * vec2[threadIdx.x];
    
    __syncthreads();
    
    int reducedSize = blockDim.x / 2;

    while (reducedSize > 0)
    {
        if (threadIdx.x < reducedSize)
        {
            cache[threadIdx.x] += cache[threadIdx.x + reducedSize];
        }
        if ((reducedSize > 1) && (reducedSize % 2) && (threadIdx.x == (reducedSize - 1)))
        {
            cache[threadIdx.x - 1] += cache[threadIdx.x];
        }
        reducedSize /= 2;
        __syncthreads();
    }
    
    if (!threadIdx.x) *out = cache[0];
}

__global__ void gradientDescentStep(const void* variables, const void* partialDerivs, size_t pitch, float learningRate, size_t subsetSize)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    float* variablesRow = (float*)(((uint8_t*)variables) + y * pitch);
    float* partialDerivsRow = (float*)(((uint8_t*)partialDerivs) + y * pitch);

    variablesRow[x] = variablesRow[x] - (learningRate / subsetSize) * partialDerivsRow[x];
}

__global__ void matVecMulAdd(const void* matrix, size_t pitch, size_t colCount, const float* vector, const float* addVec, float* out)
{
    size_t idx = threadIdx.x;
    float* row = (float*)(((uint8_t*)matrix) + idx * pitch);
    dot<<<1, colCount, colCount * sizeof(float)>>>(row, vector, &out[idx]);
    cudaDeviceSynchronize();
    out[idx] += addVec[idx];
}

__global__ void layerActivation(const void* matrix, size_t pitch, size_t colCount, const float* vector, const float* addVec, float* out)
{
    size_t idx = threadIdx.x;
    float* row = (float*)(((uint8_t*)matrix) + idx * pitch);
    dot<<<1, colCount, colCount * sizeof(float)>>>(row, vector, &out[idx]);
    cudaDeviceSynchronize();
    out[idx] = sigmoid(out[idx] + addVec[idx]);
}

__global__ void columnProduct(const float* vector1, const float* vector2, void* out, size_t pitch)
{
    size_t x = threadIdx.x;
    size_t y = threadIdx.y;

    float* outRow = (float*)(((uint8_t*)out) + y * pitch);
    outRow[x] = vector1[y] * vector2[x];
}

__global__ void calcOutputError(const float* a, const float* y, const float* z, float* out)
{
    float costDeriv = quadraticCostDeriv(a[threadIdx.x], y[threadIdx.x]);
    float sigDeriv = sigmoidDeriv(z[threadIdx.x]);
    out[threadIdx.x] = costDeriv * sigDeriv;
}

__global__ void matTranspose(const void* mat, size_t pitchIn, void* out, size_t pitchOut)
{
    size_t xIn = blockIdx.x * blockDim.x + threadIdx.x;
    size_t yIn = blockIdx.y * blockDim.y + threadIdx.y;
    
    size_t xOut = yIn;
    size_t yOut = xIn;

    const float* inRow = (float*)(((uint8_t*)mat) + yIn * pitchIn);
    float* outRow = (float*)(((uint8_t*)out) + yOut * pitchOut);

    outRow[xOut] = inRow[xIn];
}

__global__ void calcLayerError(const void* wTrans, size_t pitch, size_t colCount, const float* nextLayerError, const float* z, float* out)
{
    size_t idx = threadIdx.x;
    float* row = (float*)(((uint8_t*)wTrans) + idx * pitch);
    dot<<<1, colCount, colCount * sizeof(float)>>>(row, nextLayerError, &out[idx]);
    cudaDeviceSynchronize();
    out[idx] *= sigmoidDeriv(z[idx]);
}

class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<uint32_t>& pNeuronCounts)
    {
        m_neuronCounts = pNeuronCounts;
        m_layerCount = m_neuronCounts.size();

        std::default_random_engine generator;
        std::normal_distribution<float> distribution(0.0f, 1.0f);

        // Allocate GPU mem for weights and biases and cost function partial derivatives with respect to them.
        for (size_t layer = 0; layer < m_layerCount - 1; layer++)
        {
            int32_t rows = m_neuronCounts[layer + 1];
            int32_t cols = m_neuronCounts[layer];

            // Initialize weights and copy to GPU
            float* wHostData;
            checkCudaErrors(cudaMallocHost(&wHostData, rows * cols * sizeof(float)));
            for (int i = 0; i < rows * cols; i++)
            {
                wHostData[i] = distribution(generator);
            }

            cudaPitchedPtr wDevice;
            cudaExtent extent = make_cudaExtent(cols * sizeof(float), rows, 1);
            checkCudaErrors(cudaMalloc3D(&wDevice, extent));
            m_w.push_back(wDevice);

            cudaPitchedPtr wHost;
            wHost.ptr = wHostData;
            wHost.pitch = cols * sizeof(float);
            wHost.xsize = wDevice.xsize;
            wHost.ysize = wDevice.ysize;

            cudaMemcpy3DParms cpyParms = { 0 };
            cpyParms.srcPtr = wHost;
            cpyParms.dstPtr = wDevice;
            cpyParms.extent = extent;
            cpyParms.kind = cudaMemcpyHostToDevice;
            checkCudaErrors(cudaMemcpy3D(&cpyParms));
            checkCudaErrors(cudaFreeHost(wHostData));

            // Allocate and initialize cost function partial derivatives with rispect to weights
            checkCudaErrors(cudaMalloc3D(&wDevice, extent));
            checkCudaErrors(cudaMemset3D(wDevice, 0, extent));
            m_wPartDerivs.push_back(wDevice);

            // Initialize biases and copy to GPU
            float* bHost;
            checkCudaErrors(cudaMallocHost(&bHost, rows * sizeof(float)));
            for (int i = 0; i < rows; i++)
            {
                bHost[i] = distribution(generator);
            }

            float* bDevice;
            checkCudaErrors(cudaMalloc(&bDevice, rows * sizeof(float)));
            m_b.push_back(bDevice);

            checkCudaErrors(cudaMemcpy(bDevice, bHost, rows * sizeof(float), cudaMemcpyHostToDevice));

            checkCudaErrors(cudaMalloc(&bDevice, rows * sizeof(float)));
            checkCudaErrors(cudaMemset(bDevice, 0, rows * sizeof(float)));
            m_bPartDerivs.push_back(bDevice);
        }
    }

    ~NeuralNetwork()
    {
        for (size_t layer = 0; layer < m_w.size() - 1; layer++)
        {
            checkCudaErrors(cudaFree(m_w[layer].ptr));
            checkCudaErrors(cudaFree(m_b[layer]));
        }
    }

    uint8_t recognizeDigit(float* x)
    {
        std::vector<float*> as;

        for (size_t layer = 0; layer < m_layerCount; layer++)
        {
            float* a;
            checkCudaErrors(cudaMalloc(&a, m_neuronCounts[layer] * sizeof(float)));
            if (!layer)
            {
                checkCudaErrors(cudaMemcpy(a, x, m_neuronCounts[layer] * sizeof(float), cudaMemcpyHostToDevice));
            }
            else
            {
                checkCudaErrors(cudaMemset(a, 0, m_neuronCounts[layer] * sizeof(float)));
            }
            as.push_back(a);
        }

        for (size_t layer = 0; layer < m_layerCount - 1; layer++)
        {
            int32_t rows = m_neuronCounts[layer + 1];
            int32_t cols = m_neuronCounts[layer];

            layerActivation<<<1, rows>>>(m_w[layer].ptr, m_w[layer].pitch, cols, as[layer], m_b[layer], as[layer + 1]);
        }

        std::vector<float> result(m_neuronCounts[m_layerCount - 1]);
        checkCudaErrors(cudaMemcpy(&result[0], as[as.size() - 1], result.size() * sizeof(float), cudaMemcpyDeviceToHost));

        for (auto a : as) checkCudaErrors(cudaFree(a));

        return std::max_element(result.begin(), result.begin() + 10) - result.begin();
    }

    void learn(std::vector<float*>& xs, std::vector<float*>& ys, uint32_t epochCount, uint32_t subsetSize, float learningRate)
    {
        // Copy the training data into GPU memory
        std::vector<float*> xsDevice;
        std::vector<float*> ysDevice;
        xsDevice.resize(xs.size());
        ysDevice.resize(ys.size());
        for (int i = 0; i < xs.size(); i++)
        {
            checkCudaErrors(cudaMalloc(&xsDevice[i], m_neuronCounts[0] * sizeof(float)));
            checkCudaErrors(cudaMalloc(&ysDevice[i], m_neuronCounts[m_neuronCounts.size() - 1] * sizeof(float)));
            checkCudaErrors(cudaMemcpy(xsDevice[i], xs[i], m_neuronCounts[0] * sizeof(float), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(ysDevice[i], ys[i], m_neuronCounts[m_neuronCounts.size() - 1] * sizeof(float), cudaMemcpyHostToDevice));
        }

        uint32_t subsetCount = static_cast<uint32_t>(std::ceil(xsDevice.size() / (float)subsetSize));

        // Use stochastic gradient descent to learn the weights and biases
        // Rearrange the training data for each epoch and divide into subsets
        for (uint32_t epoch = 0; epoch < epochCount; epoch++)
        {
            printf("Start epoch %d\n", epoch);
            auto rngX = std::default_random_engine{};
            auto rngY = rngX;
            std::shuffle(std::begin(xsDevice), std::end(xsDevice), rngX);
            std::shuffle(std::begin(ysDevice), std::end(ysDevice), rngY);

            uint32_t elementsLeft = static_cast<uint32_t>(xsDevice.size());

            // Teach the network with a subset of the learning data
            for (uint32_t subset = 0; subset < subsetCount; subset++)
            {
                auto xStart = xsDevice.begin() + subset * subsetSize;
                auto xEnd = xStart + std::min(subsetSize, elementsLeft);
                std::vector<float*> xSubset(xStart, xEnd);

                auto yStart = ysDevice.begin() + subset * subsetSize;
                auto yEnd = yStart + std::min(subsetSize, elementsLeft);
                std::vector<float*> ySubset(yStart, yEnd);

                elementsLeft -= subsetSize;

                updateSubset(xSubset, ySubset, learningRate);
            }
        }
    }

    void resetPartialDerivatives()
    {
        for (size_t layer = 0; layer < m_layerCount - 1; layer++)
        {
            int32_t rows = m_neuronCounts[layer + 1];
            int32_t cols = m_neuronCounts[layer];
            cudaExtent extent = make_cudaExtent(cols * sizeof(float), rows, 1);
            checkCudaErrors(cudaMemset3D(m_wPartDerivs[layer], 0, extent));
            checkCudaErrors(cudaMemset(m_bPartDerivs[layer], 0, rows * sizeof(float)));
        }
    }

    void updateSubset(const std::vector<float*>& xs, const std::vector<float*>& ys, float learningRate)
    {
        resetPartialDerivatives();

        // Calculate partial derivatives
        for (int i = 0; i < xs.size(); i++)
        {
            float* x = xs[i];
            float* y = ys[i];

            backpropagate(x, y);
        }
        // Update weights and biases
        for (size_t layer = 0; layer < m_layerCount - 1; layer++)
        {
            int32_t rows = m_neuronCounts[layer + 1];
            int32_t cols = m_neuronCounts[layer];
            dim3 gridSize(static_cast<uint32_t>(std::ceil(cols / rows)));
            dim3 blockSize(rows, rows);
            gradientDescentStep<<<gridSize, blockSize>>>(m_w[layer].ptr, m_wPartDerivs[layer].ptr, m_w[layer].pitch, learningRate, xs.size());
            gradientDescentStep<<<1, rows>>>(m_b[layer], m_bPartDerivs[layer], sizeof(float), learningRate, xs.size());
        }
    }

    void backpropagate(float* x, float* y)
    {
        std::vector<float*> as = { x };
        std::vector<float*> zs;

        for (size_t layer = 0; layer < m_layerCount - 1; layer++)
        {
            float* z;
            checkCudaErrors(cudaMalloc(&z, m_neuronCounts[layer + 1] * sizeof(float)));
            zs.push_back(z);

            float* a;
            checkCudaErrors(cudaMalloc(&a, m_neuronCounts[layer + 1] * sizeof(float)));
            as.push_back(a);
        }
        for (size_t layer = 0; layer < m_layerCount - 1; layer++)
        {
            int32_t rows = m_neuronCounts[layer + 1];
            int32_t cols = m_neuronCounts[layer];

            float* a = as[layer];
            float* aNext = as[layer + 1];
            float* z = zs[layer];

            matVecMulAdd<<<1, rows>>>(m_w[layer].ptr, m_w[layer].pitch, cols, a, m_b[layer], z);
            sigmoid<<<1, rows>>>(z, aNext);
        }
        uint32_t lastLayerSize = m_neuronCounts[m_neuronCounts.size() - 1];
        calcOutputError<<<1, lastLayerSize>>>(as[as.size() - 1], y, zs[zs.size() - 1], m_bPartDerivs[m_bPartDerivs.size() - 1]);
        
        dim3 blockSize(m_neuronCounts[m_neuronCounts.size() - 2], m_neuronCounts[m_neuronCounts.size() - 1]);
        columnProduct<<<1, blockSize >>>(m_bPartDerivs[m_bPartDerivs.size() - 1], as[as.size() - 2], m_wPartDerivs[m_wPartDerivs.size() - 1].ptr, m_wPartDerivs[m_wPartDerivs.size() - 1].pitch);

        for (size_t layer = 2; layer < m_layerCount; layer++)
        {
            float* z = zs[zs.size() - layer];

            int32_t rows = m_neuronCounts[m_neuronCounts.size() - layer + 1];
            int32_t cols = m_neuronCounts[m_neuronCounts.size() - layer];
            cudaExtent wTransExtent = make_cudaExtent(rows * sizeof(float), cols, 1);
            cudaPitchedPtr wTrans;
            checkCudaErrors(cudaMalloc3D(&wTrans, wTransExtent));
            dim3 blockSize(cols, rows);
            matTranspose<<<1, blockSize>>>(m_w[m_w.size() - layer + 1].ptr, m_w[m_w.size() - layer + 1].pitch, wTrans.ptr, wTrans.pitch);
            calcLayerError<<<1, cols>>>(wTrans.ptr, wTrans.pitch, rows, m_bPartDerivs[m_bPartDerivs.size() - layer + 1], z, m_bPartDerivs[m_bPartDerivs.size() - layer]);

            blockSize = dim3(m_neuronCounts[m_neuronCounts.size() - layer - 1], m_neuronCounts[m_neuronCounts.size() - layer]);
            columnProduct<<<1, blockSize>>>(m_bPartDerivs[m_bPartDerivs.size() - layer], as[as.size() - layer - 1], m_wPartDerivs[m_wPartDerivs.size() - layer].ptr, m_wPartDerivs[m_wPartDerivs.size() - layer].pitch);

            checkCudaErrors(cudaFree(wTrans.ptr));
        }

        for (int i = 0; i < zs.size(); i++)
        {
            checkCudaErrors(cudaFree(as[i + 1]));
            checkCudaErrors(cudaFree(zs[i]));
        }
    }

private:
    size_t m_layerCount;
    std::vector<uint32_t> m_neuronCounts;

    std::vector<cudaPitchedPtr> m_w;
    std::vector<float*> m_b;

    std::vector<cudaPitchedPtr> m_wPartDerivs;
    std::vector<float*> m_bPartDerivs;
};

int main()
{
    checkCudaErrors(cudaSetDevice(0));

    std::vector<float*> xs;
    std::vector<float*> ys;
    uint32_t elementSize = 0;
    loadData(L"MNIST/train-images.idx3-ubyte", xs, L"MNIST/train-labels.idx1-ubyte", ys, elementSize);

    std::vector<uint32_t> neuronCounts = { elementSize, 30, 10 };
    NeuralNetwork network(neuronCounts);

    network.learn(xs, ys, 2, 10, 3.0f);

    for (int i = 0; i < xs.size(); i++)
    {
        delete[] xs[i];
        delete[] ys[i];
    }

    std::vector<float*> testImages;
    std::vector<float*> testLabels;
    loadData(L"MNIST/t10k-images.idx3-ubyte", testImages, L"MNIST/t10k-labels.idx1-ubyte", testLabels, elementSize);
    int corrects = 0;
    for (int i = 0; i < testImages.size(); i++)
    {
        /*for (int y = 0; y < 28; y++)
        {
            for (int x = 0; x < 28; x++)
            {
                if (testImages[0][y * 28 + x] > 0) printf("0");
                else printf(" ");
            }
            printf("\n");
        }*/
        int image = i;
        uint8_t label = std::max_element(testLabels[image], testLabels[image] + 10) - testLabels[image];
        uint8_t digit = network.recognizeDigit(testImages[image]);
        if (label == digit) corrects++;
        //printf("Real value is %d, network thinks it's %d\n", label, digit);
    }
    printf("%d were correct", corrects);

    for (int i = 0; i < testImages.size(); i++)
    {
        delete[] testImages[i];
        delete[] testLabels[i];
    }

    /*int cols = 784;
    int rows = 30;

    float* hostPtr;
    checkCudaErrors(cudaMallocHost(&hostPtr, rows * cols * sizeof(float)));
    for (int i = 0; i < rows * cols; i++) hostPtr[i] = ((i % (matSize + 1)) == 0) ? 1.0f : 0.0f;

    cudaPitchedPtr deviceMatrix;
    cudaExtent extent = make_cudaExtent(cols * sizeof(float), rows, 1);
    checkCudaErrors(cudaMalloc3D(&deviceMatrix, extent));

    cudaPitchedPtr hostMatrix;
    hostMatrix.ptr = hostPtr;
    hostMatrix.pitch = matSize * sizeof(float);
    hostMatrix.xsize = deviceMatrix.xsize;
    hostMatrix.ysize = deviceMatrix.ysize;

    cudaMemcpy3DParms cpyParms = { 0 };
    cpyParms.srcPtr = hostMatrix;
    cpyParms.dstPtr = deviceMatrix;
    cpyParms.extent = extent;
    cpyParms.kind = cudaMemcpyHostToDevice;
    checkCudaErrors(cudaMemcpy3D(&cpyParms));

    float* vector;
    checkCudaErrors(cudaMallocManaged(&vector, matSize * sizeof(float)));
    for (int i = 0; i < matSize; i++) vector[i] = (i + 1.0f) * -0.1f;

    float* result;
    checkCudaErrors(cudaMallocManaged(&result, matSize * sizeof(float)));

    layerActivation<<<1, matSize>>>(deviceMatrix.ptr, deviceMatrix.pitch, matSize, vector, vector, result);
    cudaDeviceSynchronize();

    for (int i = 0; i < matSize; i++)
        output("%f, ", result[i]);
    output("\n");*/

    return 0;
}
