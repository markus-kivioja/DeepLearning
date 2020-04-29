#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "include/helper_cuda.h"
#include "include/utils.h"

__device__ double sigmoid(double z)
{
    return 1.0 / (1.0 + expf(-z));
}

__global__ void sigmoid(const double* in, double* out)
{
    out[threadIdx.x] = 1.0 / (1.0 + expf(-in[threadIdx.x]));
}

__device__ double sigmoidDeriv(double z)
{
    double temp = sigmoid(z);
    return temp * (1.0 - temp);
}

__device__ double quadraticCostDeriv(double a, double y)
{
    return a - y;
}

__global__ void dot(const double* vec1, const double* vec2, double* out)
{
    extern __shared__ double cache[];
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

__global__ void gradientDescentStep(const void* variables, const void* partialDerivs, size_t pitch, double learningRate, size_t subsetSize, size_t cols)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cols) return;

    double* variablesRow = (double*)(((uint8_t*)variables) + y * pitch);
    double* partialDerivsRow = (double*)(((uint8_t*)partialDerivs) + y * pitch);

    variablesRow[x] = variablesRow[x] - (learningRate / subsetSize) * partialDerivsRow[x];
}

__global__ void matVecMulAdd(const void* matrix, size_t pitch, size_t colCount, const double* vector, const double* addVec, double* out)
{
    size_t idx = threadIdx.x;
    double* row = (double*)(((uint8_t*)matrix) + idx * pitch);
    dot<<<1, colCount, colCount * sizeof(double)>>>(row, vector, &out[idx]);
    cudaDeviceSynchronize();
    out[idx] += addVec[idx];
}

__global__ void layerActivation(const void* matrix, size_t pitch, size_t colCount, const double* vector, const double* addVec, double* out)
{
    size_t idx = threadIdx.x;
    double* row = (double*)(((uint8_t*)matrix) + idx * pitch);
    dot<<<1, colCount, colCount * sizeof(double)>>>(row, vector, &out[idx]);
    cudaDeviceSynchronize();
    out[idx] = sigmoid(out[idx] + addVec[idx]);
}

__global__ void columnProduct(const double* vector1, const double* vector2, void* out, size_t pitch, size_t rows, size_t cols)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cols || y >= rows) return;

    double* outRow = (double*)(((uint8_t*)out) + y * pitch);
    outRow[x] = vector1[y] * vector2[x];
}

__global__ void calcOutputError(const double* a, const double* y, const double* z, double* out)
{
    double costDeriv = quadraticCostDeriv(a[threadIdx.x], y[threadIdx.x]);
    double sigDeriv = sigmoidDeriv(z[threadIdx.x]);
    out[threadIdx.x] = costDeriv * sigDeriv;
}

__global__ void matTranspose(const void* mat, size_t pitchIn, void* out, size_t pitchOut)
{
    size_t xIn = blockIdx.x * blockDim.x + threadIdx.x;
    size_t yIn = blockIdx.y * blockDim.y + threadIdx.y;
    
    size_t xOut = yIn;
    size_t yOut = xIn;

    const double* inRow = (double*)(((uint8_t*)mat) + yIn * pitchIn);
    double* outRow = (double*)(((uint8_t*)out) + yOut * pitchOut);

    outRow[xOut] = inRow[xIn];
}

__global__ void calcLayerError(const void* wTrans, size_t pitch, size_t colCount, const double* nextLayerError, const double* z, double* out)
{
    size_t idx = threadIdx.x;
    double* row = (double*)(((uint8_t*)wTrans) + idx * pitch);
    dot<<<1, colCount, colCount * sizeof(double)>>>(row, nextLayerError, &out[idx]);
    cudaDeviceSynchronize();
    out[idx] *= sigmoidDeriv(z[idx]);
}

__global__ void accumulate(void* mat1, const void* mat2, size_t rows, size_t cols, size_t pitch)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cols || y >= rows) return;

    double* row1 = (double*)(((uint8_t*)mat1) + y * pitch);
    const double* row2 = (double*)(((uint8_t*)mat2) + y * pitch);

    row1[x] += row2[x];
}

class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<uint32_t>& pNeuronCounts)
    {
        m_neuronCounts = pNeuronCounts;
        m_layerCount = m_neuronCounts.size();

        std::default_random_engine generator;
        std::normal_distribution<double> distribution(0.0, 1.0);

        // Allocate GPU mem for weights and biases and cost function partial derivatives with respect to them.
        for (size_t layer = 0; layer < m_layerCount - 1; layer++)
        {
            int32_t rows = m_neuronCounts[layer + 1];
            int32_t cols = m_neuronCounts[layer];

            // Initialize weights and copy to GPU
            double* wHostData;
            checkCudaErrors(cudaMallocHost(&wHostData, rows * cols * sizeof(double)));
            for (int i = 0; i < rows * cols; i++)
            {
                wHostData[i] = distribution(generator);
            }

            cudaPitchedPtr wDevice;
            cudaExtent extent = make_cudaExtent(cols * sizeof(double), rows, 1);
            checkCudaErrors(cudaMalloc3D(&wDevice, extent));
            m_w.push_back(wDevice);

            cudaPitchedPtr wHost;
            wHost.ptr = wHostData;
            wHost.pitch = cols * sizeof(double);
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
            checkCudaErrors(cudaMalloc3D(&wDevice, extent));
            m_wPartDerivsDelta.push_back(wDevice);

            // Initialize biases and copy to GPU
            double* bHost;
            checkCudaErrors(cudaMallocHost(&bHost, rows * sizeof(double)));
            for (int i = 0; i < rows; i++)
            {
                bHost[i] = distribution(generator);
            }

            double* bDevice;
            checkCudaErrors(cudaMalloc(&bDevice, rows * sizeof(double)));
            m_b.push_back(bDevice);

            checkCudaErrors(cudaMemcpy(bDevice, bHost, rows * sizeof(double), cudaMemcpyHostToDevice));

            checkCudaErrors(cudaMalloc(&bDevice, rows * sizeof(double)));
            checkCudaErrors(cudaMemset(bDevice, 0, rows * sizeof(double)));
            m_bPartDerivs.push_back(bDevice);
            checkCudaErrors(cudaMalloc(&bDevice, rows * sizeof(double)));
            m_bPartDerivsDelta.push_back(bDevice);
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

    uint8_t recognizeDigit(double* x)
    {
        std::vector<double*> as;

        for (size_t layer = 0; layer < m_layerCount; layer++)
        {
            double* a;
            checkCudaErrors(cudaMalloc(&a, m_neuronCounts[layer] * sizeof(double)));
            if (!layer)
            {
                checkCudaErrors(cudaMemcpy(a, x, m_neuronCounts[layer] * sizeof(double), cudaMemcpyHostToDevice));
            }
            else
            {
                checkCudaErrors(cudaMemset(a, 0, m_neuronCounts[layer] * sizeof(double)));
            }
            as.push_back(a);
        }

        for (size_t layer = 0; layer < m_layerCount - 1; layer++)
        {
            int32_t rows = m_neuronCounts[layer + 1];
            int32_t cols = m_neuronCounts[layer];

            layerActivation<<<1, rows>>>(m_w[layer].ptr, m_w[layer].pitch, cols, as[layer], m_b[layer], as[layer + 1]);
        }

        std::vector<double> result(m_neuronCounts[m_layerCount - 1]);
        checkCudaErrors(cudaMemcpy(&result[0], as[as.size() - 1], result.size() * sizeof(double), cudaMemcpyDeviceToHost));

        for (auto a : as) checkCudaErrors(cudaFree(a));

        return static_cast<uint8_t>(std::max_element(result.begin(), result.begin() + 10) - result.begin());
    }

    void learn(std::vector<double*>& xs, std::vector<double*>& ys, uint32_t epochCount, uint32_t subsetSize, double learningRate)
    {
        // Copy the training data into GPU memory
        std::vector<double*> xsDevice;
        std::vector<double*> ysDevice;
        xsDevice.resize(xs.size());
        ysDevice.resize(ys.size());
        for (int i = 0; i < xs.size(); i++)
        {
            checkCudaErrors(cudaMalloc(&xsDevice[i], m_neuronCounts[0] * sizeof(double)));
            checkCudaErrors(cudaMalloc(&ysDevice[i], m_neuronCounts[m_neuronCounts.size() - 1] * sizeof(double)));
            checkCudaErrors(cudaMemcpy(xsDevice[i], xs[i], m_neuronCounts[0] * sizeof(double), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(ysDevice[i], ys[i], m_neuronCounts[m_neuronCounts.size() - 1] * sizeof(double), cudaMemcpyHostToDevice));
        }

        uint32_t subsetCount = static_cast<uint32_t>(std::ceil(xsDevice.size() / (double)subsetSize));

        // Use stochastic gradient descent to learn the weights and biases
        // Rearrange the training data for each epoch and divide into subsets
        auto rngX = std::default_random_engine{};
        auto rngY = rngX;
        for (uint32_t epoch = 0; epoch < epochCount; epoch++)
        {
            printf("Start epoch %d\n", epoch);
            std::shuffle(std::begin(xsDevice), std::end(xsDevice), rngX);
            std::shuffle(std::begin(ysDevice), std::end(ysDevice), rngY);

            uint32_t elementsLeft = static_cast<uint32_t>(xsDevice.size());

            // Teach the network with a subset of the learning data
            for (uint32_t subset = 0; subset < subsetCount; subset++)
            {
                auto xStart = xsDevice.begin() + subset * subsetSize;
                auto xEnd = xStart + std::min(subsetSize, elementsLeft);
                std::vector<double*> xSubset(xStart, xEnd);

                auto yStart = ysDevice.begin() + subset * subsetSize;
                auto yEnd = yStart + std::min(subsetSize, elementsLeft);
                std::vector<double*> ySubset(yStart, yEnd);

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
            cudaExtent extent = make_cudaExtent(cols * sizeof(double), rows, 1);
            checkCudaErrors(cudaMemset3D(m_wPartDerivs[layer], 0, extent));
            checkCudaErrors(cudaMemset(m_bPartDerivs[layer], 0, rows * sizeof(double)));
        }
    }

    void updateSubset(const std::vector<double*>& xs, const std::vector<double*>& ys, double learningRate)
    {
        resetPartialDerivatives();

        // Calculate partial derivatives
        for (int i = 0; i < xs.size(); i++)
        {
            double* x = xs[i];
            double* y = ys[i];

            backpropagate(x, y);

            for (size_t layer = 0; layer < m_layerCount - 1; layer++)
            {
                int32_t rows = m_neuronCounts[layer + 1];
                int32_t cols = m_neuronCounts[layer];
                dim3 gridSize(static_cast<uint32_t>(std::ceil((double)cols / (double)rows)));
                dim3 blockSize(rows, rows);
                accumulate<<<gridSize, blockSize>>>(m_wPartDerivs[layer].ptr, m_wPartDerivsDelta[layer].ptr, rows, cols, m_wPartDerivs[layer].pitch);
                accumulate<<<1, rows>>>(m_bPartDerivs[layer], m_bPartDerivsDelta[layer], 1, rows, sizeof(double));
            }
        }
        // Update weights and biases
        for (size_t layer = 0; layer < m_layerCount - 1; layer++)
        {
            int32_t rows = m_neuronCounts[layer + 1];
            int32_t cols = m_neuronCounts[layer];
            dim3 gridSize(static_cast<uint32_t>(std::ceil((double)cols / (double)rows)));
            dim3 blockSize(rows, rows);
            gradientDescentStep<<<gridSize, blockSize>>>(m_w[layer].ptr, m_wPartDerivs[layer].ptr, m_w[layer].pitch, learningRate, xs.size(), cols);
            gradientDescentStep<<<1, rows>>>(m_b[layer], m_bPartDerivs[layer], sizeof(double), learningRate, xs.size(), cols);
        }
    }

    void backpropagate(double* x, double* y)
    {
        std::vector<double*> as = { x };
        std::vector<double*> zs;

        for (size_t layer = 0; layer < m_layerCount - 1; layer++)
        {
            double* z;
            checkCudaErrors(cudaMalloc(&z, m_neuronCounts[layer + 1] * sizeof(double)));
            zs.push_back(z);

            double* a;
            checkCudaErrors(cudaMalloc(&a, m_neuronCounts[layer + 1] * sizeof(double)));
            as.push_back(a);
        }
        for (size_t layer = 0; layer < m_layerCount - 1; layer++)
        {
            int32_t rows = m_neuronCounts[layer + 1];
            int32_t cols = m_neuronCounts[layer];

            double* a = as[layer];
            double* aNext = as[layer + 1];
            double* z = zs[layer];

            matVecMulAdd<<<1, rows>>>(m_w[layer].ptr, m_w[layer].pitch, cols, a, m_b[layer], z);
            sigmoid<<<1, rows>>>(z, aNext);
        }
        uint32_t lastLayerSize = m_neuronCounts[m_neuronCounts.size() - 1];
        calcOutputError<<<1, lastLayerSize>>>(as[as.size() - 1], y, zs[zs.size() - 1], m_bPartDerivsDelta[m_bPartDerivsDelta.size() - 1]);
        
        dim3 blockSize(m_neuronCounts[m_neuronCounts.size() - 2], m_neuronCounts[m_neuronCounts.size() - 1]);
        columnProduct<<<1, blockSize>>>(m_bPartDerivsDelta[m_bPartDerivsDelta.size() - 1], as[as.size() - 2], 
            m_wPartDerivsDelta[m_wPartDerivsDelta.size() - 1].ptr, m_wPartDerivsDelta[m_wPartDerivsDelta.size() - 1].pitch,
            m_neuronCounts[m_neuronCounts.size() - 1], m_neuronCounts[m_neuronCounts.size() - 2]);

        for (size_t layer = 2; layer < m_layerCount; layer++)
        {
            double* z = zs[zs.size() - layer];

            int32_t rows = m_neuronCounts[m_neuronCounts.size() - layer + 1];
            int32_t cols = m_neuronCounts[m_neuronCounts.size() - layer];
            cudaExtent wTransExtent = make_cudaExtent(rows * sizeof(double), cols, 1);
            cudaPitchedPtr wTrans;
            checkCudaErrors(cudaMalloc3D(&wTrans, wTransExtent));
            dim3 blockSize(cols, rows);
            matTranspose<<<1, blockSize>>>(m_w[m_w.size() - layer + 1].ptr, m_w[m_w.size() - layer + 1].pitch, wTrans.ptr, wTrans.pitch);
            calcLayerError<<<1, cols>>>(wTrans.ptr, wTrans.pitch, rows, m_bPartDerivsDelta[m_bPartDerivsDelta.size() - layer + 1], z, m_bPartDerivsDelta[m_bPartDerivsDelta.size() - layer]);

            rows = m_neuronCounts[m_neuronCounts.size() - layer];
            cols = m_neuronCounts[m_neuronCounts.size() - layer - 1];
            dim3 gridSize(static_cast<uint32_t>(std::ceil((double)cols / (double)rows)));
            blockSize = dim3(rows, rows);
            columnProduct<<<gridSize, blockSize>>>(m_bPartDerivsDelta[m_bPartDerivsDelta.size() - layer], as[as.size() - layer - 1], 
                m_wPartDerivsDelta[m_wPartDerivsDelta.size() - layer].ptr, m_wPartDerivsDelta[m_wPartDerivsDelta.size() - layer].pitch, rows, cols);

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
    std::vector<double*> m_b;

    std::vector<cudaPitchedPtr> m_wPartDerivs;
    std::vector<double*> m_bPartDerivs;

    std::vector<cudaPitchedPtr> m_wPartDerivsDelta;
    std::vector<double*> m_bPartDerivsDelta;
};

int main()
{
    checkCudaErrors(cudaSetDevice(0));

    std::vector<double*> xs;
    std::vector<double*> ys;
    uint32_t elementSize = 0;
    loadData(L"MNIST/train-images.idx3-ubyte", xs, L"MNIST/train-labels.idx1-ubyte", ys, elementSize);

    std::vector<uint32_t> neuronCounts = { elementSize, 30, 10 };
    NeuralNetwork network(neuronCounts);

    network.learn(xs, ys, 2, 10, 3.0);

    for (int i = 0; i < xs.size(); i++)
    {
        delete[] xs[i];
        delete[] ys[i];
    }

    std::vector<double*> testImages;
    std::vector<double*> testLabels;
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
        uint8_t label = static_cast<uint8_t>(std::max_element(testLabels[image], testLabels[image] + 10) - testLabels[image]);
        uint8_t digit = network.recognizeDigit(testImages[image]);
        if (label == digit) corrects++;
        //printf("Real value is %d, network thinks it's %d\n", label, digit);
    }
    printf("%f were correct", (double)corrects / (double)testImages.size());

    for (int i = 0; i < testImages.size(); i++)
    {
        delete[] testImages[i];
        delete[] testLabels[i];
    }

    /*int cols = 784;
    int rows = 30;

    double* hostPtr;
    checkCudaErrors(cudaMallocHost(&hostPtr, rows * cols * sizeof(double)));
    for (int i = 0; i < rows * cols; i++) hostPtr[i] = ((i % (matSize + 1)) == 0) ? 1.0 : 0.0;

    cudaPitchedPtr deviceMatrix;
    cudaExtent extent = make_cudaExtent(cols * sizeof(double), rows, 1);
    checkCudaErrors(cudaMalloc3D(&deviceMatrix, extent));

    cudaPitchedPtr hostMatrix;
    hostMatrix.ptr = hostPtr;
    hostMatrix.pitch = matSize * sizeof(double);
    hostMatrix.xsize = deviceMatrix.xsize;
    hostMatrix.ysize = deviceMatrix.ysize;

    cudaMemcpy3DParms cpyParms = { 0 };
    cpyParms.srcPtr = hostMatrix;
    cpyParms.dstPtr = deviceMatrix;
    cpyParms.extent = extent;
    cpyParms.kind = cudaMemcpyHostToDevice;
    checkCudaErrors(cudaMemcpy3D(&cpyParms));

    double* vector;
    checkCudaErrors(cudaMallocManaged(&vector, matSize * sizeof(double)));
    for (int i = 0; i < matSize; i++) vector[i] = (i + 1.0) * - 0.1;

    double* result;
    checkCudaErrors(cudaMallocManaged(&result, matSize * sizeof(double)));

    layerActivation<<<1, matSize>>>(deviceMatrix.ptr, deviceMatrix.pitch, matSize, vector, vector, result);
    cudaDeviceSynchronize();

    for (int i = 0; i < matSize; i++)
        output("%f, ", result[i]);
    output("\n");*/

    return 0;
}
