#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "include/helper_cuda.h"
#include "include/utils.h"

namespace
{
struct Matrix {
    void* data;
    size_t pitch;
    size_t rows;
    size_t cols;
};
}

__device__ double sigmoid(double z)
{
    return 1.0 / (1.0 + expf(-z));
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

__global__ void dotCDP(const double* vec1, const double* vec2, double* out)
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

__device__ void dot(const double* vec1, const double* vec2, double* cache)
{
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
}

__global__ void gradientDescentStepW(Matrix variables, Matrix partialDerivs, double learningRate, size_t subsetSize)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= variables.cols) return;

    double* variablesRow = (double*)(((uint8_t*)variables.data) + y * variables.pitch);
    double* partialDerivsRow = (double*)(((uint8_t*)partialDerivs.data) + y * partialDerivs.pitch);

    variablesRow[x] = variablesRow[x] - (learningRate / subsetSize) * partialDerivsRow[x];
}

__global__ void gradientDescentStepB(double* variables, const double* partialDerivs, double learningRate, size_t subsetSize)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    variables[x] = variables[x] - (learningRate / subsetSize) * partialDerivs[x];
}

__global__ void layerActivationCDP(Matrix matrix, const double* vector, const double* addVec, double* zOut, double* out)
{
    size_t idx = threadIdx.x;
    double* row = (double*)(((uint8_t*)matrix.data) + idx * matrix.pitch);
    dotCDP<<<1, matrix.cols, matrix.cols * sizeof(double)>>>(row, vector, &out[idx]);
    cudaDeviceSynchronize();
    zOut[idx] = out[idx] + addVec[idx];
    out[idx] = sigmoid(zOut[idx]);
}

__global__ void columnProduct(const double* vector1, const double* vector2, Matrix out)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= out.cols || y >= out.rows) return;

    double* outRow = (double*)(((uint8_t*)out.data) + y * out.pitch);
    outRow[x] = vector1[y] * vector2[x];
}

__global__ void calcOutputError(const double* a, const double* y, const double* z, double* out)
{
    double costDeriv = quadraticCostDeriv(a[threadIdx.x], y[threadIdx.x]);
    double sigDeriv = sigmoidDeriv(z[threadIdx.x]);
    out[threadIdx.x] = costDeriv * sigDeriv;
}

__global__ void matTranspose(Matrix mat, Matrix out)
{
    size_t xIn = blockIdx.x * blockDim.x + threadIdx.x;
    size_t yIn = blockIdx.y * blockDim.y + threadIdx.y;
    
    size_t xOut = yIn;
    size_t yOut = xIn;

    const double* inRow = (double*)(((uint8_t*)mat.data) + yIn * mat.pitch);
    double* outRow = (double*)(((uint8_t*)out.data) + yOut * out.pitch);

    outRow[xOut] = inRow[xIn];
}

__global__ void calcLayerErrorCDP(Matrix wTrans, const double* nextLayerError, const double* z, double* out)
{
    size_t idx = threadIdx.x;
    double* row = (double*)(((uint8_t*)wTrans.data) + idx * wTrans.pitch);
    dotCDP<<<1, wTrans.cols, wTrans.cols * sizeof(double)>>>(row, nextLayerError, &out[idx]);
    cudaDeviceSynchronize();
    out[idx] *= sigmoidDeriv(z[idx]);
}

__global__ void accumulateMat(Matrix mat1, Matrix mat2)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= mat1.cols || y >= mat2.rows) return;

    double* row1 = (double*)(((uint8_t*)mat1.data) + y * mat1.pitch);
    const double* row2 = (double*)(((uint8_t*)mat2.data) + y * mat2.pitch);

    row1[x] += row2[x];
}

__global__ void accumulateVec(double* vec1, const double* vec2)
{
    vec1[threadIdx.x] += vec2[threadIdx.x];
}

class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<uint32_t>& pNeuronCounts)
    {
        m_neuronCounts = pNeuronCounts;
        m_layerCount = m_neuronCounts.size();

        std::default_random_engine generator;
        std::normal_distribution<double> distribution(0.0, 1.0);

        m_as.push_back(nullptr);

        // Allocate GPU mem for weights and biases and cost function partial derivatives with respect to them.
        for (size_t layer = 0; layer < m_layerCount - 1; layer++)
        {
            size_t rows = m_neuronCounts[layer + 1];
            size_t cols = m_neuronCounts[layer];

            // Initialize weights and copy to GPU
            double* wHostData;
            checkCudaErrors(cudaMallocHost(&wHostData, rows * cols * sizeof(double)));
            for (size_t i = 0; i < rows * cols; i++)
            {
                wHostData[i] = distribution(generator);
            }

            Matrix w = {nullptr, 0, rows, cols};
            checkCudaErrors(cudaMallocPitch(&w.data, &w.pitch, cols * sizeof(double), rows));
            m_w.push_back(w);

            checkCudaErrors(cudaMemcpy2D(w.data, w.pitch, wHostData, cols * sizeof(double), cols * sizeof(double), rows, cudaMemcpyHostToDevice));
            checkCudaErrors(cudaFreeHost(wHostData));

            // Allocate and initialize cost function partial derivatives with rispect to weights
            checkCudaErrors(cudaMallocPitch(&w.data, &w.pitch, cols * sizeof(double), rows));
            checkCudaErrors(cudaMemset2D(w.data, w.pitch, 0, cols * sizeof(double), rows));
            m_wPartDerivs.push_back(w);
            checkCudaErrors(cudaMallocPitch(&w.data, &w.pitch, cols * sizeof(double), rows));
            m_wPartDerivsDelta.push_back(w);

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

            double* z;
            checkCudaErrors(cudaMalloc(&z, m_neuronCounts[layer + 1] * sizeof(double)));
            m_zs.push_back(z);

            double* a;
            checkCudaErrors(cudaMalloc(&a, m_neuronCounts[layer + 1] * sizeof(double)));
            m_as.push_back(a);

            checkCudaErrors(cudaMallocPitch(&m_wTrans.data, &m_wTrans.pitch, rows * sizeof(double), cols));
            m_wTrans.cols = rows;
            m_wTrans.rows = cols;
        }
    }

    ~NeuralNetwork()
    {
        for (size_t layer = 0; layer < m_w.size() - 1; layer++)
        {
            checkCudaErrors(cudaFree(m_w[layer].data));
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

            layerActivationCDP<<<1, rows>>>(m_w[layer], as[layer], m_b[layer], m_zs[layer], as[layer + 1]);
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
            checkCudaErrors(cudaMemset2D(m_wPartDerivs[layer].data, m_wPartDerivs[layer].pitch, 0, cols * sizeof(double), rows));
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
                accumulateMat<<<gridSize, blockSize>>>(m_wPartDerivs[layer], m_wPartDerivsDelta[layer]);
                accumulateVec<<<1, rows>>>(m_bPartDerivs[layer], m_bPartDerivsDelta[layer]);
            }
        }
        // Update weights and biases
        for (size_t layer = 0; layer < m_layerCount - 1; layer++)
        {
            int32_t rows = m_neuronCounts[layer + 1];
            int32_t cols = m_neuronCounts[layer];
            dim3 gridSize(static_cast<uint32_t>(std::ceil((double)cols / (double)rows)));
            dim3 blockSize(rows, rows);
            gradientDescentStepW<<<gridSize, blockSize>>>(m_w[layer], m_wPartDerivs[layer], learningRate, xs.size());
            gradientDescentStepB<<<1, rows>>>(m_b[layer], m_bPartDerivs[layer], learningRate, xs.size());
        }
    }

    void backpropagate(double* x, double* y)
    {
        m_as[0] = x;

        for (size_t layer = 0; layer < m_layerCount - 1; layer++)
        {
            int32_t rows = m_neuronCounts[layer + 1];
            int32_t cols = m_neuronCounts[layer];

            double* a = m_as[layer];
            double* aNext = m_as[layer + 1];
            double* z = m_zs[layer];

            layerActivationCDP<<<1, rows>>>(m_w[layer], a, m_b[layer], z, aNext);
        }
        uint32_t lastLayerSize = m_neuronCounts[m_neuronCounts.size() - 1];
        calcOutputError<<<1, lastLayerSize>>>(m_as[m_as.size() - 1], y, m_zs[m_zs.size() - 1], m_bPartDerivsDelta[m_bPartDerivsDelta.size() - 1]);
        
        dim3 blockSize(m_neuronCounts[m_neuronCounts.size() - 2], m_neuronCounts[m_neuronCounts.size() - 1]);
        columnProduct<<<1, blockSize>>>(m_bPartDerivsDelta[m_bPartDerivsDelta.size() - 1], m_as[m_as.size() - 2], m_wPartDerivsDelta[m_wPartDerivsDelta.size() - 1]);

        for (size_t layer = 2; layer < m_layerCount; layer++)
        {
            double* z = m_zs[m_zs.size() - layer];

            int32_t rows = m_neuronCounts[m_neuronCounts.size() - layer + 1];
            int32_t cols = m_neuronCounts[m_neuronCounts.size() - layer];
            dim3 blockSize(cols, rows);
            matTranspose<<<1, blockSize>>>(m_w[m_w.size() - layer + 1], m_wTrans);
            calcLayerErrorCDP<<<1, cols>>>(m_wTrans, m_bPartDerivsDelta[m_bPartDerivsDelta.size() - layer + 1], z, m_bPartDerivsDelta[m_bPartDerivsDelta.size() - layer]);

            rows = m_neuronCounts[m_neuronCounts.size() - layer];
            cols = m_neuronCounts[m_neuronCounts.size() - layer - 1];
            dim3 gridSize(static_cast<uint32_t>(std::ceil((double)cols / (double)rows)));
            blockSize = dim3(rows, rows);
            columnProduct<<<gridSize, blockSize>>>(m_bPartDerivsDelta[m_bPartDerivsDelta.size() - layer], m_as[m_as.size() - layer - 1], m_wPartDerivsDelta[m_wPartDerivsDelta.size() - layer]);
        }
    }

private:
    size_t m_layerCount;
    std::vector<uint32_t> m_neuronCounts;

    std::vector<Matrix> m_w;
    std::vector<double*> m_b;

    std::vector<Matrix> m_wPartDerivs;
    std::vector<double*> m_bPartDerivs;

    std::vector<Matrix> m_wPartDerivsDelta;
    std::vector<double*> m_bPartDerivsDelta;

    std::vector<double*> m_as;
    std::vector<double*> m_zs;
    Matrix m_wTrans;
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

    network.learn(xs, ys, 1, 10, 3.0);

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
    printf("%.2f%% were correct", (double)corrects / (double)testImages.size() * 100.0f);

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

    layerActivationCDP<<<1, matSize>>>(deviceMatrix.ptr, deviceMatrix.pitch, matSize, vector, vector, result);

    for (int i = 0; i < matSize; i++)
        output("%f, ", result[i]);
    output("\n");*/

    return 0;
}
