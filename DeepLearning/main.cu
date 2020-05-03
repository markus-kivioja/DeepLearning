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

__global__ void gradientDescentStepW(Matrix w, Matrix partialDerivs, double learningRate, size_t subsetSize)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= w.cols || row >= w.rows) return;

    double* wRow = (double*)(((uint8_t*)w.data) + row * w.pitch);
    double* partialDerivsRow = (double*)(((uint8_t*)partialDerivs.data) + row * partialDerivs.pitch);

    wRow[col] = wRow[col] - (learningRate / subsetSize) * partialDerivsRow[col];
}

__global__ void gradientDescentStepB(double* b, Matrix partialDerivs, double learningRate, size_t subsetSize)
{
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;

    double pdSum = 0;
    for (int col = 0; col < partialDerivs.rows; col++)
    {
        double* pdCol = (double*)(((uint8_t*)partialDerivs.data) + col * partialDerivs.pitch);
        pdSum += pdCol[row];
    }
    b[row] = b[row] - (learningRate / (double)subsetSize) * pdSum;
}

__global__ void layerActivationCDP(Matrix w, const double* a, const double* b, double* zOut, double* out)
{
    size_t idx = threadIdx.x;
    double* row = (double*)(((uint8_t*)w.data) + idx * w.pitch);
    dot<<<1, w.cols, w.cols * sizeof(double)>>>(row, a, &out[idx]);
    cudaDeviceSynchronize();
    zOut[idx] = out[idx] + b[idx];
    out[idx] = sigmoid(zOut[idx]);
}

__global__ void layerActivation(Matrix w, const double* a, const double* b, double* zOut, double* out)
{
    size_t idx = threadIdx.x;
    double* row = (double*)(((uint8_t*)w.data) + idx * w.pitch);
    double result = 0;
    for (int col = 0; col < w.cols; col++)
    {
        result += row[col] * a[col];
    }
    result += b[idx];
    zOut[idx] = result;
    out[idx] = sigmoid(result);
}

__global__ void layerActivation(Matrix w, const Matrix a, const double* b, Matrix zOut, Matrix out)
{
    size_t rowIdx = threadIdx.x;
    size_t colIdx = blockIdx.x;
    double* wRow = (double*)(((uint8_t*)w.data) + rowIdx * w.pitch);
    double* aCol = (double*)(((uint8_t*)a.data) + colIdx * a.pitch);
    double* zCol = (double*)(((uint8_t*)zOut.data) + colIdx * zOut.pitch);
    double* outCol = (double*)(((uint8_t*)out.data) + colIdx * out.pitch);

    double result = 0;
    for (int element = 0; element < w.cols; element++)
    {
        result += wRow[element] * aCol[element];
    }
    result += b[rowIdx];

    zCol[rowIdx] = result;
    outCol[rowIdx] = sigmoid(result);
}

__global__ void calcWPartiaDerivs(Matrix bPartialDerivs, Matrix a, Matrix out)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= out.cols || row >= out.rows) return;

    double result = 0;
    for (int element = 0; element < bPartialDerivs.rows; element++)
    {
        double* bpdCol = (double*)(((uint8_t*)bPartialDerivs.data) + element * bPartialDerivs.pitch);
        double* aRow = (double*)(((uint8_t*)a.data) + element * a.pitch);
        result += bpdCol[row] * aRow[col];
    }

    double* outRow = (double*)(((uint8_t*)out.data) + row * out.pitch);
    outRow[col] = result;
}

__global__ void calcOutputError(Matrix a, Matrix y, Matrix z, Matrix out)
{
    size_t rowIdx = threadIdx.x;
    size_t colIdx = blockIdx.x;

    double* aCol = (double*)(((uint8_t*)a.data) + colIdx * a.pitch);
    double* yCol = (double*)(((uint8_t*)y.data) + colIdx * y.pitch);
    double* zCol = (double*)(((uint8_t*)z.data) + colIdx * z.pitch);
    double* outCol = (double*)(((uint8_t*)out.data) + colIdx * out.pitch);

    double costDeriv = quadraticCostDeriv(aCol[rowIdx], yCol[rowIdx]);
    double sigDeriv = sigmoidDeriv(zCol[rowIdx]);
    outCol[rowIdx] = costDeriv * sigDeriv;
}

__global__ void calcLayerErrorCDP(Matrix wTrans, const double* nextLayerError, const double* z, double* out)
{
    size_t idx = threadIdx.x;
    double* row = (double*)(((uint8_t*)wTrans.data) + idx * wTrans.pitch);
    dot<<<1, wTrans.cols, wTrans.cols * sizeof(double)>>>(row, nextLayerError, &out[idx]);
    cudaDeviceSynchronize();
    out[idx] *= sigmoidDeriv(z[idx]);
}

__global__ void calcLayerError(Matrix w, Matrix bPartialDerivs, Matrix z, Matrix out)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= out.cols || row >= out.rows) return;

    double* bpdCol = (double*)(((uint8_t*)bPartialDerivs.data) + col * bPartialDerivs.pitch);
    double* zCol = (double*)(((uint8_t*)z.data) + col * z.pitch);
    double* outCol = (double*)(((uint8_t*)out.data) + col * out.pitch);

    double result = 0;
    for (int element = 0; element < w.rows; element++)
    {
        double* wTransRow = (double*)(((uint8_t*)w.data) + element * w.pitch);
        result += wTransRow[row] * bpdCol[element];
    }
    outCol[row] = result * sigmoidDeriv(zCol[row]);
}

class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<uint32_t>& pNeuronCounts, uint32_t subsetSize) :
        m_subsetSize(subsetSize)
    {
        m_neuronCounts = pNeuronCounts;
        m_layerCount = m_neuronCounts.size();

        std::default_random_engine generator;
        std::normal_distribution<double> distribution(0.0, 1.0);

        m_as.push_back(Matrix{0});

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

            // Allocate cost function partial derivatives with respect to weights
            checkCudaErrors(cudaMallocPitch(&w.data, &w.pitch, cols * sizeof(double), rows));
            m_wPartDerivs.push_back(w);

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

            // Allocate cost function partial derivatives with respect to biases
            Matrix bPartDerivs = { nullptr, 0, m_subsetSize, rows };
            checkCudaErrors(cudaMallocPitch(&bPartDerivs.data, &bPartDerivs.pitch, rows * sizeof(double), m_subsetSize));
            m_bPartDerivs.push_back(bPartDerivs);

            Matrix z = { nullptr, 0, m_subsetSize, rows };
            checkCudaErrors(cudaMallocPitch(&z.data, &z.pitch, rows * sizeof(double), m_subsetSize));
            m_zs.push_back(z);

            Matrix a = { nullptr, 0, m_subsetSize, rows };
            checkCudaErrors(cudaMallocPitch(&a.data, &a.pitch, rows * sizeof(double), m_subsetSize));
            m_as.push_back(a);
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

        double* dummyZ;
        checkCudaErrors(cudaMalloc(&dummyZ, m_neuronCounts[0] * sizeof(double)));

        for (size_t layer = 0; layer < m_layerCount - 1; layer++)
        {
            int32_t rows = m_neuronCounts[layer + 1];
            int32_t cols = m_neuronCounts[layer];

            layerActivation<<<1, rows>>>(m_w[layer], as[layer], m_b[layer], dummyZ, as[layer + 1]);
        }

        std::vector<double> result(m_neuronCounts[m_layerCount - 1]);
        checkCudaErrors(cudaMemcpy(&result[0], as[as.size() - 1], result.size() * sizeof(double), cudaMemcpyDeviceToHost));

        for (auto a : as) checkCudaErrors(cudaFree(a));

        return static_cast<uint8_t>(std::max_element(result.begin(), result.begin() + 10) - result.begin());
    }

    void learn(std::vector<double*>& xs, std::vector<double*>& ys, uint32_t epochCount, double learningRate)
    {
        uint32_t subsetCount = static_cast<uint32_t>(std::ceil(xs.size() / (double)m_subsetSize));

        Matrix xSubset = { nullptr, 0, m_subsetSize, m_neuronCounts[0] };
        Matrix ySubset = { nullptr, 0, m_subsetSize, m_neuronCounts[m_neuronCounts.size() - 1] };
        checkCudaErrors(cudaMallocPitch(&xSubset.data, &xSubset.pitch, m_neuronCounts[0] * sizeof(double), m_subsetSize));
        checkCudaErrors(cudaMallocPitch(&ySubset.data, &ySubset.pitch, m_neuronCounts[m_neuronCounts.size() - 1] * sizeof(double), m_subsetSize));

        // Use stochastic gradient descent to learn the weights and biases
        auto rngX = std::default_random_engine{};
        auto rngY = rngX;
        for (uint32_t epoch = 0; epoch < epochCount; epoch++)
        {
            printf("Start epoch %d\n", epoch);

            // Rearrange the training data for each epoch
            std::shuffle(std::begin(xs), std::end(xs), rngX);
            std::shuffle(std::begin(ys), std::end(ys), rngY);

            uint32_t elementsLeft = static_cast<uint32_t>(xs.size());

            // Teach the network with a subset of the learning data
            for (uint32_t subset = 0; subset < subsetCount; subset++)
            {
                // Copy the subset into the GPU memory
                for (uint32_t i = 0; i < m_subsetSize; i++)
                {
                    checkCudaErrors(cudaMemcpy(((uint8_t*)xSubset.data) + i * xSubset.pitch,
                        xs[subset * m_subsetSize + i], m_neuronCounts[0] * sizeof(double), cudaMemcpyHostToDevice));
                    checkCudaErrors(cudaMemcpy(((uint8_t*)ySubset.data) + i * ySubset.pitch,
                        ys[subset * m_subsetSize + i], m_neuronCounts[m_neuronCounts.size() - 1] * sizeof(double), cudaMemcpyHostToDevice));
                }

                updateSubset(xSubset, ySubset, learningRate);
            }
        }
    }

    void updateSubset(const Matrix& xs, const Matrix& ys, double learningRate)
    {
        // Calculate partial derivatives
        backpropagate(xs, ys);

        // Update weights and biases
        for (size_t layer = 0; layer < m_layerCount - 1; layer++)
        {
            int32_t rows = m_neuronCounts[layer + 1];
            int32_t cols = m_neuronCounts[layer];
            dim3 gridSize(static_cast<uint32_t>(std::ceil((double)cols / (double)rows)));
            dim3 blockSize(rows, rows);
            gradientDescentStepW<<<gridSize, blockSize>>>(m_w[layer], m_wPartDerivs[layer], learningRate, m_subsetSize);
            gradientDescentStepB<<<1, rows>>>(m_b[layer], m_bPartDerivs[layer], learningRate, m_subsetSize);
        }
    }

    void backpropagate(const Matrix& x, const Matrix& y)
    {
        m_as[0] = x;

        for (size_t layer = 0; layer < m_layerCount - 1; layer++)
        {
            uint32_t rows = m_neuronCounts[layer + 1];
            uint32_t cols = m_neuronCounts[layer];

            Matrix a = m_as[layer];
            Matrix aNext = m_as[layer + 1];
            Matrix z = m_zs[layer];

            uint32_t gridCount = static_cast<uint32_t>(a.rows);
            layerActivation<<<gridCount, rows>>>(m_w[layer], a, m_b[layer], z, aNext);
        }
        uint32_t lastLayerSize = m_neuronCounts[m_neuronCounts.size() - 1];
        uint32_t gridCount = static_cast<uint32_t>(y.rows);
        calcOutputError<<<gridCount, lastLayerSize>>>(m_as[m_as.size() - 1], y, m_zs[m_zs.size() - 1], m_bPartDerivs[m_bPartDerivs.size() - 1]);
        
        dim3 blockSize(static_cast<uint32_t>(m_wPartDerivs[m_wPartDerivs.size() - 1].cols),
            static_cast<uint32_t>(m_wPartDerivs[m_wPartDerivs.size() - 1].rows));
        calcWPartiaDerivs<<<1, blockSize>>>(m_bPartDerivs[m_bPartDerivs.size() - 1], m_as[m_as.size() - 2], m_wPartDerivs[m_wPartDerivs.size() - 1]);

        for (size_t layer = 2; layer < m_layerCount; layer++)
        {
            Matrix z = m_zs[m_zs.size() - layer];

            dim3 blockSize(static_cast<uint32_t>(m_bPartDerivs[m_bPartDerivs.size() - layer].cols),
                static_cast<uint32_t>(m_bPartDerivs[m_bPartDerivs.size() - layer].rows));
            calcLayerError<<<1, blockSize>>>(m_w[m_w.size() - layer + 1], m_bPartDerivs[m_bPartDerivs.size() - layer + 1], z, m_bPartDerivs[m_bPartDerivs.size() - layer]);

            size_t rows = m_wPartDerivs[m_wPartDerivs.size() - layer].rows;
            size_t cols = m_wPartDerivs[m_wPartDerivs.size() - layer].cols;
            dim3 gridSize(static_cast<uint32_t>(std::ceil((double)cols / (double)rows)));
            blockSize = dim3(static_cast<uint32_t>(rows), static_cast<uint32_t>(rows));
            calcWPartiaDerivs<<<gridSize, blockSize>>>(m_bPartDerivs[m_bPartDerivs.size() - layer], m_as[m_as.size() - layer - 1], m_wPartDerivs[m_wPartDerivs.size() - layer]);
        }
    }

private:
    size_t m_layerCount;
    std::vector<uint32_t> m_neuronCounts;
    uint32_t m_subsetSize;

    std::vector<Matrix> m_w;
    std::vector<double*> m_b;

    std::vector<Matrix> m_wPartDerivs;
    std::vector<Matrix> m_bPartDerivs;

    std::vector<Matrix> m_as;
    std::vector<Matrix> m_zs;
};

int main()
{
    checkCudaErrors(cudaSetDevice(0));

    std::vector<double*> xs;
    std::vector<double*> ys;
    uint32_t elementSize = 0;
    loadData(L"MNIST/train-images.idx3-ubyte", xs, L"MNIST/train-labels.idx1-ubyte", ys, elementSize);

    std::vector<uint32_t> neuronCounts = { elementSize, 30, 30, 10 };
    NeuralNetwork network(neuronCounts, 25);

    network.learn(xs, ys, 10, 3.0);

    for (int i = 0; i < xs.size(); i++)
    {
        delete[] xs[i];
        delete[] ys[i];
    }

    printf("Finished learning. Now checking test images...\n");

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

    layerActivation<<<1, matSize>>>(deviceMatrix.ptr, deviceMatrix.pitch, matSize, vector, vector, result);

    for (int i = 0; i < matSize; i++)
        output("%f, ", result[i]);
    output("\n");*/

    return 0;
}
