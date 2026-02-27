/***************
 * @file: kompute_dispatch.inl
 * @brief: Kompute GPU着色器调度的实现
 ***************/

#include <fstream>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <cstring>

namespace yt::kompute {

// ======================== SPIR-V加载 ========================

namespace {
    inline std::string& shaderDirRef() {
        static std::string dir = "./spirv";
        return dir;
    }

    inline std::unordered_map<std::string, std::vector<uint32_t>>& spvCache() {
        static std::unordered_map<std::string, std::vector<uint32_t>> cache;
        return cache;
    }

    inline std::mutex& spvMutex() {
        static std::mutex mtx;
        return mtx;
    }
} // anonymous namespace

inline void setShaderDir(const std::string& dir) {
    shaderDirRef() = dir;
}

inline const std::string& getShaderDir() {
    return shaderDirRef();
}

inline std::vector<uint32_t> loadSPIRV(const std::string& spvFilename) {
    std::lock_guard<std::mutex> lock(spvMutex());
    auto& cache = spvCache();
    auto it = cache.find(spvFilename);
    if (it != cache.end()) return it->second;

    std::string path = getShaderDir() + "/" + spvFilename;
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("[kompute_dispatch] Cannot open SPIR-V file: " + path);
    }
    size_t fileSize = static_cast<size_t>(file.tellg());
    if (fileSize == 0 || fileSize % sizeof(uint32_t) != 0) {
        throw std::runtime_error("[kompute_dispatch] Invalid SPIR-V file: " + path);
    }
    std::vector<uint32_t> spirv(fileSize / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(spirv.data()), static_cast<std::streamsize>(fileSize));
    cache[spvFilename] = spirv;
    return spirv;
}

// ======================== Kompute GPU调度实现 ========================

#if YT_USE_KOMPUTE

inline kp::Manager& getSharedManager() {
    static kp::Manager mgr;
    return mgr;
}

inline void dispatchBinaryFloat(const float* inputA, const float* inputB, float* outputC,
                                size_t count, const std::string& spvFile) {
    if (count == 0) return;
    auto& mgr = getSharedManager();
    auto spirv = loadSPIRV(spvFile);

    // 创建GPU张量
    std::vector<float> vecA(inputA, inputA + count);
    std::vector<float> vecB(inputB, inputB + count);
    std::vector<float> vecC(count, 0.0f);

    auto tensorA = mgr.tensorT<float>(vecA);
    auto tensorB = mgr.tensorT<float>(vecB);
    auto tensorC = mgr.tensorT<float>(vecC);

    std::vector<std::shared_ptr<kp::Memory>> params = {tensorA, tensorB, tensorC};

    auto algo = mgr.algorithm(params, spirv, kp::Workgroup({static_cast<uint32_t>((count + 63) / 64), 1, 1}));

    mgr.sequence()
        ->record<kp::OpSyncDevice>(params)
        ->record<kp::OpAlgoDispatch>(algo)
        ->record<kp::OpSyncLocal>({tensorC})
        ->eval();

    std::memcpy(outputC, tensorC->vector().data(), count * sizeof(float));
}

inline void dispatchBinaryInt(const int32_t* inputA, const int32_t* inputB, int32_t* outputC,
                              size_t count, const std::string& spvFile) {
    if (count == 0) return;
    auto& mgr = getSharedManager();
    auto spirv = loadSPIRV(spvFile);

    std::vector<int32_t> vecA(inputA, inputA + count);
    std::vector<int32_t> vecB(inputB, inputB + count);
    std::vector<int32_t> vecC(count, 0);

    auto tensorA = mgr.tensorT<int32_t>(vecA);
    auto tensorB = mgr.tensorT<int32_t>(vecB);
    auto tensorC = mgr.tensorT<int32_t>(vecC);

    std::vector<std::shared_ptr<kp::Memory>> params = {tensorA, tensorB, tensorC};

    auto algo = mgr.algorithm(params, spirv, kp::Workgroup({static_cast<uint32_t>((count + 63) / 64), 1, 1}));

    mgr.sequence()
        ->record<kp::OpSyncDevice>(params)
        ->record<kp::OpAlgoDispatch>(algo)
        ->record<kp::OpSyncLocal>({tensorC})
        ->eval();

    std::memcpy(outputC, tensorC->vector().data(), count * sizeof(int32_t));
}

inline void dispatchMatmul(const float* A, const float* B, float* C,
                           uint32_t M, uint32_t N, uint32_t K) {
    if (M == 0 || N == 0 || K == 0) return;
    auto& mgr = getSharedManager();
    auto spirv = loadSPIRV("matmul.spv");

    std::vector<float> vecA(A, A + M * K);
    std::vector<float> vecB(B, B + K * N);
    std::vector<float> vecC(M * N, 0.0f);

    auto tensorA = mgr.tensorT<float>(vecA);
    auto tensorB = mgr.tensorT<float>(vecB);
    auto tensorC = mgr.tensorT<float>(vecC);

    // push constants: M, N, K
    std::vector<float> pushConsts(3);
    uint32_t dims[3] = {M, N, K};
    std::memcpy(pushConsts.data(), dims, sizeof(dims));

    std::vector<std::shared_ptr<kp::Memory>> params = {tensorA, tensorB, tensorC};

    auto algo = mgr.algorithm(
        params, spirv,
        kp::Workgroup({(N + 7) / 8, (M + 7) / 8, 1}),
        {},  // specialization constants
        pushConsts
    );

    mgr.sequence()
        ->record<kp::OpSyncDevice>(params)
        ->record<kp::OpAlgoDispatch>(algo)
        ->record<kp::OpSyncLocal>({tensorC})
        ->eval();

    std::memcpy(C, tensorC->vector().data(), M * N * sizeof(float));
}

inline void dispatchReductionSum(const float* input, float* output,
                                 uint32_t axisLength, uint32_t outputSize) {
    if (outputSize == 0) return;
    auto& mgr = getSharedManager();
    auto spirv = loadSPIRV("reduction_sum.spv");

    std::vector<float> vecIn(input, input + static_cast<size_t>(axisLength) * outputSize);
    std::vector<float> vecOut(outputSize, 0.0f);

    auto tensorIn = mgr.tensorT<float>(vecIn);
    auto tensorOut = mgr.tensorT<float>(vecOut);

    // push constants: axisLength, outputSize
    std::vector<float> pushConsts(2);
    uint32_t dims[2] = {axisLength, outputSize};
    std::memcpy(pushConsts.data(), dims, sizeof(dims));

    std::vector<std::shared_ptr<kp::Memory>> params = {tensorIn, tensorOut};

    auto algo = mgr.algorithm(
        params, spirv,
        kp::Workgroup({(outputSize + 63) / 64, 1, 1}),
        {},
        pushConsts
    );

    mgr.sequence()
        ->record<kp::OpSyncDevice>(params)
        ->record<kp::OpAlgoDispatch>(algo)
        ->record<kp::OpSyncLocal>({tensorOut})
        ->eval();

    std::memcpy(output, tensorOut->vector().data(), outputSize * sizeof(float));
}

inline void dispatchReductionMax(const float* input, float* maxValues, uint32_t* argmax,
                                 uint32_t axisLength, uint32_t outputSize) {
    if (outputSize == 0) return;
    auto& mgr = getSharedManager();
    auto spirv = loadSPIRV("reduction_max.spv");

    std::vector<float> vecIn(input, input + static_cast<size_t>(axisLength) * outputSize);
    std::vector<float> vecMax(outputSize, 0.0f);
    std::vector<uint32_t> vecArg(outputSize, 0u);

    auto tensorIn = mgr.tensorT<float>(vecIn);
    auto tensorMax = mgr.tensorT<float>(vecMax);
    auto tensorArg = mgr.tensorT<uint32_t>(vecArg);

    std::vector<float> pushConsts(2);
    uint32_t dims[2] = {axisLength, outputSize};
    std::memcpy(pushConsts.data(), dims, sizeof(dims));

    std::vector<std::shared_ptr<kp::Memory>> paramsAll = {tensorIn, tensorMax, tensorArg};

    auto algo = mgr.algorithm(
        paramsAll, spirv,
        kp::Workgroup({(outputSize + 63) / 64, 1, 1}),
        {},
        pushConsts
    );

    std::vector<std::shared_ptr<kp::Memory>> syncOut = {tensorMax, tensorArg};

    mgr.sequence()
        ->record<kp::OpSyncDevice>(paramsAll)
        ->record<kp::OpAlgoDispatch>(algo)
        ->record<kp::OpSyncLocal>(syncOut)
        ->eval();

    std::memcpy(maxValues, tensorMax->vector().data(), outputSize * sizeof(float));
    std::memcpy(argmax, tensorArg->vector().data(), outputSize * sizeof(uint32_t));
}

inline void dispatchCmpFloat(const float* inputA, const float* inputB, uint32_t* outputC,
                             size_t count, const std::string& spvFile) {
    if (count == 0) return;
    auto& mgr = getSharedManager();
    auto spirv = loadSPIRV(spvFile);

    std::vector<float> vecA(inputA, inputA + count);
    std::vector<float> vecB(inputB, inputB + count);
    std::vector<uint32_t> vecC(count, 0u);

    auto tensorA = mgr.tensorT<float>(vecA);
    auto tensorB = mgr.tensorT<float>(vecB);
    auto tensorC = mgr.tensorT<uint32_t>(vecC);

    std::vector<std::shared_ptr<kp::Memory>> params = {tensorA, tensorB, tensorC};

    auto algo = mgr.algorithm(params, spirv, kp::Workgroup({static_cast<uint32_t>((count + 63) / 64), 1, 1}));

    mgr.sequence()
        ->record<kp::OpSyncDevice>(params)
        ->record<kp::OpAlgoDispatch>(algo)
        ->record<kp::OpSyncLocal>({tensorC})
        ->eval();

    std::memcpy(outputC, tensorC->vector().data(), count * sizeof(uint32_t));
}

#endif // YT_USE_KOMPUTE

} // namespace yt::kompute
