#include "Utility.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <mpi/mpi.h>
#include <numeric>
#include <unordered_map>

namespace svp {
    struct ProfileResult {
        std::string name;
        double elapsedTime;
    };

    class Profiler {
    private:
        Profiler() = default;
        std::vector<std::pair<std::string, std::vector<double>>> mExecutions;
        std::string mSession;

    private:
        void flush() const;
    public:
        ~Profiler();
        static Profiler *getInstance();
        void startSession(const char *pSession);
        void endSession();
        bool isSessionActive() const;
        void log(const ProfileResult &profileResult);
    };
}

static svp::Profiler *pInstance = nullptr;

svp::Profiler::~Profiler() {
    endSession();
}

svp::Profiler* svp::Profiler::getInstance() {
    if(pInstance == nullptr) {
        pInstance = new Profiler();
    }
    return pInstance;
}

void svp::Profiler::flush() const {
    printf("|----------------------------------|------------------------------------------------------------------|------------|--------------|--------------|--------------|\n");
    printf(
        "| "
        MAGENTA("Session") "                          | "
        MAGENTA("Scope") "                                                            | "
        MAGENTA("Iterations") " | "
        MAGENTA("Avg") "          | "
        MAGENTA("Min") "          | "
        MAGENTA("Max") "          |\n"
    );
    printf("|----------------------------------|------------------------------------------------------------------|------------|--------------|--------------|--------------|\n");

    for(const auto &it: mExecutions) {
        auto &executionTime = it.second;
        printf(
            "| " CYAN("%-32.32s") " | " YELLOW("%-64.64s") " | %-10zu | " BLUE("%.9f") "s | " GREEN("%.9f") "s | " RED("%.9f") "s |\n",
            mSession.c_str(),
            it.first.c_str(),
            executionTime.size(),
            std::accumulate(executionTime.begin(), executionTime.end(), 0.0)/executionTime.size(),
            *std::min_element(executionTime.begin(), executionTime.end()),
            *std::max_element(executionTime.begin(), executionTime.end())
        );
    }

    printf("|----------------------------------|------------------------------------------------------------------|------------|--------------|--------------|--------------|\n\n");
}

void svp::Profiler::startSession(const char *pSession) {
    if(!mSession.empty() and mSession != pSession) endSession();
    mSession = pSession;
}

void svp::Profiler::endSession() {
    flush();
    mSession.clear();
    mExecutions.clear();
}

bool svp::Profiler::isSessionActive() const {
    return !mSession.empty();
}

void svp::Profiler::log(const ProfileResult &profileResult) {
    if(!isSessionActive()) return;
    for(auto &execution: mExecutions) {
        if(execution.first == profileResult.name) {
            execution.second.push_back(profileResult.elapsedTime);
            return;
        }
    }
    // couldn't find it
    mExecutions.emplace_back(std::make_pair(profileResult.name, std::vector<double>{profileResult.elapsedTime}));
}

svp::BenchMarker::BenchMarker(const char *pName) {
    Profiler::getInstance()->startSession(pName);
}

svp::BenchMarker::~BenchMarker() {
    Profiler::getInstance()->endSession();
}

svp::Timer::Timer(const char *pName)
    : mName(pName)
    , mStartPoint(std::chrono::high_resolution_clock::now()) {
}

svp::Timer::~Timer() {
    auto endPoint = std::chrono::high_resolution_clock::now();
    Profiler::getInstance()->log(ProfileResult {
        .name = mName,
        .elapsedTime = double(std::chrono::duration_cast<std::chrono::nanoseconds>(endPoint - mStartPoint).count()) / 1.e9
    });
}

svp::OpenCL_Timer::OpenCL_Timer(const char *pName, const cl::Event &event) {
    cl_int status = CL_SUCCESS;
    verifyOpenCL_Status(event.wait());
    auto start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>(&status);
    verifyOpenCL_Status(status);
    auto end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>(&status);
    verifyOpenCL_Status(status);

    Profiler::getInstance()->log(ProfileResult {
        .name = pName,
        .elapsedTime = double(end-start) / 1.e9
    });
}

svp::OpenCL_Exception::OpenCL_Exception(cl_int error) {
    mErrorMessage = "[Error] Code, Msg = (" + std::to_string(error) + ", " + svp::getOpenCL_ErrorMessage(error) + ")";
}

const char* svp::OpenCL_Exception::what() const noexcept {
    return mErrorMessage.c_str();
}

void svp::OpenCL_Base::init() {
    if(mIsInitialised) return;

    cl_int status = CL_SUCCESS;

    mContext = cl::Context(CL_DEVICE_TYPE_GPU, nullptr, nullptr, nullptr, &status);
    svp::verifyOpenCL_Status(status);

    auto devices = mContext.getInfo<CL_CONTEXT_DEVICES>(&status);
    svp::verifyOpenCL_Status(status);
    mDevice = devices.front();

    // 512 on my machine
    mWorkGroupSize1d = mDevice.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(&status);
    svp::verifyOpenCL_Status(status);
    for(size_t i = 1, foundWorkGroupSize3d = false; ; i<<=1) {
        if(mWorkGroupSize1d < i * i) {
            mWorkGroupSize2d = i>>1;
            break;
        }
        if(!foundWorkGroupSize3d and mWorkGroupSize1d < i * i * i) {
            mWorkGroupSize3d = i>>1;
            foundWorkGroupSize3d = true;
        }
    }
#if NDEBUG
    cl_command_queue_properties properties = 0;
#else
    cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE;
#endif
    mCommandQueue = cl::CommandQueue(mContext, mDevice, properties, &status);
    svp::verifyOpenCL_Status(status);

    mIsInitialised = true;
}

void svp::OpenCL_Base::loadProgramSource(const char *pProgramSourceCode) {
    cl_int status = CL_SUCCESS;

    mProgram = cl::Program(
        mContext,
        pProgramSourceCode,
        false,
        &status
    );
    svp::verifyOpenCL_Status(status);
    svp::verifyOpenCL_Status(mProgram.build("-cl-std=CL1.2"));
}

void svp::OpenCL_Base::loadProgram(const char *pProgramFile) {
    loadProgramSource(readScript(pProgramFile).c_str());
}

std::string svp::readScript(const std::string &scriptFilePath) {
    std::ifstream scriptFile(scriptFilePath);
    return std::string(std::istreambuf_iterator<char>(scriptFile), (std::istreambuf_iterator<char>()));
}

std::string svp::getOpenCL_ErrorMessage(cl_int error) {
    switch(error) {
        // run-time and JIT compiler errors
        case 0: return "CL_SUCCESS";
        case -1: return "CL_DEVICE_NOT_FOUND";
        case -2: return "CL_DEVICE_NOT_AVAILABLE";
        case -3: return "CL_COMPILER_NOT_AVAILABLE";
        case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case -5: return "CL_OUT_OF_RESOURCES";
        case -6: return "CL_OUT_OF_HOST_MEMORY";
        case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case -8: return "CL_MEM_COPY_OVERLAP";
        case -9: return "CL_IMAGE_FORMAT_MISMATCH";
        case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case -11: return "CL_BUILD_PROGRAM_FAILURE";
        case -12: return "CL_MAP_FAILURE";
        case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case -15: return "CL_COMPILE_PROGRAM_FAILURE";
        case -16: return "CL_LINKER_NOT_AVAILABLE";
        case -17: return "CL_LINK_PROGRAM_FAILURE";
        case -18: return "CL_DEVICE_PARTITION_FAILED";
        case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

            // compile-time errors
        case -30: return "CL_INVALID_VALUE";
        case -31: return "CL_INVALID_DEVICE_TYPE";
        case -32: return "CL_INVALID_PLATFORM";
        case -33: return "CL_INVALID_DEVICE";
        case -34: return "CL_INVALID_CONTEXT";
        case -35: return "CL_INVALID_QUEUE_PROPERTIES";
        case -36: return "CL_INVALID_COMMAND_QUEUE";
        case -37: return "CL_INVALID_HOST_PTR";
        case -38: return "CL_INVALID_MEM_OBJECT";
        case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case -40: return "CL_INVALID_IMAGE_SIZE";
        case -41: return "CL_INVALID_SAMPLER";
        case -42: return "CL_INVALID_BINARY";
        case -43: return "CL_INVALID_BUILD_OPTIONS";
        case -44: return "CL_INVALID_PROGRAM";
        case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case -46: return "CL_INVALID_KERNEL_NAME";
        case -47: return "CL_INVALID_KERNEL_DEFINITION";
        case -48: return "CL_INVALID_KERNEL";
        case -49: return "CL_INVALID_ARG_INDEX";
        case -50: return "CL_INVALID_ARG_VALUE";
        case -51: return "CL_INVALID_ARG_SIZE";
        case -52: return "CL_INVALID_KERNEL_ARGS";
        case -53: return "CL_INVALID_WORK_DIMENSION";
        case -54: return "CL_INVALID_WORK_GROUP_SIZE";
        case -55: return "CL_INVALID_WORK_ITEM_SIZE";
        case -56: return "CL_INVALID_GLOBAL_OFFSET";
        case -57: return "CL_INVALID_EVENT_WAIT_LIST";
        case -58: return "CL_INVALID_EVENT";
        case -59: return "CL_INVALID_OPERATION";
        case -60: return "CL_INVALID_GL_OBJECT";
        case -61: return "CL_INVALID_BUFFER_SIZE";
        case -62: return "CL_INVALID_MIP_LEVEL";
        case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
        case -64: return "CL_INVALID_PROPERTY";
        case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
        case -66: return "CL_INVALID_COMPILER_OPTIONS";
        case -67: return "CL_INVALID_LINKER_OPTIONS";
        case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

            // extension errors
        case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
        case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
        case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
        case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
        case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
        case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
        default: return "Unknown OpenCL error";
    }
}

void svp::verifyOpenCL_Status(cl_int status) {
    if(status != CL_SUCCESS) {
        throw svp::OpenCL_Exception(status);
    }
}

void svp::printOpenCL_DeviceInfo(const cl::Device &device) {
    printf("Device Profile                               : %s\n", device.getInfo<CL_DEVICE_PROFILE>().c_str());
    printf("Device Max Compute Units                     : %u\n", device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>());

    auto maxWorkItemSizes = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
    printf("Device Max Work Item Sizes                   : ");
    for(auto sz: maxWorkItemSizes) {
        printf("%lu, ", sz);
    }
    printf("\b\b\n");

    printf("Device Max Work Group Size                   : %lu\n", device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>());
    printf("Device Max Work Item Dimensions              : %u\n", device.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>());

    printf("Device Local Memory Size                     : %lu\n", device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>());
    printf("Device Global Memory Size                    : %lu\n", device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>());
    printf("Device Local Memory Size/Compute Unit        : %u\n", device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE_PER_COMPUTE_UNIT_AMD>());

    printf("Device Extensions support                    : %s\n", device.getInfo<CL_DEVICE_EXTENSIONS>().c_str());
    printf("Device Max Clock Frequency                   : %d Hz\n", device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>());
    printf("\n");
}

void svp::printOpenCL_KernelWorkGroupInfo(const cl::Kernel &kernel, const cl::Device &device) {
    printf("Kernel Work Group Size                       : %lu\n", kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device));
    printf("Kernel Preferred Max Work Group Size Multiple: %lu\n", kernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device));
    auto kernelCompileWorkGroupSize = kernel.getWorkGroupInfo<CL_KERNEL_COMPILE_WORK_GROUP_SIZE>(device);
    printf("Kernel Compiled Work Group Size              : %lu, %lu, %lu\n", kernelCompileWorkGroupSize[0], kernelCompileWorkGroupSize[1], kernelCompileWorkGroupSize[2]);
    printf("Kernel Private Memory Size                   : %lu\n", kernel.getWorkGroupInfo<CL_KERNEL_PRIVATE_MEM_SIZE>(device));
    printf("Kernel Local Memory Size                     : %lu\n", kernel.getWorkGroupInfo<CL_KERNEL_LOCAL_MEM_SIZE>(device));
    printf("\n");
}

svp::MPI_GlobalLockGuard::MPI_GlobalLockGuard(int32_t argc, char **argv) {
#if not VALGRIND_ON
    int32_t flag = false;
    if(auto status = MPI_Initialized(&flag); status != MPI_SUCCESS or flag == false) {
        if(MPI_Init(&argc, &argv) == MPI_SUCCESS) {
            if(svp::isMpiRootPid()) printf("[MPI_GlobalLockGuard] " GREEN("MPI initialized\n"));
        }
    }
#endif
}

svp::MPI_GlobalLockGuard::~MPI_GlobalLockGuard() {
#if not VALGRIND_ON
    int32_t flag = false;
    if(auto status = MPI_Initialized(&flag); status == MPI_SUCCESS and flag) {
        if(MPI_Finalize() == MPI_SUCCESS) {
            if(svp::isMpiRootPid()) printf("[MPI_GlobalLockGuard] " RED("MPI exited\n"));
        }
    }
#endif
}

static int32_t sIsMpiRootPid = -1;

bool svp::isMpiRootPid() {
#if not VALGRIND_ON
    if(sIsMpiRootPid == -1) {
        int32_t flag = false;
        if(auto status = MPI_Initialized(&flag); status == MPI_SUCCESS and flag) {
            int32_t processId;
            MPI_Comm_rank(MPI_COMM_WORLD, &processId);
            sIsMpiRootPid = (processId == 0);
        }
    }
    return (sIsMpiRootPid == -1) or sIsMpiRootPid;
#else
    return true;
#endif
}
