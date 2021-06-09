#ifndef TRYOPENCL_UTILITY_H
#define TRYOPENCL_UTILITY_H


#include <chrono>
#include <CL/cl.hpp>
#include <unordered_map>

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"

#define RED(x) ANSI_COLOR_RED x ANSI_COLOR_RESET
#define GREEN(x) ANSI_COLOR_GREEN x ANSI_COLOR_RESET
#define YELLOW(x) ANSI_COLOR_YELLOW x ANSI_COLOR_RESET
#define BLUE(x) ANSI_COLOR_BLUE x ANSI_COLOR_RESET
#define MAGENTA(x) ANSI_COLOR_MAGENTA x ANSI_COLOR_RESET
#define CYAN(x) ANSI_COLOR_CYAN x ANSI_COLOR_RESET

namespace svp {
    /**
     * RAII
     */
    class BenchMarker {
    public:
        explicit BenchMarker(const char *pName);
        ~BenchMarker();
    };

    /**
     * RAII
     */
    class Timer {
    private:
        std::string mName;
        std::chrono::high_resolution_clock::time_point mStartPoint;
    public:
        explicit Timer(const char *pName);
        ~Timer();
    };

    class OpenCL_Timer {
    public:
        explicit OpenCL_Timer(const char *pName, const cl::Event &event);
    };

    class OpenCL_Exception : public std::exception {
    private:
        std::string mErrorMessage;
    public:
        explicit OpenCL_Exception(cl_int error);
        [[nodiscard]] const char *what() const _GLIBCXX_TXN_SAFE_DYN _GLIBCXX_USE_NOEXCEPT override;
    };

    class OpenCL_Base {
    protected:
        bool mIsInitialised = false;
        size_t mWorkGroupSize1d;
        size_t mWorkGroupSize2d;
        size_t mWorkGroupSize3d;
        cl::Context mContext;
        cl::Device mDevice;
        cl::Program mProgram;
        cl::CommandQueue mCommandQueue;

    protected:
        virtual void init();
        void loadProgramSource(const char *pProgramSourceCode);
        void loadProgram(const char *pProgramFile);
    };

    std::string readScript(const std::string &scriptFilePath);

    std::string getOpenCL_ErrorMessage(cl_int error);

    void verifyOpenCL_Status(cl_int status);

    void printOpenCL_DeviceInfo(const cl::Device &device);

    void printOpenCL_KernelWorkGroupInfo(const cl::Kernel &kernel, const cl::Device &device);

/**
 * RAII
 * Works similar to std::lock_guard
 * Create this object only once.
 * It's lifecycle usually should be the entirety of the process execution.
 * Preferably make a MPI_GlobalLockGuard stack object at the beginning of the main function
 */
    class MPI_GlobalLockGuard {
    public:
        MPI_GlobalLockGuard(int32_t argc, char **argv);
        ~MPI_GlobalLockGuard();
    };

    bool isMpiRootPid();
}

#define printf(...) if(svp::isMpiRootPid()) printf(__VA_ARGS__)
#if NDEBUG
    #define dprintf(...) printf(__VA_ARGS__)
#else
    #define dprintf(...) fprintf(stdout, __VA_ARGS__)
#endif

#define SVP_PROFILING 1
#if SVP_PROFILING
    #define TOKEN_PASTE_(x, y) x##y
    #define CONCAT_(x, y) TOKEN_PASTE_(x, y)

    #define SVP_START_BENCHMARKING_SESSION(name) svp::BenchMarker CONCAT_(benchMarker, __LINE__)(name)
    #define SVP_START_BENCHMARKING_ITERATIONS(iterations) \
    for(uint32_t svp_iteration = 0, svp_iterations = iterations; svp_iteration < svp_iterations; ++svp_iteration)

    #define SVP_PRINT_BENCHMARKING_ITERATION()                      \
    printf("Iteration : %u/%u\r", svp_iteration+1, svp_iterations); \
    fflush(stdout)

    #define SVP_PROFILE_SCOPE(name) svp::Timer CONCAT_(timer, __LINE__)(name)
    #define SVP_PROFILE_FUNC() SVP_PROFILE_SCOPE(__PRETTY_FUNCTION__)
    #define SVP_PROFILE_OPENCL(event) OpenCL_Timer(#event, event)
#else
    #define SVP_START_BENCHMARKING_SESSION(name)
    #define SVP_START_BENCHMARKING_ITERATIONS(iterations)
    #define SVP_PRINT_BENCHMARKING_ITERATION()
    #define SVP_PROFILE_SCOPE(name)
    #define SVP_PROFILE_FUNC()
    #define SVP_PROFILE_OPENCL(event)
#endif

#define VALGRIND_ON 0

#endif //TRYOPENCL_UTILITY_H
