#ifndef TRYOPENCL_UTILITY_H
#define TRYOPENCL_UTILITY_H


#include <CL/cl.hpp>

class OpenCL_Exception: public std::exception {
private:
    std::string mErrorMessage;
public:
    explicit OpenCL_Exception(cl_int error);
    explicit OpenCL_Exception(std::string errorMessage);
    [[nodiscard]] const char *what() const _GLIBCXX_TXN_SAFE_DYN _GLIBCXX_USE_NOEXCEPT override;
};

std::string readScript(const std::string &scriptFilePath);

std::string getOpenCL_ErrorMessage(cl_int error);

void verifyOpenCL_Status(cl_int status);

void printOpenCL_DeviceInfo(const cl::Device &device);

void printOpenCL_KernelWorkGroupInfo(const cl::Kernel &kernel, const cl::Device &device);

/**
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

#endif //TRYOPENCL_UTILITY_H
