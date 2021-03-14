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


#endif //TRYOPENCL_UTILITY_H
