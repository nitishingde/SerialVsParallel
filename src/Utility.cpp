#include "Utility.h"
#include <fstream>
#include <iostream>
#include <mpi/mpi.h>

svp::OpenCL_Exception::OpenCL_Exception(cl_int error) {
    mErrorMessage = "[Error] Code, Msg = (" + std::to_string(error) + ", " + svp::getOpenCL_ErrorMessage(error) + ")";
}

const char* svp::OpenCL_Exception::what() const noexcept {
    return mErrorMessage.c_str();
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
    int32_t flag = false;
    if(auto status = MPI_Initialized(&flag); status != MPI_SUCCESS or flag == false) {
        if(MPI_Init(&argc, &argv) == MPI_SUCCESS) {
            if(svp::isMpiRootPid()) printf("[MPI_GlobalLockGuard] MPI initialized\n\n");
        }
    }
}

svp::MPI_GlobalLockGuard::~MPI_GlobalLockGuard() {
    int32_t flag = false;
    if(auto status = MPI_Initialized(&flag); status == MPI_SUCCESS and flag) {
        if(MPI_Finalize() == MPI_SUCCESS) {
            if(svp::isMpiRootPid()) printf("[MPI_GlobalLockGuard] MPI exited\n\n");
        }
    }
}

static int32_t sIsMpiRootPid = -1;

bool svp::isMpiRootPid() {
    if(sIsMpiRootPid == -1) {
        int32_t flag = false;
        if(auto status = MPI_Initialized(&flag); status == MPI_SUCCESS and flag) {
            int32_t processId;
            MPI_Comm_rank(MPI_COMM_WORLD, &processId);
            sIsMpiRootPid = (processId == 0);
        }
    }
    return (sIsMpiRootPid == -1) or sIsMpiRootPid;
}
