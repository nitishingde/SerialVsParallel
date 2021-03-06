#include "../Utility.h"
#include <tuple>

void hello_world() {
    cl_int status = CL_SUCCESS;

    cl::Context context(CL_DEVICE_TYPE_GPU, nullptr, nullptr, nullptr, &status);
    svp::verifyOpenCL_Status(status);

    auto devices = context.getInfo<CL_CONTEXT_DEVICES>(&status);
    svp::verifyOpenCL_Status(status);
    auto device = devices.front();
    svp::printOpenCL_DeviceInfo(device);

    cl::Program program(context, svp::readScript("resources/HelloWorld.cl"), false, &status);
    svp::verifyOpenCL_Status(status);
    status = program.build("-cl-std=CL1.2");
    svp::verifyOpenCL_Status(status);

    char buf[16] = "000111222333444";
    cl::Buffer memBuf(context, CL_MEM_WRITE_ONLY, sizeof(buf), nullptr, &status);
    svp::verifyOpenCL_Status(status);

    cl::Kernel kernel(program, "helloWorld", &status);
    svp::verifyOpenCL_Status(status);
    svp::verifyOpenCL_Status(kernel.setArg(0, memBuf));
    svp::printOpenCL_KernelWorkGroupInfo(kernel, device);

    cl::CommandQueue queue(context, device, 0, &status);
    svp::verifyOpenCL_Status(status);

    // try commenting the writeBuffer line below, observe how it behaves
    svp::verifyOpenCL_Status(queue.enqueueWriteBuffer(memBuf, CL_TRUE, 0, sizeof(buf), buf));
    svp::verifyOpenCL_Status(queue.enqueueTask(kernel, nullptr, nullptr));
    svp::verifyOpenCL_Status(queue.enqueueReadBuffer(memBuf, CL_TRUE, 0, sizeof(buf), buf, nullptr));

    printf("%s\n\n", buf);
}

void visualise_execution_model() {
    cl_int status;
    auto printMatrix = [](float *matrix, size_t dim_x, size_t dim_y) {
        for(size_t i = 0; i < dim_x; ++i) {
            for(size_t j=0; j < dim_y; ++j) {
                printf(" %02.0f ", matrix[i*dim_y+j]);
            }
            printf("\n");
        }
    };

    cl::Context context(CL_DEVICE_TYPE_GPU, nullptr, nullptr, nullptr, &status);
    svp::verifyOpenCL_Status(status);

    auto devices = context.getInfo<CL_CONTEXT_DEVICES>(&status);
    svp::verifyOpenCL_Status(status);
    auto device = devices.front();

    size_t globalDimX = 9, globalDimY = 8;
    size_t localDimX = 3, localDimY = 2;
    float matrix[globalDimX][globalDimY];
    memset(matrix, 0, sizeof(matrix));
    cl::Buffer matrixMem(context, CL_MEM_HOST_READ_ONLY|CL_MEM_WRITE_ONLY, sizeof(matrix), nullptr, &status);
    svp::verifyOpenCL_Status(status);

    cl::Program program(context, svp::readScript("resources/VisualiseExecutionModel.cl"), false, &status);
    svp::verifyOpenCL_Status(status);
    svp::verifyOpenCL_Status(program.build("-cl-std=CL1.2"));

    cl::CommandQueue commandQueue(context, device, 0 &status);
    svp::verifyOpenCL_Status(status);

    for(auto [kernelName, message]: {
        std::make_tuple("visualiseWorkItemsInGlobalSpace", "Visualise work-items in global space of a 2-dim NDRange"),
        std::make_tuple("visualiseWorkGroups", "Visualise work-groups of a 2-dim NDRange"),
        std::make_tuple("visualiseWorkItemsInWorkGroupSpace", "Visualise work-items in work-group space of a 2-dim NDRange"),
    }) {
        cl::Kernel kernel(program, kernelName, &status);
        svp::verifyOpenCL_Status(status);
        svp::verifyOpenCL_Status(kernel.setArg(0, matrixMem));
        svp::verifyOpenCL_Status(commandQueue.enqueueNDRangeKernel(
            kernel,
            cl::NullRange,
            cl::NDRange(globalDimX, globalDimY/*, 1*/),
            cl::NDRange(localDimX, localDimY/*, 1*/),
            nullptr,
            nullptr
        ));
        svp::verifyOpenCL_Status(commandQueue.enqueueReadBuffer(matrixMem, CL_TRUE, 0, sizeof(matrix), matrix, nullptr));
        printf("%s\n", message);
        printf("Global dimensions: (%zu, %zu, 1)\n", globalDimX, globalDimY);
        printf("Local dimensions : (%zu, %zu, 1)\n", localDimX, localDimY);
        printMatrix((float *)matrix, globalDimX, globalDimY);
        printf("\n");
    }

    // Though the matrix is q 2d array, we don't necessarily have to create a 2d NDRange kernel to handle it
    for(auto [kernelName, message]: {
        std::make_tuple("visualiseWorkItemsInGlobalSpace", "Visualise work-items in global space of a 1-dim NDRange"),
        std::make_tuple("visualiseWorkGroups", "Visualise work-groups of a 1-dim NDRange"),
        std::make_tuple("visualiseWorkItemsInWorkGroupSpace", "Visualise work-items in work-group space of a 1-dim NDRange"),
    }) {
        cl::Kernel kernel(program, kernelName, &status);
        svp::verifyOpenCL_Status(status);
        svp::verifyOpenCL_Status(kernel.setArg(0, matrixMem));
        svp::verifyOpenCL_Status(commandQueue.enqueueNDRangeKernel(
            kernel,
            cl::NullRange,
            cl::NDRange(globalDimX * globalDimY),
            cl::NDRange(localDimX * localDimX),
            nullptr,
            nullptr
        ));
        svp::verifyOpenCL_Status(commandQueue.enqueueReadBuffer(matrixMem, CL_TRUE, 0, sizeof(matrix), matrix, nullptr));
        printf("%s\n", message);
        printf("Global dimensions: (%zu, 1, 1)\n", globalDimX * globalDimY);
        printf("Local dimensions : (%zu, 1, 1)\n", localDimX * localDimY);
        printMatrix((float *)matrix, globalDimX, globalDimY);
        printf("\n");
    }

    for(auto [kernelName, message]: {
        std::make_tuple("visualiseSequenceOfWorkItemsInGlobalSpace", "Visualise sequence of work-items in global space of a 2-dim NDRange"),
        std::make_tuple("visualiseSequenceOfWorkGroups", "Visualise sequence of work-groups of a 2-dim NDRange"),
        std::make_tuple("visualiseSequenceOfWorkItemsInWorkGroupSpace", "Visualise sequence of work-items in work-group space of a 2-dim NDRange"),
    }) {
        cl::Kernel kernel(program, kernelName, &status);
        svp::verifyOpenCL_Status(status);
        printf("%s\n", message);
        printf("[");
        commandQueue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(globalDimX, globalDimY), cl::NDRange(localDimX, localDimY));
        commandQueue.finish();
        printf("\b\b]\n\n");
    }
}

int main() {
    hello_world();
    visualise_execution_model();

    return 0;
}
