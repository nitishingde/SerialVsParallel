#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include "../Utility.h"
#include <tuple>

void hello_world() {
    cl_int status = CL_SUCCESS;

    cl::Context context(CL_DEVICE_TYPE_GPU, nullptr, nullptr, nullptr, &status);
    verifyOpenCL_Status(status);

    auto devices = context.getInfo<CL_CONTEXT_DEVICES>(&status);
    verifyOpenCL_Status(status);
    auto device = devices.front();

    cl::Program program(context, readScript("HelloWorld.cl"), false, &status);
    verifyOpenCL_Status(status);
    status = program.build("-cl-std=CL1.2");
    verifyOpenCL_Status(status);

    char buf[16] = "000111222333444";
    cl::Buffer memBuf(context, CL_MEM_WRITE_ONLY, sizeof(buf), nullptr, &status);
    verifyOpenCL_Status(status);

    cl::Kernel kernel(program, "helloWorld", &status);
    verifyOpenCL_Status(status);
    verifyOpenCL_Status(kernel.setArg(0, memBuf));

    cl::CommandQueue queue(context, device, 0 &status);
    verifyOpenCL_Status(status);

    // try commenting the writeBuffer line below, observe how it behaves
    verifyOpenCL_Status(queue.enqueueWriteBuffer(memBuf, CL_TRUE, 0, sizeof(buf), buf));
    std::vector<cl::Event> blockers(1);
    verifyOpenCL_Status(queue.enqueueTask(kernel, nullptr, &blockers.front()));
    verifyOpenCL_Status(queue.enqueueReadBuffer(memBuf, CL_TRUE, 0, sizeof(buf), buf, &blockers));
    verifyOpenCL_Status(queue.finish());

    printf("%s\n\n", buf);
}

void visualise_execution_model() {
    cl_int status;

    cl::Context context(CL_DEVICE_TYPE_GPU, nullptr, nullptr, nullptr, &status);
    verifyOpenCL_Status(status);

    auto devices = context.getInfo<CL_CONTEXT_DEVICES>(&status);
    verifyOpenCL_Status(status);
    auto device = devices.front();

    size_t dim_x = 16, dim_y = 4;
    float matrix[dim_x][dim_y];
    memset(matrix, 0, sizeof(matrix));
    cl::Buffer matrixMem(context, CL_MEM_HOST_READ_ONLY|CL_MEM_WRITE_ONLY, sizeof(matrix), nullptr, &status);
    verifyOpenCL_Status(status);

    cl::Program program(context, readScript("VisualiseExecutionModel.cl"), false, &status);
    verifyOpenCL_Status(status);
    verifyOpenCL_Status(program.build("-cl-std=CL1.2"));

    cl::CommandQueue commandQueue(context, device, 0 &status);
    verifyOpenCL_Status(status);

    for(auto [kernelName, message]: {
        std::make_tuple("visualiseWorkItemsInGlobalSpace", "Visualise work-items in global space:"),
        std::make_tuple("visualiseWorkItemsInWorkGroupSpace", "Visualise work-items in work-group space:"),
        std::make_tuple("visualiseWorkGroups", "Visualise work groups:"),
    }) {
        cl::Kernel kernel(program, kernelName, &status);
        verifyOpenCL_Status(status);
        verifyOpenCL_Status(kernel.setArg(0, matrixMem));
        std::vector<cl::Event> blockers(1);
        verifyOpenCL_Status(commandQueue.enqueueNDRangeKernel(
            kernel,
            cl::NullRange,
            cl::NDRange(dim_x, dim_y/*, 1*/),
            cl::NDRange(2, 2/*, 1*/),
            nullptr,
            &blockers.front()
        ));
        verifyOpenCL_Status(commandQueue.enqueueReadBuffer(matrixMem, CL_TRUE, 0, sizeof(matrix), matrix, &blockers));
        verifyOpenCL_Status(commandQueue.finish());
        printf("%s\n", message);
        for(size_t i = 0; i < dim_x; ++i) {
            for(size_t j=0; j < dim_y; ++j) {
                printf(" %02.0f ", matrix[i][j]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

int main() {
    hello_world();
    visualise_execution_model();

    return 0;
}
