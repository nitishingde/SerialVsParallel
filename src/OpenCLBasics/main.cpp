#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include "../Utility.h"

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
    kernel.setArg(0, memBuf);

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

int main() {
    hello_world();

    return 0;
}
