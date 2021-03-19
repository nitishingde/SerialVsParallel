#include "Pi.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <mpi/mpi.h>
#include <numeric>
#include <omp.h>
#include <regex>
#include <vector>
#include "../Utility.h"

#define CACHE_PADDING 8

double SerialPiStrategy::calculatePi(uint32_t steps) {
    const double delta = 1.0 / steps;
    double area = 0.0;

    for(uint32_t step = 0; step < steps; ++step) {
        double x = (step + 0.5) * delta;
        area += 4.0 / (1.0 + x*x);
    }

    return area * delta;
}

std::string SerialPiStrategy::toString() {
    return "Calculate Pi using serial code";
}

double OpenMP_PiStrategy::calculatePi(uint32_t steps) {
    const double delta = 1.0 / steps;
    std::vector<double> area(omp_get_max_threads()*2, 0.0);

    omp_set_num_threads(omp_get_max_threads());
    #pragma omp parallel firstprivate(steps, delta) shared(area) default(none)
    {
        auto totalThreads = omp_get_num_threads();
        auto threadID = omp_get_thread_num();
        for(uint32_t step = threadID; step < steps; step+=totalThreads) {
            double x = (step + 0.5) * delta;
            area[threadID] += 4.0 / (1.0 + x*x);
        }
    }

    return std::accumulate(area.begin(), area.end(), 0.0) * delta;
}

std::string OpenMP_PiStrategy::toString() {
    return "Calculate Pi using OpenMP, simplified";
}

double CacheFriendlyOpenMP_PiStrategy::calculatePi(uint32_t steps) {
    const double delta = 1.0 / steps;
    uint32_t maxThreadsPossible = omp_get_max_threads();
    double area[maxThreadsPossible][CACHE_PADDING];
    for(uint32_t i = 0; i < maxThreadsPossible; ++i) {
        area[i][0] = 0;
    }

    omp_set_num_threads(omp_get_max_threads());
    #pragma omp parallel firstprivate(steps, delta) shared(area) default(none)
    {
        auto totalThreads = omp_get_num_threads();
        auto threadID = omp_get_thread_num();
        for(uint32_t step = threadID; step < steps; step+=totalThreads) {
            double x = (step + 0.5) * delta;
            area[threadID][0] += 4.0 / (1.0 + x*x);
        }
    }

    double totalArea = 0.0;
    for(uint32_t i = 0; i < maxThreadsPossible; ++i) {
        totalArea += area[i][0];
    }

    return totalArea * delta;
}

std::string CacheFriendlyOpenMP_PiStrategy::toString() {
    return "Calculate Pi using OpenMP, using cache friendly options";
}

double AtomicBarrierOpenMP_PiStrategy::calculatePi(uint32_t steps) {
    const double delta = 1.0 / steps;
    double area = 0.0;

    omp_set_num_threads(omp_get_max_threads());
    #pragma omp parallel firstprivate(steps, delta) shared(area) default(none)
    {
        auto totalThreads = omp_get_num_threads();
        double area_t = 0.0;
        for(uint32_t step = omp_get_thread_num(); step < steps; step+=totalThreads) {
            double x = (step + 0.5) * delta;
            area_t += 4.0 / (1.0 + x*x);
        }
    #pragma omp atomic
        area += area_t;
    }

    return area * delta;
}

std::string AtomicBarrierOpenMP_PiStrategy::toString() {
    return "Calculate Pi using OpenMP, using atomic barrier";
}

double ReductionOpenMP_PiStrategy::calculatePi(uint32_t steps) {
    const double delta = 1.0 / steps;
    double area = 0.0;

    omp_set_num_threads(omp_get_max_threads());
    #pragma omp parallel for reduction(+:area) firstprivate(steps, delta) default(none)
    for(uint32_t step = 0; step < steps; ++step) {
        double x = (step + 0.5) * delta;
        area += 4.0 / (1.0 + x*x);
    }

    return area * delta;
}

std::string ReductionOpenMP_PiStrategy::toString() {
    return "Calculate Pi using OpenMP, using reduction technique";
}

void OpenCL_PiStrategy::init() {
    if(isInitialised) return;

    cl_int status = CL_SUCCESS;

    mContext = cl::Context(CL_DEVICE_TYPE_GPU, nullptr, nullptr, nullptr, &status);
    verifyOpenCL_Status(status);

    auto devices = mContext.getInfo<CL_CONTEXT_DEVICES>(&status);
    verifyOpenCL_Status(status);
    mDevice = devices.front();

    // 512 on my machine
    mWorkGroupSize = mDevice.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(&status);
    verifyOpenCL_Status(status);
#if not NDEBUG
    printf("[DEBUG] Work Group size: %zu\n", mWorkGroupSize);
#endif

    cl::Program program(
        mContext,
        std::regex_replace(readScript("Pi.cl"), std::regex("%workGroupSize%"), std::to_string(mWorkGroupSize)),
        false,
        &status
    );
    verifyOpenCL_Status(status);
    verifyOpenCL_Status(program.build("-cl-std=CL1.2"));
    mKernel = cl::Kernel(program, "calculatePi", &status);
    verifyOpenCL_Status(status);

    mCommandQueue = cl::CommandQueue(mContext, mDevice, 0, &status);
    verifyOpenCL_Status(status);

    isInitialised = true;
}

OpenCL_PiStrategy::OpenCL_PiStrategy() {
    init();
}

double OpenCL_PiStrategy::calculatePi(uint32_t steps) {
    cl_int status = CL_SUCCESS;
    const double delta = 1.0 / steps;

    size_t N = ceil((double)steps/(double)mWorkGroupSize);

    std::vector<float> workGroupArea(N, 0.f);
    cl::Buffer workGroupAreaBuffer(
        mContext,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        workGroupArea.size() * sizeof(decltype(workGroupArea)::value_type),
        workGroupArea.data(),
        &status
    );
    verifyOpenCL_Status(status);

    verifyOpenCL_Status(mKernel.setArg(0, workGroupAreaBuffer));
    verifyOpenCL_Status(mKernel.setArg(1, (uint32_t)N));
    verifyOpenCL_Status(mKernel.setArg(2, (uint32_t)steps));

    VECTOR_CLASS<cl::Event> blockers(1);
    verifyOpenCL_Status(mCommandQueue.enqueueNDRangeKernel(
        mKernel,
        cl::NullRange,
        cl::NDRange(N*mWorkGroupSize),
        cl::NDRange(mWorkGroupSize),
        nullptr,
        &blockers.front()
    ));
    verifyOpenCL_Status(mCommandQueue.enqueueReadBuffer(
        workGroupAreaBuffer,
        CL_TRUE,
        0,
        workGroupArea.size() * sizeof(decltype(workGroupArea)::value_type),
        workGroupArea.data(),
        &blockers
    ));

    return std::accumulate(workGroupArea.begin(), workGroupArea.end(), 0.0) * delta;
}

std::string OpenCL_PiStrategy::toString() {
    return "Calculate Pi using OpenCL";
}

double MPI_PiStrategy::calculatePi(uint32_t steps) {
    double pi = 0.0;
    double delta = 1.0 / steps;
    double area = 0.0;

    int32_t processId, noOfProcesses;
    MPI_Comm_size(MPI_COMM_WORLD, &noOfProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &processId);
    MPI_Bcast(&steps, 1, MPI_INT, 0, MPI_COMM_WORLD);

#if not NDEBUG
    if(isMpiRootPid()) {
        printf("[Debug] No of processes = %d\n", noOfProcesses);
    }
    printf("[Debug] Process Id = %d\n", processId);
#endif

    for (size_t step = processId; step < steps; step += noOfProcesses) {
        double x = (step + 0.5) * delta;
        area += 4.0 / (1.0 + x*x);
    }
    area *= delta;

    MPI_Reduce(&area, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    return pi;
}

std::string MPI_PiStrategy::toString() {
    return "Calculate Pi using MPI";
}

double HybridMpiOpenMP_PiStrategy::calculatePi(uint32_t steps) {
    double pi = 0.0;
    double delta = 1.0 / steps;
    double area = 0.0;

    int32_t processId, noOfProcesses;
    MPI_Comm_size(MPI_COMM_WORLD, &noOfProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &processId);
    MPI_Bcast(&steps, 1, MPI_INT, 0, MPI_COMM_WORLD);

#if not NDEBUG
    if(isMpiRootPid()) {
        printf("[Debug] No of processes = %d\n", noOfProcesses);
    }
    printf("[Debug] Process Id      = %d\n", processId);
#endif

    omp_set_num_threads(omp_get_max_threads());
    #pragma omp parallel for reduction(+:area) firstprivate(steps, delta, processId, noOfProcesses) default(none)
    for (size_t step = processId; step < steps; step += noOfProcesses) {
        double x = (step + 0.5) * delta;
        area += 4.0 / (1.0 + x*x);
    }
    area *= delta;

    MPI_Reduce(&area, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    return pi;
}

std::string HybridMpiOpenMP_PiStrategy::toString() {
    return "Calculate Pi using MPI and OpenMP";
}

PiBenchMarker::PiBenchMarker(std::unique_ptr<PiStrategy> pPiStrategy)
    : mpPiStrategy(std::move(pPiStrategy))
{}

void PiBenchMarker::setPiStrategy(std::unique_ptr<PiStrategy> pPiStrategy) {
    mpPiStrategy = std::move(pPiStrategy);
}

void PiBenchMarker::benchmarkCalculatePi(uint32_t iterations, uint32_t steps) const {
    std::vector<double> executionTime(iterations, 0.0);
    double pi = 0.0;

    for(uint32_t iteration = 0; iteration < executionTime.size(); ++iteration) {
#if NDEBUG
        if(isMpiRootPid()) {
            printf("\rIteration: %u/%u", iteration+1, iterations);
            fflush(stdout);
        }
#endif

        auto start = std::chrono::high_resolution_clock::now();
        pi += mpPiStrategy->calculatePi(steps);
        auto end = std::chrono::high_resolution_clock::now();
        executionTime[iteration] = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count()/1.e9;
#if not NDEBUG
        if(isMpiRootPid()) {
            printf("[Debug] Execution Time for iteration (%u, %u): %0.9gs\n", iteration+1, iterations, executionTime[iteration]);
        }
#endif

    }
    pi /= iterations;

    if(!isMpiRootPid()) return;
    printf("\r");
    printf("> Strategy        : %s\n", mpPiStrategy->toString().c_str());
    printf("> Iterations      : %u\n", iterations);
    printf("> Steps           : %u\n", steps);
    printf("Pi                : %0.17g\n", pi);
    printf("Error margin      : %0.17g %%\n", (std::abs(pi-M_PI)*100.0)/M_PI);
    printf("Avg Execution Time: %.9gs\n", std::accumulate(executionTime.begin(), executionTime.end(), 0.0)/executionTime.size());
    printf("Min Execution Time: %.9gs\n", *std::min_element(executionTime.begin(), executionTime.end()));
    printf("Max Execution Time: %.9gs\n", *std::max_element(executionTime.begin(), executionTime.end()));
    printf("\n");
}
