#include "Pi.h"
#include <chrono>
#include <cstdio>
#include <cmath>
#include <mpi/mpi.h>
#include <numeric>
#include <omp.h>
#include <regex>
#include <vector>

#define CACHE_PADDING 8

double svp::SerialPiStrategy::calculatePi(uint32_t steps) {
    SVP_PROFILE_SCOPE(toString().c_str());

    const double delta = 1.0 / steps;
    double area = 0.0;

    for(uint32_t step = 0; step < steps; ++step) {
        double x = (step + 0.5) * delta;
        area += 4.0 / (1.0 + x*x);
    }

    return area * delta;
}

std::string svp::SerialPiStrategy::toString() {
    return "Calculate Pi using serial code";
}

double svp::OpenMP_PiStrategy::calculatePi(uint32_t steps) {
    SVP_PROFILE_SCOPE(toString().c_str());

    const double delta = 1.0 / steps;
    std::vector<double> area(omp_get_max_threads(), 0.0);

    omp_set_num_threads(omp_get_max_threads());
    #pragma omp parallel default(none) firstprivate(steps, delta) shared(area)
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

std::string svp::OpenMP_PiStrategy::toString() {
    return "Calculate Pi using OpenMP, simplified";
}

double svp::CacheFriendlyOpenMP_PiStrategy::calculatePi(uint32_t steps) {
    SVP_PROFILE_SCOPE(toString().c_str());

    const double delta = 1.0 / steps;
    uint32_t maxThreadsPossible = omp_get_max_threads();
    double area[maxThreadsPossible][CACHE_PADDING];
    for(uint32_t i = 0; i < maxThreadsPossible; ++i) {
        area[i][0] = 0;
    }

    omp_set_num_threads(omp_get_max_threads());
    #pragma omp parallel default(none) firstprivate(steps, delta) shared(area)
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

std::string svp::CacheFriendlyOpenMP_PiStrategy::toString() {
    return "Calculate Pi using OpenMP, using cache friendly options";
}

double svp::AtomicBarrierOpenMP_PiStrategy::calculatePi(uint32_t steps) {
    SVP_PROFILE_SCOPE(toString().c_str());

    const double delta = 1.0 / steps;
    double area = 0.0;

    omp_set_num_threads(omp_get_max_threads());
    #pragma omp parallel default(none) firstprivate(steps, delta) shared(area)
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

std::string svp::AtomicBarrierOpenMP_PiStrategy::toString() {
    return "Calculate Pi using OpenMP, using atomic barrier";
}

double svp::ReductionOpenMP_PiStrategy::calculatePi(uint32_t steps) {
    SVP_PROFILE_SCOPE(toString().c_str());

    const double delta = 1.0 / steps;
    double area = 0.0;

    omp_set_num_threads(omp_get_max_threads());
    #pragma omp parallel for reduction(+:area) default(none) firstprivate(steps, delta)
    for(uint32_t step = 0; step < steps; ++step) {
        double x = (step + 0.5) * delta;
        area += 4.0 / (1.0 + x*x);
    }

    return area * delta;
}

std::string svp::ReductionOpenMP_PiStrategy::toString() {
    return "Calculate Pi using OpenMP, using reduction technique";
}

void svp::OpenCL_PiStrategy::init() {
    OpenCL_Base::init();
    cl_int status = CL_SUCCESS;
    OpenCL_Base::loadProgramSource(
        std::regex_replace(svp::readScript("resources/Pi.cl"), std::regex("%workGroupSize%"), std::to_string(mWorkGroupSize1d)).c_str()
    );
    mKernel = cl::Kernel(mProgram, "calculatePi", &status);
    svp::verifyOpenCL_Status(status);
}

svp::OpenCL_PiStrategy::OpenCL_PiStrategy() {
    init();
}

double svp::OpenCL_PiStrategy::calculatePi(uint32_t steps) {
    SVP_PROFILE_SCOPE(toString().c_str());

    cl_int status = CL_SUCCESS;
    const double delta = 1.0 / steps;

    size_t N = ceil((double)steps/(double)mWorkGroupSize1d);

    std::vector<float> workGroupArea(N, 0.f);
    cl::Buffer workGroupAreaBuffer(
        mContext,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        workGroupArea.size() * sizeof(decltype(workGroupArea)::value_type),
        workGroupArea.data(),
        &status
    );
    svp::verifyOpenCL_Status(status);

    svp::verifyOpenCL_Status(mKernel.setArg(0, workGroupAreaBuffer));
    svp::verifyOpenCL_Status(mKernel.setArg(1, (uint32_t)N));
    svp::verifyOpenCL_Status(mKernel.setArg(2, (uint32_t)steps));

    svp::verifyOpenCL_Status(mCommandQueue.enqueueNDRangeKernel(
        mKernel,
        cl::NullRange,
        cl::NDRange(N*mWorkGroupSize1d),
        cl::NDRange(mWorkGroupSize1d),
        nullptr,
        nullptr
    ));
    svp::verifyOpenCL_Status(mCommandQueue.enqueueReadBuffer(
        workGroupAreaBuffer,
        CL_TRUE,
        0,
        workGroupArea.size() * sizeof(decltype(workGroupArea)::value_type),
        workGroupArea.data(),
        nullptr
    ));

    return std::accumulate(workGroupArea.begin(), workGroupArea.end(), 0.0) * delta;
}

std::string svp::OpenCL_PiStrategy::toString() {
    return "Calculate Pi using OpenCL";
}

double svp::MPI_PiStrategy::calculatePi(uint32_t steps) {
    SVP_PROFILE_SCOPE(toString().c_str());

    double pi = 0.0;
    double delta = 1.0 / steps;
    double area = 0.0;

    int32_t processId, noOfProcesses;
    MPI_Comm_size(MPI_COMM_WORLD, &noOfProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &processId);

#if not NDEBUG
    printf("[Debug] No of processes = %d\n", noOfProcesses);
    dprintf("[Debug] Process Id = %d\n", processId);
#endif

    for (size_t step = processId; step < steps; step += noOfProcesses) {
        double x = (step + 0.5) * delta;
        area += 4.0 / (1.0 + x*x);
    }
    area *= delta;

    MPI_Reduce(&area, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    return pi;
}

std::string svp::MPI_PiStrategy::toString() {
    return "Calculate Pi using MPI";
}

double svp::HybridMpiOpenMP_PiStrategy::calculatePi(uint32_t steps) {
    SVP_PROFILE_SCOPE(toString().c_str());

    double pi = 0.0;
    double delta = 1.0 / steps;
    double area = 0.0;

    int32_t processId, noOfProcesses;
    MPI_Comm_size(MPI_COMM_WORLD, &noOfProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &processId);

#if not NDEBUG
    printf("[Debug] No of processes = %d\n", noOfProcesses);
    dprintf("[Debug] Process Id      = %d\n", processId);
#endif

    omp_set_num_threads(omp_get_max_threads());
    #pragma omp parallel for reduction(+:area) default(none) firstprivate(steps, delta, processId, noOfProcesses)
    for (size_t step = processId; step < steps; step += noOfProcesses) {
        double x = (step + 0.5) * delta;
        area += 4.0 / (1.0 + x*x);
    }
    area *= delta;

    MPI_Reduce(&area, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    return pi;
}

std::string svp::HybridMpiOpenMP_PiStrategy::toString() {
    return "Calculate Pi using MPI and OpenMP";
}
