#include <omp.h>
#include <cmath>
#include "Graph.h"

std::vector<int32_t> svp::SerialBfsStrategy::search(const CsrGraph &graph, int32_t sourceNode) {
    SVP_PROFILE_FUNC();

    auto &edgeList = graph.edgeList;
    auto &csr = graph.compressedSparseRows;
    std::vector<int32_t> frontier{sourceNode};
    std::vector<int32_t> distances(csr.size()-1, -1);
    distances[sourceNode] = 0;

    for(int32_t distance = 1; !frontier.empty(); ++distance) {
        std::vector<int32_t> queue;
        // process all the unvisited neighbouring nodes
        for(const auto node :frontier) {
            for(uint32_t i = csr[node]; i < csr[node+1]; ++i) {
                const auto neighbour = edgeList[i];
                if(distances[neighbour] == -1) {
                    distances[neighbour] = distance;
                    queue.emplace_back(neighbour);
                }
            }
        }
        frontier = std::move(queue);
    }

    return distances;
}

std::string svp::SerialBfsStrategy::toString() {
    return "Do a BFS using serial code";
}

// FIXME: slower than serial code
std::vector<int32_t> svp::OpenMP_BfsStrategy::search(const CsrGraph &graph, int32_t sourceNode) {
    SVP_PROFILE_FUNC();

    omp_set_num_threads(omp_get_max_threads());

    auto &edgeList = graph.edgeList;
    auto &csr = graph.compressedSparseRows;
    std::vector<int32_t> frontier{sourceNode};
    std::vector<int32_t> distances(csr.size()-1, -1);
    distances[sourceNode] = 0;
    std::vector<uint8_t> visited(csr.size()-1, 0);
    visited[sourceNode] = true;

    for(int32_t distance = 1; !frontier.empty(); ++distance) {
        std::vector<int32_t> queue;
        // process all the unvisited neighbouring nodes
        #pragma omp parallel for schedule(static) default(none) firstprivate(distance) shared(edgeList, csr, frontier, queue, distances, visited)
        for(int32_t i = 0; i < frontier.size(); ++i) {
            for(uint32_t j = csr[frontier[i]]; j < csr[frontier[i]+1]; ++j) {
                const auto neighbour = edgeList[j];
                if(!visited[neighbour]) {
                    #pragma omp critical
                    {
                        if(distances[neighbour] == -1) {
                            distances[neighbour] = distance;
                            queue.emplace_back(neighbour);
                        }
                    }
                }
            }
        }

        for(const auto node: queue) {
            visited[node] = true;
        }
        frontier = std::move(queue);
    }

    return distances;
}

std::string svp::OpenMP_BfsStrategy::toString() {
    return "Do a BFS using OpenMP code";
}

void svp::OpenCL_BfsStrategy::init() {
    OpenCL_Base::init();
    cl_int status = CL_SUCCESS;
    loadProgram("resources/Bfs.cl");
    mKernel = cl::Kernel(mProgram, "bfsSearch", &status);
    svp::verifyOpenCL_Status(status);
}

svp::OpenCL_BfsStrategy::OpenCL_BfsStrategy() {
    init();
}

std::vector<int32_t> svp::OpenCL_BfsStrategy::search(const CsrGraph &graph, int32_t sourceNode) {
    SVP_PROFILE_FUNC();

    cl_int status = CL_SUCCESS;

    auto &edgeList = graph.edgeList;
    auto &csr = graph.compressedSparseRows;

    cl::Buffer edgeListBuffer(
        mContext,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        edgeList.size() * sizeof(decltype(graph.edgeList)::value_type),
        (void*)edgeList.data(),
        &status
    );
    verifyOpenCL_Status(status);

    cl::Buffer csrBuffer(
        mContext,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        csr.size() * sizeof(decltype(graph.compressedSparseRows)::value_type),
        (void*)csr.data(),
        &status
    );
    verifyOpenCL_Status(status);

    std::vector<int32_t> distances(graph.getVertexCount(), -1);
    distances[sourceNode] = 0;
    cl::Buffer distanceBuffer(
        mContext,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        distances.size() * sizeof(decltype(distances)::value_type),
        distances.data(),
        &status
    );
    verifyOpenCL_Status(status);

    int32_t frontierCount = 1;
    cl::Buffer frontierCountBuffer(
        mContext,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(int32_t),
        &frontierCount,
        &status
    );
    verifyOpenCL_Status(status);

    verifyOpenCL_Status(mKernel.setArg(0, edgeListBuffer));
    verifyOpenCL_Status(mKernel.setArg(1, csrBuffer));
    verifyOpenCL_Status(mKernel.setArg(2, uint32_t(graph.getVertexCount())));
    verifyOpenCL_Status(mKernel.setArg(3, distanceBuffer));
    verifyOpenCL_Status(mKernel.setArg(4, frontierCountBuffer));

    uint32_t globalWorkSize = mWorkGroupSize*((graph.getVertexCount()+mWorkGroupSize-1)/mWorkGroupSize);
    for(int32_t level = 0; 0 < frontierCount; ++level) {
        verifyOpenCL_Status(mKernel.setArg(5, int32_t(level)));
        verifyOpenCL_Status(mCommandQueue.enqueueNDRangeKernel(
            mKernel,
            cl::NullRange,
            cl::NDRange(globalWorkSize),
            cl::NDRange(mWorkGroupSize)
        ));
        verifyOpenCL_Status(mCommandQueue.enqueueReadBuffer(frontierCountBuffer, CL_TRUE, 0, sizeof(int32_t), &frontierCount));
    }

    verifyOpenCL_Status(mCommandQueue.enqueueReadBuffer(
        distanceBuffer,
        CL_TRUE,
        0,
        distances.size() * sizeof(decltype(distances)::value_type),
        distances.data()
    ));

    return distances;
}

std::string svp::OpenCL_BfsStrategy::toString() {
    return "Do a BFS using OpenCL";
}
