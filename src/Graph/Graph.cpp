#include <cfloat>
#include <cmath>
#include <omp.h>
#include <queue>
#include "Graph.h"

svp::Tree::Tree(uint32_t nodes) {
    costs = std::vector<float>(nodes, FLT_MAX);
    parents = std::vector<int32_t>(nodes, -1);
}

bool svp::verifyLineage(const svp::CsrGraph &graph, const svp::Tree &lineageTree) {
    const auto &edgeList = graph.edgeList;
    const auto &csr = graph.compressedSparseRows;
    const auto &parents = lineageTree.parents;

    for(int32_t node = 0, isRelated = false; node < parents.size(); ++node) {
        for(uint32_t i = csr[node]; i < csr[node+1]; ++i) {
            if(edgeList[i] == parents[node]) {
                isRelated = true;
                break;
            }
        }
        if(!isRelated) return false;
    }

    return true;
}

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

svp::Tree svp::SerialDijkstraStrategy::calculate(const svp::CsrGraph &graph, int32_t sourceNode) {
    SVP_PROFILE_FUNC();

    auto &csr = graph.compressedSparseRows;
    auto &edgeList = graph.edgeList;
    auto &weightList = graph.weightList;

    Tree lineageTree(graph.getVertexCount());
    using Edge = std::pair<float, int32_t>;
    std::priority_queue<Edge, std::vector<Edge>, std::greater<>> priorityQueue;
    priorityQueue.emplace(Edge(0.f, sourceNode));
    std::vector<uint8_t> visited(graph.getVertexCount(), 0);
    auto &costs = lineageTree.costs;
    costs[sourceNode] = 0;
    auto &parents = lineageTree.parents;
    parents[sourceNode] = sourceNode;

    for(;!priorityQueue.empty();) {
        auto [weight, node] = priorityQueue.top();
        priorityQueue.pop();
        visited[node] = true;

        // optimisation
        if(costs[node] < weight) continue;

        for(uint32_t i = csr[node]; i < csr[node+1]; ++i) {
            const auto neighbour = edgeList[i];
            if(visited[neighbour]) continue;

            auto newDistance = costs[node] + weightList[i];
            if(newDistance < costs[neighbour]) {
                costs[neighbour] = newDistance;
                parents[neighbour] = node;
                priorityQueue.emplace(Edge(newDistance, neighbour));
            }
        }
    }

    return lineageTree;
}

std::string svp::SerialDijkstraStrategy::toString() {
    return "Dijkstra's algorithm using Serial code";
}

void svp::OpenCL_DijkstraStrategy::init() {
    OpenCL_Base::init();
    cl_int status = CL_SUCCESS;
    loadProgram("resources/sssp.cl");
    mCalculateCostKernel = cl::Kernel(mProgram, "calculateCost", &status);
    svp::verifyOpenCL_Status(status);
    mUpdateCostKernel = cl::Kernel(mProgram, "updateCost", &status);
    svp::verifyOpenCL_Status(status);
}

svp::OpenCL_DijkstraStrategy::OpenCL_DijkstraStrategy() {
    init();
}

svp::Tree svp::OpenCL_DijkstraStrategy::calculate(const svp::CsrGraph &graph, int32_t sourceNode) {
    SVP_PROFILE_FUNC();

    cl_int status = CL_SUCCESS;
    svp::Tree lineageTree(graph.getVertexCount());

    auto &edgeList = graph.edgeList;
    cl::Buffer edgeListBuffer(
        mContext,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        edgeList.size() * sizeof(decltype(graph.edgeList)::value_type),
        (void*)edgeList.data(),
        &status
    );
    verifyOpenCL_Status(status);

    auto &weightList = graph.weightList;
    cl::Buffer weightListBuffer(
        mContext,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        weightList.size() * sizeof(decltype(graph.weightList)::value_type),
        (void*)weightList.data(),
        &status
    );
    verifyOpenCL_Status(status);

    auto &csr = graph.compressedSparseRows;
    cl::Buffer csrBuffer(
        mContext,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        csr.size() * sizeof(decltype(graph.compressedSparseRows)::value_type),
        (void*)csr.data(),
        &status
    );
    verifyOpenCL_Status(status);

    std::vector<uint8_t> mask(graph.getVertexCount(), 0);
    mask[sourceNode] = true;
    cl::Buffer maskBuffer(
        mContext,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        mask.size() * sizeof(decltype(mask)::value_type),
        mask.data(),
        &status
    );

    auto &costs = lineageTree.costs;
    costs[sourceNode] = 0;
    cl::Buffer costsBuffer(
        mContext,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        costs.size() * sizeof(decltype(lineageTree.costs)::value_type),
        costs.data(),
        &status
    );
    verifyOpenCL_Status(status);

    std::vector<float> updatedCosts(graph.getVertexCount(), FLT_MAX);
    updatedCosts[sourceNode] = 0;
    cl::Buffer updatedCostsBuffer(
        mContext,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        updatedCosts.size() * sizeof(decltype(updatedCosts)::value_type),
        updatedCosts.data(),
        &status
    );
    verifyOpenCL_Status(status);

    uint32_t updatedCount = 1;
    cl::Buffer updatedCountBuffer(
        mContext,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t),
        &updatedCount,
        &status
    );
    verifyOpenCL_Status(status);

    verifyOpenCL_Status(mCalculateCostKernel.setArg(0, edgeListBuffer));
    verifyOpenCL_Status(mCalculateCostKernel.setArg(1, weightListBuffer));
    verifyOpenCL_Status(mCalculateCostKernel.setArg(2, csrBuffer));
    verifyOpenCL_Status(mCalculateCostKernel.setArg(3, uint32_t(graph.getVertexCount())));
    verifyOpenCL_Status(mCalculateCostKernel.setArg(4, maskBuffer));
    verifyOpenCL_Status(mCalculateCostKernel.setArg(5, costsBuffer));
    verifyOpenCL_Status(mCalculateCostKernel.setArg(6, updatedCostsBuffer));
    verifyOpenCL_Status(mCalculateCostKernel.setArg(7, updatedCountBuffer));

    verifyOpenCL_Status(mUpdateCostKernel.setArg(0, edgeListBuffer));
    verifyOpenCL_Status(mUpdateCostKernel.setArg(1, weightListBuffer));
    verifyOpenCL_Status(mUpdateCostKernel.setArg(2, csrBuffer));
    verifyOpenCL_Status(mUpdateCostKernel.setArg(3, uint32_t(graph.getVertexCount())));
    verifyOpenCL_Status(mUpdateCostKernel.setArg(4, maskBuffer));
    verifyOpenCL_Status(mUpdateCostKernel.setArg(5, costsBuffer));
    verifyOpenCL_Status(mUpdateCostKernel.setArg(6, updatedCostsBuffer));
    verifyOpenCL_Status(mUpdateCostKernel.setArg(7, updatedCountBuffer));

    uint32_t globalWorkSize = mWorkGroupSize*((graph.getVertexCount()+mWorkGroupSize-1)/mWorkGroupSize);
    for(; 0 < updatedCount;) {
        verifyOpenCL_Status(mCommandQueue.enqueueNDRangeKernel(
            mCalculateCostKernel,
            cl::NullRange,
            cl::NDRange(globalWorkSize),
            cl::NDRange(mWorkGroupSize)
        ));

        verifyOpenCL_Status(mCommandQueue.enqueueNDRangeKernel(
            mUpdateCostKernel,
            cl::NullRange,
            cl::NDRange(globalWorkSize),
            cl::NDRange(mWorkGroupSize)
        ));

        verifyOpenCL_Status(mCommandQueue.enqueueReadBuffer(updatedCountBuffer, CL_TRUE, 0, sizeof(uint32_t), &updatedCount));
    }

    verifyOpenCL_Status(mCommandQueue.enqueueReadBuffer(
        costsBuffer,
        CL_TRUE,
        0,
        costs.size() * sizeof(decltype(lineageTree.costs)::value_type),
        costs.data()
    ));

    return lineageTree;
}

std::string svp::OpenCL_DijkstraStrategy::toString() {
    return "Dijkstra's algorithm using OpenCL";
}
