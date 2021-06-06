#include <cfloat>
#include <cmath>
#include <omp.h>
#include <queue>
#include "Graph.h"

bool svp::verifyLineage(const CsrGraph &graph, const std::vector<int32_t> &parents, const int32_t rootNode) {
    if(parents[rootNode] != rootNode) return false;

    const auto &edgeList = graph.edgeList;
    const auto &csr = graph.compressedSparseRows;

    for(int32_t node = 0, isRelated = false; node < parents.size(); ++node) {
        if(node == rootNode or parents[node] == -1) continue;
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

svp::WeightedTree<int32_t> svp::SerialBfsStrategy::search(const CsrGraph &graph, int32_t sourceNode) {
    SVP_PROFILE_FUNC();

    WeightedTree<int32_t> lineageTree(sourceNode, graph.getVertexCount(), -1);

    auto &edgeList = graph.edgeList;
    auto &csr = graph.compressedSparseRows;
    std::vector<int32_t> frontier{sourceNode};
    auto &costs = lineageTree.costs;
    costs[sourceNode] = 0;
    auto &parents = lineageTree.parents;
    parents[sourceNode] = sourceNode;

    for(int32_t distance = 1; !frontier.empty(); ++distance) {
        std::vector<int32_t> queue;
        // process all the unvisited neighbouring nodes
        for(const auto node :frontier) {
            for(uint32_t i = csr[node]; i < csr[node+1]; ++i) {
                const auto neighbour = edgeList[i];
                if(costs[neighbour] == -1) {
                    costs[neighbour] = distance;
                    parents[neighbour] = node;
                    queue.emplace_back(neighbour);
                }
            }
        }
        frontier = std::move(queue);
    }

    return lineageTree;
}

std::string svp::SerialBfsStrategy::toString() {
    return "Do a BFS using serial code";
}

// FIXME: slower than serial code
svp::WeightedTree<int32_t> svp::OpenMP_BfsStrategy::search(const CsrGraph &graph, int32_t sourceNode) {
    SVP_PROFILE_FUNC();

    WeightedTree<int32_t> lineageTree(sourceNode, graph.getVertexCount(), -1);
    omp_set_num_threads(omp_get_max_threads());

    auto &edgeList = graph.edgeList;
    auto &csr = graph.compressedSparseRows;
    std::vector<int32_t> frontier{sourceNode};
    auto &costs = lineageTree.costs;
    costs[sourceNode] = 0;
    auto &parents = lineageTree.parents;
    parents[sourceNode] = sourceNode;
    std::vector<uint8_t> visited(csr.size()-1, 0);
    visited[sourceNode] = true;

    for(int32_t distance = 1; !frontier.empty(); ++distance) {
        std::vector<int32_t> queue;
        // process all the unvisited neighbouring nodes
        #pragma omp parallel for schedule(static) default(none) firstprivate(distance) shared(edgeList, csr, frontier, queue, costs, parents, visited)
        for(int32_t i = 0; i < frontier.size(); ++i) {
            const auto node = frontier[i];
            for(uint32_t j = csr[node]; j < csr[node+1]; ++j) {
                const auto neighbour = edgeList[j];
                if(!visited[neighbour]) {
                    #pragma omp critical
                    {
                        if(costs[neighbour] == -1) {
                            costs[neighbour] = distance;
                            parents[neighbour] = node;
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

    return lineageTree;
}

std::string svp::OpenMP_BfsStrategy::toString() {
    return "Do a BFS using OpenMP code";
}

void svp::OpenCL_BfsStrategy::init() {
    OpenCL_Base::init();
    cl_int status = CL_SUCCESS;
    loadProgram("resources/bfs.cl");
    mKernel = cl::Kernel(mProgram, "bfs", &status);
    svp::verifyOpenCL_Status(status);
}

svp::OpenCL_BfsStrategy::OpenCL_BfsStrategy() {
    init();
}

svp::WeightedTree<int32_t> svp::OpenCL_BfsStrategy::search(const CsrGraph &graph, int32_t sourceNode) {
    SVP_PROFILE_FUNC();

    cl_int status = CL_SUCCESS;
    WeightedTree<int32_t> lineageTree(sourceNode, graph.getVertexCount(), -1);

    auto &edgeList = graph.edgeList;
    cl::Buffer edgeListBuffer(
        mContext,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        edgeList.size() * sizeof(decltype(graph.edgeList)::value_type),
        (void*)edgeList.data(),
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

    auto &costs = lineageTree.costs;
    memset(costs.data(), -1, costs.size() * sizeof(decltype(lineageTree.costs)::value_type));
    costs[sourceNode] = 0;
    cl::Buffer costsBuffer(
        mContext,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        costs.size() * sizeof(decltype(lineageTree.costs)::value_type),
        costs.data(),
        &status
    );
    verifyOpenCL_Status(status);

    auto &parents = lineageTree.parents;
    parents[sourceNode] = sourceNode;
    cl::Buffer parentsBuffer(
        mContext,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        parents.size() * sizeof(decltype(lineageTree.parents)::value_type),
        parents.data(),
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

    int kernelArg = 0;
    verifyOpenCL_Status(mKernel.setArg(kernelArg++, edgeListBuffer));
    verifyOpenCL_Status(mKernel.setArg(kernelArg++, csrBuffer));
    verifyOpenCL_Status(mKernel.setArg(kernelArg++, uint32_t(graph.getVertexCount())));
    verifyOpenCL_Status(mKernel.setArg(kernelArg++, costsBuffer));
    verifyOpenCL_Status(mKernel.setArg(kernelArg++, parentsBuffer));
    verifyOpenCL_Status(mKernel.setArg(kernelArg++, frontierCountBuffer));

    uint32_t globalWorkSize = mWorkGroupSize1d * ((graph.getVertexCount() + mWorkGroupSize1d - 1) / mWorkGroupSize1d);
    for(int32_t level = 0; 0 < frontierCount; ++level) {
        verifyOpenCL_Status(mKernel.setArg(kernelArg, int32_t(level)));

        cl::Event kernelEvent;
        verifyOpenCL_Status(mCommandQueue.enqueueNDRangeKernel(
            mKernel,
            cl::NullRange,
            cl::NDRange(globalWorkSize),
            cl::NDRange(mWorkGroupSize1d),
            nullptr,
            &kernelEvent
        ));

        cl::Event readBufferEvent;
        verifyOpenCL_Status(mCommandQueue.enqueueReadBuffer(frontierCountBuffer, CL_TRUE, 0, sizeof(int32_t), &frontierCount, nullptr, &readBufferEvent));
#if not NDEBUG
        SVP_PROFILE_OPENCL(kernelEvent);
        SVP_PROFILE_OPENCL(readBufferEvent);
#endif
    }

    verifyOpenCL_Status(mCommandQueue.enqueueReadBuffer(
        costsBuffer,
        CL_TRUE,
        0,
        costs.size() * sizeof(decltype(lineageTree.costs)::value_type),
        costs.data()
    ));

    verifyOpenCL_Status(mCommandQueue.enqueueReadBuffer(
        parentsBuffer,
        CL_TRUE,
        0,
        parents.size() * sizeof(decltype(lineageTree.parents)::value_type),
        parents.data()
    ));

    return lineageTree;
}

std::string svp::OpenCL_BfsStrategy::toString() {
    return "Do a BFS using OpenCL";
}

svp::WeightedTree<float> svp::SerialDijkstraStrategy::calculate(const svp::CsrGraph &graph, int32_t sourceNode) {
    SVP_PROFILE_FUNC();

    auto &csr = graph.compressedSparseRows;
    auto &edgeList = graph.edgeList;
    auto &weightList = graph.weightList;

    WeightedTree<float> lineageTree(sourceNode, graph.getVertexCount(), FLT_MAX);
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
    verifyOpenCL_Status(status);
    mUpdateCostKernel = cl::Kernel(mProgram, "updateCost", &status);
    verifyOpenCL_Status(status);
}

svp::OpenCL_DijkstraStrategy::OpenCL_DijkstraStrategy() {
    init();
}

svp::WeightedTree<float> svp::OpenCL_DijkstraStrategy::calculate(const svp::CsrGraph &graph, int32_t sourceNode) {
    SVP_PROFILE_FUNC();

    cl_int status = CL_SUCCESS;
    WeightedTree<float> lineageTree(sourceNode, graph.getVertexCount(), FLT_MAX);

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

    std::vector<uint8_t> masks(graph.getVertexCount(), 0);
    masks[sourceNode] = true;
    cl::Buffer masksBuffer(
        mContext,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        masks.size() * sizeof(decltype(masks)::value_type),
        masks.data(),
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

    auto &parents = lineageTree.parents;
    parents[sourceNode] = sourceNode;
    cl::Buffer parentsBuffer(
        mContext,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        parents.size() * sizeof(decltype(lineageTree.parents)::value_type),
        parents.data(),
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

    int kernelArg = 0;
    verifyOpenCL_Status(mCalculateCostKernel.setArg(kernelArg++, edgeListBuffer));
    verifyOpenCL_Status(mCalculateCostKernel.setArg(kernelArg++, weightListBuffer));
    verifyOpenCL_Status(mCalculateCostKernel.setArg(kernelArg++, csrBuffer));
    verifyOpenCL_Status(mCalculateCostKernel.setArg(kernelArg++, uint32_t(graph.getVertexCount())));
    verifyOpenCL_Status(mCalculateCostKernel.setArg(kernelArg++, masksBuffer));
    verifyOpenCL_Status(mCalculateCostKernel.setArg(kernelArg++, costsBuffer));
    verifyOpenCL_Status(mCalculateCostKernel.setArg(kernelArg++, updatedCostsBuffer));
    verifyOpenCL_Status(mCalculateCostKernel.setArg(kernelArg++, parentsBuffer));
    verifyOpenCL_Status(mCalculateCostKernel.setArg(kernelArg++, updatedCountBuffer));

    kernelArg = 0;
    verifyOpenCL_Status(mUpdateCostKernel.setArg(kernelArg++, uint32_t(graph.getVertexCount())));
    verifyOpenCL_Status(mUpdateCostKernel.setArg(kernelArg++, masksBuffer));
    verifyOpenCL_Status(mUpdateCostKernel.setArg(kernelArg++, costsBuffer));
    verifyOpenCL_Status(mUpdateCostKernel.setArg(kernelArg++, updatedCostsBuffer));
    verifyOpenCL_Status(mUpdateCostKernel.setArg(kernelArg++, updatedCountBuffer));

    uint32_t globalWorkSize = mWorkGroupSize1d * ((graph.getVertexCount() + mWorkGroupSize1d - 1) / mWorkGroupSize1d);
    for(; 0 < updatedCount;) {
        cl::Event calculateCostKernelEvent;
        verifyOpenCL_Status(mCommandQueue.enqueueNDRangeKernel(
            mCalculateCostKernel,
            cl::NullRange,
            cl::NDRange(globalWorkSize),
            cl::NDRange(mWorkGroupSize1d),
            nullptr,
            &calculateCostKernelEvent
        ));

        cl::Event updateCostKernelEvent;
        verifyOpenCL_Status(mCommandQueue.enqueueNDRangeKernel(
            mUpdateCostKernel,
            cl::NullRange,
            cl::NDRange(globalWorkSize),
            cl::NDRange(mWorkGroupSize1d),
            nullptr,
            &updateCostKernelEvent
        ));

        cl::Event readBufferEvent;
        verifyOpenCL_Status(mCommandQueue.enqueueReadBuffer(updatedCountBuffer, CL_TRUE, 0, sizeof(uint32_t), &updatedCount, nullptr, &readBufferEvent));
#if not NDEBUG
        SVP_PROFILE_OPENCL(calculateCostKernelEvent);
        SVP_PROFILE_OPENCL(updateCostKernelEvent);
        SVP_PROFILE_OPENCL(readBufferEvent);
#endif
    }

    verifyOpenCL_Status(mCommandQueue.enqueueReadBuffer(
        costsBuffer,
        CL_TRUE,
        0,
        costs.size() * sizeof(decltype(lineageTree.costs)::value_type),
        costs.data()
    ));

    verifyOpenCL_Status(mCommandQueue.enqueueReadBuffer(
        parentsBuffer,
        CL_TRUE,
        0,
        parents.size() * sizeof(decltype(lineageTree.parents)::value_type),
        parents.data()
    ));

    return lineageTree;
}

std::string svp::OpenCL_DijkstraStrategy::toString() {
    return "Dijkstra's algorithm using OpenCL";
}
