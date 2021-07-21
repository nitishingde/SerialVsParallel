# Graph

- [Graph](#graph)
  - [Breadth first search](#breadth-first-search)
    - [Serial Implementation](#serial-implementation)
    - [OpenMP Implementation](#openmp-implementation)
    - [OpenCL Implementation](#opencl-implementation)
  - [Single source shortest path (SSSP)](#single-source-shortest-path-sssp)
    - [Serial Implementation](#serial-implementation-1)
    - [OpenCL Implementation](#opencl-implementation-1)

## Breadth first search

- [What is Breadth First search (bfs)?](https://en.wikipedia.org/wiki/Breadth-first_search)

![animation](https://upload.wikimedia.org/wikipedia/commons/4/46/Animated_BFS.gif)
- A breadth first search traversal gives us an unweighted tree data structure. We store the parent of each node in `WeightedTree::parents` and the distance from source node in `WeightedTree::costs` (don't mistake this for an edge between 2 nodes).
- We use compressed row matrix [(csr)](http://scipy-lectures.org/advanced/scipy_sparse/csr_matrix.html) to represent our graphs.
- We could have used an adjacency list, `std::vector<std::vector<uint32_t>>` as our graph data structure, but we will explain why we chose csr instead, in the OpenCL implementation.

### Serial Implementation

```cpp
#include <cstdint>
#include <vector>

// Graph Data structure
struct CsrGraph {
    std::vector<uint32_t> edgeList;
    std::vector<float> weightList;
    std::vector<uint32_t> compressedSparseRows;
    [[nodiscard]] size_t getVertexCount() const {
        return compressedSparseRows.size()-1;
    }
};

// Output of bfs, an unweighted tree with cost/distance from source node
template<typename Weight>
struct WeightedTree {
    explicit WeightedTree(int32_t rootNode, uint32_t nodes, Weight defaultWeight) {
        static_assert(std::is_arithmetic_v<Weight>);
        costs = std::vector<Weight>(nodes, defaultWeight);
        parents = std::vector<int32_t>(nodes, -1);
        this->rootNode = parents[rootNode] = rootNode;
    }
    int32_t rootNode;
    std::vector<Weight> costs;
    std::vector<int32_t> parents;
};


WeightedTree<int32_t> bfs(const CsrGraph &graph, int32_t sourceNode) {
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
```

### OpenMP Implementation

- This algorithm looks like it has 3 nested loops.
- We try to parallelize the 2nd loop and not the outermost loop like before.
  - This is a level synchronised algorithm.
  - That's why we need to process/deal with all the nodes in frontier queue as one iteration, for this to work.

```cpp
#include <cstdint>
#include <vector>

// Graph Data structure
struct CsrGraph {
    std::vector<uint32_t> edgeList;
    std::vector<float> weightList;
    std::vector<uint32_t> compressedSparseRows;
    [[nodiscard]] size_t getVertexCount() const {
        return compressedSparseRows.size()-1;
    }
};

// Output of bfs, an unweighted tree with cost/distance from source node
template<typename Weight>
struct WeightedTree {
    explicit WeightedTree(int32_t rootNode, uint32_t nodes, Weight defaultWeight) {
        static_assert(std::is_arithmetic_v<Weight>);
        costs = std::vector<Weight>(nodes, defaultWeight);
        parents = std::vector<int32_t>(nodes, -1);
        this->rootNode = parents[rootNode] = rootNode;
    }
    int32_t rootNode;
    std::vector<Weight> costs;
    std::vector<int32_t> parents;
};


WeightedTree<int32_t> bfs(const CsrGraph &graph, int32_t sourceNode) {
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
        #pragma omp parallel for schedule(static) default(none) firstprivate(distance) shared(edgeList, csr, frontier, queue, costs, parents, visited)
        for(int32_t i = 0; i < frontier.size(); ++i) {
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
```

### OpenCL Implementation

- Why csr?
  - Let's say we are using adjacency matrix, and that adjacency matrix has `10,000` nodes.
  - Which means we will have `10,000` edge-lists on our hands. Now each edge-list individually is a contiguous block of memory. Therefore, to copy the whole graph into the buffer (RAM to VRAM), we will need to copy each edge-list separately. This means we will make `10,000` buffer write operations. As you can imagine, this is quite expensive. A weighted csr has `3` things, the edge-list as a whole, the weight-list as whole and 1 index array for the edge-list and weight-list. So making `3` buffer writes makes copying the graph a much more efficient faster.
- I used this [paper](HiPC.pdf) as reference to write the kernel.
- For this to work, the host side code, and the kernel code, both need to communicate.
- We create frontier array of size #nodes.
- Global work size dimension: (#nodes, 1, 1)
- Local work size dimension: (1, 1, 1)
- Work-item: A work-item/thread will be executed for 1 node each.
  - We will process that node, only if it is enqueued. Note: We have merged the functionality of frontier queue with the cost array, to save memory.
    - A node is enqueued if it's cost matches with the level/iteration of the kernel call.
    - To enqueue a node we set the cost of that node as level(kernel)+1, which also happens to be the actual distance from the source node.
    - It might look a little confusing, since we have merged the functionality of the frontier queue with the cost array. See the commented parts in kernel, to get a better understanding.
- 1 kernel call will deal with 1 level synchronization. Therefore the kernel will be enqueued at most the diameter(graph).
- One interesting point to note here is that the execution time of reading buffer (frontierCount) is found to be equivalent to that of kernel execution time. So to minimize the overall execution time, we read the `frontierCount` only at intervals, instead of reading it at each level. We might end up executing redundant kernels at the end, but the cost of executing few empty kernels still ends up costing less.

Host  code:

```cpp
for(int32_t level = 0; 0 < frontierCount; ++level) {
    mKernel.setArg(kernelArg, int32_t(level));
    mCommandQueue.enqueueNDRangeKernel(mKernel, cl::NullRange, cl::NDRange(graph.getVertexCount()), cl::NDRange(1));

    if(level%4 == 0) {
        mCommandQueue.enqueueReadBuffer(frontierCountBuffer, CL_TRUE, 0, sizeof(int32_t), &frontierCount);
    }
}
```

Kernel code:

```c
//#define ENQUEUED 1
#define UNVISITED (-1)

__kernel void bfs(
    __constant uint *pEdgeList,
    __constant uint *pCsr,
    __const uint vertexCount,
    //__global int *pFrontier,
    __global int *pCosts,
    __global int *pParents,
    __global int *pFrontierCount,
    __const int level
) {
    size_t vertex = get_global_id(0);
    if(vertexCount <= vertex) return;

    // if(pFrontier[vertex] != ENQUEUED) return;
    if(pCosts[vertex] != level) return;

    // remove vertex as frontier
    atomic_dec(pFrontierCount);
    for(uint i = pCsr[vertex]; i < pCsr[vertex + 1]; ++i) {
        uint neighbour = pEdgeList[i];
        if(atomic_cmpxchg(&pCosts[neighbour], UNVISITED, level + 1) == UNVISITED) {
        // if(atomic_cmpxchg(&pFrontier[neighbour], UNVISITED, ENQUEUED) == UNVISITED) {
            pParents[neighbour] = vertex;
            atomic_inc(pFrontierCount);
        }
    }
}
```

## Single source shortest path (SSSP)
- [ ] Todo: Explain SSSP briefly

### Serial Implementation
- [ ] Todo: Document code below

```cpp
WeightedTree<float> dijkstra(const CsrGraph &graph, int32_t sourceNode) {
    auto &csr = graph.compressedSparseRows;
    auto &edgeList = graph.edgeList;
    auto &weightList = graph.weightList;

    WeightedTree<float> lineageTree(sourceNode, graph.getVertexCount(), std::numeric_limits<float>::infinity());
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
```


### OpenCL Implementation
- [ ] Todo: Document kernel code below

```c
// OpenCL 1.2 specs does not support atomic_min for floats, so we implemented our own
inline float atomic_minf(volatile __global float *pFloat, float val) {
    for(float current; val < (current = *pFloat);)
        val = atomic_xchg(pFloat, min(current, val));
    return val;
}

__kernel void calculateCost(
    __constant uint *pEdgeList,
    __constant float *pWeightList,
    __constant uint *pCsr,
    __const uint vertexCount,
    __global uchar *pMasks,
    __global float *pCosts,
    __global float *pUpdatedCosts,
    __global int *pParents,
    __global uint *pUpdatedCount
) {
    size_t vertex = get_global_id(0);
    if(vertexCount <= vertex) return;
    if(!pMasks[vertex]) return;

    atomic_dec(pUpdatedCount);
    pMasks[vertex] = false;
    for(uint i = pCsr[vertex]; i < pCsr[vertex + 1]; ++i) {
        uint neighbour = pEdgeList[i];
        float cost = pCosts[vertex] + pWeightList[i];
        if(cost < atomic_minf(&pUpdatedCosts[neighbour], cost)) {
            if(cost < pCosts[neighbour]) {
                atomic_xchg(&pParents[neighbour], vertex);
            }
        }
    }
}

__kernel void updateCost(
    __const uint vertexCount,
    __global uchar *pMasks,
    __global float *pCosts,
    __global float *pUpdatedCosts,
    __global uint *pUpdatedCount
) {
    size_t vertex = get_global_id(0);
    if(vertexCount <= vertex) return;

    if(pUpdatedCosts[vertex] < pCosts[vertex]) {
        pCosts[vertex] = pUpdatedCosts[vertex];
        pMasks[vertex] = true;
        atomic_inc(pUpdatedCount);
    }
    pUpdatedCosts[vertex] = pCosts[vertex];
}
```