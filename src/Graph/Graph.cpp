#include <omp.h>
#include "Graph.h"
#include "../Utility.h"

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
