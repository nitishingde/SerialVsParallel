#include <omp.h>
#include "Graph.h"
#include "../Utility.h"

std::vector<int32_t> svp::SerialBfsStrategy::search(const std::vector<std::vector<int32_t>> &graph, int32_t sourceNode) {
    SVP_PROFILE_FUNC();

    std::vector<int32_t> frontier;
    frontier.emplace_back(sourceNode);
    std::vector<int32_t> distances(graph.size(), -1);
    distances[sourceNode] = 0;

    for(int32_t distance = 1; !frontier.empty(); ++distance) {
        std::vector<int32_t> queue;
        // process all the unvisited neighbouring nodes
        for(const auto node :frontier) {
            for(const auto neighbour: graph[node]) {
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
std::vector<int32_t> svp::OpenMP_BfsStrategy::search(const std::vector<std::vector<int32_t>> &graph, int32_t sourceNode) {
    SVP_PROFILE_FUNC();

    omp_set_num_threads(omp_get_max_threads());

    std::vector<int32_t> frontier{sourceNode};
    std::vector<int32_t> distances(graph.size(), -1);
    distances[sourceNode] = 0;
    std::vector<uint8_t> visited(graph.size(), 0);
    visited[sourceNode] = true;

    for(int32_t distance = 1; !frontier.empty(); ++distance) {
        std::vector<int32_t> queue;
        // process all the unvisited neighbouring nodes
        #pragma omp parallel for schedule(static) default(none) firstprivate(distance) shared(graph, frontier, queue, distances, visited)
        for(int32_t i = 0; i < frontier.size(); ++i) {
            for(const auto neighbour: graph[frontier[i]]) {
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
