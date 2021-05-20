#include "Graph.h"
#include "../Utility.h"

std::vector<int32_t> svp::SerialBfsStrategy::search(const std::vector<std::vector<int32_t>> &graph, int32_t sourceNode) {
    SVP_PROFILE_FUNC();

    std::vector<int32_t> frontier;
    frontier.emplace_back(sourceNode);
    std::vector<int32_t> distances(graph.size(), -1);

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
