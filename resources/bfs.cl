#define UNVISITED (-1)

__kernel void bfs(
    __constant uint *pEdgeList,
    __constant uint *pCsr,
    __const uint vertexCount,
    __global int *pCost,
    __global int *pParent,
    __global int *pFrontierCount,
    __const int level
) {
    size_t vertex = get_global_id(0);
    if(vertexCount <= vertex) return;

    if(atomic_cmpxchg(&pCost[vertex], level, level) != level) return;

    // remove vertex as frontier
    atomic_dec(&pFrontierCount[0]);
    for(uint i = pCsr[vertex]; i < pCsr[vertex + 1]; ++i) {
        uint neighbour = pEdgeList[i];
        if(atomic_cmpxchg(&pCost[neighbour], UNVISITED, level + 1) == UNVISITED) {
            pParent[neighbour] = vertex;
            atomic_inc(&pFrontierCount[0]);
        }
    }
}
