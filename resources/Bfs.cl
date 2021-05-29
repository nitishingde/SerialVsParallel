#define UNVISITED (-1)

__kernel void bfsSearch(__constant uint *pEdgeList, __constant uint *pCsr, __const uint vertexCount, __global int *pDistance, __global int *pFrontierCount, __const int level) {
    size_t vertex = get_global_id(0);
    if(vertexCount <= vertex) return;

    if(atomic_cmpxchg(&pDistance[vertex], level, level) != level) return;

    // remove vertex as frontier
    atomic_dec(&pFrontierCount[0]);
    for(uint i = pCsr[vertex]; i < pCsr[vertex + 1]; ++i) {
        uint neighbour = pEdgeList[i];
        if(atomic_cmpxchg(&pDistance[neighbour], UNVISITED, level + 1) == UNVISITED) {
            atomic_inc(&pFrontierCount[0]);
        }
    }
}
