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
