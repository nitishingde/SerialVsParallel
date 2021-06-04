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
    __global uchar *pMask,
    __global float *pCost,
    __global float *pUpdatedCost,
    __global int *pParent,
    __global uint *pUpdatedCount
) {
    size_t vertex = get_global_id(0);
    if(vertexCount <= vertex) return;
    if(!pMask[vertex]) return;

    atomic_dec(pUpdatedCount);
    pMask[vertex] = false;
    for(uint i = pCsr[vertex]; i < pCsr[vertex + 1]; ++i) {
        uint neighbour = pEdgeList[i];
        float cost = pCost[vertex] + pWeightList[i];
        if(cost < atomic_minf(&pUpdatedCost[neighbour], cost)) {
            if(cost < pCost[neighbour]) {
                atomic_xchg(&pParent[neighbour], vertex);
            }
        }
    }
}

__kernel void updateCost(
    __const uint vertexCount,
    __global uchar *pMask,
    __global float *pCost,
    __global float *pUpdatedCost,
    __global uint *pUpdatedCount
) {
    size_t vertex = get_global_id(0);
    if(vertexCount <= vertex) return;

    if(pUpdatedCost[vertex] < pCost[vertex]) {
        pCost[vertex] = pUpdatedCost[vertex];
        pMask[vertex] = true;
        atomic_inc(pUpdatedCount);
    }
    pUpdatedCost[vertex] = pCost[vertex];
}
