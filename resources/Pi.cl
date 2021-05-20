__kernel void calculatePi(__global float *workGroupArea, __const size_t N, __const size_t steps) {
    // This check is moved after the barrier function call
    // if(get_global_id(0) > steps) return;

    __local float localArea[%workGroupSize%];

    __private size_t localIndex = get_local_id(0);
    __private float x = (get_global_id(0) + 0.5f) / (float)steps;
    localArea[localIndex] = 4.0f / (1.0f + x*x);

    // Note: barrier function must be encountered by all work-items in a work-group executing the kernel.
    barrier(CLK_LOCAL_MEM_FENCE);

    if(get_global_id(0) > steps) return;

    // add all the values in localArea and store it in 'workGroupArea' array for the work-group id
    if(localIndex != 0) return;
    __private float partialSum = 0;
    for(size_t i = 0; i < %workGroupSize%; ++i) {
        partialSum += localArea[i];
    }
    workGroupArea[get_group_id(0)] = partialSum;
}
