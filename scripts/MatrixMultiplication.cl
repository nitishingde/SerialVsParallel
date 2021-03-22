__kernel void calculateDotProduct(__global float *matrix1, __global float *matrix2, __global float *result, __const size_t m, __const size_t n, __const size_t p) {
    __private size_t x = get_global_id(0);
    __private size_t y = get_global_id(1);

    if(m <= x || p <= y) return;

    __private float temp = 0.f;
    for(size_t k = 0; k < n; ++k) {
        temp += matrix1[x*n + k] * matrix2[k*p + y];
    }
    result[x*p + y] = temp;
}
