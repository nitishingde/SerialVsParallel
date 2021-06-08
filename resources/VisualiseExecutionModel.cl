__kernel void visualiseWorkItemsInGlobalSpace(__global float *matrix) {
    __private size_t x = get_global_id(0);
    __private size_t y = get_global_id(1);
    __private size_t z = get_global_id(2);
    __private size_t dim_x = get_global_size(0);
    __private size_t dim_y = get_global_size(1);
    __private size_t dim_z = get_global_size(2);

    matrix[x*dim_y + y*dim_z + z] = x*dim_y + y*dim_z + z;
}

__kernel void visualiseWorkGroups(__global float *matrix) {
    __private size_t x = get_global_id(0);
    __private size_t y = get_global_id(1);
    __private size_t z = get_global_id(2);
    __private size_t dim_x = get_global_size(0);
    __private size_t dim_y = get_global_size(1);
    __private size_t dim_z = get_global_size(2);

    __private size_t gx = get_group_id(0);
    __private size_t gy = get_group_id(1);
    __private size_t gz = get_group_id(2);
    __private size_t dim_gx = get_num_groups(0);
    __private size_t dim_gy = get_num_groups(1);
    __private size_t dim_gz = get_num_groups(2);

    matrix[x*dim_y + y*dim_z + z] = gx*dim_gy + gy*dim_gz + gz;
}

__kernel void visualiseWorkItemsInWorkGroupSpace(__global float *matrix) {
    __private size_t x = get_global_id(0);
    __private size_t y = get_global_id(1);
    __private size_t z = get_global_id(2);
    __private size_t dim_x = get_global_size(0);
    __private size_t dim_y = get_global_size(1);
    __private size_t dim_z = get_global_size(2);

    __private size_t lx = get_local_id(0);
    __private size_t ly = get_local_id(1);
    __private size_t lz = get_local_id(2);
    __private size_t dim_lx = get_local_size(0);
    __private size_t dim_ly = get_local_size(1);
    __private size_t dim_lz = get_local_size(2);

    matrix[x*dim_y + y*dim_z + z] = lx*dim_ly + ly*dim_lz + lz;
}

__kernel void visualiseSequenceOfWorkItemsInGlobalSpace() {
    __private size_t x = get_global_id(0);
    __private size_t y = get_global_id(1);
    __private size_t z = get_global_id(2);
    __private size_t dim_x = get_global_size(0);
    __private size_t dim_y = get_global_size(1);
    __private size_t dim_z = get_global_size(2);

    printf("%u, ", x*dim_y + y*dim_z + z);
}

__kernel void visualiseSequenceOfWorkGroups() {
    __private size_t gx = get_group_id(0);
    __private size_t gy = get_group_id(1);
    __private size_t gz = get_group_id(2);
    __private size_t dim_gx = get_num_groups(0);
    __private size_t dim_gy = get_num_groups(1);
    __private size_t dim_gz = get_num_groups(2);

    printf("%u, ", gx*dim_gy + gy*dim_gz + gz);
}

__kernel void visualiseSequenceOfWorkItemsInWorkGroupSpace() {
    __private size_t gx = get_group_id(0);
    __private size_t gy = get_group_id(1);
    __private size_t gz = get_group_id(2);
    __private size_t dim_gx = get_num_groups(0);
    __private size_t dim_gy = get_num_groups(1);
    __private size_t dim_gz = get_num_groups(2);

    __private size_t lx = get_local_id(0);
    __private size_t ly = get_local_id(1);
    __private size_t lz = get_local_id(2);
    __private size_t dim_lx = get_local_size(0);
    __private size_t dim_ly = get_local_size(1);
    __private size_t dim_lz = get_local_size(2);

    printf("\t{work-group ID: %u, work-item ID: %u},\n", gx*dim_gy + gy*dim_gz + gz, lx*dim_ly + ly*dim_lz + lz);
}