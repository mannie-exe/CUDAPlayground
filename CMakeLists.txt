cmake_minimum_required(VERSION 3.18)
project(CUDAPlayground VERSION 0.1.0 LANGUAGES C CUDA)

# Require C17 for all projects
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED True)

# C-only
add_executable("c_add" ./src/c_add.c)

# with CUDA
find_package(CUDAToolkit)
add_executable("cuda_add" ./src/cuda_add.cu)
# target_link_libraries("cuda_add" PRIVATE CUDA::cudart)
set_target_properties("cuda_add" PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
