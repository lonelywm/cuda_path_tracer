FILE(GLOB HEADER_AND_SOURCE 
    ${PROJECT_SOURCE_DIR}/core/*.hpp
    ${PROJECT_SOURCE_DIR}/core/kernel/*.cu
    ${PROJECT_SOURCE_DIR}/build/_deps/assimp-src/include/*.h
    ${PROJECT_SOURCE_DIR}/build/_deps/eigen-src/Eigen/*.h
)

add_executable(test_load_mesh
    ${HEADER_AND_SOURCE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_load_mesh.cu
)

add_executable(test_bvh_cuda
    ${HEADER_AND_SOURCE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_bvh_cuda.cpp
)

target_link_libraries(test_load_mesh assimp-vc142-mtd)
target_link_libraries(test_bvh_cuda assimp-vc142-mtd CUDA::cudart)
