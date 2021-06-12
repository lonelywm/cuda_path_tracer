FILE(GLOB HEADER_AND_SOURCE 
    ${PROJECT_SOURCE_DIR}/core/*.h
    ${PROJECT_SOURCE_DIR}/core/*.hpp
    ${PROJECT_SOURCE_DIR}/core/kernel/*.cu
    ${PROJECT_SOURCE_DIR}/build/_deps/assimp-src/include/*.h
    ${PROJECT_SOURCE_DIR}/build/_deps/eigen-src/Eigen/*.h
)

add_executable(test
    ${HEADER_AND_SOURCE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test.cpp
)

target_link_libraries(test assimp-vc142-mtd CUDA::cudart)
