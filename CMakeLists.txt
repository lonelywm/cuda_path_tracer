cmake_minimum_required(VERSION 3.17)

project(MPathTracing LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

add_definitions(-w)

find_package(CUDAToolkit)

include(FetchContent)

# add_definitions(-rdc)

# # Eigen
# FetchContent_Declare(
#   Eigen
#   GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
#   GIT_TAG master
#   GIT_SHALLOW TRUE
#   GIT_PROGRESS TRUE)
# set(EIGEN_BUILD_DOC OFF)
# set(BUILD_TESTING OFF)
# set(EIGEN_BUILD_PKGCONFIG OFF)
# set(OFF)
# FetchContent_MakeAvailable(Eigen)

# assimp
# FetchContent_Declare(assimp
#     GIT_REPOSITORY https://github.com/assimp/assimp
#     GIT_TAG v5.0.1)
# set(ASSIMP_BUILD_TESTS OFF)
# set(ASSIMP_BUILD_ASSIMP_TOOLS OFF)
# set(ASSIMP_INSTALL_PDB OFF)
# FetchContent_MakeAvailable(assimp)

# # stb
# FetchContent_Declare(stb
#     GIT_REPOSITORY https://github.com/nothings/stb)
# FetchContent_MakeAvailable(stb)

# include_directories(${PROJECT_SOURCE_DIR}/build/_deps/eigen-src)
include_directories(${PROJECT_SOURCE_DIR}/build/_deps/assimp-src/include)
include_directories(${PROJECT_SOURCE_DIR}/build/_deps/assimp-build/include)
include_directories(${PROJECT_SOURCE_DIR}/build/_deps/stb-src)
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/core)
include_directories(${PROJECT_SOURCE_DIR}/utils)

link_directories(${PROJECT_SOURCE_DIR}/Lib)

add_subdirectory(test)

# FILE(GLOB HEADER_AND_SOURCES 
#     ${PROJECT_SOURCE_DIR}/core/*.hpp
#     ${PROJECT_SOURCE_DIR}/include/*.hpp
#     # ${PROJECT_SOURCE_DIR}/main/main.cpp
# )
