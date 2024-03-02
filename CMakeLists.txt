cmake_minimum_required(VERSION 3.4.3)
project(CudaFE)

# need at least c++11 to compile successfully, I chose c++17 just to be cool
set (CMAKE_CXX_STANDARD 17)

# maybe not needed, but just to be safe
set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_COMPILER "clang++")

list(APPEND CMAKE_PREFIX_PATH "${LLVM_ROOT}/share/llvm/cmake")
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
SET(CMAKE_CXX_FLAGS "-Wall -fno-rtti")

# just copied from the docs
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(/u/zujunt/noelle-workspace/noelle/install/include /u/zujunt/noelle-workspace/noelle/install/install/include/svf)
list(APPEND TVM_LINKER_LIBS LLVM)
# this is very important, to link with the passes library
#llvm_map_components_to_libnames(llvm_libs passes)

add_subdirectory(MergeKernel)
add_subdirectory(NoelleReduction)
