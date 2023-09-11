add_library(LLVMCudaFE MODULE mergeKernel.cpp)

# this is very important, to link with the passes library
target_link_libraries(LLVMCudaFE ${llvm_libs})
