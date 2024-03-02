add_library(NoelleReduction MODULE Reduction.cpp)

# this is very important, to link with the passes library
target_link_libraries(NoelleReduction ${llvm_libs})
