add_library(NoelleDOALL MODULE DOALL.cpp)

# this is very important, to link with the passes library
target_link_libraries(NoelleDOALL ${llvm_libs})
