CONFIG_LLVM_DEBUG   = -G "Unix Makefiles" \
									    -DCMAKE_BUILD_TYPE="Debug" \

cudafe:
	mkdir -p build && cd build
	cd build && \
	cmake ${CONFIG_LLVM_DEBUG} .. &&\
	make -j20 

