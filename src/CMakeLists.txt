project(APE_INTERNAL)
aux_source_directory(. SRC_CXX)

cuda_add_library(ape SHARED ${SRC_CXX})
set_target_properties(ape PROPERTIES CUDA_ARCHITECTURES "80")
