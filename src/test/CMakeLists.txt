project(APE)

aux_source_directory(. SRC_CXX)
cuda_add_library(ape_test SHARED ${SRC_CXX})
