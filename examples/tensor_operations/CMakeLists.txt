#set(TEST_TARGET tensor_operations)
#add_executable(${TEST_TARGET} main.cpp)
# ${CMAKE_SOURCE_DIR} refers to the top-level directory where the CMakeLists.txt is located (tensor_operations directory in this case).
#target_include_directories(tensor_operations PRIVATE "${CMAKE_SOURCE_DIR}/../../include" "${CMAKE_SOURCE_DIR}/../")


#set(TEST_TARGET mul_mat_inline)
#add_executable(${TEST_TARGET} mul_mat_inline.cpp)
#target_link_libraries(${TEST_TARGET} PRIVATE ggml common)

#set(TEST_TARGET mul_mat_from_gguf)
#add_executable(${TEST_TARGET} mul_mat_from_gguf.cpp)
#target_link_libraries(${TEST_TARGET} PRIVATE ggml common)

set(TEST_TARGET learn_ggml_tensors)
add_executable(${TEST_TARGET} learn_ggml_tensors.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common)