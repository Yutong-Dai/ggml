set(TEST_TARGET perceiver_resampler)
add_executable(${TEST_TARGET} PerceiverResampler.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common)

set(TEST_TARGET projector_single_sample)
add_executable(${TEST_TARGET} projector_single_sample.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common)

set(TEST_TARGET io)
add_executable(${TEST_TARGET} io.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common)