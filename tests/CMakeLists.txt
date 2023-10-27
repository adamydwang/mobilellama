enable_testing()

file(GLOB_RECURSE TEST_SOURCES ${PROJECT_SOURCE_DIR}/tests/*.cc)
message(STATUS "Test sources: ${TEST_SOURCES}")
add_executable(
    ${PROJECT_NAME}_test
    ${TEST_SOURCES}
    ${SOURCES}
)
target_link_libraries(
    ${PROJECT_NAME}_test
    GTest::gtest_main
    sentencepiece
)

include(GoogleTest)
gtest_discover_tests(${PROJECT_NAME}_test)