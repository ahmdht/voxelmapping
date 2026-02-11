find_package(Python3 COMPONENTS Interpreter REQUIRED)

add_test(
    NAME voxel_mapping_pytest
    COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_LIST_DIR}
)

set_tests_properties(voxel_mapping_pytest PROPERTIES
    ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:voxel_mapping_py>:${CMAKE_SOURCE_DIR}/python"
)
