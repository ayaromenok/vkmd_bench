cmake_minimum_required(VERSION 3.10)
project(vulkan_matmul_fp16 C)

find_package(Vulkan REQUIRED)

# Compile shaders
add_custom_command(
    OUTPUT matmul_fp16.spv
    COMMAND glslc -DDATA_TYPE=float16_t ${CMAKE_CURRENT_SOURCE_DIR}/matmul.comp -o matmul_fp16.spv
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/matmul.comp
    COMMENT "Compiling fp16 compute shader"
)

add_custom_command(
    OUTPUT matmul_int16.spv
    COMMAND glslc -DDATA_TYPE=int16_t ${CMAKE_CURRENT_SOURCE_DIR}/matmul.comp -o matmul_int16.spv
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/matmul.comp
    COMMENT "Compiling int16 compute shader"
)

add_custom_command(
    OUTPUT matmul_fp32.spv
    COMMAND glslc -DDATA_TYPE=float ${CMAKE_CURRENT_SOURCE_DIR}/matmul.comp -o matmul_fp32.spv
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/matmul.comp
    COMMENT "Compiling fp32 compute shader"
)

add_custom_command(
    OUTPUT matmul_int32.spv
    COMMAND glslc -DDATA_TYPE=int ${CMAKE_CURRENT_SOURCE_DIR}/matmul.comp -o matmul_int32.spv
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/matmul.comp
    COMMENT "Compiling int32 compute shader"
)

add_executable(matmul_vk main.c args.c matmul_fp16.spv matmul_int16.spv matmul_fp32.spv matmul_int32.spv)

target_include_directories(matmul_vk PRIVATE ${Vulkan_INCLUDE_DIRS})
target_link_libraries(matmul_vk PRIVATE ${Vulkan_LIBRARIES})

# Copy spv to build dir if needed (it's already there because of add_custom_command output)
