cmake_minimum_required(VERSION 3.10)
project(vulkan_matmul_fp16 C)

find_package(Vulkan REQUIRED)

# Compile shader
add_custom_command(
    OUTPUT matmul.spv
    COMMAND glslc ${CMAKE_CURRENT_SOURCE_DIR}/matmul.comp -o matmul.spv
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/matmul.comp
    COMMENT "Compiling compute shader to SPIR-V"
)

add_executable(matmul_vk main.c args.c matmul.spv)

target_include_directories(matmul_vk PRIVATE ${Vulkan_INCLUDE_DIRS})
target_link_libraries(matmul_vk PRIVATE ${Vulkan_LIBRARIES})

# Copy spv to build dir if needed (it's already there because of add_custom_command output)
