cmake_minimum_required(VERSION 3.10)
project(vkmdbench C)

find_package(Vulkan REQUIRED)

# === Unified shaders ===
set(OPS matmul add sub div mad)
set(OP_IDS 0 1 2 3 4)
set(DATA_TYPES fp16 int16 fp32 int32)
set(DATA_TYPE_GLSL float16_t int16_t float int)

set(ALL_SPVS "")

foreach(op_idx RANGE 0 4)
    list(GET OPS ${op_idx} op_name)
    list(GET OP_IDS ${op_idx} op_id)
    foreach(dt_idx RANGE 0 3)
        list(GET DATA_TYPES ${dt_idx} dt_name)
        list(GET DATA_TYPE_GLSL ${dt_idx} dt_glsl)
        set(spv_name "shaders/${op_name}_${dt_name}.spv")
        add_custom_command(
            OUTPUT ${spv_name}
            COMMAND ${CMAKE_COMMAND} -E make_directory shaders
            COMMAND glslc -DDATA_TYPE=${dt_glsl} -DOP_TYPE=${op_id} ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/shader.comp -o ${spv_name}
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/shader.comp
            COMMENT "Compiling ${op_name} ${dt_name} shader"
        )
        list(APPEND ALL_SPVS ${spv_name})
    endforeach()
endforeach()

add_executable(vkmdbench src/main.c src/args.c
    ${ALL_SPVS}
)

target_include_directories(vkmdbench PRIVATE ${Vulkan_INCLUDE_DIRS})
target_link_libraries(vkmdbench PRIVATE ${Vulkan_LIBRARIES})

# Copy spv to build dir if needed (it's already there because of add_custom_command output)
