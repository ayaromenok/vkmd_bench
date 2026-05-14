cmake_minimum_required(VERSION 3.10)
project(vulkan_matmul_fp16 C)

find_package(Vulkan REQUIRED)

# === MatMul shaders ===
add_custom_command(
    OUTPUT matmul_fp16.spv
    COMMAND glslc -DDATA_TYPE=float16_t ${CMAKE_CURRENT_SOURCE_DIR}/matmul.comp -o matmul_fp16.spv
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/matmul.comp
    COMMENT "Compiling fp16 matmul shader"
)

add_custom_command(
    OUTPUT matmul_int16.spv
    COMMAND glslc -DDATA_TYPE=int16_t ${CMAKE_CURRENT_SOURCE_DIR}/matmul.comp -o matmul_int16.spv
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/matmul.comp
    COMMENT "Compiling int16 matmul shader"
)

add_custom_command(
    OUTPUT matmul_fp32.spv
    COMMAND glslc -DDATA_TYPE=float ${CMAKE_CURRENT_SOURCE_DIR}/matmul.comp -o matmul_fp32.spv
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/matmul.comp
    COMMENT "Compiling fp32 matmul shader"
)

add_custom_command(
    OUTPUT matmul_int32.spv
    COMMAND glslc -DDATA_TYPE=int ${CMAKE_CURRENT_SOURCE_DIR}/matmul.comp -o matmul_int32.spv
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/matmul.comp
    COMMENT "Compiling int32 matmul shader"
)

# === Element-wise operation shaders (add/sub/div/mad) ===
# OP_TYPE: 0=add, 1=sub, 2=div, 3=mad

set(ELEM_OPS add sub div mad)
set(ELEM_OP_IDS 0 1 2 3)
set(DATA_TYPES fp16 int16 fp32 int32)
set(DATA_TYPE_GLSL float16_t int16_t float int)

set(ALL_ELEM_SPVS "")

foreach(op_idx RANGE 0 3)
    list(GET ELEM_OPS ${op_idx} op_name)
    list(GET ELEM_OP_IDS ${op_idx} op_id)
    foreach(dt_idx RANGE 0 3)
        list(GET DATA_TYPES ${dt_idx} dt_name)
        list(GET DATA_TYPE_GLSL ${dt_idx} dt_glsl)
        set(spv_name "${op_name}_${dt_name}.spv")
        add_custom_command(
            OUTPUT ${spv_name}
            COMMAND glslc -DDATA_TYPE=${dt_glsl} -DOP_TYPE=${op_id} ${CMAKE_CURRENT_SOURCE_DIR}/elemop.comp -o ${spv_name}
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/elemop.comp
            COMMENT "Compiling ${op_name} ${dt_name} shader"
        )
        list(APPEND ALL_ELEM_SPVS ${spv_name})
    endforeach()
endforeach()

add_executable(matmul_vk main.c args.c
    matmul_fp16.spv matmul_int16.spv matmul_fp32.spv matmul_int32.spv
    ${ALL_ELEM_SPVS}
)

target_include_directories(matmul_vk PRIVATE ${Vulkan_INCLUDE_DIRS})
target_link_libraries(matmul_vk PRIVATE ${Vulkan_LIBRARIES})

# Copy spv to build dir if needed (it's already there because of add_custom_command output)
