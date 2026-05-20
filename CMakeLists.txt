cmake_minimum_required(VERSION 3.10)
project(vkmdbench C)

find_package(Vulkan REQUIRED)

# Get git hash
execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

# Get commit count
execute_process(
    COMMAND git rev-list --count HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_COUNT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

if(NOT GIT_HASH)
    set(GIT_HASH "unknown")
endif()
if(NOT GIT_COMMIT_COUNT)
    set(GIT_COMMIT_COUNT "0")
endif()

string(TIMESTAMP BUILD_DATETIME "%Y%m%d%H%M")

# === Unified shaders ===
set(OPS mul add sub div mad matmul matadd matsub matdiv matmad)
set(OP_IDS 0 1 2 3 4 5 6 7 8 9)
set(DATA_TYPES fp16 int16 fp32 int32 int8)
set(DATA_TYPE_GLSL float16_t int16_t float int int8_t)
set(DIMS 1 2 3)
set(DIM_NAMES 1d 2d 3d)

set(ALL_SPVS "")

foreach(op_idx RANGE 0 9)
    list(GET OPS ${op_idx} op_name)
    list(GET OP_IDS ${op_idx} op_id)
    foreach(dt_idx RANGE 0 4)
        list(GET DATA_TYPES ${dt_idx} dt_name)
        list(GET DATA_TYPE_GLSL ${dt_idx} dt_glsl)
        foreach(dim_idx RANGE 0 2)
            list(GET DIMS ${dim_idx} dim_val)
            list(GET DIM_NAMES ${dim_idx} dim_name)
            set(spv_name "shaders/${op_name}_${dt_name}_${dim_name}.spv")
            add_custom_command(
                OUTPUT ${spv_name}
                COMMAND ${CMAKE_COMMAND} -E make_directory shaders
                COMMAND glslc -DDATA_TYPE=${dt_glsl} -DOP_TYPE=${op_id} -DDIMENSION=${dim_val} ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/shader.comp -o ${spv_name}
                DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/shader.comp
                COMMENT "Compiling ${op_name} ${dt_name} ${dim_name} shader"
            )
            list(APPEND ALL_SPVS ${spv_name})
        endforeach()
    endforeach()
endforeach()

add_executable(vkmdbench src/main.c src/args.c
    ${ALL_SPVS}
)

target_include_directories(vkmdbench PRIVATE ${Vulkan_INCLUDE_DIRS})
target_link_libraries(vkmdbench PRIVATE ${Vulkan_LIBRARIES})

target_compile_definitions(vkmdbench PRIVATE
    APP_VERSION_HASH="${GIT_HASH}"
    APP_VERSION_COMMIT_COUNT="${GIT_COMMIT_COUNT}"
    APP_VERSION_BUILD_DATETIME="${BUILD_DATETIME}"
)

# Copy spv to build dir if needed (it's already there because of add_custom_command output)
