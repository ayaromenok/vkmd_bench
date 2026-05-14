#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include <vulkan/vulkan.h>
#include "args.h"

//it's not a WG size from GPU
#define WORKGROUP_SIZE 16
#define ELEMOP_WORKGROUP_SIZE 256

double get_time_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Simple float32 to float16 conversion (IEEE 754)
uint16_t float32_to_float16(float f) {
    uint32_t i = *(uint32_t*)&f;
    uint32_t s = (i >> 16) & 0x8000;
    uint32_t e = ((i >> 23) & 0xff);
    uint32_t m = i & 0x7fffff;

    if (e == 0) return s; // flush to zero
    if (e == 0xff) return s | 0x7c00 | (m ? 1 : 0); // inf/nan

    int new_e = (int)e - 127 + 15;
    if (new_e <= 0) return s; // flush to zero
    if (new_e >= 31) return s | 0x7c00; // inf

    return (uint16_t)(s | (new_e << 10) | (m >> 13));
}

float float16_to_float32(uint16_t h) {
    uint32_t s = (h & 0x8000) << 16;
    uint32_t e = (h & 0x7c00) >> 10;
    uint32_t m = (h & 0x03ff) << 13;

    if (e == 0) {
        if (m == 0) return *(float*)&s;
        while (!(m & 0x00800000)) { m <<= 1; e--; }
        e++; m &= ~0x00800000;
    } else if (e == 0x1f) {
        e = 0xff;
    } else {
        e += (127 - 15);
    }
    uint32_t i = s | (e << 23) | m;
    return *(float*)&i;
}

typedef struct {
    uint32_t M, N, K;
} PushConstants;

void createBuffer(VkDevice device, VkPhysicalDevice physicalDevice, size_t size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer* buffer, VkDeviceMemory* memory) {
    VkBufferCreateInfo bufferInfo = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VkResult resBuf = vkCreateBuffer(device, &bufferInfo, NULL, buffer);
    if (resBuf != VK_SUCCESS) {
        fprintf(stderr, "Failed to create buffer (res: %d)\n", resBuf);
        exit(1);
    }

    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(device, *buffer, &memReqs);

    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);

    uint32_t memoryTypeIndex = (uint32_t)-1;
    VkResult resAlloc = VK_ERROR_OUT_OF_DEVICE_MEMORY;

    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((memReqs.memoryTypeBits & (1 << i)) && 
            (memProps.memoryTypes[i].propertyFlags & properties) == properties) {
            
            VkMemoryAllocateInfo allocInfo = {
                .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                .allocationSize = memReqs.size,
                .memoryTypeIndex = i,
            };
            resAlloc = vkAllocateMemory(device, &allocInfo, NULL, memory);
            if (resAlloc == VK_SUCCESS) {
                memoryTypeIndex = i;
                break;
            }
        }
    }

    if (resAlloc != VK_SUCCESS) {
        fprintf(stderr, "Failed to allocate memory (res: %d)\n", resAlloc);
        exit(1);
    }
    vkBindBufferMemory(device, *buffer, *memory, 0);

    const char* memTypeStr = "UNKNOWN";
    if (properties & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
        memTypeStr = "DEVICE LOCAL";
    } else if (properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        memTypeStr = "HOST VISIBLE";
    }
    //printf("Buffer allocated in %s memory (size: %zu MB)\n", memTypeStr, size/1024/1024);
}

int main(int argc, char** argv) {
    AppArgs args = parse_args(argc, argv);
    uint32_t N_SIZE = args.matrix_size;

    VkResult res;

    VkApplicationInfo appInfo = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "Vulkan MatMul FP16",
        .apiVersion = VK_API_VERSION_1_1,
    };

    VkInstanceCreateInfo createInfo = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &appInfo,
    };

    const char* type_str = "FP16";
    const char* perf_label = "GFLOPS";
    size_t elementSize = 2;
    const char* shaderFile = "matmul_fp16.spv";
    switch(args.data_type) {
        case DT_FP16: type_str = "FP16"; perf_label = "GFLOPS"; elementSize = 2; break;
        case DT_INT16: type_str = "INT16"; perf_label = "GOPS"; elementSize = 2; break;
        case DT_FP32: type_str = "FP32"; perf_label = "GFLOPS"; elementSize = 4; break;
        case DT_INT32: type_str = "INT32"; perf_label = "GOPS"; elementSize = 4; break;
    }

    const char* op_str = "MUL";
    int is_elemop = 0;  // 0 = matmul (2D dispatch), 1 = element-wise (1D dispatch)
    switch(args.operator_type) {
        case OP_MUL: op_str = "MUL"; is_elemop = 0; break;
        case OP_ADD: op_str = "ADD"; is_elemop = 1; break;
        case OP_SUB: op_str = "SUB"; is_elemop = 1; break;
        case OP_DIV: op_str = "DIV"; is_elemop = 1; break;
        case OP_MAD: op_str = "MAD"; is_elemop = 1; break;
    }

    // Select shader file based on operator + data type
    char shaderFileBuf[64];
    if (is_elemop) {
        const char* op_names[] = {"mul", "add", "sub", "div", "mad"};
        const char* dt_names[] = {"fp16", "int16", "fp32", "int32"};
        snprintf(shaderFileBuf, sizeof(shaderFileBuf), "%s_%s.spv", op_names[args.operator_type], dt_names[args.data_type]);
    } else {
        const char* dt_names[] = {"fp16", "int16", "fp32", "int32"};
        snprintf(shaderFileBuf, sizeof(shaderFileBuf), "matmul_%s.spv", dt_names[args.data_type]);
    }
    shaderFile = shaderFileBuf;

    VkInstance instance;
    vkCreateInstance(&createInfo, NULL, &instance);

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, NULL);
    if (deviceCount == 0) { fprintf(stderr, "No Vulkan devices found\n"); return 1; }
    
    VkPhysicalDevice* devices = malloc(sizeof(VkPhysicalDevice) * deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices);
    
    if (args.list_devices) {
        printf("Available Vulkan devices:\n");
        for (uint32_t i = 0; i < deviceCount; i++) {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(devices[i], &props);
            printf("[%u] %s\n", i, props.deviceName);
        }
        return 0;
    }

    if (args.device_index >= deviceCount) {
        fprintf(stderr, "Error: Selected device index %u is out of bounds (found %u devices)\n", args.device_index, deviceCount);
        printf("Available devices:\n");
        for (uint32_t i = 0; i < deviceCount; i++) {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(devices[i], &props);
            printf("[%u] %s\n", i, props.deviceName);
        }
        return 1;
    }

    VkPhysicalDevice physicalDevice = devices[args.device_index];

    VkPhysicalDeviceProperties deviceProps;
    vkGetPhysicalDeviceProperties(physicalDevice, &deviceProps);
    printf("Using device [%u]: %s\n", args.device_index, deviceProps.deviceName);

    VkPhysicalDeviceMemoryProperties deviceMemProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &deviceMemProps);
    size_t totalDeviceLocalMemory = 0;
    for (uint32_t i = 0; i < deviceMemProps.memoryHeapCount; i++) {
        if (deviceMemProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            totalDeviceLocalMemory += deviceMemProps.memoryHeaps[i].size;
        }
    }
    //printf("Maximum device local memory: %zu MB\n", totalDeviceLocalMemory / 1024 / 1024);

    //printf("Matrix size: %u x %u\n", N_SIZE, N_SIZE);
    //fflush(stdout);

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, NULL);
    VkQueueFamilyProperties* queueFamilies = malloc(sizeof(VkQueueFamilyProperties) * queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies);

    uint32_t computeQueueFamilyIndex = (uint32_t)-1;
    for (uint32_t i = 0; i < queueFamilyCount; i++) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            computeQueueFamilyIndex = i;
            break;
        }
    }

    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = computeQueueFamilyIndex,
        .queueCount = 1,
        .pQueuePriorities = &queuePriority,
    };

    VkPhysicalDeviceFeatures deviceFeatures = {
        .shaderInt16 = VK_TRUE
    };

    VkPhysicalDeviceShaderFloat16Int8Features float16Features = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
        .shaderFloat16 = VK_TRUE,
    };

    VkPhysicalDevice16BitStorageFeatures storage16Features = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
        .pNext = &float16Features,
        .storageBuffer16BitAccess = VK_TRUE,
    };

    const char* deviceExtensions[] = {
        VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME,
        VK_KHR_16BIT_STORAGE_EXTENSION_NAME,
        VK_KHR_STORAGE_BUFFER_STORAGE_CLASS_EXTENSION_NAME
    };

    VkDeviceCreateInfo deviceCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &storage16Features,
        .pEnabledFeatures = &deviceFeatures,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queueCreateInfo,
        .enabledExtensionCount = 3,
        .ppEnabledExtensionNames = deviceExtensions,
    };

    VkDevice device;
    VkResult devRes = vkCreateDevice(physicalDevice, &deviceCreateInfo, NULL, &device);
    if (devRes != VK_SUCCESS) {
        fprintf(stderr, "Failed to create Vulkan device (res: %d)\n", devRes);
        exit(1);
    }

    VkQueue queue;
    vkGetDeviceQueue(device, computeQueueFamilyIndex, 0, &queue);

    size_t matrixSize = (size_t)N_SIZE * N_SIZE * elementSize;
    VkBuffer bufferA, bufferB, bufferC;
    VkDeviceMemory memoryA, memoryB, memoryC;

    //printf("Creating device buffers...\n"); fflush(stdout);
    createBuffer(device, physicalDevice, matrixSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &bufferA, &memoryA);
    createBuffer(device, physicalDevice, matrixSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &bufferB, &memoryB);
    createBuffer(device, physicalDevice, matrixSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &bufferC, &memoryC);

    VkBuffer stagingA, stagingB, stagingC;
    VkDeviceMemory stagingMemoryA, stagingMemoryB, stagingMemoryC;
    //printf("Creating staging buffers...\n"); fflush(stdout);
    createBuffer(device, physicalDevice, matrixSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &stagingA, &stagingMemoryA);
    createBuffer(device, physicalDevice, matrixSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &stagingB, &stagingMemoryB);
    createBuffer(device, physicalDevice, matrixSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &stagingC, &stagingMemoryC);

    //printf("Mapping staging memory...\n"); fflush(stdout);
    void *dataA_mapped = NULL, *dataB_mapped = NULL;
    VkResult resA = vkMapMemory(device, stagingMemoryA, 0, matrixSize, 0, &dataA_mapped);
    VkResult resB = vkMapMemory(device, stagingMemoryB, 0, matrixSize, 0, &dataB_mapped);
    
    if (resA != VK_SUCCESS || dataA_mapped == NULL) {
        fprintf(stderr, "Failed to map staging memory A (res: %d)\n", resA);
        exit(1);
    }
    if (resB != VK_SUCCESS || dataB_mapped == NULL) {
        fprintf(stderr, "Failed to map staging memory B (res: %d)\n", resB);
        exit(1);
    }

    if (args.data_type == DT_FP16) {
        uint16_t* hA = (uint16_t*)dataA_mapped;
        uint16_t* hB = (uint16_t*)dataB_mapped;
        for (uint32_t i = 0; i < N_SIZE * N_SIZE; i++) {
            hA[i] = float32_to_float16(1.0f);
            hB[i] = float32_to_float16(2.0f);
        }
    } else if (args.data_type == DT_INT16) {
        int16_t* iA = (int16_t*)dataA_mapped;
        int16_t* iB = (int16_t*)dataB_mapped;
        for (uint32_t i = 0; i < N_SIZE * N_SIZE; i++) {
            iA[i] = 1;
            iB[i] = 2;
        }
    } else if (args.data_type == DT_FP32) {
        float* fA = (float*)dataA_mapped;
        float* fB = (float*)dataB_mapped;
        for (uint32_t i = 0; i < N_SIZE * N_SIZE; i++) {
            fA[i] = 1.0f;
            fB[i] = 2.0f;
        }
    } else if (args.data_type == DT_INT32) {
        int32_t* iA = (int32_t*)dataA_mapped;
        int32_t* iB = (int32_t*)dataB_mapped;
        for (uint32_t i = 0; i < N_SIZE * N_SIZE; i++) {
            iA[i] = 1;
            iB[i] = 2;
        }
    }
    vkUnmapMemory(device, memoryA);
    vkUnmapMemory(device, memoryB);

    FILE* f = fopen(shaderFile, "rb");
    if (!f) { fprintf(stderr, "Error: Could not open %s. Make sure it is in the current directory.\n", shaderFile); return 1; }
    fseek(f, 0, SEEK_END);
    long length = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint32_t* code = malloc(length);
    if (!code) { fprintf(stderr, "Failed to allocate memory for shader code\n"); return 1; }
    fread(code, 1, length, f);
    fclose(f);

    VkDescriptorSetLayoutBinding bindings[3] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
    };
    VkDescriptorSetLayoutCreateInfo layoutInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3,
        .pBindings = bindings,
    };
    VkDescriptorSetLayout descriptorSetLayout;
    vkCreateDescriptorSetLayout(device, &layoutInfo, NULL, &descriptorSetLayout);

    VkDescriptorPoolSize poolSize = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
    VkDescriptorPoolCreateInfo poolInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &poolSize,
    };
    VkDescriptorPool descriptorPool;
    vkCreateDescriptorPool(device, &poolInfo, NULL, &descriptorPool);

    VkDescriptorSetAllocateInfo setAllocInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptorPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &descriptorSetLayout,
    };
    VkDescriptorSet descriptorSet;
    vkAllocateDescriptorSets(device, &setAllocInfo, &descriptorSet);

    VkDescriptorBufferInfo bufferInfos[3] = {
        {bufferA, 0, matrixSize},
        {bufferB, 0, matrixSize},
        {bufferC, 0, matrixSize},
    };
    VkWriteDescriptorSet writes[3] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = descriptorSet, .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bufferInfos[0]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = descriptorSet, .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bufferInfos[1]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = descriptorSet, .dstBinding = 2, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bufferInfos[2]},
    };
    vkUpdateDescriptorSets(device, 3, writes, 0, NULL);


    VkShaderModuleCreateInfo shaderInfo = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = length,
        .pCode = code,
    };
    VkShaderModule shaderModule;
    vkCreateShaderModule(device, &shaderInfo, NULL, &shaderModule);

    VkPushConstantRange pushConstantRange = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(PushConstants),
    };

    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &descriptorSetLayout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pushConstantRange,
    };
    VkPipelineLayout pipelineLayout;
    vkCreatePipelineLayout(device, &pipelineLayoutInfo, NULL, &pipelineLayout);

    VkComputePipelineCreateInfo pipelineInfo = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shaderModule,
            .pName = "main",
        },
        .layout = pipelineLayout,
    };
    VkPipeline pipeline;
    vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, NULL, &pipeline);

    VkCommandPoolCreateInfo commandPoolInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = computeQueueFamilyIndex,
    };
    VkCommandPool commandPool;
    vkCreateCommandPool(device, &commandPoolInfo, NULL, &commandPool);

    VkCommandBufferAllocateInfo cmdAllocInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = commandPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &cmdAllocInfo, &commandBuffer);


    FILE* csv_file = NULL;
    if (args.save_csv) {
        FILE* pipe = popen("lact cli profile", "r");
        char filename[256] = "results.csv";
        if (pipe) {
            char buffer[128];
            if (fgets(buffer, 128, pipe)) {
                size_t len = strlen(buffer);
                while (len > 0 && (buffer[len-1] == '\n' || buffer[len-1] == '\r')) {
                    buffer[--len] = '\0';
                }
                snprintf(filename, sizeof(filename), "%s_%s.csv", buffer, type_str);
            }
            pclose(pipe);
        }
        csv_file = fopen(filename, "w");
        if (csv_file) {
            fprintf(csv_file, "Matrix Size,Performance (%s)\n", perf_label);
            printf("Saving results to %s\n", filename);
        } else {
            fprintf(stderr, "Warning: Could not open %s for writing\n", filename);
        }
    }

    VkCommandBufferBeginInfo beginInfo = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    VkBufferCopy copyRegion = { .srcOffset = 0, .dstOffset = 0, .size = matrixSize };
    vkCmdCopyBuffer(commandBuffer, stagingA, bufferA, 1, &copyRegion);
    vkCmdCopyBuffer(commandBuffer, stagingB, bufferB, 1, &copyRegion);
    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo copySubmitInfo = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &commandBuffer,
    };
    vkQueueSubmit(queue, 1, &copySubmitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);

    printf("Benchmarking %s %s from %ux%u to %ux%u with step %u...\n\n", op_str, type_str, args.matrix_start_size, args.matrix_start_size, N_SIZE, N_SIZE, args.matrix_step_size);
    printf("| Matrix Size | Perf, %s |\n", perf_label);
    printf("|-------------|--------------|\n");

    VkSubmitInfo submitInfo = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &commandBuffer,
    };

    for (uint32_t current_n = args.matrix_start_size; ; ) {
        if (current_n > N_SIZE) break;
        // Record command buffer for current size
        VkCommandBufferBeginInfo beginInfo = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        vkBeginCommandBuffer(commandBuffer, &beginInfo);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, NULL);
        PushConstants pc = {current_n, current_n, current_n};
        vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pc);
        if (is_elemop) {
            uint32_t total_elements = current_n * current_n;
            vkCmdDispatch(commandBuffer, (total_elements + ELEMOP_WORKGROUP_SIZE - 1) / ELEMOP_WORKGROUP_SIZE, 1, 1);
        } else {
            vkCmdDispatch(commandBuffer, (current_n + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE, (current_n + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE, 1);
        }
        vkEndCommandBuffer(commandBuffer);

        // Warm-up
        vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(queue);

        // Run for approximately 1 seconds
        uint32_t count = 0;
        double start_time = get_time_sec();
        while (get_time_sec() - start_time < 1.0) {
            vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
            vkQueueWaitIdle(queue);
            count++;
        }
        double end_time = get_time_sec();
        double total_time = end_time - start_time;
        double avg_time = total_time / count;
        double gops;
        if (is_elemop) {
            // Element-wise: N*N operations (MAD counts as 2 ops)
            double ops_per_element = (args.operator_type == OP_MAD) ? 2.0 : 1.0;
            gops = (ops_per_element * (double)current_n * current_n) / (avg_time * 1e9);
        } else {
            // MatMul: 2*N^3 operations
            gops = (2.0 * (double)current_n * current_n * current_n) / (avg_time * 1e9);
        }

        printf("| %4u x %-4u | %12.3f |\n", current_n, current_n, gops);
        fflush(stdout);

        if (csv_file) {
            fprintf(csv_file, "%u,%f\n", current_n, gops);
            fflush(csv_file);
        }

        if (current_n >= N_SIZE) break;
        uint32_t next_n = current_n + args.matrix_step_size;
        if (next_n > N_SIZE) next_n = N_SIZE;
        current_n = next_n;
        
        sleep(1);
    }
    printf("\n");
    if (csv_file) fclose(csv_file);

    VkCommandBufferBeginInfo beginInfo2 = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    vkBeginCommandBuffer(commandBuffer, &beginInfo2);
    VkBufferCopy copyRegionC = { .srcOffset = 0, .dstOffset = 0, .size = matrixSize };
    vkCmdCopyBuffer(commandBuffer, bufferC, stagingC, 1, &copyRegionC);
    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo copySubmitInfo2 = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &commandBuffer,
    };
    vkQueueSubmit(queue, 1, &copySubmitInfo2, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);

    void* dataC_mapped;
    vkMapMemory(device, stagingMemoryC, 0, matrixSize, 0, &dataC_mapped);

    // Compute expected values based on operator
    // A values = 1.0, B values = 2.0
    float expected_fp;
    int expected_int;
    switch(args.operator_type) {
        case OP_MUL: expected_fp = (float)N_SIZE * 1.0f * 2.0f; expected_int = (int)N_SIZE * 1 * 2; break;
        case OP_ADD: expected_fp = 1.0f + 2.0f; expected_int = 1 + 2; break;
        case OP_SUB: expected_fp = 1.0f - 2.0f; expected_int = 1 - 2; break;
        case OP_DIV: expected_fp = 1.0f / 2.0f; expected_int = 1 / 2; break;
        case OP_MAD: expected_fp = 1.0f * 2.0f + 1.0f; expected_int = 1 * 2 + 1; break;
    }
/*
    if (args.data_type == DT_FP16) {
        uint16_t* hC = (uint16_t*)dataC_mapped;
        printf("Result [0,0]: %f\n", float16_to_float32(hC[0]));
        printf("Expected [0,0]: %f\n", expected_fp);
    } else if (args.data_type == DT_INT16) {
        int16_t* iC = (int16_t*)dataC_mapped;
        printf("Result [0,0]: %d\n", (int)iC[0]);
        printf("Expected [0,0]: %d\n", expected_int);
    } else if (args.data_type == DT_FP32) {
        float* fC = (float*)dataC_mapped;
        printf("Result [0,0]: %f\n", fC[0]);
        printf("Expected [0,0]: %f\n", expected_fp);
    } else if (args.data_type == DT_INT32) {
        int32_t* iC = (int32_t*)dataC_mapped;
        printf("Result [0,0]: %d\n", (int)iC[0]);
        printf("Expected [0,0]: %d\n", expected_int);
    }
*/
    vkUnmapMemory(device, stagingMemoryC);

    return 0;
}
