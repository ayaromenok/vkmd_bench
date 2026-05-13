#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include <vulkan/vulkan.h>
#include "args.h"

#define WORKGROUP_SIZE 16

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

void createBuffer(VkDevice device, VkPhysicalDevice physicalDevice, size_t size, VkBuffer* buffer, VkDeviceMemory* memory) {
    VkBufferCreateInfo bufferInfo = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    vkCreateBuffer(device, &bufferInfo, NULL, buffer);

    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(device, *buffer, &memReqs);

    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);

    uint32_t memoryTypeIndex = (uint32_t)-1;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((memReqs.memoryTypeBits & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))) {
            memoryTypeIndex = i;
            break;
        }
    }

    VkMemoryAllocateInfo allocInfo = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = memReqs.size,
        .memoryTypeIndex = memoryTypeIndex,
    };
    vkAllocateMemory(device, &allocInfo, NULL, memory);
    vkBindBufferMemory(device, *buffer, *memory, 0);
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
        case DT_FP16: type_str = "FP16"; perf_label = "GFLOPS"; elementSize = 2; shaderFile = "matmul_fp16.spv"; break;
        case DT_INT16: type_str = "INT16"; perf_label = "GOPS"; elementSize = 2; shaderFile = "matmul_int16.spv"; break;
        case DT_FP32: type_str = "FP32"; perf_label = "GFLOPS"; elementSize = 4; shaderFile = "matmul_fp32.spv"; break;
        case DT_INT32: type_str = "INT32"; perf_label = "GOPS"; elementSize = 4; shaderFile = "matmul_int32.spv"; break;
    }

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
    printf("Matrix size: %u x %u\n", N_SIZE, N_SIZE);

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
    vkCreateDevice(physicalDevice, &deviceCreateInfo, NULL, &device);

    VkQueue queue;
    vkGetDeviceQueue(device, computeQueueFamilyIndex, 0, &queue);

    size_t matrixSize = (size_t)N_SIZE * N_SIZE * elementSize;
    VkBuffer bufferA, bufferB, bufferC;
    VkDeviceMemory memoryA, memoryB, memoryC;

    createBuffer(device, physicalDevice, matrixSize, &bufferA, &memoryA);
    createBuffer(device, physicalDevice, matrixSize, &bufferB, &memoryB);
    createBuffer(device, physicalDevice, matrixSize, &bufferC, &memoryC);

    void *dataA_mapped, *dataB_mapped;
    vkMapMemory(device, memoryA, 0, matrixSize, 0, &dataA_mapped);
    vkMapMemory(device, memoryB, 0, matrixSize, 0, &dataB_mapped);
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

    printf("Benchmarking %s from 32x32 to %ux%u with step 32...\n\n", type_str, N_SIZE, N_SIZE);
    printf("| Matrix Size | Perf, %s |\n", perf_label);
    printf("|-------------|--------------|\n");

    VkSubmitInfo submitInfo = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &commandBuffer,
    };

    for (uint32_t current_n = 32; ; ) {
        if (current_n > N_SIZE) break;
        // Record command buffer for current size
        VkCommandBufferBeginInfo beginInfo = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        vkBeginCommandBuffer(commandBuffer, &beginInfo);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, NULL);
        PushConstants pc = {current_n, current_n, current_n};
        vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pc);
        vkCmdDispatch(commandBuffer, (current_n + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE, (current_n + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE, 1);
        vkEndCommandBuffer(commandBuffer);

        // Warm-up
        vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(queue);

        // Run for approximately 2 seconds
        uint32_t count = 0;
        double start_time = get_time_sec();
        while (get_time_sec() - start_time < 2.0) {
            vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
            vkQueueWaitIdle(queue);
            count++;
        }
        double end_time = get_time_sec();
        double total_time = end_time - start_time;
        double avg_time = total_time / count;
        double gops = (2.0 * (double)current_n * current_n * current_n) / (avg_time * 1e9);

        printf("| %4u x %-4u | %12.3f |\n", current_n, current_n, gops);
        fflush(stdout);

        if (csv_file) {
            fprintf(csv_file, "%u,%f\n", current_n, gops);
            fflush(csv_file);
        }

        if (current_n >= N_SIZE) break;
        uint32_t next_n = current_n + 32;
        if (next_n > N_SIZE) next_n = N_SIZE;
        current_n = next_n;
        
        sleep(2);
    }
    printf("\n");
    if (csv_file) fclose(csv_file);

    void* dataC_mapped;
    vkMapMemory(device, memoryC, 0, matrixSize, 0, &dataC_mapped);
    if (args.data_type == DT_FP16) {
        uint16_t* hC = (uint16_t*)dataC_mapped;
        printf("Result [0,0]: %f\n", float16_to_float32(hC[0]));
        printf("Expected [0,0]: %f\n", (float)N_SIZE * 1.0f * 2.0f);
    } else if (args.data_type == DT_INT16) {
        int16_t* iC = (int16_t*)dataC_mapped;
        printf("Result [0,0]: %d\n", (int)iC[0]);
        printf("Expected [0,0]: %d\n", (int)N_SIZE * 1 * 2);
    } else if (args.data_type == DT_FP32) {
        float* fC = (float*)dataC_mapped;
        printf("Result [0,0]: %f\n", fC[0]);
        printf("Expected [0,0]: %f\n", (float)N_SIZE * 1.0f * 2.0f);
    } else if (args.data_type == DT_INT32) {
        int32_t* iC = (int32_t*)dataC_mapped;
        printf("Result [0,0]: %d\n", (int)iC[0]);
        printf("Expected [0,0]: %d\n", (int)N_SIZE * 1 * 2);
    }
    vkUnmapMemory(device, memoryC);

    return 0;
}
