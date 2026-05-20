#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include <vulkan/vulkan.h>
#include "args.h"
#include <sys/stat.h>
#include <sys/types.h>

//it's not a WG size from GPU
#define WORKGROUP_SIZE 16
#define ELEMOP_WORKGROUP_SIZE 256

double get_time_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void get_device_name(uint32_t device_index, char* out_name, size_t max_len) {
    snprintf(out_name, max_len, "Device %u", device_index);
    VkInstanceCreateInfo createInfo = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
    };
    VkInstance instance;
    if (vkCreateInstance(&createInfo, NULL, &instance) == VK_SUCCESS) {
        uint32_t count = 0;
        vkEnumeratePhysicalDevices(instance, &count, NULL);
        if (count > device_index) {
            VkPhysicalDevice* devices = malloc(sizeof(VkPhysicalDevice) * count);
            if (vkEnumeratePhysicalDevices(instance, &count, devices) == VK_SUCCESS) {
                VkPhysicalDeviceProperties props;
                vkGetPhysicalDeviceProperties(devices[device_index], &props);
                snprintf(out_name, max_len, "%s", props.deviceName);
            }
            free(devices);
        }
        vkDestroyInstance(instance, NULL);
    }
}

static const char* get_operator_name(OperatorType op) {
    switch (op) {
        case OP_MUL: return "MUL";
        case OP_ADD: return "ADD";
        case OP_SUB: return "SUB";
        case OP_DIV: return "DIV";
        case OP_MAD: return "MAD";
        default: return "UNKNOWN";
    }
}

static const char* get_datatype_name(DataType dt) {
    switch (dt) {
        case DT_FP16: return "FP16";
        case DT_INT16: return "INT16";
        case DT_FP32: return "FP32";
        case DT_INT32: return "INT32";
        default: return "UNKNOWN";
    }
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
}

// core benchmarking logic extracted into a reusable function
double* run_benchmark_on_device(AppArgs args, uint32_t target_device, int silent, uint32_t* out_count) {
    uint32_t N_SIZE = args.matrix_size;

    VkResult res;

    VkApplicationInfo appInfo = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "vkmdbench",
        .apiVersion = VK_API_VERSION_1_1,
    };

    VkInstanceCreateInfo createInfo = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &appInfo,
    };

    const char* type_str = "FP16";
    const char* perf_label = "GFLOPS";
    size_t elementSize = 2;
    const char* shaderFile = "shaders/matmul_fp16.spv";
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
        snprintf(shaderFileBuf, sizeof(shaderFileBuf), "shaders/%s_%s.spv", op_names[args.operator_type], dt_names[args.data_type]);
    } else {
        const char* dt_names[] = {"fp16", "int16", "fp32", "int32"};
        snprintf(shaderFileBuf, sizeof(shaderFileBuf), "shaders/matmul_%s.spv", dt_names[args.data_type]);
    }
    shaderFile = shaderFileBuf;

    VkInstance instance;
    res = vkCreateInstance(&createInfo, NULL, &instance);
    if (res != VK_SUCCESS) {
        fprintf(stderr, "Failed to create Vulkan instance (res: %d)\n", res);
        return NULL;
    }

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, NULL);
    if (deviceCount == 0) {
        fprintf(stderr, "No Vulkan devices found\n");
        vkDestroyInstance(instance, NULL);
        return NULL;
    }
    
    VkPhysicalDevice* devices = malloc(sizeof(VkPhysicalDevice) * deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices);
    
    if (args.list_devices) {
        if (!silent) {
            printf("Available Vulkan devices:\n");
            for (uint32_t i = 0; i < deviceCount; i++) {
                VkPhysicalDeviceProperties props;
                vkGetPhysicalDeviceProperties(devices[i], &props);
                printf("[%u] %s\n", i, props.deviceName);
            }
        }
        free(devices);
        vkDestroyInstance(instance, NULL);
        if (out_count) *out_count = 0;
        return NULL;
    }

    if (target_device >= deviceCount) {
        fprintf(stderr, "Error: Selected device index %u is out of bounds (found %u devices)\n", target_device, deviceCount);
        if (!silent) {
            printf("Available devices:\n");
            for (uint32_t i = 0; i < deviceCount; i++) {
                VkPhysicalDeviceProperties props;
                vkGetPhysicalDeviceProperties(devices[i], &props);
                printf("[%u] %s\n", i, props.deviceName);
            }
        }
        free(devices);
        vkDestroyInstance(instance, NULL);
        return NULL;
    }

    VkPhysicalDevice physicalDevice = devices[target_device];

    VkPhysicalDeviceProperties deviceProps;
    vkGetPhysicalDeviceProperties(physicalDevice, &deviceProps);
    if (!silent) {
        printf("Running benchmarks on Device %u (%s) - %s %s...\n", target_device, deviceProps.deviceName, op_str, type_str);
    }

    VkPhysicalDeviceMemoryProperties deviceMemProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &deviceMemProps);
    size_t totalDeviceLocalMemory = 0;
    for (uint32_t i = 0; i < deviceMemProps.memoryHeapCount; i++) {
        if (deviceMemProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            totalDeviceLocalMemory += deviceMemProps.memoryHeaps[i].size;
        }
    }

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
        free(queueFamilies);
        free(devices);
        vkDestroyInstance(instance, NULL);
        return NULL;
    }

    VkQueue queue;
    vkGetDeviceQueue(device, computeQueueFamilyIndex, 0, &queue);

    size_t matrixSize = (size_t)N_SIZE * N_SIZE * elementSize;
    VkBuffer bufferA, bufferB, bufferC;
    VkDeviceMemory memoryA, memoryB, memoryC;

    createBuffer(device, physicalDevice, matrixSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &bufferA, &memoryA);
    createBuffer(device, physicalDevice, matrixSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &bufferB, &memoryB);
    createBuffer(device, physicalDevice, matrixSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &bufferC, &memoryC);

    VkBuffer stagingA, stagingB, stagingC;
    VkDeviceMemory stagingMemoryA, stagingMemoryB, stagingMemoryC;
    createBuffer(device, physicalDevice, matrixSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &stagingA, &stagingMemoryA);
    createBuffer(device, physicalDevice, matrixSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &stagingB, &stagingMemoryB);
    createBuffer(device, physicalDevice, matrixSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &stagingC, &stagingMemoryC);

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
    vkUnmapMemory(device, stagingMemoryA);
    vkUnmapMemory(device, stagingMemoryB);

    FILE* f_shader = fopen(shaderFile, "rb");
    if (!f_shader) {
        fprintf(stderr, "Error: Could not open %s. Make sure it is in the current directory.\n", shaderFile);
        // clean up basic resources
        vkDestroyBuffer(device, bufferA, NULL); vkFreeMemory(device, memoryA, NULL);
        vkDestroyBuffer(device, bufferB, NULL); vkFreeMemory(device, memoryB, NULL);
        vkDestroyBuffer(device, bufferC, NULL); vkFreeMemory(device, memoryC, NULL);
        vkDestroyBuffer(device, stagingA, NULL); vkFreeMemory(device, stagingMemoryA, NULL);
        vkDestroyBuffer(device, stagingB, NULL); vkFreeMemory(device, stagingMemoryB, NULL);
        vkDestroyBuffer(device, stagingC, NULL); vkFreeMemory(device, stagingMemoryC, NULL);
        vkDestroyDevice(device, NULL);
        free(queueFamilies);
        free(devices);
        vkDestroyInstance(instance, NULL);
        return NULL;
    }
    fseek(f_shader, 0, SEEK_END);
    long length = ftell(f_shader);
    fseek(f_shader, 0, SEEK_SET);
    uint32_t* code = malloc(length);
    if (!code) {
        fprintf(stderr, "Failed to allocate memory for shader code\n");
        fclose(f_shader);
        return NULL;
    }
    fread(code, 1, length, f_shader);
    fclose(f_shader);

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
    // CSV saving logic is only handled in non-dual-bench mode here.
    // If dual bench is requested, main() will do the side-by-side CSV writing.
    if (args.save_csv && !args.multi_bench) {
        FILE* pipe = popen("lact cli profile", "r");
        char filename[256] = "output/results.csv";
        if (pipe) {
            char buffer[128];
            if (fgets(buffer, 128, pipe)) {
                size_t len = strlen(buffer);
                while (len > 0 && (buffer[len-1] == '\n' || buffer[len-1] == '\r')) {
                     buffer[--len] = '\0';
                }
                snprintf(filename, sizeof(filename), "output/%s_%s.csv", buffer, type_str);
            }
            pclose(pipe);
        }
        csv_file = fopen(filename, "w");
        if (csv_file) {
            fprintf(csv_file, "Operator: %s\n", op_str);
            fprintf(csv_file, "DataType: %s\n", type_str);
            fprintf(csv_file, "LACT Profile: %s\n", args.lact_profile);
            fprintf(csv_file, "Matrix Size,Performance (%s)\n", perf_label);
            if (!silent) printf("Saving results to %s\n", filename);
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

    if (!silent) {
        printf("LACT Profile: %s\n", args.lact_profile);
        printf("Benchmarking %s %s from %ux%u to %ux%u with step %u...\n\n", op_str, type_str, args.matrix_start_size, args.matrix_start_size, N_SIZE, N_SIZE, args.matrix_step_size);
        printf("| Matrix Size | Perf, %s |\n", perf_label);
        printf("|-------------|--------------|\n");
    }

    // Allocate result array
    uint32_t step_count = 0;
    for (uint32_t current_n = args.matrix_start_size; ; ) {
        step_count++;
        if (current_n >= N_SIZE) break;
        uint32_t next_n = current_n + args.matrix_step_size;
        if (next_n > N_SIZE) next_n = N_SIZE;
        current_n = next_n;
    }
    double* results = malloc(sizeof(double) * step_count);
    uint32_t step_idx = 0;

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

        // Run for approximately 1 second
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

        if (!silent) {
            printf("| %4u x %-4u | %12.3f |\n", current_n, current_n, gops);
            fflush(stdout);
        }

        if (step_idx < step_count) {
            results[step_idx++] = gops;
        }

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
    if (!silent) {
        printf("\n");
    }
    if (csv_file) fclose(csv_file);

    // Clean up Vulkan resources completely to prevent memory/resource leaks
    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    vkDestroyCommandPool(device, commandPool, NULL);
    vkDestroyPipeline(device, pipeline, NULL);
    vkDestroyPipelineLayout(device, pipelineLayout, NULL);
    vkDestroyShaderModule(device, shaderModule, NULL);
    vkDestroyDescriptorPool(device, descriptorPool, NULL);
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, NULL);
    
    vkDestroyBuffer(device, bufferA, NULL); vkFreeMemory(device, memoryA, NULL);
    vkDestroyBuffer(device, bufferB, NULL); vkFreeMemory(device, memoryB, NULL);
    vkDestroyBuffer(device, bufferC, NULL); vkFreeMemory(device, memoryC, NULL);
    vkDestroyBuffer(device, stagingA, NULL); vkFreeMemory(device, stagingMemoryA, NULL);
    vkDestroyBuffer(device, stagingB, NULL); vkFreeMemory(device, stagingMemoryB, NULL);
    vkDestroyBuffer(device, stagingC, NULL); vkFreeMemory(device, stagingMemoryC, NULL);
    
    vkDestroyDevice(device, NULL);
    free(code);
    free(queueFamilies);
    free(devices);
    vkDestroyInstance(instance, NULL);

    if (out_count) *out_count = step_idx;
    return results;
}

int main(int argc, char** argv) {
    AppArgs args = parse_args(argc, argv);
    mkdir("output", 0777);

    if (args.list_devices) {
        uint32_t count = 0;
        run_benchmark_on_device(args, 0, 0, &count);
        return 0;
    }

    if (!args.multi_bench) {
        // Single device benchmark run for each operator and data type
        for (uint32_t op_idx = 0; op_idx < args.multi_operator_count; op_idx++) {
            for (uint32_t dt_idx = 0; dt_idx < args.multi_data_type_count; dt_idx++) {
                AppArgs temp_args = args;
                temp_args.operator_type = args.multi_operators[op_idx];
                temp_args.data_type = args.multi_data_types[dt_idx];
                uint32_t count = 0;
                double* res = run_benchmark_on_device(temp_args, temp_args.device_index, 0, &count);
                if (res) {
                    free(res);
                }
            }
        }
        return 0;
    }

    // Multi Benchmarking Mode
    printf("--- Starting Multi Device Benchmarking ---\n");
    
    for (uint32_t op_idx = 0; op_idx < args.multi_operator_count; op_idx++) {
        for (uint32_t dt_idx = 0; dt_idx < args.multi_data_type_count; dt_idx++) {
            AppArgs temp_args = args;
            temp_args.operator_type = args.multi_operators[op_idx];
            temp_args.data_type = args.multi_data_types[dt_idx];

            double* results[32] = {NULL};
            uint32_t counts[32] = {0};
            char device_names[32][256];

            for (uint32_t d = 0; d < temp_args.multi_device_count; d++) {
                uint32_t dev_idx = temp_args.multi_devices[d];
                const char* profile = (d < temp_args.multi_profile_count) ? temp_args.multi_profiles[d] : "default";
                get_device_name(dev_idx, device_names[d], sizeof(device_names[d]));

                if (d > 0) printf("\n");

                if (strcmp(profile, "default") != 0) {
                    char cmd[256];
                    snprintf(cmd, sizeof(cmd), "lact cli -g %u profile set \"%s\"", dev_idx, profile);
                    printf("Configuring Device %u profile: \"%s\"\n", dev_idx, profile);
                    fflush(stdout);
                    int sys_res = system(cmd);
                    if (sys_res != 0) {
                        fprintf(stderr, "Warning: lact command returned non-zero status %d. Continuing...\n", sys_res);
                    }
                } else {
                    printf("Configuring Device %u profile: \"default\" (skipping)\n", dev_idx);
                    fflush(stdout);
                }
                
                printf("Running benchmarks on Device %u (%s) - %s %s...\n", dev_idx, device_names[d], get_operator_name(temp_args.operator_type), get_datatype_name(temp_args.data_type));
                fflush(stdout);
                
                results[d] = run_benchmark_on_device(temp_args, dev_idx, 1, &counts[d]);
                if (!results[d]) {
                    fprintf(stderr, "Error: Failed to run benchmarks on Device %u\n", dev_idx);
                    for (uint32_t k = 0; k < d; k++) {
                        if (results[k]) free(results[k]);
                    }
                    return 1;
                }

                if (d > 0 && counts[d] != counts[0]) {
                    fprintf(stderr, "Error: Mismatched result sizes between devices (%u vs %u)\n", counts[0], counts[d]);
                    for (uint32_t k = 0; k <= d; k++) {
                        if (results[k]) free(results[k]);
                    }
                    return 1;
                }
            }

            // Format headers
            const char* perf_label = "GFLOPS";
            if (temp_args.data_type == DT_INT16 || temp_args.data_type == DT_INT32) {
                perf_label = "GOPS";
            }

            const char* type_str = "FP16";
            switch(temp_args.data_type) {
                case DT_FP16: type_str = "FP16"; break;
                case DT_INT16: type_str = "INT16"; break;
                case DT_FP32: type_str = "FP32"; break;
                case DT_INT32: type_str = "INT32"; break;
            }

            const char* op_str = "MUL";
            switch(temp_args.operator_type) {
                case OP_MUL: op_str = "MUL"; break;
                case OP_ADD: op_str = "ADD"; break;
                case OP_SUB: op_str = "SUB"; break;
                case OP_DIV: op_str = "DIV"; break;
                case OP_MAD: op_str = "MAD"; break;
            }

            // Display Markdown table side-by-side
            printf("\nLACT Profiles: ");
            for (uint32_t d = 0; d < temp_args.multi_device_count; d++) {
                const char* profile = (d < temp_args.multi_profile_count) ? temp_args.multi_profiles[d] : "default";
                printf("%s%s", profile, (d + 1 < temp_args.multi_device_count) ? ", " : "");
            }
            printf("\n### Multi Benchmark Results: %s %s\n\n", op_str, type_str);
            printf("| Matrix Size");
            for (uint32_t d = 0; d < temp_args.multi_device_count; d++) {
                printf(" | Device %u (%s) [%s]", temp_args.multi_devices[d], device_names[d], perf_label);
            }
            printf(" |\n");

            printf("| :---");
            for (uint32_t d = 0; d < temp_args.multi_device_count; d++) {
                printf(" | :---:");
            }
            printf(" |\n");

            uint32_t n = temp_args.matrix_start_size;
            for (uint32_t i = 0; i < counts[0]; i++) {
                printf("| %u x %u", n, n);
                for (uint32_t d = 0; d < temp_args.multi_device_count; d++) {
                    printf(" | %.3f", results[d][i]);
                }
                printf(" |\n");
                
                uint32_t next_n = n + temp_args.matrix_step_size;
                if (next_n > temp_args.matrix_size) next_n = temp_args.matrix_size;
                if (n >= temp_args.matrix_size) break;
                n = next_n;
            }
            printf("\n");

            // Save CSV side-by-side if required
            if (temp_args.save_csv) {
                char filename[256];
                snprintf(filename, sizeof(filename), "output/multi_bench_%s_%s.csv", op_str, type_str);
                FILE* csv_file = fopen(filename, "w");
                if (csv_file) {
                    fprintf(csv_file, "Operator: %s\n", op_str);
                    fprintf(csv_file, "DataType: %s\n", type_str);
                    fprintf(csv_file, "LACT Profiles: ");
                    for (uint32_t d = 0; d < temp_args.multi_device_count; d++) {
                        const char* profile = (d < temp_args.multi_profile_count) ? temp_args.multi_profiles[d] : "default";
                        fprintf(csv_file, "%s%s", profile, (d + 1 < temp_args.multi_device_count) ? ", " : "");
                    }
                    fprintf(csv_file, "\n");
                    fprintf(csv_file, "Matrix Size");
                    for (uint32_t d = 0; d < temp_args.multi_device_count; d++) {
                        fprintf(csv_file, ",Device %u (%s) [%s]", temp_args.multi_devices[d], device_names[d], perf_label);
                    }
                    fprintf(csv_file, "\n");
                    
                    n = temp_args.matrix_start_size;
                    for (uint32_t i = 0; i < counts[0]; i++) {
                        fprintf(csv_file, "%u", n);
                        for (uint32_t d = 0; d < temp_args.multi_device_count; d++) {
                            fprintf(csv_file, ",%.3f", results[d][i]);
                        }
                        fprintf(csv_file, "\n");
                        
                        uint32_t next_n = n + temp_args.matrix_step_size;
                        if (next_n > temp_args.matrix_size) next_n = temp_args.matrix_size;
                        if (n >= temp_args.matrix_size) break;
                        n = next_n;
                    }
                    fclose(csv_file);
                    printf("Saved side-by-side CSV results to: %s\n", filename);
                } else {
                    fprintf(stderr, "Error: Could not open %s for writing CSV\n", filename);
                }
            }

            for (uint32_t d = 0; d < temp_args.multi_device_count; d++) {
                if (results[d]) free(results[d]);
            }
        }
    }

    return 0;
}
