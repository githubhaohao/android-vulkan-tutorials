//
// Created by 字节流动 on 2024/8/11.
//

#ifndef ANDROID_VULKAN_TUTORIALS_VkSample17_MultiThreading_H
#define ANDROID_VULKAN_TUTORIALS_VkSample17_MultiThreading_H
#include "vulkanexamplebase.h"
#include "VkSampleBase.h"
#include "threadpool.hpp"

using namespace glm;
using namespace std;

#define MATH_PI     3.1415926535897932384626433832802
#define ANGLE_SPAN  9
#define RADIUS 0.5
#define RADIAN(angle) ((angle) / 180 * MATH_PI)

class VkSample17_MultiThreading : public VkSampleBase {
public:
    VkSample17_MultiThreading();
    virtual ~VkSample17_MultiThreading();
    void loadTexture();
    void generateSpheres();
    void setupDescriptors();
    void createPipelines();
    float rnd(float range);
    void prepareMultiThreadedRenderer();
    void threadRenderCode(uint32_t threadIndex, uint32_t cmdBufferIndex, VkCommandBufferInheritanceInfo inheritanceInfo);
    void updateCommandBuffers(VkFramebuffer frameBuffer);

    virtual void updateTransformMatrix(float rotateX, float rotateY, float scaleX, float scaleY);
    virtual void prepare();
    virtual void render();
public:
    // Vertex layout used in this example
    struct Vertex {
        vec3 pos;
        vec2 uv;
        vec3 normal;
    };
    vks::Buffer vertexBuffer;
    uint32_t vertexCount{ 0 };

    struct MVPMatrix {
        glm::mat4 projectionMatrix;
        glm::mat4 modelMatrix;
        glm::mat4 viewMatrix;
    } mvpMatrix;

    VkPipeline pipeline{ VK_NULL_HANDLE };
    VkPipelineLayout pipelineLayout{ VK_NULL_HANDLE };
    VkDescriptorSet descriptorSet{ VK_NULL_HANDLE };
    VkDescriptorSetLayout descriptorSetLayout{ VK_NULL_HANDLE };
    vks::Texture2D* pTexture2D;

    VkCommandBuffer primaryCommandBuffer{ VK_NULL_HANDLE };

    // Secondary scene command buffers used to store backdrop and user interface
    struct SecondaryCommandBuffers {
        VkCommandBuffer background{ VK_NULL_HANDLE };
        VkCommandBuffer ui{ VK_NULL_HANDLE };
    } secondaryCommandBuffers;

    // Number of animated objects to be renderer
    // by using threads and secondary command buffers
    uint32_t numObjectsPerThread{ 0 };

    // Multi threaded stuff
    // Max. number of concurrent threads
    uint32_t numThreads{ 0 };

    // Use push constants to update shader
    // parameters on a per-thread base
    struct ThreadPushConstantBlock {
        glm::mat4 mvp;
        glm::vec3 color;
    };

    struct ObjectData {
        glm::mat4 model;
        glm::vec3 pos;
        glm::vec3 rotation;
        float rotationDir;
        float rotationSpeed;
        float scale;
        float deltaT;
        float stateT = 0;
        bool visible = true;
    };

    struct ThreadData {
        VkCommandPool commandPool{ VK_NULL_HANDLE };
        // One command buffer per render object
        std::vector<VkCommandBuffer> commandBuffer;
        // One push constant block per render object
        std::vector<ThreadPushConstantBlock> pushConstBlock;
        // Per object information (position, rotation, etc.)
        std::vector<ObjectData> objectData;
    };
    std::vector<ThreadData> threadData;

    vks::ThreadPool threadPool;

    // Fence to wait for all command buffers to finish before
    // presenting to the swap chain
    VkFence renderFence{ VK_NULL_HANDLE };

    std::default_random_engine rndEngine;
};
#endif //ANDROID_VULKAN_TUTORIALS_VkSample17_MultiThreading_H
