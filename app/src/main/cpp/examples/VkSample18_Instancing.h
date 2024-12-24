//
// Created by 字节流动 on 2024/8/11.
//

#ifndef ANDROID_VULKAN_TUTORIALS_VkSample18_Instancing_H
#define ANDROID_VULKAN_TUTORIALS_VkSample18_Instancing_H
#include "vulkanexamplebase.h"
#include "VkSampleBase.h"

using namespace glm;
using namespace std;

#define MATH_PI     3.1415926535897932384626433832802
#define ANGLE_SPAN  9
#define RADIUS 0.5
#define RADIAN(angle) ((angle) / 180 * MATH_PI)
#define INSTANCE_COUNT 1024
class VkSample18_Instancing : public VkSampleBase {
public:
    VkSample18_Instancing();
    virtual ~VkSample18_Instancing();
    void loadTexture();
    void generateSpheres();
    void createUniformBuffers();
    void setupDescriptors();
    void createPipelines();
    float rnd(float range);
    void prepareInstanceData();
    void updateInstanceData();
    void buildCommandBuffers();

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

    // Uniform buffer block object
    struct UniformBuffer {
        VkDeviceMemory memory;
        VkBuffer buffer;
        // We keep a pointer to the mapped buffer, so we can easily update it's contents via a memcpy
        uint8_t *mapped{nullptr};
    };
    // We use one UBO per frame, so we can have a frame overlap and make sure that uniforms aren't updated while still in use
    UniformBuffer uniformBuffer;

    struct UBO {
        glm::mat4 mvpMatrix;
    } uboData;

    VkPipeline pipeline{ VK_NULL_HANDLE };
    VkPipelineLayout pipelineLayout{ VK_NULL_HANDLE };
    VkDescriptorSet descriptorSet{ VK_NULL_HANDLE };
    VkDescriptorSetLayout descriptorSetLayout{ VK_NULL_HANDLE };
    vks::Texture2D* pTexture2D;

    // We provide position, rotation and scale per mesh instance
    struct InstanceData {
        glm::vec3 pos;
        glm::vec3 rot;
        float scale{ 0.0f };
        glm::vec3 col;
    };
    // Contains the instanced data
    vks::Buffer instanceBuffer;
    std::vector<InstanceData> instanceData;

    // Fence to wait for all command buffers to finish before
    // presenting to the swap chain
    VkFence renderFence{ VK_NULL_HANDLE };

    std::default_random_engine rndEngine;

};
#endif //ANDROID_VULKAN_TUTORIALS_VkSample18_Instancing_H
