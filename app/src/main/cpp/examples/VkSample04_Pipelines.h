//
// Created by 字节流动 on 2024/7/11.
//

#ifndef ANDROID_VULKAN_TUTORIALS_VKSAMPLE04_PIPELINES_H
#define ANDROID_VULKAN_TUTORIALS_VKSAMPLE04_PIPELINES_H
#include "vulkanexamplebase.h"
#include "VkSampleBase.h"

class VkSample04_Pipelines : public VkSampleBase {
public:
    VkSample04_Pipelines();
    virtual ~VkSample04_Pipelines();
    void loadTexture();
    void createVertexBuffer();
    void createDescriptors();
    void createPipelines();
    void createUniformBuffers();
    void buildCommandBuffers();

    virtual void updateTransformMatrix(float rotateX, float rotateY, float scaleX, float scaleY);
    virtual void prepare();
    virtual void render();
public:
    // Vertex layout used in this example
    struct Vertex {
        float pos[3];
        float uv[2];
        float normal[3];
    };

    // Vertex buffer and attributes
    struct {
        VkDeviceMemory memory{VK_NULL_HANDLE}; // Handle to the device memory for this buffer
        VkBuffer buffer;                         // Handle to the Vulkan buffer object that the memory is bound to
    } vertices;

    // Index buffer
    struct {
        VkDeviceMemory memory{VK_NULL_HANDLE};
        VkBuffer buffer;
        uint32_t count{0};
    } indices;

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
        glm::mat4 projectionMatrix;
        glm::mat4 modelMatrix;
        glm::mat4 viewMatrix;
        float iTime;
    } uboData;

    VkPipelineLayout pipelineLayout{VK_NULL_HANDLE};
    VkPipeline pipeline{VK_NULL_HANDLE}, pipeline2;
    VkDescriptorSetLayout descriptorSetLayout{VK_NULL_HANDLE};
    VkDescriptorSet descriptorSet{ VK_NULL_HANDLE };

    vks::Texture2D* pTexture2D;
};
#endif //ANDROID_VULKAN_TUTORIALS_VKSAMPLE04_PIPELINES_H