//
// Created by 字节流动 on 2024/7/11.
//

#ifndef ANDROID_VULKAN_TUTORIALS_VKSAMPLE11_CUBEMAP_H
#define ANDROID_VULKAN_TUTORIALS_VKSAMPLE11_CUBEMAP_H
#include "vulkanexamplebase.h"
#include "VkSampleBase.h"

class VkSample11_CubeMap : public VkSampleBase {
public:
    VkSample11_CubeMap();
    virtual ~VkSample11_CubeMap();
    void loadTexture();
    void createVertexBuffer();
    void createDescriptors();
    void createPipelines();
    void createUniformBuffers();
    void buildCommandBuffers();
    void loadCubeMap(std::string filename, VkFormat format);

    virtual void updateTransformMatrix(float rotateX, float rotateY, float scaleX, float scaleY);
    virtual void prepare();
    virtual void render();
public:
    // Vertex layout used in this example
    struct Vertex {
        float pos[3];
    };

    // Vertex buffer and attributes
    struct {
        VkDeviceMemory memory{VK_NULL_HANDLE}; // Handle to the device memory for this buffer
        VkBuffer buffer;                         // Handle to the Vulkan buffer object that the memory is bound to
        uint32_t count{0};
    } vertices;

    // Uniform buffer block object
    struct UniformBuffer {
        VkDeviceMemory memory;
        VkBuffer buffer;
        // We keep a pointer to the mapped buffer, so we can easily update it's contents via a memcpy
        uint8_t *mapped{nullptr};
    };
    // We use one UBO per frame, so we can have a frame overlap and make sure that uniforms aren't updated while still in use
    UniformBuffer uniformBuffer;

    // For simplicity we use the same uniform block layout as in the shader:
    //
    //	layout(set = 0, binding = 0) uniform UBO
    //	{
    //		mat4 projectionMatrix;
    //		mat4 modelMatrix;
    //		mat4 viewMatrix;
    //	} ubo;
    //
    // This way we can just memcopy the ubo data to the ubo
    // Note: You should use data types that align with the GPU in order to avoid manual padding (vec4, mat4)
    struct ShaderData {
        glm::mat4 projectionMatrix;
        glm::mat4 modelMatrix;
        glm::mat4 viewMatrix;
    } mvpMatrix;

    VkPipelineLayout pipelineLayout{VK_NULL_HANDLE};
    VkPipeline pipeline{VK_NULL_HANDLE};
    VkDescriptorSetLayout descriptorSetLayout{VK_NULL_HANDLE};
    VkDescriptorSet descriptorSet{ VK_NULL_HANDLE };

    vks::Texture cubeMap;

};
#endif //ANDROID_VULKAN_TUTORIALS_VKSAMPLE11_CUBEMAP_H
