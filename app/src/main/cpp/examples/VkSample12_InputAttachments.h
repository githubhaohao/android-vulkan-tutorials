//
// Created by 字节流动 on 2025/7/11.
//

#ifndef ANDROID_VULKAN_TUTORIALS_VKSAMPLE12_INPUTATTACHMENTS_H
#define ANDROID_VULKAN_TUTORIALS_VKSAMPLE12_INPUTATTACHMENTS_H
#include "vulkanexamplebase.h"
#include "VkSampleBase.h"
struct FrameBufferAttachment {
    VkImage image{ VK_NULL_HANDLE };
    VkDeviceMemory memory{ VK_NULL_HANDLE };
    VkImageView view{ VK_NULL_HANDLE };
    VkFormat format;
};
struct Attachments {
    FrameBufferAttachment color, depth;
};
class VkSample12_InputAttachments : public VkSampleBase {
public:
    VkSample12_InputAttachments();
    virtual ~VkSample12_InputAttachments();
    void loadTexture();
    void createVertexBuffer();
    void createPipelines();
    void createUniformBuffers();
    void buildCommandBuffers();

    void createAttachment(VkFormat format, VkImageUsageFlags usage, FrameBufferAttachment *attachment);
    void clearAttachment(FrameBufferAttachment* attachment);
    void updateAttachmentDescriptors(uint32_t index);
    void setupDescriptors();

    virtual void updateTransformMatrix(float rotateX, float rotateY, float scaleX, float scaleY);
    virtual void setupRenderPass();
    virtual void setupFrameBuffer();
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
        VkBuffer buffer;                       // Handle to the Vulkan buffer object that the memory is bound to
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

    std::vector<Attachments> attachments;
    VkExtent2D attachmentSize{};

    VkPipelineLayout pipelineLayout{VK_NULL_HANDLE}, pipelineLayout2{VK_NULL_HANDLE};;
    VkPipeline pipeline{VK_NULL_HANDLE}, pipeline2{VK_NULL_HANDLE};
    VkDescriptorSetLayout descriptorSetLayout{VK_NULL_HANDLE}, descriptorSetLayout2{VK_NULL_HANDLE};
    VkDescriptorSet descriptorSet{ VK_NULL_HANDLE };
    std::vector<VkDescriptorSet> descriptorSet2s;

    vks::Texture2D* pTexture2D;

    const VkFormat colorFormat = VK_FORMAT_R8G8B8A8_UNORM;
};
#endif //ANDROID_VULKAN_TUTORIALS_VKSAMPLE12_INPUTATTACHMENTS_H
