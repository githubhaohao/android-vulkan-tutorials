//
// Created by 字节流动 on 2024/7/11.
//

#ifndef ANDROID_VULKAN_TUTORIALS_VkSample20_ComputeShader_H
#define ANDROID_VULKAN_TUTORIALS_VkSample20_ComputeShader_H
#include "vulkanexamplebase.h"
#include "VkSampleBase.h"

class VkSample20_ComputeShader : public VkSampleBase {
public:
    VkSample20_ComputeShader();
    virtual ~VkSample20_ComputeShader();
    void loadTexture();
    void createVertexBuffer();
    void createUniformBuffers();
    void buildCommandBuffers();
    void prepareStorageImage();
    void setupDescriptorPool();
    void prepareGraphics();
    void prepareCompute();
    void buildComputeCommandBuffer();

    virtual void updateTransformMatrix(float rotateX, float rotateY, float scaleX, float scaleY);
    virtual void prepare();
    virtual void render();
public:
    // Vertex layout used in this example
    struct Vertex {
        float pos[3];
        float uv[2];
    };

    struct UBO {
        glm::mat4 mvpMatrix;
        float iTime;
    } uboData;

    struct MVPMatrix {
        glm::mat4 projectionMatrix;
        glm::mat4 modelMatrix;
        glm::mat4 viewMatrix;
    } mvpMatrix;

    // Storage image that the compute shader uses to apply the filter effect to
    vks::Texture2D storageImage;
    vks::Texture2D storageImage2;

    // Resources for the graphics part of the example
    struct Graphics {
        VkDescriptorSetLayout descriptorSetLayout{ VK_NULL_HANDLE };	// Image display shader binding layout
        VkDescriptorSet descriptorSetPreCompute{ VK_NULL_HANDLE };		// Image display shader bindings before compute shader image manipulation
        VkDescriptorSet descriptorSetPostCompute{ VK_NULL_HANDLE };		// Image display shader bindings after compute shader image manipulation
        VkDescriptorSet descriptorSetPostCompute2{ VK_NULL_HANDLE };		// Image display shader bindings after compute shader image manipulation
        VkPipeline pipeline{ VK_NULL_HANDLE };							// Image display pipeline
        VkPipelineLayout pipelineLayout{ VK_NULL_HANDLE };				// Layout of the graphics pipeline
        VkSemaphore semaphore{ VK_NULL_HANDLE };						// Execution dependency between compute & graphic submission
    } graphics;

    // Resources for the compute part of the example
    struct Compute {
        VkQueue queue{ VK_NULL_HANDLE };								// Separate queue for compute commands (queue family may differ from the one used for graphics)
        VkCommandPool commandPool{ VK_NULL_HANDLE };					// Use a separate command pool (queue family may differ from the one used for graphics)
        VkCommandBuffer commandBuffer{ VK_NULL_HANDLE };				// Command buffer storing the dispatch commands and barriers
        VkSemaphore semaphore{ VK_NULL_HANDLE };						// Execution dependency between compute & graphic submission
        VkDescriptorSetLayout descriptorSetLayout{ VK_NULL_HANDLE };	// Compute shader binding layout
        VkDescriptorSet descriptorSet{ VK_NULL_HANDLE };				// Compute shader bindings
        VkDescriptorSet descriptorSet2{ VK_NULL_HANDLE };				// Compute shader bindings
        VkPipelineLayout pipelineLayout{ VK_NULL_HANDLE };				// Layout of the compute pipeline
        std::vector<VkPipeline> pipelines{};							// Compute pipelines for image filters
        int32_t pipelineIndex{ 0 };										// Current image filtering compute pipeline index
    } compute;

    vks::Buffer vertexBuffer;
    vks::Buffer indexBuffer;
    vks::Buffer uniformBuffer;
    uint32_t indexCount{ 0 };
    uint32_t vertexBufferSize{ 0 };

    vks::Texture2D* pTexture2D;

    std::vector<std::string> filterNames{};
};
#endif //ANDROID_VULKAN_TUTORIALS_VkSample20_ComputeShader_H
