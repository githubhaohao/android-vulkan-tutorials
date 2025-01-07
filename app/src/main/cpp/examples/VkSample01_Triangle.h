/**
 *
 * Created by 公众号：字节流动 on 2025/01/12.
 * https://github.com/githubhaohao/android-vulkan-tutorials
 * 最新文章首发于公众号：字节流动，有疑问或者技术交流可以添加微信 Byte-Flow ,领取视频教程, 拉你进技术交流群
 *
 * */

#ifndef ANDROID_VULKAN_TUTORIALS_VKSAMPLE01_TRIANGLE_H
#define ANDROID_VULKAN_TUTORIALS_VKSAMPLE01_TRIANGLE_H
#include "vulkanexamplebase.h"
#include "VkSampleBase.h"

class VkSample01_Triangle : public VkSampleBase {
public:
    VkSample01_Triangle();
    virtual ~VkSample01_Triangle();
    void createVertexBuffer();
    void createDescriptorPool();
    void createDescriptorSetLayout();
    void createDescriptorSets();
    void createPipelines();
    void createUniformBuffers();
    void buildCommandBuffers();

    virtual void updateTransformMatrix(float rotateX, float rotateY, float scaleX, float scaleY);

    virtual void setupDepthStencil();
    virtual void setupFrameBuffer();
    virtual void setupRenderPass();
    virtual void prepare();
    virtual void render();
public:
    // Vertex layout used in this example
    struct Vertex {
        float position[3];
        float color[3];
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
        // The descriptor set stores the resources bound to the binding points in a shader
        // It connects the binding points of the different shaders with the buffers and images used for those bindings
        VkDescriptorSet descriptorSet;
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

    // The pipeline layout is used by a pipeline to access the descriptor sets
    // It defines interface (without binding any actual data) between the shader stages used by the pipeline and the shader resources
    // A pipeline layout can be shared among multiple pipelines as long as their interfaces match
    VkPipelineLayout pipelineLayout{VK_NULL_HANDLE};

    // Pipelines (often called "pipeline state objects") are used to bake all states that affect a pipeline
    // While in OpenGL every state can be changed at (almost) any time, Vulkan requires to layout the graphics (and compute) pipeline states upfront
    // So for each combination of non-dynamic pipeline states you need a new pipeline (there are a few exceptions to this not discussed here)
    // Even though this adds a new dimension of planning ahead, it's a great opportunity for performance optimizations by the driver
    VkPipeline pipeline{VK_NULL_HANDLE};

    // The descriptor set layout describes the shader binding layout (without actually referencing descriptor)
    // Like the pipeline layout it's pretty much a blueprint and can be used with different descriptor sets as long as their layout matches
    VkDescriptorSetLayout descriptorSetLayout{VK_NULL_HANDLE};

    // Fence to wait for all command buffers to finish before
    // presenting to the swap chain
    VkFence renderFence{ VK_NULL_HANDLE };
};
#endif //ANDROID_VULKAN_TUTORIALS_VKSAMPLE01_TRIANGLE_H
