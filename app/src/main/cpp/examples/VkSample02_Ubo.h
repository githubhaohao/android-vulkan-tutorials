/**
 *
 * Created by 公众号：字节流动 on 2025/01/12.
 * https://github.com/githubhaohao/android-vulkan-tutorials
 * 最新文章首发于公众号：字节流动，有疑问或者技术交流可以添加微信 Byte-Flow ,领取视频教程, 拉你进技术交流群
 *
 * */

#ifndef ANDROID_VULKAN_TUTORIALS_VKSAMPLE02_UBO_H
#define ANDROID_VULKAN_TUTORIALS_VKSAMPLE02_UBO_H
#include "vulkanexamplebase.h"
#include "VkSampleBase.h"

#define OBJECT_INSTANCES 9

class VkSample02_Ubo : public VkSampleBase {
public:
    VkSample02_Ubo();
    virtual ~VkSample02_Ubo();
    void createVertexBuffer();
    void createDescriptors();
    void createPipelines();
    void createUniformBuffers();
    void updateDynamicUniformBuffer();
    void updateUniformBuffers();
    void buildCommandBuffers();

    // Wrapper functions for aligned memory allocation
// There is currently no standard for this in C++ that works across all platforms and vendors, so we abstract this
    void* alignedAlloc(size_t size, size_t alignment)
    {
        void *data = nullptr;
#if defined(_MSC_VER) || defined(__MINGW32__)
        data = _aligned_malloc(size, alignment);
#else
        int res = posix_memalign(&data, alignment, size);
        if (res != 0)
            data = nullptr;
#endif
        return data;
    }

    void alignedFree(void* data)
    {
#if	defined(_MSC_VER) || defined(__MINGW32__)
        _aligned_free(data);
#else
        free(data);
#endif
    }

    virtual void loadImage(NativeImage *pImage);
    virtual void updateTransformMatrix(float rotateX, float rotateY, float scaleX, float scaleY);
    virtual void prepare();
    virtual void render();
public:
    // Vertex layout for this example
    struct Vertex {
        float pos[3];
        float color[3];
    };

    vks::Buffer vertexBuffer;
    vks::Buffer indexBuffer;
    uint32_t indexCount{ 0 };

    struct {
        vks::Buffer view;
        vks::Buffer dynamic;
    } uniformBuffers;

    struct {
        glm::mat4 projection;
        glm::mat4 view;
    } uboVS;

    glm::mat4 globalModelMat;

    // Store random per-object rotations
    glm::vec3 rotations[OBJECT_INSTANCES];
    glm::vec3 rotationSpeeds[OBJECT_INSTANCES];

    // One big uniform buffer that contains all matrices
    // Note that we need to manually allocate the data to cope for GPU-specific uniform buffer offset alignments
    struct UboDataDynamic {
        glm::mat4* model{ nullptr };
    } uboDataDynamic;

    VkPipeline pipeline{ VK_NULL_HANDLE };
    VkPipelineLayout pipelineLayout{ VK_NULL_HANDLE };
    VkDescriptorSet descriptorSet{ VK_NULL_HANDLE };
    VkDescriptorSetLayout descriptorSetLayout{ VK_NULL_HANDLE };

    float animationTimer{ 0.0f };

    size_t dynamicAlignment{ 0 };
};
#endif //ANDROID_VULKAN_TUTORIALS_VKSAMPLE02_UBO_H
