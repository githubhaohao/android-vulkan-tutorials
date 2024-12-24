//
// Created by 字节流动 on 2024/8/11.
//

#ifndef ANDROID_VULKAN_TUTORIALS_VKSAMPLE05_PUSHCONSTANTS_H
#define ANDROID_VULKAN_TUTORIALS_VKSAMPLE05_PUSHCONSTANTS_H
#include "vulkanexamplebase.h"
#include "VkSampleBase.h"

using namespace glm;
using namespace std;

#define MATH_PI     3.1415926535897932384626433832802
#define ANGLE_SPAN  9
#define RADIUS 0.5
#define RADIAN(angle) ((angle) / 180 * MATH_PI)

class VkSample05_PushConstants : public VkSampleBase {
public:
    VkSample05_PushConstants();
    virtual ~VkSample05_PushConstants();
    void destroyTextureImage(Texture texture);
    void loadTexture();
    void generateSpheres();
    void setupDescriptors();
    void createPipelines();
    void prepareUniformBuffers();
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

    struct UniformData {
        glm::mat4 mvpMatrix;
    } uniformData;

    struct MVPMatrix {
        glm::mat4 projectionMatrix;
        glm::mat4 modelMatrix;
        glm::mat4 viewMatrix;
    } mvpMatrix;

    vks::Buffer uniformBuffer;

    VkPipeline pipeline{ VK_NULL_HANDLE };
    VkPipelineLayout pipelineLayout{ VK_NULL_HANDLE };
    VkDescriptorSet descriptorSet{ VK_NULL_HANDLE };
    VkDescriptorSetLayout descriptorSetLayout{ VK_NULL_HANDLE };
    vks::Texture2D* pTexture2D;

    // Color and position data for each sphere is uploaded using push constants
    struct SpherePushConstantData {
        glm::vec4 color;
        glm::vec4 position;
    };
    std::array<SpherePushConstantData, 6> spheres;
};
#endif //ANDROID_VULKAN_TUTORIALS_VKSAMPLE05_PUSHCONSTANTS_H
