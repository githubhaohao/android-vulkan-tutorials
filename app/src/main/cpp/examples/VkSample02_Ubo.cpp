/**
 *
 * Created by 公众号：字节流动 on 2025/01/12.
 * https://github.com/githubhaohao/android-vulkan-tutorials
 * 最新文章首发于公众号：字节流动，有疑问或者技术交流可以添加微信 Byte-Flow ,领取视频教程, 拉你进技术交流群
 *
 * */


#include <fstream>
#include <vector>
#include <exception>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vulkan/vulkan.h>
#include "vulkanexamplebase.h"
#include "VkSample02_Ubo.h"

VkSample02_Ubo::VkSample02_Ubo() {

    title = "VkSample02_Ubo";
    camera.type = Camera::CameraType::lookat;
    camera.setPosition(glm::vec3(0.0f, 0.0f, -18.0f));
    camera.setRotation(glm::vec3(0.0f));
}

VkSample02_Ubo::~VkSample02_Ubo() {
    if (device) {
        if (uboDataDynamic.model) {
            alignedFree(uboDataDynamic.model);
        }
        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        vertexBuffer.destroy();
        indexBuffer.destroy();
        uniformBuffers.view.destroy();
        uniformBuffers.dynamic.destroy();
    }
    NativeImageUtil::FreeNativeImage(&renderImage);
}

void VkSample02_Ubo::updateTransformMatrix(float rotateX, float rotateY, float scaleX, float scaleY)
{
    LOGCATE("VkSample02_Ubo::updateTransformMatrix()");
    float radiansX = static_cast<float>(MATH_PI / 180.0f * rotateX);
    float radiansY = static_cast<float>(MATH_PI / 180.0f * rotateY);
    glm::mat4 Model = glm::mat4(1.0f);
    Model = glm::scale(Model, glm::vec3(scaleX, scaleX, 1.0f));
    Model = glm::rotate(Model, radiansX, glm::vec3(1.0f, 0.0f, 0.0f));
    Model = glm::rotate(Model, radiansY, glm::vec3(0.0f, 1.0f, 0.0f));
    Model = glm::translate(Model, glm::vec3(0.0f, 0.0f, 0.0f));
}

void VkSample02_Ubo::loadImage(NativeImage *pImage)
{
//    LOGCATE("VkSample02_Ubo::loadImage() pImage = %p", pImage->ppPlane[0]);
//    assert(pImage->format == IMAGE_FORMAT_I444);
//    if (pImage)
//    {
//        renderImage.width = pImage->width;
//        renderImage.height = pImage->height;
//        renderImage.format = pImage->format;
//        NativeImageUtil::CopyNativeImage(pImage, &renderImage);
//    }
}
// Prepare vertex and index buffers for an indexed triangle
// Also uploads them to device local memory using staging and initializes vertex input and attribute binding to match the vertex shader
void VkSample02_Ubo::createVertexBuffer() {
    // Setup vertices indices for a colored cube
    std::vector<Vertex> vertices = {
            { { -1.0f, -1.0f,  1.0f },{ 1.0f, 0.0f, 0.0f } },
            { {  1.0f, -1.0f,  1.0f },{ 0.0f, 1.0f, 0.0f } },
            { {  1.0f,  1.0f,  1.0f },{ 0.0f, 0.0f, 1.0f } },
            { { -1.0f,  1.0f,  1.0f },{ 1.0f, 0.0f, 1.0f } },
            { { -1.0f, -1.0f, -1.0f },{ 1.0f, 0.0f, 0.0f } },
            { {  1.0f, -1.0f, -1.0f },{ 0.0f, 1.0f, 0.0f } },
            { {  1.0f,  1.0f, -1.0f },{ 0.0f, 0.0f, 1.0f } },
            { { -1.0f,  1.0f, -1.0f },{ 1.0f, 0.0f, 1.0f } },
    };

    std::vector<uint32_t> indices = {
            0,1,2, 2,3,0, 1,5,6, 6,2,1, 7,6,5, 5,4,7, 4,0,3, 3,7,4, 4,5,1, 1,0,4, 3,2,6, 6,7,3,
    };

    indexCount = static_cast<uint32_t>(indices.size());

    // Create buffers
    // For the sake of simplicity we won't stage the vertex data to the gpu memory
    // Vertex buffer
    VK_CHECK_RESULT(vulkanDevice->createBuffer(
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            &vertexBuffer,
            vertices.size() * sizeof(Vertex),
            vertices.data()));
    // Index buffer
    VK_CHECK_RESULT(vulkanDevice->createBuffer(
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            &indexBuffer,
            indices.size() * sizeof(uint32_t),
            indices.data()));
}

void VkSample02_Ubo::createDescriptors() {
    // Pool
    std::vector<VkDescriptorPoolSize> poolSizes = {
            vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1),
            // Dynamic uniform buffer
            vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1)
    };

    VkDescriptorPoolCreateInfo descriptorPoolInfo = vks::initializers::descriptorPoolCreateInfo(poolSizes, 2);
    VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));

    // Layout
    std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
            vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0),
            // Dynamic uniform buffer
            vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, VK_SHADER_STAGE_VERTEX_BIT, 1)
    };

    VkDescriptorSetLayoutCreateInfo descriptorLayout = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &descriptorSetLayout));

    // Set
    VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayout, 1);
    VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet));

    std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
            // Binding 0 : Projection/View matrix as uniform buffer
            vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &uniformBuffers.view.descriptor),
            // Binding 1 : Instance matrix as dynamic uniform buffer
            vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, &uniformBuffers.dynamic.descriptor),
    };
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
}

void VkSample02_Ubo::createPipelines() {
    // Layout
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayout, 1);
    VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout));

    // Pipeline
    VkPipelineInputAssemblyStateCreateInfo inputAssemblyState = vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0,  VK_FALSE);
    VkPipelineRasterizationStateCreateInfo rasterizationState = vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, 0);
    VkPipelineColorBlendAttachmentState blendAttachmentState = vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
    VkPipelineColorBlendStateCreateInfo colorBlendState = vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);
    VkPipelineDepthStencilStateCreateInfo depthStencilState = vks::initializers::pipelineDepthStencilStateCreateInfo(VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS_OR_EQUAL);
    VkPipelineViewportStateCreateInfo viewportState = vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);
    VkPipelineMultisampleStateCreateInfo multisampleState = vks::initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT, 0);
    std::vector<VkDynamicState> dynamicStateEnables = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamicState = vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);
    std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages;

    // Vertex bindings and attributes
    VkVertexInputBindingDescription vertexInputBinding = {
            vks::initializers::vertexInputBindingDescription(0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX)
    };
    std::vector<VkVertexInputAttributeDescription> vertexInputAttributes = {
            vks::initializers::vertexInputAttributeDescription(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos)),	// Location 0 : Position
            vks::initializers::vertexInputAttributeDescription(0, 1, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color)),	// Location 1 : Color
    };
    VkPipelineVertexInputStateCreateInfo vertexInputStateCI = vks::initializers::pipelineVertexInputStateCreateInfo();
    vertexInputStateCI.vertexBindingDescriptionCount = 1;
    vertexInputStateCI.pVertexBindingDescriptions = &vertexInputBinding;
    vertexInputStateCI.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexInputAttributes.size());
    vertexInputStateCI.pVertexAttributeDescriptions = vertexInputAttributes.data();

    // Shaders
    shaderStages[0] = compileShader(device, getShadersPath() + "ubo/ubo.vert", VK_SHADER_STAGE_VERTEX_BIT);
    shaderStages[1] = compileShader(device,getShadersPath() + "ubo/ubo.frag", VK_SHADER_STAGE_FRAGMENT_BIT);

    VkGraphicsPipelineCreateInfo pipelineCreateInfo = vks::initializers::pipelineCreateInfo(pipelineLayout, renderPass, 0);
    pipelineCreateInfo.pVertexInputState = &vertexInputStateCI;
    pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
    pipelineCreateInfo.pRasterizationState = &rasterizationState;
    pipelineCreateInfo.pColorBlendState = &colorBlendState;
    pipelineCreateInfo.pMultisampleState = &multisampleState;
    pipelineCreateInfo.pViewportState = &viewportState;
    pipelineCreateInfo.pDepthStencilState = &depthStencilState;
    pipelineCreateInfo.pDynamicState = &dynamicState;
    pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
    pipelineCreateInfo.pStages = shaderStages.data();
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipeline));
}

void VkSample02_Ubo::createUniformBuffers() {
    // Calculate required alignment based on minimum device offset alignment
    size_t minUboAlignment = vulkanDevice->properties.limits.minUniformBufferOffsetAlignment;
    dynamicAlignment = sizeof(glm::mat4);
    if (minUboAlignment > 0) {
        dynamicAlignment = (dynamicAlignment + minUboAlignment - 1) & ~(minUboAlignment - 1);
    }

    size_t bufferSize = OBJECT_INSTANCES * dynamicAlignment;

    uboDataDynamic.model = (glm::mat4*)alignedAlloc(bufferSize, dynamicAlignment);
    assert(uboDataDynamic.model);

    std::cout << "minUniformBufferOffsetAlignment = " << minUboAlignment << std::endl;
    std::cout << "dynamicAlignment = " << dynamicAlignment << std::endl;

    // Vertex shader uniform buffer block

    // Static shared uniform buffer object with projection and view matrix
    VK_CHECK_RESULT(vulkanDevice->createBuffer(
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            &uniformBuffers.view,
            sizeof(uboVS)));

    // Uniform buffer object with per-object matrices
    VK_CHECK_RESULT(vulkanDevice->createBuffer(
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
            &uniformBuffers.dynamic,
            bufferSize));

    // Override descriptor range to [base, base + dynamicAlignment]
    uniformBuffers.dynamic.descriptor.range = dynamicAlignment;

    // Map persistent
    VK_CHECK_RESULT(uniformBuffers.view.map());
    VK_CHECK_RESULT(uniformBuffers.dynamic.map());

    // Prepare per-object matrices with offsets and random rotations
    std::default_random_engine rndEngine(benchmark.active ? 0 : (unsigned)time(nullptr));
    std::normal_distribution<float> rndDist(-1.0f, 1.0f);
    for (uint32_t i = 0; i < OBJECT_INSTANCES; i++) {
        rotations[i] = glm::vec3(rndDist(rndEngine), rndDist(rndEngine), rndDist(rndEngine)) * 2.0f * (float)M_PI;
        rotationSpeeds[i] = glm::vec3(rndDist(rndEngine), rndDist(rndEngine), rndDist(rndEngine));
    }

    updateUniformBuffers();
    updateDynamicUniformBuffer();
}


void VkSample02_Ubo::updateUniformBuffers()
{
    // Fixed ubo with projection and view matrices
    uboVS.projection = camera.matrices.perspective;
    uboVS.view = camera.matrices.view;

    memcpy(uniformBuffers.view.mapped, &uboVS, sizeof(uboVS));
}

void VkSample02_Ubo::updateDynamicUniformBuffer()
{
    // Update at max. 60 fps
    animationTimer += 0.016f;
    if (animationTimer <= 1.0f / 60.0f) {
        return;
    }

    // Dynamic ubo with per-object model matrices indexed by offsets in the command buffer
    uint32_t dim = static_cast<uint32_t>(pow(OBJECT_INSTANCES, (1.0f / 2.0f)));
    glm::vec3 offset(5.0f);

    for (uint32_t x = 0; x < dim; x++)
    {
        for (uint32_t y = 0; y < dim; y++)
        {
            uint32_t index = x * dim + y;

            // Aligned offset
            glm::mat4* modelMat = (glm::mat4*)(((uint64_t)uboDataDynamic.model + (index * dynamicAlignment)));

            // Update rotations
            rotations[index] += animationTimer * rotationSpeeds[index];

            // Update matrices
            glm::vec3 pos = glm::vec3(-((dim * offset.x) / 2.0f) + offset.x / 2.0f + x * offset.x, -((dim * offset.y) / 2.0f) + offset.y / 2.0f + y * offset.y, -((dim * offset.z) / 2.0f) + offset.z / 2.0f);
            *modelMat = glm::translate(glm::mat4(1.0f), pos);
            *modelMat = glm::rotate(*modelMat, rotations[index].x, glm::vec3(1.0f, 1.0f, 0.0f));
            *modelMat = glm::rotate(*modelMat, rotations[index].y, glm::vec3(0.0f, 1.0f, 0.0f));
            *modelMat = glm::rotate(*modelMat, rotations[index].z, glm::vec3(0.0f, 0.0f, 1.0f));
        }
    }

    animationTimer = 0.0f;

    memcpy(uniformBuffers.dynamic.mapped, uboDataDynamic.model, uniformBuffers.dynamic.size);
    // Flush to make changes visible to the host
    VkMappedMemoryRange memoryRange = vks::initializers::mappedMemoryRange();
    memoryRange.memory = uniformBuffers.dynamic.memory;
    memoryRange.size = uniformBuffers.dynamic.size;
    vkFlushMappedMemoryRanges(device, 1, &memoryRange);
}

void VkSample02_Ubo::buildCommandBuffers()
{
    VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

    VkClearValue clearValues[2];
    clearValues[0].color = defaultClearColor;
    clearValues[1].depthStencil = { 1.0f, 0 };

    VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
    renderPassBeginInfo.renderPass = renderPass;
    renderPassBeginInfo.renderArea.offset.x = 0;
    renderPassBeginInfo.renderArea.offset.y = 0;
    renderPassBeginInfo.renderArea.extent.width = outputWidth;
    renderPassBeginInfo.renderArea.extent.height = outputHeight;
    renderPassBeginInfo.clearValueCount = 2;
    renderPassBeginInfo.pClearValues = clearValues;
    //录制渲染指令，存在上下文概念
    for (int32_t i = 0; i < drawCmdBuffers.size(); ++i)
    {
        renderPassBeginInfo.framebuffer = frameBuffers[i];

        VK_CHECK_RESULT(vkBeginCommandBuffer(drawCmdBuffers[i], &cmdBufInfo));

        vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

        VkViewport viewport = vks::initializers::viewport((float)outputWidth, (float)outputHeight, 0.0f, 1.0f);
        vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);

        VkRect2D scissor = vks::initializers::rect2D(outputWidth, outputHeight, 0, 0);
        vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);

        vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

        VkDeviceSize offsets[1] = { 0 };
        vkCmdBindVertexBuffers(drawCmdBuffers[i], 0, 1, &vertexBuffer.buffer, offsets);
        vkCmdBindIndexBuffer(drawCmdBuffers[i], indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

        // Render multiple objects using different model matrices by dynamically offsetting into one uniform buffer
        for (uint32_t j = 0; j < OBJECT_INSTANCES; j++)
        {
            // One dynamic offset per dynamic descriptor to offset into the ubo containing all model matrices
            uint32_t dynamicOffset = j * static_cast<uint32_t>(dynamicAlignment);
            // Bind the descriptor set for rendering a mesh using the dynamic offset
            vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet, 1, &dynamicOffset);

            vkCmdDrawIndexed(drawCmdBuffers[i], indexCount, 1, 0, 0, 0);
        }

        vkCmdEndRenderPass(drawCmdBuffers[i]);

        VK_CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
    }
}

void VkSample02_Ubo::prepare() {
    float viewportRatio = (float) outputWidth / (float) outputHeight;
    camera.setPerspective(60.0f, viewportRatio, 1.0f, 256.0f);

    VulkanExampleBase::prepare();
    createVertexBuffer();
    createUniformBuffers();
    createDescriptors();
    createPipelines();
    buildCommandBuffers();
    prepared = true;
}

void VkSample02_Ubo::render() {
    if (!prepared)
        return;
    updateUniformBuffers();
    updateDynamicUniformBuffer();

    VulkanExampleBase::prepareFrame();
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &drawCmdBuffers[currentBufferIdx];
    VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
    VulkanExampleBase::submitFrame();
}


