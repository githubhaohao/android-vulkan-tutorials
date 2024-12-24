#include <fstream>
#include <vector>
#include <exception>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vulkan/vulkan.h>
#include "vulkanexamplebase.h"
#include "VkSample05_PushConstants.h"

VkSample05_PushConstants::VkSample05_PushConstants() {
    LOGCATE("VkSample05_PushConstants::VkSample05_PushConstants()");
    title = "VkSample05_PushConstants";
    pTexture2D = nullptr;
}

VkSample05_PushConstants::~VkSample05_PushConstants() {
    LOGCATE("VkSample05_PushConstants::~VkSample05_PushConstants()");
    if (device) {
        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        vertexBuffer.destroy();
        uniformBuffer.destroy();

        vkDestroyBuffer(device, uniformBuffer.buffer, nullptr);
        vkFreeMemory(device, uniformBuffer.memory, nullptr);
    }

    if(pTexture2D) {
        pTexture2D->destroy();
        delete pTexture2D;
        pTexture2D = nullptr;
    }
}

void VkSample05_PushConstants::destroyTextureImage(Texture texture)
{
    vkDestroyImageView(device, texture.view, nullptr);
    vkDestroyImage(device, texture.image, nullptr);
    vkDestroySampler(device, texture.sampler, nullptr);
    vkFreeMemory(device, texture.deviceMemory, nullptr);
}

void VkSample05_PushConstants::updateTransformMatrix(float rotateX, float rotateY, float scaleX, float scaleY)
{
    LOGCATE("VkSample05_PushConstants::updateTransformMatrix()");
    float radiansX = static_cast<float>(MATH_PI / 180.0f * rotateX);
    float radiansY = static_cast<float>(MATH_PI / 180.0f * rotateY);
    glm::mat4 Model = glm::mat4(1.0f);
    Model = glm::scale(Model, glm::vec3(scaleX, scaleY, scaleY));
    Model = glm::rotate(Model, radiansX, glm::vec3(1.0f, 0.0f, 0.0f));
    Model = glm::rotate(Model, radiansY, glm::vec3(0.0f, 1.0f, 0.0f));
    Model = glm::translate(Model, glm::vec3(0.0f, 0.0f, 0.0f));
    mvpMatrix.modelMatrix = Model;
    uniformData.mvpMatrix = mvpMatrix.projectionMatrix * mvpMatrix.viewMatrix * mvpMatrix.modelMatrix;
}

void VkSample05_PushConstants::buildCommandBuffers()
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

    for (int32_t i = 0; i < drawCmdBuffers.size(); ++i)
    {
        // Set target frame buffer
        renderPassBeginInfo.framebuffer = frameBuffers[i];

        VK_CHECK_RESULT(vkBeginCommandBuffer(drawCmdBuffers[i], &cmdBufInfo));

        vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

        VkViewport viewport = vks::initializers::viewport((float)outputWidth, (float)outputHeight, 0.0f, 1.0f);
        vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);

        VkRect2D scissor = vks::initializers::rect2D(outputWidth, outputHeight, 0, 0);
        vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);

        vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
        vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

        VkDeviceSize offsets[1] = { 0 };
        vkCmdBindVertexBuffers(drawCmdBuffers[i], 0, 1, &vertexBuffer.buffer, offsets);

        uint32_t spherecount = static_cast<uint32_t>(spheres.size());
        for (uint32_t j = 0; j < spherecount; j++) {
            vkCmdPushConstants(
                    drawCmdBuffers[i],
                    pipelineLayout,
                    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                    0,
                    sizeof(SpherePushConstantData),
                    &spheres[j]);
            vkCmdDraw(drawCmdBuffers[i], vertexCount, 1, 0, 0);
        }

        vkCmdEndRenderPass(drawCmdBuffers[i]);

        VK_CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
    }
}

void VkSample05_PushConstants::generateSpheres()
{
    vector<vec3> m_VertexCoords;
    vector<vec2> m_TextureCoords;
    //构建顶点坐标
    for (float vAngle = 90; vAngle > -90; vAngle = vAngle - ANGLE_SPAN) {//垂直方向每隔 ANGLE_SPAN 度计算一次
        for (float hAngle = 360; hAngle > 0; hAngle = hAngle - ANGLE_SPAN) {//水平方向每隔 ANGLE_SPAN 度计算一次
            double xozLength = RADIUS * cos(RADIAN(vAngle));
            float x1 = (float) (xozLength * cos(RADIAN(hAngle)));
            float z1 = (float) (xozLength * sin(RADIAN(hAngle)));
            float y1 = (float) (RADIUS * sin(RADIAN(vAngle)));
            xozLength = RADIUS * cos(RADIAN(vAngle - ANGLE_SPAN));
            float x2 = (float) (xozLength * cos(RADIAN(hAngle)));
            float z2 = (float) (xozLength * sin(RADIAN(hAngle)));
            float y2 = (float) (RADIUS * sin(RADIAN(vAngle - ANGLE_SPAN)));
            xozLength = RADIUS * cos(RADIAN(vAngle - ANGLE_SPAN));
            float x3 = (float) (xozLength * cos(RADIAN(hAngle - ANGLE_SPAN)));
            float z3 = (float) (xozLength * sin(RADIAN(hAngle - ANGLE_SPAN)));
            float y3 = (float) (RADIUS * sin(RADIAN(vAngle - ANGLE_SPAN)));
            xozLength = RADIUS * cos(RADIAN(vAngle));
            float x4 = (float) (xozLength * cos(RADIAN(hAngle - ANGLE_SPAN)));
            float z4 = (float) (xozLength * sin(RADIAN(hAngle - ANGLE_SPAN)));
            float y4 = (float) (RADIUS * sin(RADIAN(vAngle)));

            //球面小矩形的四个点
            vec3 v1(x1, y1, z1);
            vec3 v2(x2, y2, z2);
            vec3 v3(x3, y3, z3);
            vec3 v4(x4, y4, z4);

            //构建第一个三角形
            m_VertexCoords.push_back(v1);
            m_VertexCoords.push_back(v2);
            m_VertexCoords.push_back(v4);
            //构建第二个三角形
            m_VertexCoords.push_back(v4);
            m_VertexCoords.push_back(v2);
            m_VertexCoords.push_back(v3);
        }
    }

    //构建纹理坐标，球面展开后的矩形
    int width = 360 / ANGLE_SPAN;//列数
    int height = 180 / ANGLE_SPAN;//行数
    float dw = 1.0f / width;
    float dh = 1.0f / height;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            //每一个小矩形，由两个三角形构成，共六个点
            float s = j * dw;
            float t = i * dh;
            vec2 v1(s, t);
            vec2 v2(s, t + dh);
            vec2 v3(s + dw, t + dh);
            vec2 v4(s + dw, t);

            //构建第一个三角形
            m_TextureCoords.push_back(v1);
            m_TextureCoords.push_back(v2);
            m_TextureCoords.push_back(v4);
            //构建第二个三角形
            m_TextureCoords.push_back(v4);
            m_TextureCoords.push_back(v2);
            m_TextureCoords.push_back(v3);
        }
    }

    std::vector<Vertex> vertices;
    for (int i = 0; i < m_VertexCoords.size(); ++i) {
        vertices.push_back({m_VertexCoords[i], m_TextureCoords[i], {0.0f, 0.0f, 1.0f}});
    }

    vertexCount = m_VertexCoords.size();
    // Setup random colors and fixed positions for every sphere in the scene
    std::random_device rndDevice;
    std::default_random_engine rndEngine(rndDevice());
    std::uniform_real_distribution<float> rndDist(0.1f, 1.0f);
    float delta = 1.0 / spheres.size();
    for (uint32_t i = 0; i < spheres.size(); i++) {
        spheres[i].color = glm::vec4(glm::vec3(delta * i + 0.2), 1.0f);
        spheres[i].position = glm::vec4(glm::vec3(0, -0.65 + i * 0.45, 0.0f) * 3.5f, 1.0f);
    }

    // Create buffers and upload data to the GPU
    struct StagingBuffers {
        vks::Buffer vertices;
    } stagingBuffers;

    // Host visible source buffers (staging)
    VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &stagingBuffers.vertices, vertices.size() * sizeof(Vertex), vertices.data()));

    // Device local destination buffers
    VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &vertexBuffer, vertices.size() * sizeof(Vertex)));

    // Copy from host do device
    vulkanDevice->copyBuffer(&stagingBuffers.vertices, &vertexBuffer, queue);

    // Clean up
    stagingBuffers.vertices.destroy();
}

void VkSample05_PushConstants::setupDescriptors()
{
    // Pool
    std::vector<VkDescriptorPoolSize> poolSizes = {
            vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1),
            // The sample uses a combined image + sampler descriptor to sample the texture in the fragment shader
            vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1)
    };
    VkDescriptorPoolCreateInfo descriptorPoolInfo = vks::initializers::descriptorPoolCreateInfo(poolSizes, 2);
    VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));

    // Layout
    std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
            // Binding 0 : Vertex shader uniform buffer
            vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0),
            // Binding 1 : Fragment shader image sampler
            vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1)
    };
    VkDescriptorSetLayoutCreateInfo descriptorLayout = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &descriptorSetLayout));

    // Set
    VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayout, 1);
    VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet));

    std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
            // Binding 0 : Vertex shader uniform buffer
            vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &uniformBuffer.descriptor),
            // Binding 1 : Fragment shader texture sampler
            //	Fragment shader: layout (binding = 1) uniform sampler2D samplerColor;
            vks::initializers::writeDescriptorSet(descriptorSet,
                                                  VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,		// The descriptor set will use a combined image sampler (as opposed to splitting image and sampler)
                                                  1,												// Shader binding point 1
                                                  &pTexture2D->descriptor)								// Pointer to the descriptor image for our texture
    };
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
}

void VkSample05_PushConstants::createPipelines()
{
    // Layout
    // [POI] Define the push constant range used by the pipeline layout
    // Note that the spec only requires a minimum of 128 bytes, so for passing larger blocks of data you'd use UBOs or SSBOs
    VkPushConstantRange pushConstantRange;
    // Push constants will only be accessible at the selected pipeline stages, for this sample it's the vertex shader that reads them
    pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(SpherePushConstantData);
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayout, 1);
    pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
    pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
    VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout));

    // Pipeline
    VkPipelineInputAssemblyStateCreateInfo inputAssemblyState = vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);
    VkPipelineRasterizationStateCreateInfo rasterizationState = vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, 0);
    VkPipelineColorBlendAttachmentState blendAttachmentState = vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
    VkPipelineColorBlendStateCreateInfo colorBlendState = vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);
    VkPipelineDepthStencilStateCreateInfo depthStencilState = vks::initializers::pipelineDepthStencilStateCreateInfo(VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS_OR_EQUAL);
    VkPipelineViewportStateCreateInfo viewportState = vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);
    VkPipelineMultisampleStateCreateInfo multisampleState = vks::initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT, 0);
    std::vector<VkDynamicState> dynamicStateEnables = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamicState = vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);
    std::array<VkPipelineShaderStageCreateInfo,2> shaderStages;

    // Shaders
    shaderStages[0] = compileShader(device, getShadersPath() + "pushconstants/pushconstants.vert", VK_SHADER_STAGE_VERTEX_BIT);
    shaderStages[1] = compileShader(device, getShadersPath() + "pushconstants/pushconstants.frag", VK_SHADER_STAGE_FRAGMENT_BIT);

    // Vertex input state
    std::vector<VkVertexInputBindingDescription> vertexInputBindings = {
            vks::initializers::vertexInputBindingDescription(0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX)
    };
    std::vector<VkVertexInputAttributeDescription> vertexInputAttributes = {
            vks::initializers::vertexInputAttributeDescription(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos)),
            vks::initializers::vertexInputAttributeDescription(0, 1, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv)),
            vks::initializers::vertexInputAttributeDescription(0, 2, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)),
    };
    VkPipelineVertexInputStateCreateInfo vertexInputState = vks::initializers::pipelineVertexInputStateCreateInfo();
    vertexInputState.vertexBindingDescriptionCount = static_cast<uint32_t>(vertexInputBindings.size());
    vertexInputState.pVertexBindingDescriptions = vertexInputBindings.data();
    vertexInputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexInputAttributes.size());
    vertexInputState.pVertexAttributeDescriptions = vertexInputAttributes.data();

    VkGraphicsPipelineCreateInfo pipelineCreateInfo = vks::initializers::pipelineCreateInfo(pipelineLayout, renderPass, 0);
    pipelineCreateInfo.pVertexInputState = &vertexInputState;
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

// Prepare and initialize uniform buffer containing shader uniforms
void VkSample05_PushConstants::prepareUniformBuffers()
{
    // Vertex shader uniform buffer block
    VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &uniformBuffer, sizeof(uniformData), &uniformData));
    VK_CHECK_RESULT(uniformBuffer.map());
}

void VkSample05_PushConstants::loadTexture()
{
    if(pTexture2D) {
        pTexture2D->destroy();
        delete pTexture2D;
    }

    pTexture2D = new vks::Texture2D();
    VK_CHECK_RESULT(loadRGBTexture2DFromFile("image/earth.jpg", pTexture2D))
    float viewportRatio = (float) outputWidth / (float) outputHeight;
    mvpMatrix.projectionMatrix = glm::perspective(45.0f,viewportRatio, 0.1f,256.f);;
    mvpMatrix.viewMatrix = glm::lookAt(
            glm::vec3(0, 0, 4), // Camera is at (0,0,9), in World Space
            glm::vec3(0, 0, 0), // and looks at the origin
            glm::vec3(0, 1, 0)  // Head is up (set to 0,-1,0 to look upside-down)
    );
    mvpMatrix.modelMatrix = glm::mat4(1.0f);
    uniformData.mvpMatrix = mvpMatrix.projectionMatrix * mvpMatrix.viewMatrix * mvpMatrix.modelMatrix;
}

void VkSample05_PushConstants::prepare() {

    VulkanExampleBase::prepare();
    loadTexture();
    generateSpheres();
    prepareUniformBuffers();
    setupDescriptors();
    createPipelines();
    buildCommandBuffers();
    prepared = true;
}

void VkSample05_PushConstants::render() {
    if (!prepared)
        return;
    VulkanExampleBase::prepareFrame();
    memcpy(uniformBuffer.mapped, &uniformData, sizeof(uniformData));
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &drawCmdBuffers[currentBufferIdx];
    VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
    VulkanExampleBase::submitFrame();
}


