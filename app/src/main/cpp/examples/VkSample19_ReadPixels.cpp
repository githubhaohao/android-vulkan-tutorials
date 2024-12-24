#include <fstream>
#include <vector>
#include <exception>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vulkan/vulkan.h>
#include "vulkanexamplebase.h"
#include "VkSample19_ReadPixels.h"

VkSample19_ReadPixels::VkSample19_ReadPixels() {
    LOGCATE("VkSample19_ReadPixels::VkSample19_ReadPixels()");
    title = "VkSample19_ReadPixels";
    pTexture2D = nullptr;
    rndEngine.seed(benchmark.active ? 0 : (unsigned)time(nullptr));
}

VkSample19_ReadPixels::~VkSample19_ReadPixels() {
    LOGCATE("VkSample19_ReadPixels::~VkSample19_ReadPixels()");
    if (device) {
        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        vertexBuffer.destroy();
        instanceBuffer.destroy();

        vkDestroyFence(device, renderFence, nullptr);

        vkDestroyBuffer(device, uniformBuffer.buffer, nullptr);
        vkFreeMemory(device, uniformBuffer.memory, nullptr);
    }

    if(pTexture2D) {
        pTexture2D->destroy();
        delete pTexture2D;
        pTexture2D = nullptr;
    }
}

float VkSample19_ReadPixels::rnd(float range)
{
    std::uniform_real_distribution<float> rndDist(0.0f, range);
    return rndDist(rndEngine);
}

void VkSample19_ReadPixels::updateTransformMatrix(float rotateX, float rotateY, float scaleX, float scaleY)
{
    LOGCATE("VkSample19_ReadPixels::updateTransformMatrix()");
    float radiansX = static_cast<float>(MATH_PI / 180.0f * rotateX);
    float radiansY = static_cast<float>(MATH_PI / 180.0f * rotateY);
    glm::mat4 Model = glm::mat4(1.0f);
    Model = glm::scale(Model, glm::vec3(scaleX, scaleY, scaleY));
    Model = glm::rotate(Model, radiansX, glm::vec3(1.0f, 0.0f, 0.0f));
    Model = glm::rotate(Model, radiansY, glm::vec3(0.0f, 1.0f, 0.0f));
    Model = glm::translate(Model, glm::vec3(0.0f, 0.0f, 0.0f));
    mvpMatrix.modelMatrix = Model;
    uboData.mvpMatrix = mvpMatrix.projectionMatrix * mvpMatrix.viewMatrix * mvpMatrix.modelMatrix;
}

// Create a buffer with per-instance data that is sourced in the shaders
void VkSample19_ReadPixels::prepareInstanceData()
{
    instanceData.resize(INSTANCE_COUNT);

    std::default_random_engine rndGenerator(benchmark.active ? 0 : (unsigned)time(nullptr));
    std::uniform_real_distribution<float> uniformDist(0.0, 1.0);
    std::uniform_int_distribution<uint32_t> rndTextureIndex(0, 2);

    // Distribute rocks randomly on two different rings
    for (auto i = 0; i < INSTANCE_COUNT / 2; i++) {
        glm::vec2 ring0 { 7.0f, 11.0f };
        glm::vec2 ring1 { 14.0f, 18.0f };

        float rho, theta;

        // Inner ring
        rho = sqrt((pow(ring0[1], 2.0f) - pow(ring0[0], 2.0f)) * uniformDist(rndGenerator) + pow(ring0[0], 2.0f));
        theta = static_cast<float>(2.0f * M_PI * uniformDist(rndGenerator));
        instanceData[i].pos = glm::vec3(rho*cos(theta), uniformDist(rndGenerator) * 0.5f - 0.25f, rho*sin(theta));
        instanceData[i].rot = glm::vec3(M_PI * uniformDist(rndGenerator), M_PI * uniformDist(rndGenerator), M_PI * uniformDist(rndGenerator));
        instanceData[i].scale = 1.5f + uniformDist(rndGenerator) - uniformDist(rndGenerator);
        instanceData[i].scale *= 0.3f;
        instanceData[i].col = glm::vec3(rnd(1.0f), rnd(1.0f), rnd(1.0f));

        // Outer ring
        rho = sqrt((pow(ring1[1], 2.0f) - pow(ring1[0], 2.0f)) * uniformDist(rndGenerator) + pow(ring1[0], 2.0f));
        theta = static_cast<float>(2.0f * M_PI * uniformDist(rndGenerator));
        instanceData[i + INSTANCE_COUNT / 2].pos = glm::vec3(rho*cos(theta), uniformDist(rndGenerator) * 0.5f - 0.25f, rho*sin(theta));
        instanceData[i + INSTANCE_COUNT / 2].rot = glm::vec3(M_PI * uniformDist(rndGenerator), M_PI * uniformDist(rndGenerator), M_PI * uniformDist(rndGenerator));
        instanceData[i + INSTANCE_COUNT / 2].scale = 1.5f + uniformDist(rndGenerator) - uniformDist(rndGenerator);
        instanceData[i + INSTANCE_COUNT / 2].scale *= 0.3f;

        instanceData[i + INSTANCE_COUNT / 2].col = glm::vec3(rnd(1.0f), rnd(1.0f), rnd(1.0f));
    }

    int memSize = instanceData.size() * sizeof(InstanceData);
    VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &instanceBuffer, memSize))
    instanceBuffer.map();
    memcpy(instanceBuffer.mapped, instanceData.data(), memSize);
}

void VkSample19_ReadPixels::updateInstanceData() {
    for (auto i = 0; i < INSTANCE_COUNT / 2; i++) {
        instanceData[i].pos.y = sin(glm::radians(0.1f * this->iTime * 360.0f + i)) * 7.0f * abs( cos(18346 * i * 0.1));
        instanceData[i + INSTANCE_COUNT / 2].pos.y = sin(glm::radians(0.1f * this->iTime * 360.0f + i)) * 8.0f * abs( cos(89346 * i));
    }
    memcpy(instanceBuffer.mapped, instanceData.data(), instanceData.size() * sizeof(InstanceData));
}

void VkSample19_ReadPixels::createUniformBuffers() {
    // Prepare and initialize the per-frame uniform buffer blocks containing shader uniforms
    // Single uniforms like in OpenGL are no longer present in Vulkan. All Shader uniforms are passed via uniform buffer blocks
    VkMemoryRequirements memReqs;

    // Vertex shader uniform buffer block
    VkBufferCreateInfo bufferInfo{};
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.pNext = nullptr;
    allocInfo.allocationSize = 0;
    allocInfo.memoryTypeIndex = 0;

    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = sizeof(UBO);
    // This buffer will be used as a uniform buffer
    bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

    // Create the buffers
    VK_CHECK_RESULT(vkCreateBuffer(device, &bufferInfo, nullptr, &uniformBuffer.buffer));
    // Get memory requirements including size, alignment and memory type
    vkGetBufferMemoryRequirements(device, uniformBuffer.buffer, &memReqs);
    allocInfo.allocationSize = memReqs.size;
    // Get the memory type index that supports host visible memory access
    // Most implementations offer multiple memory types and selecting the correct one to allocate memory from is crucial
    // We also want the buffer to be host coherent so we don't have to flush (or sync after every update.
    // Note: This may affect performance so you might not want to do this in a real world application that updates buffers on a regular base
    allocInfo.memoryTypeIndex = getMemoryTypeIndex(memReqs.memoryTypeBits,
                                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    // Allocate memory for the uniform buffer
    VK_CHECK_RESULT(vkAllocateMemory(device, &allocInfo, nullptr, &(uniformBuffer.memory)));
    // Bind memory to buffer
    VK_CHECK_RESULT(
            vkBindBufferMemory(device, uniformBuffer.buffer, uniformBuffer.memory, 0));
    // We map the buffer once, so we can update it without having to map it again
    VK_CHECK_RESULT(vkMapMemory(device, uniformBuffer.memory, 0, sizeof(UBO), 0,
                                (void **) &uniformBuffer.mapped));
}

void VkSample19_ReadPixels::generateSpheres()
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
//    std::random_device rndDevice;
//    std::default_random_engine rndEngine(rndDevice());
//    std::uniform_real_distribution<float> rndDist(0.1f, 1.0f);
//    float delta = 1.0 / spheres.size();
//    for (uint32_t i = 0; i < spheres.size(); i++) {
//        spheres[i].color = glm::vec4(glm::vec3(delta * i + 0.2), 1.0f);
//        spheres[i].position = glm::vec4(glm::vec3(0, -0.65 + i * 0.45, 0.0f) * 3.5f, 1.0f);
//    }

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

void VkSample19_ReadPixels::setupDescriptors()
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

    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = uniformBuffer.buffer;
    bufferInfo.range = sizeof(UBO);
    bufferInfo.offset = 0;

    std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
            // Binding 0 : Vertex shader uniform buffer
            vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &bufferInfo),
            // Binding 1 : Fragment shader texture sampler
            //	Fragment shader: layout (binding = 1) uniform sampler2D samplerColor;
            vks::initializers::writeDescriptorSet(descriptorSet,
                                                  VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,		// The descriptor set will use a combined image sampler (as opposed to splitting image and sampler)
                                                  1,												// Shader binding point 1
                                                  &pTexture2D->descriptor)								// Pointer to the descriptor image for our texture
    };
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
}

void VkSample19_ReadPixels::createPipelines()
{
    // Layout
    // [POI] Define the push constant range used by the pipeline layout
    // Note that the spec only requires a minimum of 128 bytes, so for passing larger blocks of data you'd use UBOs or SSBOs
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayout, 1);
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
    shaderStages[0] = compileShader(device, getShadersPath() + "instancing/instancing.vert", VK_SHADER_STAGE_VERTEX_BIT);
    shaderStages[1] = compileShader(device, getShadersPath() + "instancing/instancing.frag", VK_SHADER_STAGE_FRAGMENT_BIT);

    // Vertex input state
    std::vector<VkVertexInputBindingDescription> vertexInputBindings = {
            // Binding point 0: Mesh vertex layout description at per-vertex rate
            vks::initializers::vertexInputBindingDescription(0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX),
            // Binding point 1: Instanced data at per-instance rate
            vks::initializers::vertexInputBindingDescription(1, sizeof(InstanceData), VK_VERTEX_INPUT_RATE_INSTANCE)
    };

    // Vertex attribute bindings
    // Note that the shader declaration for per-vertex and per-instance attributes is the same, the different input rates are only stored in the bindings:
    // instanced.vert:
    //	layout (location = 0) in vec3 inPos;		Per-Vertex
    //	...
    //	layout (location = 4) in vec3 instancePos;	Per-Instance
    std::vector<VkVertexInputAttributeDescription> vertexInputAttributes = {
            vks::initializers::vertexInputAttributeDescription(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos)),
            vks::initializers::vertexInputAttributeDescription(0, 1, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv)),
            vks::initializers::vertexInputAttributeDescription(0, 2, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)),

            // Per-Instance attributes
            // These are advanced for each instance rendered
            vks::initializers::vertexInputAttributeDescription(1, 3, VK_FORMAT_R32G32B32_SFLOAT, 0),					// Location 3: Position
            vks::initializers::vertexInputAttributeDescription(1, 4, VK_FORMAT_R32G32B32_SFLOAT, sizeof(float) * 3),	// Location 4: Rotation
            vks::initializers::vertexInputAttributeDescription(1, 5, VK_FORMAT_R32_SFLOAT,sizeof(float) * 6),			// Location 5: Scale
            vks::initializers::vertexInputAttributeDescription(1, 6, VK_FORMAT_R32G32B32_SFLOAT, sizeof(float) * 7),	// Location 6: Color

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

void VkSample19_ReadPixels::buildCommandBuffers()
{
    VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

    VkClearValue clearValues[2];
    defaultClearColor = { { 0.0f, 0.0f, 0.0f, 0.0f } };
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

        vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1,
                                &descriptorSet, 0, nullptr);
        vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        VkDeviceSize offsets[1] = {0};
        vkCmdBindVertexBuffers(drawCmdBuffers[i], 0, 1, &vertexBuffer.buffer, offsets);
        // Binding point 1 : Instance data buffer
        vkCmdBindVertexBuffers(drawCmdBuffers[i], 1, 1, &instanceBuffer.buffer, offsets);
        vkCmdDraw(drawCmdBuffers[i], vertexCount, INSTANCE_COUNT, 0, 0);

        vkCmdEndRenderPass(drawCmdBuffers[i]);

        VK_CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
    }
}

void VkSample19_ReadPixels::loadTexture()
{
    if(pTexture2D) {
        pTexture2D->destroy();
        delete pTexture2D;
    }

    pTexture2D = new vks::Texture2D();
    VK_CHECK_RESULT(loadRGBTexture2DFromFile("image/moon.jpg", pTexture2D))
    float viewportRatio = (float) outputWidth / (float) outputHeight;
    mvpMatrix.projectionMatrix = glm::perspective(45.0f,viewportRatio, 0.1f,256.f);;
    mvpMatrix.viewMatrix = glm::lookAt(
            glm::vec3(0, 0, 12.0), // Camera is at (0,0,9), in World Space
            glm::vec3(0, 0, 0), // and looks at the origin
            glm::vec3(0, 1, 0)  // Head is up (set to 0,-1,0 to look upside-down)
    );
    mvpMatrix.modelMatrix = glm::mat4(1.0f);
    uboData.mvpMatrix = mvpMatrix.projectionMatrix * mvpMatrix.viewMatrix * mvpMatrix.modelMatrix;
}

// Take a screenshot from the current swapchain image
// This is done using a blit from the swapchain image to a linear image whose memory content is then saved as a ppm image
// Getting the image date directly from a swapchain image wouldn't work as they're usually stored in an implementation dependent optimal tiling format
// Note: This requires the swapchain images to be created with the VK_IMAGE_USAGE_TRANSFER_SRC_BIT flag (see VulkanSwapChain::create)
void VkSample19_ReadPixels::readPixels(const char *filename)
{
    //screenshotSaved = false;
    bool supportsBlit = false;

    // Check blit support for source and destination
    VkFormatProperties formatProps;

		// Check if the device supports blitting from optimal images (the swapchain images are in optimal format)
		vkGetPhysicalDeviceFormatProperties(physicalDevice, swapChain.colorFormat, &formatProps);
		if (!(formatProps.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT)) {
			std::cerr << "Device does not support blitting from optimal tiled images, using copy instead of blit!" << std::endl;
			supportsBlit = false;
		}

		// Check if the device supports blitting to linear images
		vkGetPhysicalDeviceFormatProperties(physicalDevice, VK_FORMAT_R8G8B8A8_UNORM, &formatProps);
		if (!(formatProps.linearTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT)) {
			std::cerr << "Device does not support blitting to linear tiled images, using copy instead of blit!" << std::endl;
			supportsBlit = false;
		}

    // Source for the copy is the last rendered swapchain image
    VkImage srcImage = swapChain.images[currentBufferIdx];

    // Create the linear tiled destination image to copy to and to read the memory from
    VkImageCreateInfo imageCreateCI(vks::initializers::imageCreateInfo());
    imageCreateCI.imageType = VK_IMAGE_TYPE_2D;
    // Note that vkCmdBlitImage (if supported) will also do format conversions if the swapchain color format would differ
    imageCreateCI.format = VK_FORMAT_R8G8B8A8_UNORM;
    imageCreateCI.extent.width = outputWidth;
    imageCreateCI.extent.height = outputHeight;
    imageCreateCI.extent.depth = 1;
    imageCreateCI.arrayLayers = 1;
    imageCreateCI.mipLevels = 1;
    imageCreateCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageCreateCI.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreateCI.tiling = VK_IMAGE_TILING_LINEAR;
    imageCreateCI.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    // Create the image
    VkImage dstImage;
    VK_CHECK_RESULT(vkCreateImage(device, &imageCreateCI, nullptr, &dstImage));
    // Create memory to back up the image
    VkMemoryRequirements memRequirements;
    VkMemoryAllocateInfo memAllocInfo(vks::initializers::memoryAllocateInfo());
    VkDeviceMemory dstImageMemory;
    vkGetImageMemoryRequirements(device, dstImage, &memRequirements);
    memAllocInfo.allocationSize = memRequirements.size;
    // Memory must be host visible to copy from
    memAllocInfo.memoryTypeIndex = vulkanDevice->getMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VK_CHECK_RESULT(vkAllocateMemory(device, &memAllocInfo, nullptr, &dstImageMemory));
    VK_CHECK_RESULT(vkBindImageMemory(device, dstImage, dstImageMemory, 0));

    // Do the actual blit from the swapchain image to our host visible destination image
    VkCommandBuffer copyCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

    // Transition destination image to transfer destination layout
    vks::tools::insertImageMemoryBarrier(
            copyCmd,
            dstImage,
            0,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 });

    // Transition swapchain image from present to transfer source layout
    vks::tools::insertImageMemoryBarrier(
            copyCmd,
            srcImage,
            VK_ACCESS_MEMORY_READ_BIT,
            VK_ACCESS_TRANSFER_READ_BIT,
            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 });

    // If source and destination support blit we'll blit as this also does automatic format conversion (e.g. from BGR to RGB)
    if (supportsBlit)
    {
        // Define the region to blit (we will blit the whole swapchain image)
        VkOffset3D blitSize;
        blitSize.x = outputWidth;
        blitSize.y = outputHeight;
        blitSize.z = 1;
        VkImageBlit imageBlitRegion{};
        imageBlitRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageBlitRegion.srcSubresource.layerCount = 1;
        imageBlitRegion.srcOffsets[1] = blitSize;
        imageBlitRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageBlitRegion.dstSubresource.layerCount = 1;
        imageBlitRegion.dstOffsets[1] = blitSize;

        // Issue the blit command
        vkCmdBlitImage(
                copyCmd,
                srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                dstImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1,
                &imageBlitRegion,
                VK_FILTER_NEAREST);
    }
    else
    {
        // Otherwise use image copy (requires us to manually flip components)
        VkImageCopy imageCopyRegion{};
        imageCopyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageCopyRegion.srcSubresource.layerCount = 1;
        imageCopyRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageCopyRegion.dstSubresource.layerCount = 1;
        imageCopyRegion.extent.width = outputWidth;
        imageCopyRegion.extent.height = outputHeight;
        imageCopyRegion.extent.depth = 1;

        // Issue the copy command
        vkCmdCopyImage(
                copyCmd,
                srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                dstImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1,
                &imageCopyRegion);
    }

    // Transition destination image to general layout, which is the required layout for mapping the image memory later on
    vks::tools::insertImageMemoryBarrier(
            copyCmd,
            dstImage,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_MEMORY_READ_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 });

    // Transition back the swap chain image after the blit is done
    vks::tools::insertImageMemoryBarrier(
            copyCmd,
            srcImage,
            VK_ACCESS_TRANSFER_READ_BIT,
            VK_ACCESS_MEMORY_READ_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 });

    vulkanDevice->flushCommandBuffer(copyCmd, queue);

    // Get layout of the image (including row pitch)
    VkImageSubresource subResource { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0 };
    VkSubresourceLayout subResourceLayout;
    vkGetImageSubresourceLayout(device, dstImage, &subResource, &subResourceLayout);

    // Map image memory so we can start copying from it
    const char* data;
    vkMapMemory(device, dstImageMemory, 0, VK_WHOLE_SIZE, 0, (void**)&data);
    data += subResourceLayout.offset;

    std::ofstream file(filename, std::ios::out | std::ios::binary);

    // ppm header
    file << "P6\n" << outputWidth << "\n" << outputHeight << "\n" << 255 << "\n";

    // If source is BGR (destination is always RGB) and we can't use blit (which does automatic conversion), we'll have to manually swizzle color components
    bool colorSwizzle = false;
    // Check if source is BGR
    // Note: Not complete, only contains most common and basic BGR surface formats for demonstration purposes
    if (!supportsBlit)
    {
        std::vector<VkFormat> formatsBGR = { VK_FORMAT_B8G8R8A8_SRGB, VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_B8G8R8A8_SNORM };
        colorSwizzle = (std::find(formatsBGR.begin(), formatsBGR.end(), swapChain.colorFormat) != formatsBGR.end());
    }

    NativeImage nativeImage;
    nativeImage.width = outputWidth;
    nativeImage.height = outputHeight;
    nativeImage.format = IMAGE_FORMAT_RGBA;
    NativeImageUtil::AllocNativeImage(&nativeImage);

    // ppm binary pixel data
    for (uint32_t y = 0; y < outputHeight; y++)
    {

        unsigned int *row = (unsigned int*)data;
        unsigned int *dstRow = (unsigned int*)nativeImage.ppPlane[0] + y * nativeImage.width;
        memcpy(dstRow, row, outputWidth * 4);
        for (uint32_t x = 0; x < outputWidth; x++)
        {
            if (colorSwizzle)
            {
                file.write((char*)row+2, 1);
                file.write((char*)row+1, 1);
                file.write((char*)row, 1);
            }
            else
            {
                file.write((char*)row, 3);
            }
            row++;
        }
        data += subResourceLayout.rowPitch;
    }
    file.close();

    //NativeImageUtil::DumpNativeImage(&nativeImage, "/sdcard/Android/data/com.byteflow.vkapp/files/Download", "VK_readbuffer");
    NativeImageUtil::FreeNativeImage(&nativeImage);

    std::cout << "Screenshot saved to disk" << std::endl;

    // Clean up resources
    vkUnmapMemory(device, dstImageMemory);
    vkFreeMemory(device, dstImageMemory, nullptr);
    vkDestroyImage(device, dstImage, nullptr);

    //screenshotSaved = true;
}

void VkSample19_ReadPixels::prepare() {

    VulkanExampleBase::prepare();
    // Create a fence for synchronization
    VkFenceCreateInfo fenceCreateInfo = vks::initializers::fenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT);
    vkCreateFence(device, &fenceCreateInfo, nullptr, &renderFence);
    loadTexture();
    generateSpheres();
    createUniformBuffers();
    setupDescriptors();
    createPipelines();
    prepareInstanceData();
    buildCommandBuffers();
    prepared = true;
}

void VkSample19_ReadPixels::render() {
    if (!prepared)
        return;
    // Wait for fence to signal that all command buffers are ready
    VkResult fenceRes;
    do {
        fenceRes = vkWaitForFences(device, 1, &renderFence, VK_TRUE, 100000000);
    } while (fenceRes == VK_TIMEOUT);
    VK_CHECK_RESULT(fenceRes);
    vkResetFences(device, 1, &renderFence);

    VulkanExampleBase::prepareFrame();
    memcpy(uniformBuffer.mapped, &uboData, sizeof(UBO));
    updateInstanceData();
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &drawCmdBuffers[currentBufferIdx];
    VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, renderFence));
    VulkanExampleBase::submitFrame();
    if(fract(iTime) < 0.0001f) { //每秒钟读取一次
        readPixels("/sdcard/Android/data/com.byteflow.vkapp/files/Download/screenshot.ppm");
    }
}


