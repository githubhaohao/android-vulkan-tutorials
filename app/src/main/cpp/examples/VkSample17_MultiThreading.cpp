#include <fstream>
#include <vector>
#include <exception>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vulkan/vulkan.h>
#include "vulkanexamplebase.h"
#include "VkSample17_MultiThreading.h"

VkSample17_MultiThreading::VkSample17_MultiThreading() {
    LOGCATE("VkSample17_MultiThreading::VkSample17_MultiThreading()");
    title = "VkSample17_MultiThreading";
    pTexture2D = nullptr;
    // Get number of max. concurrent threads
    numThreads = std::thread::hardware_concurrency();
    assert(numThreads > 0);
    threadPool.setThreadCount(numThreads);
    numObjectsPerThread = 512 / numThreads;
    rndEngine.seed(benchmark.active ? 0 : (unsigned)time(nullptr));
}

VkSample17_MultiThreading::~VkSample17_MultiThreading() {
    LOGCATE("VkSample17_MultiThreading::~VkSample17_MultiThreading()");
    if (device) {
        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        vertexBuffer.destroy();

        for (auto& thread : threadData) {
            vkFreeCommandBuffers(device, thread.commandPool, static_cast<uint32_t>(thread.commandBuffer.size()), thread.commandBuffer.data());
            vkDestroyCommandPool(device, thread.commandPool, nullptr);
        }

        vkDestroyFence(device, renderFence, nullptr);
    }

    if(pTexture2D) {
        pTexture2D->destroy();
        delete pTexture2D;
        pTexture2D = nullptr;
    }
}

float VkSample17_MultiThreading::rnd(float range)
{
    std::uniform_real_distribution<float> rndDist(0.0f, range);
    return rndDist(rndEngine);
}

// Create all threads and initialize shader push constants
void VkSample17_MultiThreading::prepareMultiThreadedRenderer()
{
    // Since this demo updates the command buffers on each frame
    // we don't use the per-framebuffer command buffers from the
    // base class, and create a single primary command buffer instead
    VkCommandBufferAllocateInfo cmdBufAllocateInfo =
            vks::initializers::commandBufferAllocateInfo(
                    cmdPool,
                    VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                    1);
    VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &cmdBufAllocateInfo, &primaryCommandBuffer));

    // Create additional secondary CBs for background and ui
    cmdBufAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
    VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &cmdBufAllocateInfo, &secondaryCommandBuffers.background));
    VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &cmdBufAllocateInfo, &secondaryCommandBuffers.ui));

    threadData.resize(numThreads);

    float maxX = static_cast<float>(std::floor(std::sqrt(numThreads * numObjectsPerThread)));
    uint32_t posX = 0;
    uint32_t posZ = 0;

    /// 每个线程包含多个 Obj , 每个 Obj 包含一个 commandBuffer 和一个 pushConstBlock

    for (uint32_t i = 0; i < numThreads; i++) {
        ThreadData *thread = &threadData[i];

        // Create one command pool for each thread
        VkCommandPoolCreateInfo cmdPoolInfo = vks::initializers::commandPoolCreateInfo();
        cmdPoolInfo.queueFamilyIndex = swapChain.queueNodeIndex;
        cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        VK_CHECK_RESULT(vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &thread->commandPool));

        // One secondary command buffer per object that is updated by this thread
        thread->commandBuffer.resize(numObjectsPerThread);
        // Generate secondary command buffers for each thread
        VkCommandBufferAllocateInfo secondaryCmdBufAllocateInfo =
                vks::initializers::commandBufferAllocateInfo(
                        thread->commandPool,
                        VK_COMMAND_BUFFER_LEVEL_SECONDARY,
                        static_cast<uint32_t>(thread->commandBuffer.size()));
        VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &secondaryCmdBufAllocateInfo, thread->commandBuffer.data()));

        thread->pushConstBlock.resize(numObjectsPerThread);
        thread->objectData.resize(numObjectsPerThread);

        for (uint32_t j = 0; j < numObjectsPerThread; j++) {
            float theta = 2.0f * float(M_PI) * rnd(1.0f);
            float phi = acos(1.0f - 2.0f * rnd(1.0f));
            thread->objectData[j].pos = glm::vec3(sin(phi) * cos(theta), 0.0f, cos(phi)) * 35.0f;

            thread->objectData[j].rotation = glm::vec3(0.0f, rnd(360.0f), 0.0f);
            thread->objectData[j].deltaT = rnd(1.0f);
            thread->objectData[j].rotationDir = (rnd(100.0f) < 50.0f) ? 1.0f : -1.0f;
            thread->objectData[j].rotationSpeed = (2.0f + rnd(4.0f)) * thread->objectData[j].rotationDir;
            thread->objectData[j].scale = 0.75f + rnd(0.5f);

            thread->pushConstBlock[j].color = glm::vec3(rnd(1.0f), rnd(1.0f), rnd(1.0f));
        }
    }

}

// Builds the secondary command buffer for each thread
void VkSample17_MultiThreading::threadRenderCode(uint32_t threadIndex, uint32_t cmdBufferIndex, VkCommandBufferInheritanceInfo inheritanceInfo)
{
    ThreadData *thread = &threadData[threadIndex];
    ObjectData *objectData = &thread->objectData[cmdBufferIndex];

    VkCommandBufferBeginInfo commandBufferBeginInfo = vks::initializers::commandBufferBeginInfo();
    commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT;
    commandBufferBeginInfo.pInheritanceInfo = &inheritanceInfo;

    VkCommandBuffer cmdBuffer = thread->commandBuffer[cmdBufferIndex];

    VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuffer, &commandBufferBeginInfo));

    VkViewport viewport = vks::initializers::viewport((float)outputWidth, (float)outputHeight, 0.0f, 1.0f);
    vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);

    VkRect2D scissor = vks::initializers::rect2D(outputWidth, outputHeight, 0, 0);
    vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

    // Update
    if (!paused) {
        objectData->rotation.y += 0.01f * objectData->rotationSpeed * this->iTime;
        if (objectData->rotation.y > 360.0f) {
            objectData->rotation.y -= 360.0f;
        }
        objectData->deltaT += 0.001f * this->iTime;
        if (objectData->deltaT > 1.0f)
            objectData->deltaT -= 1.0f;
        objectData->pos.y = sin(glm::radians(objectData->deltaT * 360.0f)) * 2.5f;
    }

    objectData->model = glm::translate(glm::mat4(1.0f), objectData->pos);
    objectData->model = glm::rotate(objectData->model, -sinf(glm::radians(objectData->deltaT * 360.0f)) * 0.25f, glm::vec3(objectData->rotationDir, 0.0f, 0.0f));
    objectData->model = glm::rotate(objectData->model, glm::radians(objectData->rotation.y), glm::vec3(0.0f, objectData->rotationDir, 0.0f));
    objectData->model = glm::rotate(objectData->model, glm::radians(objectData->deltaT * 360.0f), glm::vec3(0.0f, objectData->rotationDir, 0.0f));
    objectData->model = glm::scale(objectData->model, glm::vec3(objectData->scale));

    thread->pushConstBlock[cmdBufferIndex].mvp = mvpMatrix.projectionMatrix * mvpMatrix.viewMatrix * mvpMatrix.modelMatrix * objectData->model;

    // Update shader push constant block
    // Contains model view matrix
    vkCmdPushConstants(
            cmdBuffer,
            pipelineLayout,
            VK_SHADER_STAGE_VERTEX_BIT,
            0,
            sizeof(ThreadPushConstantBlock),
            &thread->pushConstBlock[cmdBufferIndex]);

    VkDeviceSize offsets[1] = { 0 };
    vkCmdBindVertexBuffers(cmdBuffer, 0, 1, &vertexBuffer.buffer, offsets);
    vkCmdDraw(cmdBuffer, vertexCount, 1, 0, 0);

    VK_CHECK_RESULT(vkEndCommandBuffer(cmdBuffer));
}

void VkSample17_MultiThreading::updateTransformMatrix(float rotateX, float rotateY, float scaleX, float scaleY)
{
    LOGCATE("VkSample17_MultiThreading::updateTransformMatrix()");
    float radiansX = static_cast<float>(MATH_PI / 180.0f * rotateX);
    float radiansY = static_cast<float>(MATH_PI / 180.0f * rotateY);
    glm::mat4 Model = glm::mat4(1.0f);
    Model = glm::scale(Model, glm::vec3(scaleX, scaleY, scaleY));
    Model = glm::rotate(Model, radiansX, glm::vec3(1.0f, 0.0f, 0.0f));
    Model = glm::rotate(Model, radiansY, glm::vec3(0.0f, 1.0f, 0.0f));
    Model = glm::translate(Model, glm::vec3(0.0f, 0.0f, 0.0f));
    mvpMatrix.modelMatrix = Model;
    //uniformData.mvpMatrix = mvpMatrix.projectionMatrix * mvpMatrix.viewMatrix * mvpMatrix.modelMatrix;
}

// Updates the secondary command buffers using a thread pool
// and puts them into the primary command buffer that's
// lat submitted to the queue for rendering
void VkSample17_MultiThreading::updateCommandBuffers(VkFramebuffer frameBuffer)
{
    // Contains the list of secondary command buffers to be submitted
    std::vector<VkCommandBuffer> commandBuffers;

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
    renderPassBeginInfo.framebuffer = frameBuffer;

    // Set target frame buffer

    VK_CHECK_RESULT(vkBeginCommandBuffer(primaryCommandBuffer, &cmdBufInfo));

    // The primary command buffer does not contain any rendering commands
    // These are stored (and retrieved) from the secondary command buffers
    /// 主命令缓冲区不包含任何渲染命令这些命令从辅助命令缓冲区存储（和检索）
    vkCmdBeginRenderPass(primaryCommandBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS);
    //
    /// VkCommandBufferInheritanceInfo 是 Vulkan API 中的一个结构体，它在次级命令缓冲区（secondary command buffer）被记录时，
    /// 用于指定从主命令缓冲区（primary command buffer）继承的状态。这个结构体主要用于优化和提高命令缓冲区的重用效率，尤其是在渲染过程中。
    // Inheritance info for the secondary command buffers
    VkCommandBufferInheritanceInfo inheritanceInfo = vks::initializers::commandBufferInheritanceInfo();
    inheritanceInfo.renderPass = renderPass;
    // Secondary command buffer also use the currently active framebuffer
    inheritanceInfo.framebuffer = frameBuffer;


    // Add a job to the thread's queue for each object to be rendered
    for (uint32_t t = 0; t < numThreads; t++)
    {
        for (uint32_t i = 0; i < numObjectsPerThread; i++)
        {
            threadPool.threads[t]->addJob([=] { threadRenderCode(t, i, inheritanceInfo); });
        }
    }

    threadPool.wait();

    // Only submit if object is within the current view frustum
    for (uint32_t t = 0; t < numThreads; t++)
    {
        for (uint32_t i = 0; i < numObjectsPerThread; i++)
        {
            if (threadData[t].objectData[i].visible)
            {
                commandBuffers.push_back(threadData[t].commandBuffer[i]);
            }
        }
    }

    ///vkCmdExecuteCommands 函数在 Vulkan 中用于在主命令缓冲区中执行次级命令缓冲区。这允许在一个主命令缓冲区中嵌入预先录制的次级命令缓冲区，从而提高效率和代码组织的灵活性。
    // Execute render commands from the secondary command buffer
    vkCmdExecuteCommands(primaryCommandBuffer, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());

    vkCmdEndRenderPass(primaryCommandBuffer);

    VK_CHECK_RESULT(vkEndCommandBuffer(primaryCommandBuffer));
}

void VkSample17_MultiThreading::generateSpheres()
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

void VkSample17_MultiThreading::setupDescriptors()
{
    // Pool
    std::vector<VkDescriptorPoolSize> poolSizes = {
            vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1)
    };
    VkDescriptorPoolCreateInfo descriptorPoolInfo = vks::initializers::descriptorPoolCreateInfo(poolSizes, 1);
    VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));

    // Layout
    std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
            // Binding 0 : Fragment shader image sampler
            vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0)
    };
    VkDescriptorSetLayoutCreateInfo descriptorLayout = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &descriptorSetLayout));

    // Set
    VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayout, 1);
    VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet));

    std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
            // Binding 0 : Fragment shader texture sampler
            //	Fragment shader: layout (binding = 0) uniform sampler2D samplerColor;
            vks::initializers::writeDescriptorSet(descriptorSet,
                                                  VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,		// The descriptor set will use a combined image sampler (as opposed to splitting image and sampler)
                                                  0,												// Shader binding point 1
                                                  &pTexture2D->descriptor)								// Pointer to the descriptor image for our texture
    };
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
}

void VkSample17_MultiThreading::createPipelines()
{
    // Layout
    // [POI] Define the push constant range used by the pipeline layout
    // Note that the spec only requires a minimum of 128 bytes, so for passing larger blocks of data you'd use UBOs or SSBOs

    // Push constants for model matrices
    VkPushConstantRange pushConstantRange = vks::initializers::pushConstantRange( VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(ThreadPushConstantBlock), 0);
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
    shaderStages[0] = compileShader(device, getShadersPath() + "multithreading/multithreading.vert", VK_SHADER_STAGE_VERTEX_BIT);
    shaderStages[1] = compileShader(device, getShadersPath() + "multithreading/multithreading.frag", VK_SHADER_STAGE_FRAGMENT_BIT);

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

void VkSample17_MultiThreading::loadTexture()
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
            glm::vec3(0, 0, 8), // Camera is at (0,0,9), in World Space
            glm::vec3(0, 0, 0), // and looks at the origin
            glm::vec3(0, 1, 0)  // Head is up (set to 0,-1,0 to look upside-down)
    );
    mvpMatrix.modelMatrix = glm::mat4(1.0f);
    //uniformData.mvpMatrix = mvpMatrix.projectionMatrix * mvpMatrix.viewMatrix * mvpMatrix.modelMatrix;
}

void VkSample17_MultiThreading::prepare() {

    VulkanExampleBase::prepare();
    // Create a fence for synchronization
    VkFenceCreateInfo fenceCreateInfo = vks::initializers::fenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT);
    vkCreateFence(device, &fenceCreateInfo, nullptr, &renderFence);
    loadTexture();
    generateSpheres();
    setupDescriptors();
    createPipelines();
    prepareMultiThreadedRenderer();
    prepared = true;
}

void VkSample17_MultiThreading::render() {
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

    updateCommandBuffers(frameBuffers[currentBufferIdx]);

    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &primaryCommandBuffer;
    VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, renderFence));
    VulkanExampleBase::submitFrame();
}


