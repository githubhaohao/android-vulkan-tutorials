#include <fstream>
#include <vector>
#include <exception>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vulkan/vulkan.h>
#include "vulkanexamplebase.h"
#include "VkSample14_DepthTesting.h"

VkSample14_DepthTesting::VkSample14_DepthTesting() {

    title = "VkSample14_DepthTesting";
    pTexture2D = nullptr;
}

VkSample14_DepthTesting::~VkSample14_DepthTesting() {
    // Clean up used Vulkan resources
    // Note: Inherited destructor cleans up resources stored in base class
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipeline(device, pipeline2, nullptr);
    vkDestroyPipeline(device, pipeline3, nullptr);

    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

    vkDestroyBuffer(device, vertices.buffer, nullptr);
    vkFreeMemory(device, vertices.memory, nullptr);

    vkDestroyBuffer(device, uniformBuffer.buffer, nullptr);
    vkFreeMemory(device, uniformBuffer.memory, nullptr);

    if(pTexture2D) {
        pTexture2D->destroy();
        delete pTexture2D;
        pTexture2D = nullptr;
    }
    //destroyTexture(texture);
}

void VkSample14_DepthTesting::updateTransformMatrix(float rotateX, float rotateY, float scaleX, float scaleY)
{
    LOGCATE("VkSample02_Texture::updateTransformMatrix()");
    float radiansX = static_cast<float>(MATH_PI / 180.0f * rotateX);
    float radiansY = static_cast<float>(MATH_PI / 180.0f * rotateY);
    glm::mat4 Model = glm::mat4(1.0f);
    Model = glm::scale(Model, glm::vec3(scaleX, scaleX, 1.0f));
    Model = glm::rotate(Model, radiansX, glm::vec3(1.0f, 0.0f, 0.0f));
    Model = glm::rotate(Model, radiansY, glm::vec3(0.0f, 1.0f, 0.0f));
    Model = glm::translate(Model, glm::vec3(0.0f, 0.0f, 0.0f));
    uboData.modelMatrix = Model;
}

// Prepare vertex and index buffers for an indexed triangle
// Also uploads them to device local memory using staging and initializes vertex input and attribute binding to match the vertex shader
void VkSample14_DepthTesting::createVertexBuffer() {
    // A note on memory management in Vulkan in general:
    //	This is a very complex topic and while it's fine for an example application to small individual memory allocations that is not
    //	what should be done a real-world application, where you should allocate large chunks of memory at once instead.

    // Setup vertices
    std::vector<Vertex> vertexBuffer{
            {{ -0.5f, -0.5f, -0.5f},   {0.0f, 0.0f},     { 0.0f,  0.0f, -1.0f} },
            {{  0.5f, -0.5f, -0.5f},   {1.0f, 0.0f},     { 0.0f,  0.0f, -1.0f} },
            {{  0.5f,  0.5f, -0.5f},   {1.0f, 1.0f},     { 0.0f,  0.0f, -1.0f} },
            {{  0.5f,  0.5f, -0.5f},   {1.0f, 1.0f},     { 0.0f,  0.0f, -1.0f} },
            {{ -0.5f,  0.5f, -0.5f},   {0.0f, 1.0f},     { 0.0f,  0.0f, -1.0f} },
            {{ -0.5f, -0.5f, -0.5f},   {0.0f, 0.0f},     { 0.0f,  0.0f, -1.0f} },

            {{ -0.5f, -0.5f, 0.5f},    {0.0f, 0.0f},     { 0.0f,  0.0f,  1.0f} },
            {{  0.5f, -0.5f, 0.5f},    {1.0f, 0.0f},     { 0.0f,  0.0f,  1.0f} },
            {{  0.5f,  0.5f, 0.5f},    {1.0f, 1.0f},     { 0.0f,  0.0f,  1.0f} },
            {{  0.5f,  0.5f, 0.5f},    {1.0f, 1.0f},     { 0.0f,  0.0f,  1.0f} },
            {{ -0.5f,  0.5f, 0.5f},    {0.0f, 1.0f},     { 0.0f,  0.0f,  1.0f} },
            {{ -0.5f, -0.5f, 0.5f},    {0.0f, 0.0f},     { 0.0f,  0.0f,  1.0f} },

            {{ -0.5f,  0.5f,  0.5f},   {1.0f, 0.0f},     {-1.0f,  0.0f,  0.0f} },
            {{ -0.5f,  0.5f, -0.5f},   {1.0f, 1.0f},     {-1.0f,  0.0f,  0.0f} },
            {{ -0.5f, -0.5f, -0.5f},   {0.0f, 1.0f},     {-1.0f,  0.0f,  0.0f} },
            {{ -0.5f, -0.5f, -0.5f},   {0.0f, 1.0f},     {-1.0f,  0.0f,  0.0f} },
            {{ -0.5f, -0.5f,  0.5f},   {0.0f, 0.0f},     {-1.0f,  0.0f,  0.0f} },
            {{ -0.5f,  0.5f,  0.5f},   {1.0f, 0.0f},     {-1.0f,  0.0f,  0.0f} },

            {{ 0.5f,  0.5f,  0.5f},    {1.0f, 0.0f},     { 1.0f,  0.0f,  0.0f} },
            {{ 0.5f,  0.5f, -0.5f},    {1.0f, 1.0f},     { 1.0f,  0.0f,  0.0f} },
            {{ 0.5f, -0.5f, -0.5f},    {0.0f, 1.0f},     { 1.0f,  0.0f,  0.0f} },
            {{ 0.5f, -0.5f, -0.5f},    {0.0f, 1.0f},     { 1.0f,  0.0f,  0.0f} },
            {{ 0.5f, -0.5f,  0.5f},    {0.0f, 0.0f},     { 1.0f,  0.0f,  0.0f} },
            {{ 0.5f,  0.5f,  0.5f},    {1.0f, 0.0f},     { 1.0f,  0.0f,  0.0f} },

            {{ -0.5f, -0.5f, -0.5f},   {0.0f, 1.0f},     { 0.0f, -1.0f,  0.0f} },
            {{  0.5f, -0.5f, -0.5f},   {1.0f, 1.0f},     { 0.0f, -1.0f,  0.0f} },
            {{  0.5f, -0.5f,  0.5f},   {1.0f, 0.0f},     { 0.0f, -1.0f,  0.0f} },
            {{  0.5f, -0.5f,  0.5f},   {1.0f, 0.0f},     { 0.0f, -1.0f,  0.0f} },
            {{ -0.5f, -0.5f,  0.5f},   {0.0f, 0.0f},     { 0.0f, -1.0f,  0.0f} },
            {{ -0.5f, -0.5f, -0.5f},   {0.0f, 1.0f},     { 0.0f, -1.0f,  0.0f} },

            {{ -0.5f, 0.5f, -0.5f},    {0.0f, 1.0f},     { 0.0f,  1.0f,  0.0f} },
            {{  0.5f, 0.5f, -0.5f},    {1.0f, 1.0f},     { 0.0f,  1.0f,  0.0f} },
            {{  0.5f, 0.5f,  0.5f},    {1.0f, 0.0f},     { 0.0f,  1.0f,  0.0f} },
            {{  0.5f, 0.5f,  0.5f},    {1.0f, 0.0f},     { 0.0f,  1.0f,  0.0f} },
            {{ -0.5f, 0.5f,  0.5f},    {0.0f, 0.0f},     { 0.0f,  1.0f,  0.0f} },
            {{ -0.5f, 0.5f, -0.5f},    {0.0f, 1.0f},     { 0.0f,  1.0f,  0.0f} },
    };

    //为了展示效果，再绘制一个小立方体
    std::vector<Vertex> vertexBuffe2 = vertexBuffer;
    for (int i = 0; i < vertexBuffe2.size(); ++i) {
        vertexBuffe2[i].pos[0] *= 0.5f;
        vertexBuffe2[i].pos[1] *= 0.5f;
        vertexBuffe2[i].pos[2] *= 0.5f;
        vertexBuffe2[i].pos[0] += 0.5f;
        vertexBuffe2[i].pos[2] += 0.5f;
        vertexBuffer.push_back(vertexBuffe2[i]);
    }

    uint32_t vertexBufferSize = static_cast<uint32_t>(vertexBuffer.size()) * sizeof(Vertex);
    vertices.count = vertexBuffer.size();

    VkMemoryAllocateInfo memAlloc{};
    memAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    VkMemoryRequirements memReqs;

    void *data;

    // Create a vertex buffer
    VkBufferCreateInfo vertexBufferInfoCI{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .size = vertexBufferSize,
            .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 1,
            .pQueueFamilyIndices = &vulkanDevice->queueFamilyIndices.graphics,
    };

    VK_CHECK_RESULT(
            vkCreateBuffer(device, &vertexBufferInfoCI, nullptr, &vertices.buffer));
    vkGetBufferMemoryRequirements(device, vertices.buffer, &memReqs);
    memAlloc.allocationSize = memReqs.size;
    // Assign the proper memory type for that buffer
    mapMemoryTypeToIndex(memReqs.memoryTypeBits,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         &memAlloc.memoryTypeIndex);
    VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &vertices.memory));
    // Map and copy
    VK_CHECK_RESULT(
            vkMapMemory(device, vertices.memory, 0, memAlloc.allocationSize, 0,
                        &data));
    memcpy(data, vertexBuffer.data(), vertexBufferSize);
    vkUnmapMemory(device, vertices.memory);
    VK_CHECK_RESULT(vkBindBufferMemory(device, vertices.buffer,
                                       vertices.memory, 0));
}

void VkSample14_DepthTesting::createDescriptors() {
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
    // Setup a descriptor image info for the current texture to be used as a combined image sampler
//    VkDescriptorImageInfo textureDescriptor;
//    // The image's view (images are never directly accessed by the shader, but rather through views defining subresources)
//    textureDescriptor.imageView = texture.view;
//    // The sampler (Telling the pipeline how to sample the texture, including repeat, border, etc.)
//    textureDescriptor.sampler = texture.sampler;
//    // The current layout of the image(Note: Should always fit the actual use, e.g.shader read)
//    textureDescriptor.imageLayout = texture.imageLayout;
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
    LOGCATE("VkSample02_Texture::createDescriptors() %d", __LINE__);
}

void VkSample14_DepthTesting::createPipelines() {
    // Layout
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
    shaderStages[0] = compileShader(device, getShadersPath() + "depthtesting/texture.vert", VK_SHADER_STAGE_VERTEX_BIT);
    shaderStages[1] = compileShader(device, getShadersPath() + "depthtesting/texture.frag", VK_SHADER_STAGE_FRAGMENT_BIT);

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

    // Prepare specialization constants data
    // Host data to take specialization constants from
    struct SpecializationData {
        //设置是否显示深度值
        uint32_t model{ 0 };
    } specializationData;

    // Each shader constant of a shader stage corresponds to one map entry
    std::array<VkSpecializationMapEntry, 1> specializationMapEntries;
    // Shader bindings based on specialization constants are marked by the new "constant_id" layout qualifier:
    //layout (constant_id = 0) const int MODEL = 0;

    // Map entry for the model to be used by the fragment shader
    specializationMapEntries[0].constantID = 0;
    specializationMapEntries[0].size = sizeof(specializationData.model);
    specializationMapEntries[0].offset = 0;

    // VkSpecializationInfo 可以在管线创建时，设置特定的常量值，这样可以在运行时动态改变着色器的行为
    VkSpecializationInfo specializationInfo{};
    specializationInfo.dataSize = sizeof(specializationData);
    specializationInfo.mapEntryCount = static_cast<uint32_t>(specializationMapEntries.size());
    specializationInfo.pMapEntries = specializationMapEntries.data();
    specializationInfo.pData = &specializationData;
    shaderStages[1].pSpecializationInfo = &specializationInfo;
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipeline));

    //创建第二个 pipeline, 显示深度值
    specializationData.model = 1;
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipeline2));

    //创建第三个 pipeline, 关闭深度测试
    depthStencilState.depthTestEnable = VK_FALSE;
    specializationData.model = 0;
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipeline3));
}

void VkSample14_DepthTesting::createUniformBuffers() {
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

void VkSample14_DepthTesting::buildCommandBuffers()
{
    VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

    VkClearValue clearValues[2];
    defaultClearColor = { { 1.0f, 1.0f, 1.0f, 1.0f } };
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
        // viewport is set to half the screen
        viewport.height = outputHeight / 3;
        vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);

        VkRect2D scissor = vks::initializers::rect2D(outputWidth, outputHeight, 0, 0);
        vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);

        vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1,
                                &descriptorSet, 0, nullptr);
        vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        VkDeviceSize offsets[1] = {0};
        vkCmdBindVertexBuffers(drawCmdBuffers[i], 0, 1, &vertices.buffer, offsets);
        vkCmdDraw(drawCmdBuffers[i], vertices.count, 1, 0, 0);

        // Switch to use the second pipeline for drawing. IndexBuffer, vertexBuffer, and uniformBuffer use the above settings by default.
        viewport.y = outputHeight / 3;
        vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);
        vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline2);
        vkCmdDraw(drawCmdBuffers[i], vertices.count, 1, 0, 0);

        viewport.y = outputHeight / 3 * 2;
        vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);
        vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline3);
        vkCmdDraw(drawCmdBuffers[i], vertices.count, 1, 0, 0);

        vkCmdEndRenderPass(drawCmdBuffers[i]);

        VK_CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
    }
}

void VkSample14_DepthTesting::loadTexture()
{
    if(pTexture2D) {
        pTexture2D->destroy();
        delete pTexture2D;
    }

    pTexture2D = new vks::Texture2D();
    VK_CHECK_RESULT(loadRGBTexture2DFromFile("image/board.png", pTexture2D))
    float viewportRatio = (float) outputWidth / (float) outputHeight * 3.0;
    float imageRatio = (float) pTexture2D->width / (float) pTexture2D->height;
    float ratio = viewportRatio / imageRatio;
    //glm::mat4 Projection = glm::ortho(-ratio, ratio, -1.0f, 1.0f, 0.0f, 100.0f);
    //glm::mat4 Projection = glm::frustum(-ratio, ratio, -1.0f, 1.0f, 4.0f, 100.0f);
    glm::mat4 Projection = glm::perspective(45.0f, ratio, 1.0f, 256.f);

    // View matrix
    glm::mat4 View = glm::lookAt(
            glm::vec3(0, 0, 2.5), // Camera is at (0,0,1), in World Space
            glm::vec3(0, 0, 0), // and looks at the origin
            glm::vec3(0, 1, 0)  // Head is up (set to 0,-1,0 to look upside-down)
    );

    uboData.projectionMatrix = Projection;
    uboData.viewMatrix = View;
    uboData.modelMatrix = glm::mat4(1.0f);
}

void VkSample14_DepthTesting::prepare() {
    VulkanExampleBase::prepare();
    loadTexture();
    createVertexBuffer();
    createUniformBuffers();
    createDescriptors();
    createPipelines();
    buildCommandBuffers();
    prepared = true;
}

void VkSample14_DepthTesting::render() {
    if (!prepared)
        return;
    VulkanExampleBase::prepareFrame();
    uboData.iTime = this->iTime;
    memcpy(uniformBuffer.mapped, &uboData, sizeof(UBO));
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &drawCmdBuffers[currentBufferIdx];
    VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
    VulkanExampleBase::submitFrame();
}


