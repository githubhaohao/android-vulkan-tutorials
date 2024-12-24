#include <fstream>
#include <vector>
#include <exception>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vulkan/vulkan.h>
#include "vulkanexamplebase.h"
#include "VkSample12_InputAttachments.h"

VkSample12_InputAttachments::VkSample12_InputAttachments() {

    title = "VkSample12_InputAttachments";
    pTexture2D = nullptr;
}

VkSample12_InputAttachments::~VkSample12_InputAttachments() {
    // Clean up used Vulkan resources
    // Note: Inherited destructor cleans up resources stored in base class
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipeline(device, pipeline2, nullptr);

    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout2, nullptr);

    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout2, nullptr);

    vkDestroyBuffer(device, vertices.buffer, nullptr);
    vkFreeMemory(device, vertices.memory, nullptr);

    vkDestroyBuffer(device, indices.buffer, nullptr);
    vkFreeMemory(device, indices.memory, nullptr);

    vkDestroyBuffer(device, uniformBuffer.buffer, nullptr);
    vkFreeMemory(device, uniformBuffer.memory, nullptr);

    for (uint32_t i = 0; i < attachments.size(); i++) {
        vkDestroyImageView(device, attachments[i].color.view, nullptr);
        vkDestroyImage(device, attachments[i].color.image, nullptr);
        vkFreeMemory(device, attachments[i].color.memory, nullptr);
        vkDestroyImageView(device, attachments[i].depth.view, nullptr);
        vkDestroyImage(device, attachments[i].depth.image, nullptr);
        vkFreeMemory(device, attachments[i].depth.memory, nullptr);
    }

    if(pTexture2D) {
        pTexture2D->destroy();
        delete pTexture2D;
        pTexture2D = nullptr;
    }
    //destroyTexture(texture);
}

void VkSample12_InputAttachments::clearAttachment(FrameBufferAttachment* attachment)
{
    vkDestroyImageView(device, attachment->view, nullptr);
    vkDestroyImage(device, attachment->image, nullptr);
    vkFreeMemory(device, attachment->memory, nullptr);
}

// Create a frame buffer attachment
void VkSample12_InputAttachments::createAttachment(VkFormat format, VkImageUsageFlags usage, FrameBufferAttachment *attachment)
{
    VkImageAspectFlags aspectMask = 0;
    VkImageLayout imageLayout;

    attachment->format = format;

    if (usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) {
        aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }
    if (usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) {
        aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    }

    VkImageCreateInfo imageCI = vks::initializers::imageCreateInfo();
    imageCI.imageType = VK_IMAGE_TYPE_2D;
    imageCI.format = format;
    imageCI.extent.width = outputWidth;
    imageCI.extent.height = outputHeight;
    imageCI.extent.depth = 1;
    imageCI.mipLevels = 1;
    imageCI.arrayLayers = 1;
    imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
    // VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT flag is required for input attachments;
    imageCI.usage = usage | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
    imageCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VK_CHECK_RESULT(vkCreateImage(device, &imageCI, nullptr, &attachment->image));

    VkMemoryAllocateInfo memAlloc = vks::initializers::memoryAllocateInfo();
    VkMemoryRequirements memReqs;
    vkGetImageMemoryRequirements(device, attachment->image, &memReqs);
    memAlloc.allocationSize = memReqs.size;
    memAlloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &attachment->memory));
    VK_CHECK_RESULT(vkBindImageMemory(device, attachment->image, attachment->memory, 0));

    VkImageViewCreateInfo imageViewCI = vks::initializers::imageViewCreateInfo();
    imageViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
    imageViewCI.format = format;
    imageViewCI.subresourceRange = {};
    imageViewCI.subresourceRange.aspectMask = aspectMask;
    imageViewCI.subresourceRange.baseMipLevel = 0;
    imageViewCI.subresourceRange.levelCount = 1;
    imageViewCI.subresourceRange.baseArrayLayer = 0;
    imageViewCI.subresourceRange.layerCount = 1;
    imageViewCI.image = attachment->image;
    VK_CHECK_RESULT(vkCreateImageView(device, &imageViewCI, nullptr, &attachment->view));
}

void VkSample12_InputAttachments::updateTransformMatrix(float rotateX, float rotateY, float scaleX, float scaleY)
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
void VkSample12_InputAttachments::createVertexBuffer() {
    // A note on memory management in Vulkan in general:
    //	This is a very complex topic and while it's fine for an example application to small individual memory allocations that is not
    //	what should be done a real-world application, where you should allocate large chunks of memory at once instead.

    // Setup vertices
    std::vector<Vertex> vertexBuffer{
            { {  1.0f,  1.0f, 0.0f }, { 1.0f, 1.0f },{ 0.0f, 0.0f, 1.0f } },
            { { -1.0f,  1.0f, 0.0f }, { 0.0f, 1.0f },{ 0.0f, 0.0f, 1.0f } },
            { { -1.0f, -1.0f, 0.0f }, { 0.0f, 0.0f },{ 0.0f, 0.0f, 1.0f } },
            { {  1.0f, -1.0f, 0.0f }, { 1.0f, 0.0f },{ 0.0f, 0.0f, 1.0f } }
    };
    uint32_t vertexBufferSize = static_cast<uint32_t>(vertexBuffer.size()) * sizeof(Vertex);

    // Setup indices
    std::vector<uint32_t> indexBuffer { 0,1,2, 2,3,0 };
    indices.count = static_cast<uint32_t>(indexBuffer.size());
    uint32_t indexBufferSize = indices.count * sizeof(uint32_t);

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

    // Index buffer
    VkBufferCreateInfo indexbufferCI{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .size = indexBufferSize,
            .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 1,
            .pQueueFamilyIndices = &vulkanDevice->queueFamilyIndices.graphics,
    };

    VK_CHECK_RESULT(
            vkCreateBuffer(device, &indexbufferCI, nullptr, &indices.buffer));
    vkGetBufferMemoryRequirements(device, indices.buffer, &memReqs);
    memAlloc.allocationSize = memReqs.size;
    mapMemoryTypeToIndex(memReqs.memoryTypeBits,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         &memAlloc.memoryTypeIndex);
    VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &indices.memory));
    VK_CHECK_RESULT(
            vkMapMemory(device, indices.memory, 0, indexBufferSize, 0, &data));
    memcpy(data, indexBuffer.data(), indexBufferSize);
    vkUnmapMemory(device, indices.memory);
    VK_CHECK_RESULT(
            vkBindBufferMemory(device, indices.buffer, indices.memory,
                               0));
}

void VkSample12_InputAttachments::updateAttachmentDescriptors(uint32_t index)
{
    // Image descriptors for the input attachments read by the shader
    std::vector<VkDescriptorImageInfo> descriptors = {
            vks::initializers::descriptorImageInfo(VK_NULL_HANDLE, attachments[index].color.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
            vks::initializers::descriptorImageInfo(VK_NULL_HANDLE, attachments[index].depth.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
    };

    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = uniformBuffer.buffer;
    bufferInfo.range = sizeof(UBO);
    bufferInfo.offset = 0;

    std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
            // Binding 0: Color input attachment
            vks::initializers::writeDescriptorSet(descriptorSet2s[index], VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 0, &descriptors[0]),
            // Binding 1: Depth input attachment
            vks::initializers::writeDescriptorSet(descriptorSet2s[index], VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1, &descriptors[1]),
            // Binding 2: Display parameters uniform buffer
            vks::initializers::writeDescriptorSet(descriptorSet2s[index], VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2, &bufferInfo),
    };
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
}

void VkSample12_InputAttachments::setupDescriptors()
{
    /*
        Pool 描述符池可以有冗余
    */
    std::vector<VkDescriptorPoolSize> poolSizes = {
            vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, static_cast<uint32_t>(attachments.size()) + 1),
            vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1),
            vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, static_cast<uint32_t>(attachments.size()) * 2 + 1),
    };
    VkDescriptorPoolCreateInfo descriptorPoolInfo = vks::initializers::descriptorPoolCreateInfo(static_cast<uint32_t>(poolSizes.size()), poolSizes.data(), static_cast<uint32_t>(attachments.size()) + 1);
    VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));

    /*
        Attachment write
    */
    {
        std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
                // Binding 0 : Vertex shader uniform buffer
                vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0),
                // Binding 1 : Fragment shader image sampler
                vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1)
        };
        VkDescriptorSetLayoutCreateInfo descriptorLayout = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &descriptorSetLayout));

        VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayout, 1);
        VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout));

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

    /*
        Attachment read
    */
    std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
            // Binding 0: Color input attachment
            vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, VK_SHADER_STAGE_FRAGMENT_BIT, 0),
            // Binding 1: Depth input attachment
            vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, VK_SHADER_STAGE_FRAGMENT_BIT, 1),
            // Binding 2: Display parameters uniform buffer
            vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 2),
    };
    VkDescriptorSetLayoutCreateInfo descriptorLayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayoutCI, nullptr, &descriptorSetLayout2));

    VkPipelineLayoutCreateInfo pipelineLayoutCI = vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayout2, 1);
    VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCI, nullptr, &pipelineLayout2));

    descriptorSet2s.resize(attachments.size());
    for (auto i = 0; i < descriptorSet2s.size(); i++) {
        VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayout2, 1);
        VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet2s[i]));
        updateAttachmentDescriptors(i);
    }
}

void VkSample12_InputAttachments::createPipelines() {
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
    shaderStages[0] = compileShader(device, getShadersPath() + "inputattachments/shader1.vert", VK_SHADER_STAGE_VERTEX_BIT);
    shaderStages[1] = compileShader(device, getShadersPath() + "inputattachments/shader1.frag", VK_SHADER_STAGE_FRAGMENT_BIT);

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
    pipelineCreateInfo.subpass = 0;
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipeline));

    //第二个子通道对应的 pipeline
    pipelineCreateInfo.subpass = 1;
    pipelineCreateInfo.layout = pipelineLayout2;

    VkPipelineVertexInputStateCreateInfo emptyInputStateCI{};
    emptyInputStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    pipelineCreateInfo.pVertexInputState = &emptyInputStateCI;
    colorBlendState.attachmentCount = 1;
    shaderStages[0] = compileShader(device, getShadersPath() + "inputattachments/shader2.vert", VK_SHADER_STAGE_VERTEX_BIT);
    shaderStages[1] = compileShader(device, getShadersPath() + "inputattachments/shader2.frag", VK_SHADER_STAGE_FRAGMENT_BIT);
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipeline2));
}

void VkSample12_InputAttachments::createUniformBuffers() {
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


// Override framebuffer setup from base class
void VkSample12_InputAttachments::setupFrameBuffer()
{
    VkImageView views[3];

    VkFramebufferCreateInfo frameBufferCI{};
    frameBufferCI.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    frameBufferCI.renderPass = renderPass;
    frameBufferCI.attachmentCount = 3;
    frameBufferCI.pAttachments = views;
    frameBufferCI.width = outputWidth;
    frameBufferCI.height = outputHeight;
    frameBufferCI.layers = 1;

    frameBuffers.resize(swapChain.imageCount);
    for (uint32_t i = 0; i < frameBuffers.size(); i++)
    {
        views[0] = swapChain.buffers[i].view;
        views[1] = attachments[i].color.view;
        views[2] = attachments[i].depth.view;
        VK_CHECK_RESULT(vkCreateFramebuffer(device, &frameBufferCI, nullptr, &frameBuffers[i]));
    }
}

// Override render pass setup from base class
void VkSample12_InputAttachments::setupRenderPass()
{
    DEBUG_LOGCATE();
    attachmentSize = {outputWidth, outputHeight};

    attachments.resize(swapChain.imageCount);
    for (auto i = 0; i < attachments.size(); i++) {
        createAttachment(colorFormat, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, &attachments[i].color);
        createAttachment(depthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, &attachments[i].depth);
    }

    std::array<VkAttachmentDescription, 3> attachmentDescs{};

    // Swap chain image color attachment
    // Will be transitioned to present layout
    attachmentDescs[0].format = swapChain.colorFormat;
    attachmentDescs[0].samples = VK_SAMPLE_COUNT_1_BIT;
    attachmentDescs[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachmentDescs[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachmentDescs[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachmentDescs[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachmentDescs[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachmentDescs[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    // Input attachments
    // These will be written in the first subpass, transitioned to input attachments
    // and then read in the secod subpass

    // Color
    attachmentDescs[1].format = colorFormat;
    attachmentDescs[1].samples = VK_SAMPLE_COUNT_1_BIT;
    attachmentDescs[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachmentDescs[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachmentDescs[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachmentDescs[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachmentDescs[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachmentDescs[1].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    // Depth
    attachmentDescs[2].format = depthFormat;
    attachmentDescs[2].samples = VK_SAMPLE_COUNT_1_BIT;
    attachmentDescs[2].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachmentDescs[2].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachmentDescs[2].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachmentDescs[2].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachmentDescs[2].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachmentDescs[2].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    std::array<VkSubpassDescription,2> subpassDescriptions{};

    /*
        First subpass
        Fill the color and depth attachments
    */
    VkAttachmentReference colorReference = { 1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
    VkAttachmentReference depthReference = { 2, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };

    subpassDescriptions[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpassDescriptions[0].colorAttachmentCount = 1;
    subpassDescriptions[0].pColorAttachments = &colorReference;
    subpassDescriptions[0].pDepthStencilAttachment = &depthReference;

    /*
        Second subpass
        Input attachment read and swap chain color attachment write
    */

    // Color reference (target) for this sub pass is the swap chain color attachment
    VkAttachmentReference colorReferenceSwapchain = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };

    subpassDescriptions[1].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpassDescriptions[1].colorAttachmentCount = 1;
    subpassDescriptions[1].pColorAttachments = &colorReferenceSwapchain;

    // Color and depth attachment written to in first sub pass will be used as input attachments to be read in the fragment shader
    VkAttachmentReference inputReferences[2];
    inputReferences[0] = { 1, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    inputReferences[1] = { 2, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };

    // Use the attachments filled in the first pass as input attachments
    subpassDescriptions[1].inputAttachmentCount = 2;
    subpassDescriptions[1].pInputAttachments = inputReferences;
    DEBUG_LOGCATE();

    /*
        Subpass dependencies for layout transitions
     在 Vulkan 中，VkSubpassDependency 结构体用于描述子通道（subpass）之间的依赖关系。这在多子通道渲染中非常重要，因为它确保了渲染操作按照预期的顺序进行，并且正确处理资源访问和同步问题。
    */
    std::array<VkSubpassDependency, 3> dependencies;

    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0;
    dependencies[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependencies[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    // This dependency transitions the input attachment from color attachment to shader read
    dependencies[1].srcSubpass = 0;
    dependencies[1].dstSubpass = 1;
    dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    dependencies[2].srcSubpass = 0;
    dependencies[2].dstSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[2].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[2].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    dependencies[2].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[2].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    dependencies[2].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
    DEBUG_LOGCATE();
    VkRenderPassCreateInfo renderPassInfoCI{};
    renderPassInfoCI.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfoCI.attachmentCount = static_cast<uint32_t>(attachmentDescs.size());
    renderPassInfoCI.pAttachments = attachmentDescs.data();
    renderPassInfoCI.subpassCount = static_cast<uint32_t>(subpassDescriptions.size());
    renderPassInfoCI.pSubpasses = subpassDescriptions.data();
    renderPassInfoCI.dependencyCount = static_cast<uint32_t>(dependencies.size());
    renderPassInfoCI.pDependencies = dependencies.data();
    DEBUG_LOGCATE();
    VK_CHECK_RESULT(vkCreateRenderPass(device, &renderPassInfoCI, nullptr, &renderPass));
}

void VkSample12_InputAttachments::buildCommandBuffers()
{
    VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

    VkClearValue clearValues[3];
    clearValues[0].color = { { 1.0f, 1.0f, 1.0f, 1.0f  } };
    clearValues[1].color = { { 1.0f, 1.0f, 1.0f, 1.0f  } };
    clearValues[2].depthStencil = { 1.0f, 0 };

    VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
    renderPassBeginInfo.renderPass = renderPass;
    renderPassBeginInfo.renderArea.offset.x = 0;
    renderPassBeginInfo.renderArea.offset.y = 0;
    renderPassBeginInfo.renderArea.extent.width = outputWidth;
    renderPassBeginInfo.renderArea.extent.height = outputHeight;
    renderPassBeginInfo.clearValueCount = 3;
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

        //first subpass
        {
            vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1,
                                    &descriptorSet, 0, nullptr);
            vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
            VkDeviceSize offsets[1] = {0};
            vkCmdBindVertexBuffers(drawCmdBuffers[i], 0, 1, &vertices.buffer, offsets);
            vkCmdBindIndexBuffer(drawCmdBuffers[i], indices.buffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(drawCmdBuffers[i], indices.count, 1, 0, 0, 1);
        }


        //second subpass
        {
            vkCmdNextSubpass(drawCmdBuffers[i], VK_SUBPASS_CONTENTS_INLINE);
            vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline2);
            vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout2, 0, 1, &descriptorSet2s[i], 0, NULL);
            vkCmdDraw(drawCmdBuffers[i], 3, 1, 0, 0);
        }

        vkCmdEndRenderPass(drawCmdBuffers[i]);
        VK_CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
    }
}

void VkSample12_InputAttachments::loadTexture()
{
    if(pTexture2D) {
        pTexture2D->destroy();
        delete pTexture2D;
    }

    pTexture2D = new vks::Texture2D();
    VK_CHECK_RESULT(loadRGBTexture2DFromFile("image/texture.png", pTexture2D))
    float viewportRatio = (float) outputWidth / (float) outputHeight;
    float imageRatio = (float) pTexture2D->width / (float) pTexture2D->height;
    //camera.setPerspective(45.0f, viewportRatio / imageRatio, 1.0f, 256.0f);
    //glm::mat4 Projection = glm::perspective(45.0f, viewportRatio, 0.1f, 100.f);

    float ratio = viewportRatio / imageRatio;
    //glm::mat4 Projection = glm::ortho(-ratio, ratio, -1.0f, 1.0f, 0.0f, 100.0f);
    //glm::mat4 Projection = glm::frustum(-ratio, ratio, -1.0f, 1.0f, 4.0f, 100.0f);
    glm::mat4 Projection = glm::perspective(45.0f, ratio, 1.0f, 256.f);

    // View matrix
    glm::mat4 View = glm::lookAt(
            glm::vec3(0, 0, 4.0), // Camera is at (0,0,1), in World Space
            glm::vec3(0, 0, 0), // and looks at the origin
            glm::vec3(0, 1, 0)  // Head is up (set to 0,-1,0 to look upside-down)
    );
    uboData.projectionMatrix = Projection;
    uboData.viewMatrix = View;
    uboData.modelMatrix = glm::mat4(1.0f);
}

void VkSample12_InputAttachments::prepare() {
    VulkanExampleBase::prepare();
    loadTexture();
    createVertexBuffer();
    createUniformBuffers();
    setupDescriptors();
    createPipelines();
    buildCommandBuffers();
    prepared = true;
}

void VkSample12_InputAttachments::render() {
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


