
#include <fstream>
#include <vector>
#include <exception>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vulkan/vulkan.h>
#include "vulkanexamplebase.h"
#include "VkSample01_Triangle.h"

VkSample01_Triangle::VkSample01_Triangle() {

    title = "VkSample01_Triangle";
    // Setup a default look-at camera
    camera.type = Camera::CameraType::lookat;
    camera.setPosition(glm::vec3(0.0f, 0.0f, -2.5f));
    camera.setRotation(glm::vec3(0.0f));
}

VkSample01_Triangle::~VkSample01_Triangle() {
    // Clean up used Vulkan resources
    // Note: Inherited destructor cleans up resources stored in base class
    vkDestroyPipeline(device, pipeline, nullptr);

    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

    vkDestroyBuffer(device, vertices.buffer, nullptr);
    vkFreeMemory(device, vertices.memory, nullptr);

    vkDestroyBuffer(device, indices.buffer, nullptr);
    vkFreeMemory(device, indices.memory, nullptr);

    vkDestroyFence(device, renderFence, nullptr);
}

void VkSample01_Triangle::updateTransformMatrix(float rotateX, float rotateY, float scaleX, float scaleY)
{
    LOGCATE("VkSample01_Triangle::updateTransformMatrix()");
    float radiansX = static_cast<float>(MATH_PI / 180.0f * rotateX);
    float radiansY = static_cast<float>(MATH_PI / 180.0f * rotateY);
    glm::mat4 Model = glm::mat4(1.0f);
    Model = glm::scale(Model, glm::vec3(scaleX, scaleX, 1.0f));
    Model = glm::rotate(Model, radiansX, glm::vec3(1.0f, 0.0f, 0.0f));
    Model = glm::rotate(Model, radiansY, glm::vec3(0.0f, 1.0f, 0.0f));
    Model = glm::translate(Model, glm::vec3(0.0f, 0.0f, 0.0f));
    mvpMatrix.modelMatrix = Model;
}

// Prepare vertex and index buffers for an indexed triangle
// Also uploads them to device local memory using staging and initializes vertex input and attribute binding to match the vertex shader
void VkSample01_Triangle::createVertexBuffer() {
    // A note on memory management in Vulkan in general:
    //	This is a very complex topic and while it's fine for an example application to small individual memory allocations that is not
    //	what should be done a real-world application, where you should allocate large chunks of memory at once instead.

    // Setup vertices
    std::vector<Vertex> vertexBuffer{
            {{1.0f,  1.0f,  0.0f}, {1.0f, 0.0f, 0.0f}},
            {{-1.0f, 1.0f,  0.0f}, {0.0f, 1.0f, 0.0f}},
            {{0.0f,  -1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}}
    };
    uint32_t vertexBufferSize = static_cast<uint32_t>(vertexBuffer.size()) * sizeof(Vertex);

    // Setup indices
    std::vector<uint32_t> indexBuffer{0, 1, 2};
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
    VkBufferCreateInfo indexBufferCreateInfo{
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
            vkCreateBuffer(device, &indexBufferCreateInfo, nullptr, &indices.buffer));
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

// Descriptors are allocated from a pool, that tells the implementation how many and what types of descriptors we are going to use (at maximum)
void VkSample01_Triangle::createDescriptorPool() {
    // We need to tell the API the number of max. requested descriptors per type
    VkDescriptorPoolSize descriptorTypeCounts[1];
    // This example only one descriptor type (uniform buffer)
    descriptorTypeCounts[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    // We have one buffer (and as such descriptor) per frame
    descriptorTypeCounts[0].descriptorCount = 1;
    // For additional types you need to add new entries in the type count list
    // E.g. for two combined image samplers :

    // Create the global descriptor pool
    // All descriptors used in this example are allocated from this pool
    VkDescriptorPoolCreateInfo descriptorPoolCreateInfo{};
    descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptorPoolCreateInfo.pNext = nullptr;
    descriptorPoolCreateInfo.poolSizeCount = 1;
    descriptorPoolCreateInfo.pPoolSizes = descriptorTypeCounts;
    // Set the max. number of descriptor sets that can be requested from this pool (requesting beyond this limit will result in an error)
    // Our sample will create one set per uniform buffer per frame
    descriptorPoolCreateInfo.maxSets = 1;
    VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, nullptr, &descriptorPool));
}

// Descriptor set layouts define the interface between our application and the shader
// Basically connects the different shader stages to descriptors for binding uniform buffers, image samplers, etc.
// So every shader binding should map to one descriptor set layout binding
void VkSample01_Triangle::createDescriptorSetLayout() {
    // Binding 0: Uniform buffer (Vertex shader)
    VkDescriptorSetLayoutBinding layoutBinding{};
    layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    layoutBinding.descriptorCount = 1;
    layoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    layoutBinding.pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutCreateInfo descriptorLayoutCreateInfo{};
    descriptorLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorLayoutCreateInfo.pNext = nullptr;
    descriptorLayoutCreateInfo.bindingCount = 1;
    descriptorLayoutCreateInfo.pBindings = &layoutBinding;
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayoutCreateInfo, nullptr,
                                                &descriptorSetLayout));

    // Create the pipeline layout that is used to generate the rendering pipelines that are based on this descriptor set layout
    // In a more complex scenario you would have different pipeline layouts for different descriptor set layouts that could be reused
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{};
    pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCreateInfo.pNext = nullptr;
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayout;
    VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout));
}

// Shaders access data using descriptor sets that "point" at our uniform buffers
// The descriptor sets make use of the descriptor set layouts created above
void VkSample01_Triangle::createDescriptorSets() {
    // Allocate one descriptor set per frame from the global descriptor pool
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptorSetLayout;
    VK_CHECK_RESULT(
            vkAllocateDescriptorSets(device, &allocInfo, &uniformBuffer.descriptorSet));

    // Update the descriptor set determining the shader binding points
    // For every binding point used in a shader there needs to be one
    // descriptor set matching that binding point
    VkWriteDescriptorSet writeDescriptorSet{};

    // The buffer's information is passed using a descriptor info structure
    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = uniformBuffer.buffer;
    bufferInfo.range = sizeof(ShaderData);

    // Binding 0 : Uniform buffer
    writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet.dstSet = uniformBuffer.descriptorSet;
    writeDescriptorSet.descriptorCount = 1;
    writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writeDescriptorSet.pBufferInfo = &bufferInfo;
    writeDescriptorSet.dstBinding = 0;
    vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, nullptr);
}

// Create the depth (and stencil) buffer attachments used by our framebuffers
// Note: Override of virtual function in the base class and called from within VulkanExampleBase::prepare
void VkSample01_Triangle::setupDepthStencil() {
    // Create an optimal image used as the depth stencil attachment
    VkImageCreateInfo imageCI{};
    imageCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCI.imageType = VK_IMAGE_TYPE_2D;
    imageCI.format = depthFormat;
    // Use example's outputHeight and outputWidth
    imageCI.extent = {outputWidth, outputHeight, 1};
    imageCI.mipLevels = 1;
    imageCI.arrayLayers = 1;
    imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCI.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    imageCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VK_CHECK_RESULT(vkCreateImage(device, &imageCI, nullptr, &depthStencil.image));

    // Allocate memory for the image (device local) and bind it to our image
    VkMemoryAllocateInfo memAlloc{};
    memAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    VkMemoryRequirements memReqs;
    vkGetImageMemoryRequirements(device, depthStencil.image, &memReqs);
    memAlloc.allocationSize = memReqs.size;
    memAlloc.memoryTypeIndex = getMemoryTypeIndex(memReqs.memoryTypeBits,
                                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &depthStencil.mem));
    VK_CHECK_RESULT(vkBindImageMemory(device, depthStencil.image, depthStencil.mem, 0));

    // Create a view for the depth stencil image
    // Images aren't directly accessed in Vulkan, but rather through views described by a subresource range
    // This allows for multiple views of one image with differing ranges (e.g. for different layers)
    VkImageViewCreateInfo depthStencilViewCI{};
    depthStencilViewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    depthStencilViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
    depthStencilViewCI.format = depthFormat;
    depthStencilViewCI.subresourceRange = {};
    depthStencilViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    // Stencil aspect should only be set on depth + stencil formats (VK_FORMAT_D16_UNORM_S8_UINT..VK_FORMAT_D32_SFLOAT_S8_UINT)
    if (depthFormat >= VK_FORMAT_D16_UNORM_S8_UINT) {
        depthStencilViewCI.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
    }
    depthStencilViewCI.subresourceRange.baseMipLevel = 0;
    depthStencilViewCI.subresourceRange.levelCount = 1;
    depthStencilViewCI.subresourceRange.baseArrayLayer = 0;
    depthStencilViewCI.subresourceRange.layerCount = 1;
    depthStencilViewCI.image = depthStencil.image;
    VK_CHECK_RESULT(vkCreateImageView(device, &depthStencilViewCI, nullptr, &depthStencil.view));
}

// Create a frame buffer for each swap chain image
// Note: Override of virtual function in the base class and called from within VulkanExampleBase::prepare
void VkSample01_Triangle::setupFrameBuffer() {
    // Create a frame buffer for every image in the swapchain
    frameBuffers.resize(swapChain.imageCount);
    for (size_t i = 0; i < frameBuffers.size(); i++) {
        std::array<VkImageView, 2> attachments;
        // Color attachment is the view of the swapchain image
        attachments[0] = swapChain.buffers[i].view;
        // Depth/Stencil attachment is the same for all frame buffers due to how depth works with current GPUs
        attachments[1] = depthStencil.view;

        VkFramebufferCreateInfo frameBufferCI{};
        frameBufferCI.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        // All frame buffers use the same renderpass setup
        frameBufferCI.renderPass = renderPass;
        frameBufferCI.attachmentCount = static_cast<uint32_t>(attachments.size());
        frameBufferCI.pAttachments = attachments.data();
        frameBufferCI.width = outputWidth;
        frameBufferCI.height = outputHeight;
        frameBufferCI.layers = 1;
        // Create the framebuffer
        VK_CHECK_RESULT(vkCreateFramebuffer(device, &frameBufferCI, nullptr, &frameBuffers[i]));
    }
}

// Render pass setup
// Render passes are a new concept in Vulkan. They describe the attachments used during rendering and may contain multiple subpasses with attachment dependencies
// This allows the driver to know up-front what the rendering will look like and is a good opportunity to optimize especially on tile-based renderers (with multiple subpasses)
// Using sub pass dependencies also adds implicit layout transitions for the attachment used, so we don't need to add explicit image memory barriers to transform them
// Note: Override of virtual function in the base class and called from within VulkanExampleBase::prepare
void VkSample01_Triangle::setupRenderPass() {
    // This example will use a single render pass with one subpass

    // Descriptors for the attachments used by this renderpass
    std::array<VkAttachmentDescription, 2> attachments{};

    // Color attachment
    attachments[0].format = swapChain.colorFormat;                                  // Use the color format selected by the swapchain
    attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;                                 // We don't use multi sampling in this example
    attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;                            // Clear this attachment at the start of the render pass
    attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;                          // Keep its contents after the render pass is finished (for displaying it)
    attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;                 // We don't use stencil, so don't care for load
    attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;               // Same for store
    attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;                       // Layout at render pass start. Initial doesn't matter, so we use undefined
    attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;                   // Layout to which the attachment is transitioned when the render pass is finished
    // As we want to present the color buffer to the swapchain, we transition to PRESENT_KHR
    // Depth attachment
    attachments[1].format = depthFormat;                                           // A proper depth format is selected in the example base
    attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;                           // Clear depth at start of first subpass
    attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;                     // We don't need depth after render pass has finished (DONT_CARE may result in better performance)
    attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;                // No stencil
    attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;              // No Stencil
    attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;                      // Layout at render pass start. Initial doesn't matter, so we use undefined
    attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL; // Transition to depth/stencil attachment

    // Setup attachment references
    VkAttachmentReference colorReference{};
    colorReference.attachment = 0;                                    // Attachment 0 is color
    colorReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; // Attachment layout used as color during the subpass

    VkAttachmentReference depthReference{};
    depthReference.attachment = 1;                                            // Attachment 1 is color
    depthReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL; // Attachment used as depth/stencil used during the subpass

    // Setup a single subpass reference
    VkSubpassDescription subpassDescription{};
    subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpassDescription.colorAttachmentCount = 1;                            // Subpass uses one color attachment
    subpassDescription.pColorAttachments = &colorReference;                 // Reference to the color attachment in slot 0
    subpassDescription.pDepthStencilAttachment = &depthReference;           // Reference to the depth attachment in slot 1
    subpassDescription.inputAttachmentCount = 0;                            // Input attachments can be used to sample from contents of a previous subpass
    subpassDescription.pInputAttachments = nullptr;                         // (Input attachments not used by this example)
    subpassDescription.preserveAttachmentCount = 0;                         // Preserved attachments can be used to loop (and preserve) attachments through subpasses
    subpassDescription.pPreserveAttachments = nullptr;                      // (Preserve attachments not used by this example)
    subpassDescription.pResolveAttachments = nullptr;                       // Resolve attachments are resolved at the end of a sub pass and can be used for e.g. multi sampling

    // Setup subpass dependencies
    // These will add the implicit attachment layout transitions specified by the attachment descriptions
    // The actual usage layout is preserved through the layout specified in the attachment reference
    // Each subpass dependency will introduce a memory and execution dependency between the source and dest subpass described by
    // srcStageMask, dstStageMask, srcAccessMask, dstAccessMask (and dependencyFlags is set)
    // Note: VK_SUBPASS_EXTERNAL is a special constant that refers to all commands executed outside of the actual renderpass)
    std::array<VkSubpassDependency, 2> dependencies;

    // Does the transition from final to initial layout for the depth an color attachments
    // Depth attachment
    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0;
    dependencies[0].srcStageMask =
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    dependencies[0].dstStageMask =
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    dependencies[0].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependencies[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                                    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
    dependencies[0].dependencyFlags = 0;
    // Color attachment
    dependencies[1].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[1].dstSubpass = 0;
    dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[1].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[1].srcAccessMask = 0;
    dependencies[1].dstAccessMask =
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
    dependencies[1].dependencyFlags = 0;

    // Create the actual renderpass
    VkRenderPassCreateInfo renderPassCI{};
    renderPassCI.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassCI.attachmentCount = attachments.size();  // Number of attachments used by this render pass
    renderPassCI.pAttachments = attachments.data();                            // Descriptions of the attachments used by the render pass
    renderPassCI.subpassCount = 1;                                             // We only use one subpass in this example
    renderPassCI.pSubpasses = &subpassDescription;                             // Description of that subpass
    renderPassCI.dependencyCount = static_cast<uint32_t>(dependencies.size()); // Number of subpass dependencies
    renderPassCI.pDependencies = dependencies.data();                          // Subpass dependencies used by the render pass
    VK_CHECK_RESULT(vkCreateRenderPass(device, &renderPassCI, nullptr, &renderPass));
}

void VkSample01_Triangle::createPipelines() {
    // Create the graphics pipeline used in this example
    // Vulkan uses the concept of rendering pipelines to encapsulate fixed states, replacing OpenGL's complex state machine
    // A pipeline is then stored and hashed on the GPU making pipeline changes very fast
    // Note: There are still a few dynamic states that are not directly part of the pipeline (but the info that they are used is)

    VkGraphicsPipelineCreateInfo pipelineCI{};
    pipelineCI.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    // The layout used for this pipeline (can be shared among multiple pipelines using the same layout)
    pipelineCI.layout = pipelineLayout;
    // Renderpass this pipeline is attached to
    pipelineCI.renderPass = renderPass;

    // Construct the different states making up the pipeline

    // Input assembly state describes how primitives are assembled
    // This pipeline will assemble vertex data as a triangle lists (though we only use one triangle)
    VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCI{};
    inputAssemblyStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssemblyStateCI.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    // Rasterization state
    VkPipelineRasterizationStateCreateInfo rasterizationStateCI{};
    rasterizationStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizationStateCI.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizationStateCI.cullMode = VK_CULL_MODE_NONE;
    rasterizationStateCI.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizationStateCI.depthClampEnable = VK_FALSE;
    rasterizationStateCI.rasterizerDiscardEnable = VK_FALSE;
    rasterizationStateCI.depthBiasEnable = VK_FALSE;
    rasterizationStateCI.lineWidth = 1.0f;

    // Color blend state describes how blend factors are calculated (if used)
    // We need one blend attachment state per color attachment (even if blending is not used)
    VkPipelineColorBlendAttachmentState blendAttachmentState{};
    blendAttachmentState.colorWriteMask = 0xf;
    blendAttachmentState.blendEnable = VK_FALSE;
    VkPipelineColorBlendStateCreateInfo colorBlendStateCI{};
    colorBlendStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlendStateCI.attachmentCount = 1;
    colorBlendStateCI.pAttachments = &blendAttachmentState;

    // Viewport state sets the number of viewports and scissor used in this pipeline
    // Note: This is actually overridden by the dynamic states (see below)
    VkPipelineViewportStateCreateInfo viewportStateCI{};
    viewportStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportStateCI.viewportCount = 1;
    viewportStateCI.scissorCount = 1;

    // Enable dynamic states
    // Most states are baked into the pipeline, but there are still a few dynamic states that can be changed within a command buffer
    // To be able to change these we need do specify which dynamic states will be changed using this pipeline. Their actual states are set later on in the command buffer.
    // For this example we will set the viewport and scissor using dynamic states
    std::vector<VkDynamicState> dynamicStateEnables;
    dynamicStateEnables.push_back(VK_DYNAMIC_STATE_VIEWPORT);
    dynamicStateEnables.push_back(VK_DYNAMIC_STATE_SCISSOR);
    VkPipelineDynamicStateCreateInfo dynamicStateCI{};
    dynamicStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicStateCI.pDynamicStates = dynamicStateEnables.data();
    dynamicStateCI.dynamicStateCount = static_cast<uint32_t>(dynamicStateEnables.size());

    // Depth and stencil state containing depth and stencil compare and test operations
    // We only use depth tests and want depth tests and writes to be enabled and compare with less or equal
    VkPipelineDepthStencilStateCreateInfo depthStencilStateCI{};
    depthStencilStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencilStateCI.depthTestEnable = VK_TRUE;
    depthStencilStateCI.depthWriteEnable = VK_TRUE;
    depthStencilStateCI.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    depthStencilStateCI.depthBoundsTestEnable = VK_FALSE;
    depthStencilStateCI.back.failOp = VK_STENCIL_OP_KEEP;
    depthStencilStateCI.back.passOp = VK_STENCIL_OP_KEEP;
    depthStencilStateCI.back.compareOp = VK_COMPARE_OP_ALWAYS;
    depthStencilStateCI.stencilTestEnable = VK_FALSE;
    depthStencilStateCI.front = depthStencilStateCI.back;

    // Multi sampling state
    // This example does not make use of multi sampling (for anti-aliasing), the state must still be set and passed to the pipeline
    VkPipelineMultisampleStateCreateInfo multisampleStateCI{};
    multisampleStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampleStateCI.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisampleStateCI.pSampleMask = nullptr;

    // Vertex input descriptions
    // Specifies the vertex input parameters for a pipeline

    // Vertex input binding
    // This example uses a single vertex input binding at binding point 0 (see vkCmdBindVertexBuffers)
    VkVertexInputBindingDescription vertexInputBinding{};
    vertexInputBinding.binding = 0;
    vertexInputBinding.stride = sizeof(Vertex);
    vertexInputBinding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    // Input attribute bindings describe shader attribute locations and memory layouts
    std::array<VkVertexInputAttributeDescription, 2> vertexInputAttributs;
    // These match the following shader layout (see triangle.vert):
    //	layout (location = 0) in vec3 inPos;
    //	layout (location = 1) in vec3 inColor;
    // Attribute location 0: Position
    vertexInputAttributs[0].binding = 0;
    vertexInputAttributs[0].location = 0;
    // Position attribute is three 32 bit signed (SFLOAT) floats (R32 G32 B32)
    vertexInputAttributs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    vertexInputAttributs[0].offset = offsetof(Vertex, position);
    // Attribute location 1: Color
    vertexInputAttributs[1].binding = 0;
    vertexInputAttributs[1].location = 1;
    // Color attribute is three 32 bit signed (SFLOAT) floats (R32 G32 B32)
    vertexInputAttributs[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    vertexInputAttributs[1].offset = offsetof(Vertex, color);

    // Vertex input state used for pipeline creation
    VkPipelineVertexInputStateCreateInfo vertexInputStateCI{};
    vertexInputStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputStateCI.vertexBindingDescriptionCount = 1;
    vertexInputStateCI.pVertexBindingDescriptions = &vertexInputBinding;
    vertexInputStateCI.vertexAttributeDescriptionCount = 2;
    vertexInputStateCI.pVertexAttributeDescriptions = vertexInputAttributs.data();

    // Shaders
    std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages{};

    // Vertex shader
//    shaderStages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
//    // Set pipeline stage for this shader
//    shaderStages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
//    // Load binary SPIR-V shader
//    shaderStages[0].module = loadSPIRVShader(getShadersPath() + "triangle/triangle.vert.spv");
//    // Main entry point for the shader
//    shaderStages[0].pName = "main";
    shaderStages[0] = compileShader(device, getShadersPath() + "triangle/triangle.vert", VK_SHADER_STAGE_VERTEX_BIT);
    assert(shaderStages[0].module != VK_NULL_HANDLE);

    // Fragment shader
//    shaderStages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
//    // Set pipeline stage for this shader
//    shaderStages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
//    // Load binary SPIR-V shader
//    shaderStages[1].module = loadSPIRVShader(getShadersPath() + "triangle/triangle.frag.spv");
//    // Main entry point for the shader
//    shaderStages[1].pName = "main";
    shaderStages[1] = compileShader(device, getShadersPath() + "triangle/triangle.frag",VK_SHADER_STAGE_FRAGMENT_BIT);
    assert(shaderStages[1].module != VK_NULL_HANDLE);

    // Set pipeline shader stage info
    pipelineCI.stageCount = static_cast<uint32_t>(shaderStages.size());
    pipelineCI.pStages = shaderStages.data();

    // Assign the pipeline states to the pipeline creation info structure
    pipelineCI.pVertexInputState = &vertexInputStateCI;
    pipelineCI.pInputAssemblyState = &inputAssemblyStateCI;
    pipelineCI.pRasterizationState = &rasterizationStateCI;
    pipelineCI.pColorBlendState = &colorBlendStateCI;
    pipelineCI.pMultisampleState = &multisampleStateCI;
    pipelineCI.pViewportState = &viewportStateCI;
    pipelineCI.pDepthStencilState = &depthStencilStateCI;
    pipelineCI.pDynamicState = &dynamicStateCI;

    // Create rendering pipeline using the specified states
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipeline));

    // Shader modules are no longer needed once the graphics pipeline has been created
    vkDestroyShaderModule(device, shaderStages[0].module, nullptr);
    vkDestroyShaderModule(device, shaderStages[1].module, nullptr);
}

void VkSample01_Triangle::createUniformBuffers() {
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
    bufferInfo.size = sizeof(ShaderData);
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
    VK_CHECK_RESULT(vkMapMemory(device, uniformBuffer.memory, 0, sizeof(ShaderData), 0,
                                (void **) &uniformBuffer.mapped));
}

void VkSample01_Triangle::buildCommandBuffers()
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

        vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1,
                                &uniformBuffer.descriptorSet, 0, nullptr);
        // Bind the rendering pipeline
        // The pipeline (state object) contains all states of the rendering pipeline, binding it will set all the states specified at pipeline creation time
        vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        // Bind triangle vertex buffer (contains position and colors)
        VkDeviceSize offsets[1]{0};
        vkCmdBindVertexBuffers(drawCmdBuffers[i], 0, 1, &vertices.buffer, offsets);
        // Bind triangle index buffer
        vkCmdBindIndexBuffer(drawCmdBuffers[i], indices.buffer, 0, VK_INDEX_TYPE_UINT32);
        // Draw indexed triangle
        vkCmdDrawIndexed(drawCmdBuffers[i], indices.count, 1, 0, 0, 1);
        vkCmdEndRenderPass(drawCmdBuffers[i]);

        VK_CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
    }
}

void VkSample01_Triangle::prepare() {
    camera.setPerspective(75.0f, (float) outputWidth / (float) outputHeight, 1.0f, 256.0f);
    // Values not set here are initialized in the base class constructor
    mvpMatrix.projectionMatrix = camera.matrices.perspective;
    mvpMatrix.viewMatrix = camera.matrices.view;
    mvpMatrix.modelMatrix = glm::mat4(1.0f);

    VulkanExampleBase::prepare();

    // Create a fence for synchronization
    VkFenceCreateInfo fenceCreateInfo = vks::initializers::fenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT);
    vkCreateFence(device, &fenceCreateInfo, nullptr, &renderFence);

    createVertexBuffer();
    createUniformBuffers();
    createDescriptorSetLayout();
    createDescriptorPool();
    createDescriptorSets();
    createPipelines();
    buildCommandBuffers();
    prepared = true;
}

void VkSample01_Triangle::render() {
    if (!prepared)
        return;
    // Wait for renderFence to be triggered, indicating that all command buffers from the previous frame have completed execution
    VkResult fenceRes;
    do {
        // vkWaitForFences will block the current thread until the specified Fence is triggered or times out
        fenceRes = vkWaitForFences(device, 1, &renderFence, VK_TRUE, 100000000); // Timeout set to 100ms
    } while (fenceRes == VK_TIMEOUT); // If it times out without completing, keep waiting
    VK_CHECK_RESULT(fenceRes); // Ensure that the Fence state is successful
    vkResetFences(device, 1, &renderFence); // Reset the Fence state, preparing it for the next frame

    // Acquire the next available image index from the swap chain
    // Use semaphores.presentComplete to ensure the image is not used before being written to
    VK_CHECK_RESULT(swapChain.acquireNextImage(semaphores.presentComplete, &currentBufferIdx));

    // Update Uniform Buffer data by copying the new MVP matrix data into the mapped memory
    memcpy(uniformBuffer.mapped, &mvpMatrix, sizeof(ShaderData));

    // Submit the command buffer to the queue for rendering
    submitInfo.commandBufferCount = 1; // Number of command buffers to submit
    submitInfo.pCommandBuffers = &drawCmdBuffers[currentBufferIdx]; // Command buffer for the current frame
    VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, renderFence)); // Submit with renderFence for synchronization

    // Submit the rendered image to the swap chain, preparing it for display on the screen
    // Use semaphores.renderComplete to ensure the image is not switched before the rendering is complete
    VK_CHECK_RESULT(swapChain.queuePresent(queue, currentBufferIdx, semaphores.renderComplete));

    // The commented-out code ensures that the queue operations are completed before continuing
    // VK_CHECK_RESULT(vkQueueWaitIdle(queue));
}


