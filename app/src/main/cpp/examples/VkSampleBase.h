/**
 *
 * Created by 公众号：字节流动 on 2021/3/12.
 * https://github.com/githubhaohao/VKSAMPLE
 * 最新文章首发于公众号：字节流动，有疑问或者技术交流可以添加微信 Byte-Flow ,领取视频教程, 拉你进技术交流群
 *
 * */

#ifndef VKSAMPLE_VkSampleBase_H
#define VKSAMPLE_VkSampleBase_H

#include "stdint.h"
#include "../util/ImageDef.h"
#include <GLES3/gl3.h>
#include <ImageDef.h>
#include "vulkanexamplebase.h"
#include "../wrapper/vulkanexamplebase.h"
#include "tinygltf/stb_image.h"
//For PI define
#define MATH_PI 3.1415926535897932384626433832802

#define SAMPLE_TYPE                             200
#define SAMPLE_TYPE_KEY_TRIANGLE                SAMPLE_TYPE + 0
#define SAMPLE_TYPE_KEY_UBO                     SAMPLE_TYPE + 1
#define SAMPLE_TYPE_KEY_TEXTURE_MAPPING         SAMPLE_TYPE + 2
#define SAMPLE_TYPE_KEY_PIPELINES               SAMPLE_TYPE + 3
#define SAMPLE_TYPE_KEY_PUSH_CONSTANTS          SAMPLE_TYPE + 4
#define SAMPLE_TYPE_KEY_RENDER_NV21             SAMPLE_TYPE + 5
#define SAMPLE_TYPE_KEY_RENDER_I420             SAMPLE_TYPE + 6
#define SAMPLE_TYPE_KEY_RENDER_YUYV             SAMPLE_TYPE + 7
#define SAMPLE_TYPE_KEY_RENDER_I444             SAMPLE_TYPE + 8
#define SAMPLE_TYPE_KEY_SPECIALIZATION_INFO     SAMPLE_TYPE + 9
#define SAMPLE_TYPE_KEY_CUBEMAP                 SAMPLE_TYPE + 10
#define SAMPLE_TYPE_KEY_INPUT_ATTACHMENTS       SAMPLE_TYPE + 11
#define SAMPLE_TYPE_KEY_OFFSCREEN_RENDERING     SAMPLE_TYPE + 12
#define SAMPLE_TYPE_KEY_DEPTH_TESTING           SAMPLE_TYPE + 13
#define SAMPLE_TYPE_KEY_STENCIL_TESTING         SAMPLE_TYPE + 14
#define SAMPLE_TYPE_KEY_MULTISAMPLING           SAMPLE_TYPE + 15
#define SAMPLE_TYPE_KEY_MULTITHREADING          SAMPLE_TYPE + 16
#define SAMPLE_TYPE_KEY_INSTANCING              SAMPLE_TYPE + 17
#define SAMPLE_TYPE_KEY_READ_PIXELS             SAMPLE_TYPE + 18
#define SAMPLE_TYPE_KEY_COMPUTE_SHADER          SAMPLE_TYPE + 19

#define SAMPLE_TYPE_KEY_COORD_SYSTEM            SAMPLE_TYPE + 10
#define SAMPLE_TYPE_KEY_BASIC_LIGHTING          SAMPLE_TYPE + 11
#define SAMPLE_TYPE_KEY_BLENDING                SAMPLE_TYPE + 14
#define SAMPLE_TYPE_KEY_TRANSFORM_FEEDBACK      SAMPLE_TYPE + 15
#define SAMPLE_TYPE_KEY_INSTANCING3D            SAMPLE_TYPE + 17
#define SAMPLE_TYPE_KEY_PARTICLES               SAMPLE_TYPE + 18
#define SAMPLE_TYPE_KEY_SKYBOX                  SAMPLE_TYPE + 19
#define SAMPLE_TYPE_KEY_FILTER_SPLITSCREEN      SAMPLE_TYPE + 20
#define SAMPLE_TYPE_KEY_FILTER_COLOROFFSET      SAMPLE_TYPE + 21
#define SAMPLE_TYPE_KEY_FILTER_ROTATE           SAMPLE_TYPE + 22
#define SAMPLE_TYPE_KEY_LUT                     SAMPLE_TYPE + 23
#define SAMPLE_TYPE_KEY_3D_MODEL                SAMPLE_TYPE + 24
#define SAMPLE_TYPE_KEY_PBO                     SAMPLE_TYPE + 25
#define SAMPLE_TYPE_KEY_MRT                     SAMPLE_TYPE + 26
#define SAMPLE_TYPE_KEY_TBO                     SAMPLE_TYPE + 28


//#define SAMPLE_TYPE_KEY_MULTI_LIGHTS            SAMPLE_TYPE + 10
//#define SAMPLE_TYPE_KEY_PBO                     SAMPLE_TYPE + 18
//#define SAMPLE_TYPE_KEY_BEATING_HEART           SAMPLE_TYPE + 19
//#define SAMPLE_TYPE_KEY_CLOUD                   SAMPLE_TYPE + 20
//#define SAMPLE_TYPE_KEY_TIME_TUNNEL             SAMPLE_TYPE + 21
//#define SAMPLE_TYPE_KEY_BEZIER_CURVE            SAMPLE_TYPE + 22
//#define SAMPLE_TYPE_KEY_BIG_EYES                SAMPLE_TYPE + 23
//#define SAMPLE_TYPE_KEY_FACE_SLENDER            SAMPLE_TYPE + 24
//#define SAMPLE_TYPE_KEY_BIG_HEAD                SAMPLE_TYPE + 25
//#define SAMPLE_TYPE_KEY_RATARY_HEAD             SAMPLE_TYPE + 26
//#define SAMPLE_TYPE_KEY_VISUALIZE_AUDIO         SAMPLE_TYPE + 27
//#define SAMPLE_TYPE_KEY_SCRATCH_CARD            SAMPLE_TYPE + 28
//#define SAMPLE_TYPE_KEY_AVATAR                  SAMPLE_TYPE + 29
//#define SAMPLE_TYPE_KEY_SHOCK_WAVE              SAMPLE_TYPE + 30
//#define SAMPLE_TYPE_KEY_MRT                     SAMPLE_TYPE + 31
//#define SAMPLE_TYPE_KEY_FBO_BLIT                SAMPLE_TYPE + 32
//#define SAMPLE_TYPE_KEY_TBO                     SAMPLE_TYPE + 33
//#define SAMPLE_TYPE_KEY_UBO                     SAMPLE_TYPE + 34
//#define SAMPLE_TYPE_KEY_RGB2YUYV                SAMPLE_TYPE + 35
//#define SAMPLE_TYPE_KEY_MULTI_THREAD_RENDER     SAMPLE_TYPE + 36
//#define SAMPLE_TYPE_KEY_TEXT_RENDER     		SAMPLE_TYPE + 37
//#define SAMPLE_TYPE_KEY_STAY_COLOR       		SAMPLE_TYPE + 38
//#define SAMPLE_TYPE_KEY_TRANSITIONS_1      		SAMPLE_TYPE + 39
//#define SAMPLE_TYPE_KEY_TRANSITIONS_2      		SAMPLE_TYPE + 40
//#define SAMPLE_TYPE_KEY_TRANSITIONS_3      		SAMPLE_TYPE + 41
//#define SAMPLE_TYPE_KEY_TRANSITIONS_4      		SAMPLE_TYPE + 42
//#define SAMPLE_TYPE_KEY_RGB2NV21                SAMPLE_TYPE + 43
//#define SAMPLE_TYPE_KEY_RGB2I420                SAMPLE_TYPE + 44
//#define SAMPLE_TYPE_KEY_RGB2I444                SAMPLE_TYPE + 45
//#define SAMPLE_TYPE_KEY_COPY_TEXTURE            SAMPLE_TYPE + 46
//#define SAMPLE_TYPE_KEY_BLIT_FRAME_BUFFER       SAMPLE_TYPE + 47
//#define SAMPLE_TYPE_KEY_BINARY_PROGRAM          SAMPLE_TYPE + 48
//#define SAMPLE_TYPE_KEY_HWBuffer                SAMPLE_TYPE + 49
//#define SAMPLE_TYPE_KEY_RENDER_16BIT_GRAY       SAMPLE_TYPE + 50
//#define SAMPLE_TYPE_KEY_RENDER_P010             SAMPLE_TYPE + 51
//#define SAMPLE_TYPE_KEY_RENDER_I420             SAMPLE_TYPE + 53
//#define SAMPLE_TYPE_KEY_RENDER_I444             SAMPLE_TYPE + 54
//#define SAMPLE_TYPE_KEY_RENDER_YUYV             SAMPLE_TYPE + 55
//#define SAMPLE_TYPE_KEY_COMPUTE_SHADER          SAMPLE_TYPE + 56

#define SAMPLE_TYPE_KEY_SET_TOUCH_LOC           SAMPLE_TYPE + 999
#define SAMPLE_TYPE_SET_GRAVITY_XY              SAMPLE_TYPE + 1000

#define DEFAULT_OGL_ASSETS_DIR "/sdcard/Android/data/com.byteflow.vkapp/files/Download"

struct Texture {
    VkSampler sampler{ VK_NULL_HANDLE };
    VkImage image{ VK_NULL_HANDLE };
    VkImageLayout imageLayout;
    VkDeviceMemory deviceMemory{ VK_NULL_HANDLE };
    VkImageView view{ VK_NULL_HANDLE };
    uint32_t width{ 0 };
    uint32_t height{ 0 };
    uint32_t mipLevels{ 0 };
};

class VkSampleBase : public VulkanExampleBase
{
public:
	VkSampleBase()
	{

	}

	virtual ~VkSampleBase()
	{

	}

    /**
     * 加载单个图像
     * */
	virtual void loadImage(NativeImage *pImage)
	{};

	/**
	 * 加载多个图像
	 * */
	virtual void loadMultiImageWithIndex(int index, NativeImage *pImage)
	{};

	/**
	 * 更新 MVP 矩阵
	 * */
	virtual void updateTransformMatrix(float rotateX, float rotateY, float scaleX, float scaleY)
	{}

	/**
	 * 设置屏幕点击坐标
	 * */
	virtual void setTouchLocation(float x, float y)
	{}

	virtual void setGravityXy(float x, float y)
	{}

	//初始化
	virtual void onInit(ANativeWindow* window, AAssetManager* assetManager) {
        LOGCATE("VkSampleBase::onInit window=%p, assetManager=%p", window, assetManager);
        this->window = window;
        this->assetManager = assetManager;
        if (!this->initVulkan()) {
            LOGCATE("VkSampleBase::onInit initVulkan fail.");
        }
        this->iTime = 0.0f;
    }
	//视口变化
	virtual void onOutputSizeChanged(int width, int height)
	{
        LOGCATE("VkSampleBase::onOutputSizeChanged width=%d, height=%d", width, height);
        this->outputWidth = width;
        this->outputHeight = height;
        if(!this->prepared) {
            this->prepare();
        } else {
            this->windowResize();
        }
	}
	//绘制
	virtual void onDrawFrame() {
        LOGCATE("VkSampleBase::onDrawFrame");
        this->render();
        this->iTime += 0.016;
    }
    //销毁资源
	virtual void onDestroy() {
        LOGCATE("VkSampleBase::onDestroy");
        this->destroy();
    }

    // Vulkan loads its shaders from an immediate binary representation called SPIR-V
    // Shaders are compiled offline from e.g. GLSL using the reference glslang compiler
    // This function loads such a shader from a binary file and returns a shader module structure
    VkShaderModule loadSPIRVShader(std::string filename) {
        size_t shaderSize;
        char *shaderCode{nullptr};
        // Load shader from compressed asset
        AAsset *asset = AAssetManager_open(this->assetManager, filename.c_str(),
                                           AASSET_MODE_STREAMING);
        assert(asset);
        shaderSize = AAsset_getLength(asset);
        assert(shaderSize > 0);

        shaderCode = new char[shaderSize];
        AAsset_read(asset, shaderCode, shaderSize);
        AAsset_close(asset);

        if (shaderCode) {
            // Create a new shader module that will be used for pipeline creation
            VkShaderModuleCreateInfo shaderModuleCI{};
            shaderModuleCI.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            shaderModuleCI.codeSize = shaderSize;
            shaderModuleCI.pCode = (uint32_t *) shaderCode;

            VkShaderModule shaderModule;
            VK_CHECK_RESULT(vkCreateShaderModule(device, &shaderModuleCI, nullptr, &shaderModule));

            delete[] shaderCode;

            return shaderModule;
        } else {
            LOGCATE("Error: Could not open shader file %s", filename.c_str());
            return VK_NULL_HANDLE;
        }
    }

    bool mapMemoryTypeToIndex(uint32_t typeBits, VkFlags requirements_mask,
                              uint32_t* typeIndex) {
        VkPhysicalDeviceMemoryProperties memoryProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);
        // Search memtypes to find first index with those properties
        for (uint32_t i = 0; i < 32; i++) {
            if ((typeBits & 1) == 1) {
                // Type is available, does it match user properties?
                if ((memoryProperties.memoryTypes[i].propertyFlags & requirements_mask) ==
                    requirements_mask) {
                    *typeIndex = i;
                    return true;
                }
            }
            typeBits >>= 1;
        }
        return false;
    }

    // This function is used to request a device memory type that supports all the property flags we request (e.g. device local, host visible)
    // Upon success it will return the index of the memory type that fits our requested memory properties
    // This is necessary as implementations can offer an arbitrary number of memory types with different
    // memory properties.
    // You can check https://vulkan.gpuinfo.org/ for details on different memory configurations
    uint32_t getMemoryTypeIndex(uint32_t typeBits, VkMemoryPropertyFlags properties) {
        // Iterate over all memory types available for the device used in this example
        for (uint32_t i = 0; i < deviceMemoryProperties.memoryTypeCount; i++) {
            if ((typeBits & 1) == 1) {
                if ((deviceMemoryProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                    return i;
                }
            }
            typeBits >>= 1;
        }

        throw "Could not find a suitable memory type!";
    }

    // A help function to map required memory property into a VK memory type
    // memory type is an index into the array of 32 entries; or the bit index
    // for the memory type ( each BIT of an 32 bit integer is a type ).
    VkResult allocateMemoryTypeFromProperties(uint32_t typeBits,
                                              VkFlags requirements_mask,
                                              uint32_t* typeIndex) {
        // Search memtypes to find first index with those properties
        for (uint32_t i = 0; i < 32; i++) {
            if ((typeBits & 1) == 1) {
                // Type is available, does it match user properties?
                if ((deviceMemoryProperties.memoryTypes[i].propertyFlags &
                     requirements_mask) == requirements_mask) {
                    *typeIndex = i;
                    return VK_SUCCESS;
                }
            }
            typeBits >>= 1;
        }
        // No memory types matched, return failure
        return VK_ERROR_MEMORY_MAP_FAILED;
    }

    void setImageLayout(VkCommandBuffer cmdBuffer, VkImage image,
                        VkImageLayout oldImageLayout, VkImageLayout newImageLayout,
                        VkPipelineStageFlags srcStages,
                        VkPipelineStageFlags destStages) {
        VkImageMemoryBarrier imageMemoryBarrier = {
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .pNext = NULL,
                .srcAccessMask = 0,
                .dstAccessMask = 0,
                .oldLayout = oldImageLayout,
                .newLayout = newImageLayout,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = image,
                .subresourceRange =
                        {
                                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                .baseMipLevel = 0,
                                .levelCount = 1,
                                .baseArrayLayer = 0,
                                .layerCount = 1,
                        },
        };

        switch (oldImageLayout) {
            case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
                imageMemoryBarrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                break;

            case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
                imageMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                break;

            case VK_IMAGE_LAYOUT_PREINITIALIZED:
                imageMemoryBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
                break;

            default:
                break;
        }

        switch (newImageLayout) {
            case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
                imageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                break;

            case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
                imageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                break;

            case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
                imageMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                break;

            case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
                imageMemoryBarrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                break;

            case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
                imageMemoryBarrier.dstAccessMask =
                        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                break;

            default:
                break;
        }

        vkCmdPipelineBarrier(cmdBuffer, srcStages, destStages, 0, 0, NULL, 0, NULL, 1,
                             &imageMemoryBarrier);
    }

    VkResult loadRGBTexture2DFromFile(const char* filePath, vks::Texture2D* texture2D,
                                      VkFilter filter = VK_FILTER_LINEAR,
                                      VkImageUsageFlags imageUsageFlags = VK_IMAGE_USAGE_SAMPLED_BIT,
                                      VkImageLayout imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        assert(filePath);
        VkFormat kTexFmt = VK_FORMAT_R8G8B8_UNORM;
        // Read the file:
        AAsset* file = AAssetManager_open(assetManager,
                                          filePath, AASSET_MODE_BUFFER);
        size_t fileLength = AAsset_getLength(file);
        stbi_uc* fileContent = new unsigned char[fileLength];
        AAsset_read(file, fileContent, fileLength);
        AAsset_close(file);

        uint32_t imgWidth, imgHeight, n;
        unsigned char* imageData = stbi_load_from_memory(
                fileContent, fileLength, reinterpret_cast<int*>(&imgWidth),
                reinterpret_cast<int*>(&imgHeight), reinterpret_cast<int*>(&n), 0);
        assert(n == 3 || n == 4);
        if(n == 4) kTexFmt = VK_FORMAT_R8G8B8A8_UNORM;

        texture2D->fromBuffer(imageData, imgWidth * imgHeight * n, kTexFmt, imgWidth, imgHeight, vulkanDevice, queue, filter,imageUsageFlags, imageLayout);
        stbi_image_free(imageData);
        delete[] fileContent;
        return VK_SUCCESS;
    }
public:
    NativeImage renderImage;
    float iTime;
};


#endif //VKSAMPLE_VkSampleBase_H
