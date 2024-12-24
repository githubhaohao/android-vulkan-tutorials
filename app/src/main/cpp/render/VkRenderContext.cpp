/**
 *
 * Created by 公众号：字节流动 on 2024/3/12.
 * 获取视频教程，有疑问或者技术交流可以添加微信 Byte-Flow ,拉你进技术交流群
 *
 * */


#include "VkRenderContext.h"
#include "LogUtil.h"
#include <VkSample01_Triangle.h>
#include "VkSample02_Ubo.h"
#include "VkSample03_TextureMapping.h"
#include "VkSample04_Pipelines.h"
#include "VkSample05_PushConstants.h"
#include "VkSample06_RenderNV21.h"
#include "VkSample07_RenderI420.h"
#include "VkSample08_RenderYUYV.h"
#include "VkSample09_RenderI444.h"
#include "VkSample10_SpecializationInfo.h"
#include "VkSample11_CubeMap.h"
#include "VkSample12_InputAttachments.h"
#include "VkSample13_OffScreenRendering.h"
#include "VkSample14_DepthTesting.h"
#include "VkSample15_StencilTesting.h"
#include "VkSample16_MultiSampling.h"
#include "VkSample17_MultiThreading.h"
#include "VkSample18_Instancing.h"
#include "VkSample19_ReadPixels.h"
#include "VkSample20_ComputeShader.h"
#include <android/native_window_jni.h>
#include <android/asset_manager_jni.h>
#include <android/asset_manager.h>

VkRenderContext* VkRenderContext::m_pContext = nullptr;

VkRenderContext::VkRenderContext()
{
	m_pNewSample = new VkSample01_Triangle();
	m_pCurSample = nullptr;
    assetManager = nullptr;
    window = nullptr;
}

VkRenderContext::~VkRenderContext()
{
	if (m_pCurSample)
	{
        m_pCurSample->onDestroy();
		delete m_pCurSample;
		m_pCurSample = nullptr;
	}

    if(window) {
        ANativeWindow_release(window);
        window = nullptr;
    }
    assetManager = nullptr;
}

void VkRenderContext::onInit(JNIEnv *jniEnv, jobject surface, jobject assetsManager) {
    LOGCATE("VkRenderContext::onInit");
    assetManager = AAssetManager_fromJava(jniEnv, assetsManager);
    window = ANativeWindow_fromSurface(jniEnv, surface);
}

void VkRenderContext::SetParamsInt(int paramType, int value0, int value1)
{
	LOGCATE("VkRenderContext::SetParamsInt paramType = %d, value0 = %d, value1 = %d", paramType, value0, value1);

	if (paramType == SAMPLE_TYPE)
	{
		switch (value0)
		{
			case SAMPLE_TYPE_KEY_TRIANGLE:
				m_pNewSample = new VkSample01_Triangle();
				break;
            case SAMPLE_TYPE_KEY_UBO:
                m_pNewSample = new VkSample02_Ubo();
                break;
            case SAMPLE_TYPE_KEY_TEXTURE_MAPPING:
                m_pNewSample = new VkSample03_TextureMapping();
                break;
            case SAMPLE_TYPE_KEY_PIPELINES:
                m_pNewSample = new VkSample04_Pipelines();
                break;
            case SAMPLE_TYPE_KEY_PUSH_CONSTANTS:
                m_pNewSample = new VkSample05_PushConstants();
                break;
            case SAMPLE_TYPE_KEY_RENDER_NV21:
                m_pNewSample = new VkSample06_RenderNV21();
                break;
            case SAMPLE_TYPE_KEY_RENDER_I420:
                m_pNewSample = new VkSample07_RenderI420();
                break;
            case SAMPLE_TYPE_KEY_RENDER_YUYV:
                m_pNewSample = new VkSample08_RenderYUYV();
                break;
            case SAMPLE_TYPE_KEY_RENDER_I444:
                m_pNewSample = new VkSample09_RenderI444();
                break;
            case SAMPLE_TYPE_KEY_SPECIALIZATION_INFO:
                m_pNewSample = new VkSample10_SpecializationInfo();
                break;
            case SAMPLE_TYPE_KEY_CUBEMAP:
                m_pNewSample = new VkSample11_CubeMap();
                break;
            case SAMPLE_TYPE_KEY_INPUT_ATTACHMENTS:
                m_pNewSample = new VkSample12_InputAttachments();
                break;
            case SAMPLE_TYPE_KEY_OFFSCREEN_RENDERING:
                m_pNewSample = new VkSample13_OffScreenRendering();
                break;
            case SAMPLE_TYPE_KEY_DEPTH_TESTING:
                m_pNewSample = new VkSample14_DepthTesting();
                break;
            case SAMPLE_TYPE_KEY_STENCIL_TESTING:
                m_pNewSample = new VkSample15_StencilTesting();
                break;
            case SAMPLE_TYPE_KEY_MULTISAMPLING:
                m_pNewSample = new VkSample16_MultiSampling();
                break;
            case SAMPLE_TYPE_KEY_MULTITHREADING:
                m_pNewSample = new VkSample17_MultiThreading();
                break;
            case SAMPLE_TYPE_KEY_INSTANCING:
                m_pNewSample = new VkSample18_Instancing();
                break;
            case SAMPLE_TYPE_KEY_READ_PIXELS:
                m_pNewSample = new VkSample19_ReadPixels();
                break;
            case SAMPLE_TYPE_KEY_COMPUTE_SHADER:
                m_pNewSample = new VkSample20_ComputeShader();
                break;
			default:
				m_pNewSample = nullptr;
				break;
		}

		LOGCATE("VkRenderContext::SetParamsInt m_pNewSample = %p, m_pCurSample=%p", m_pNewSample, m_pCurSample);
	}
}

void VkRenderContext::SetParamsFloat(int paramType, float value0, float value1) {
	LOGCATE("VkRenderContext::SetParamsFloat paramType=%d, value0=%f, value1=%f", paramType, value0, value1);
	if(m_pCurSample)
	{
		switch (paramType)
		{
			case SAMPLE_TYPE_KEY_SET_TOUCH_LOC:
                m_pCurSample->setTouchLocation(value0, value1);
				break;
			case SAMPLE_TYPE_SET_GRAVITY_XY:
				//m_pCurSample->setGravityXy(value0, value1);
				break;
			default:
				break;

		}
	}

}

void VkRenderContext::SetParamsShortArr(short *const pShortArr, int arrSize) {
	LOGCATE("VkRenderContext::SetParamsShortArr pShortArr=%p, arrSize=%d, pShortArr[0]=%d", pShortArr, arrSize, pShortArr[0]);
	if(m_pCurSample)
	{
		//m_pCurSample->LoadShortArrData(pShortArr, arrSize);
	}

}

void VkRenderContext::UpdateTransformMatrix(float rotateX, float rotateY, float scaleX, float scaleY)
{
	LOGCATE("VkRenderContext::updateTransformMatrix [rotateX, rotateY, scaleX, scaleY] = [%f, %f, %f, %f]", rotateX, rotateY, scaleX, scaleY);
	if (m_pCurSample)
	{
        m_pCurSample->updateTransformMatrix(rotateX, rotateY, scaleX, scaleY);
	}
}

void VkRenderContext::SetImageDataWithIndex(int index, int format, int width, int height, uint8_t *pData)
{
	LOGCATE("VkRenderContext::SetImageDataWithIndex index=%d, format=%d, outputWidth=%d, outputHeight=%d, pData=%p", index, format, width, height, pData);
	NativeImage nativeImage;
	nativeImage.format = format;
	nativeImage.width = width;
	nativeImage.height = height;
	nativeImage.ppPlane[0] = pData;

	switch (format)
	{
		case IMAGE_FORMAT_NV12:
		case IMAGE_FORMAT_NV21:
			nativeImage.ppPlane[1] = nativeImage.ppPlane[0] + width * height;
			break;
		case IMAGE_FORMAT_I420:
			nativeImage.ppPlane[1] = nativeImage.ppPlane[0] + width * height;
			nativeImage.ppPlane[2] = nativeImage.ppPlane[1] + width * height / 4;
			break;
		default:
			break;
	}

	if (m_pNewSample)
	{
        m_pNewSample->loadMultiImageWithIndex(index, &nativeImage);
	}

}

void VkRenderContext::SetImageData(int format, int width, int height, uint8_t *pData)
{
	LOGCATE("VkRenderContext::SetImageData format=%d, outputWidth=%d, outputHeight=%d, pData=%p", format, width, height, pData);
	NativeImage nativeImage;
	nativeImage.format = format;
	nativeImage.width = width;
	nativeImage.height = height;
	nativeImage.ppPlane[0] = pData;

	switch (format)
	{
		case IMAGE_FORMAT_NV12:
		case IMAGE_FORMAT_NV21:
			nativeImage.ppPlane[1] = nativeImage.ppPlane[0] + width * height;
			break;
		case IMAGE_FORMAT_I420:
			nativeImage.ppPlane[1] = nativeImage.ppPlane[0] + width * height;
			nativeImage.ppPlane[2] = nativeImage.ppPlane[1] + width * height / 4;
			break;
		default:
			break;
	}

	if (m_pNewSample)
	{
        m_pNewSample->loadImage(&nativeImage);
	}

}

void VkRenderContext::OnSurfaceCreated()
{
	LOGCATE("VkRenderContext::OnSurfaceCreated");

}

void VkRenderContext::OnSurfaceChanged(int width, int height)
{
	LOGCATE("VkRenderContext::OnSurfaceChanged [w, h] = [%d, %d]", width, height);
	m_OutputWidth = width / 2 * 2;
	m_OutputHeight = height / 2 * 2;;
}

void VkRenderContext::OnDrawFrame()
{
	LOGCATE("VkRenderContext::onDrawFrame");

	if (m_pNewSample)
	{
		if(m_pCurSample) {
            m_pCurSample->onDestroy();
			delete m_pCurSample;
		}

        m_pNewSample->onInit(window, assetManager);
        m_pNewSample->onOutputSizeChanged(m_OutputWidth, m_OutputHeight);

		m_pCurSample = m_pNewSample;
		m_pNewSample = nullptr;
	}

	if (m_pCurSample)
	{
        m_pCurSample->onDrawFrame();
	}
}

VkRenderContext *VkRenderContext::GetInstance()
{
	LOGCATE("VkRenderContext::GetInstance");
	if (m_pContext == nullptr)
	{
		m_pContext = new VkRenderContext();
	}
	return m_pContext;
}

void VkRenderContext::DestroyInstance()
{
	LOGCATE("VkRenderContext::DestroyInstance");
	if (m_pContext)
	{
		delete m_pContext;
		m_pContext = nullptr;
	}

}



