/**
 *
 * Created by 公众号：字节流动 on 2024/3/12.
 * 获取视频教程，有疑问或者技术交流可以添加微信 Byte-Flow ,拉你进技术交流群
 *
 * */

#ifndef OPENGLES_3_X_MYGLRENDERCONTEXT_H
#define OPENGLES_3_X_MYGLRENDERCONTEXT_H

#include "stdint.h"
#include <GLES3/gl3.h>
#include "VkSampleBase.h"

class VkRenderContext
{
	VkRenderContext();

	~VkRenderContext();

public:
    void onInit(JNIEnv *jniEnv, jobject surface, jobject assetsManager);

	void SetImageData(int format, int width, int height, uint8_t *pData);

	void SetImageDataWithIndex(int index, int format, int width, int height, uint8_t *pData);

	void SetParamsInt(int paramType, int value0, int value1);

	void SetParamsFloat(int paramType, float value0, float value1);

	void SetParamsShortArr(short *const pShortArr, int arrSize);

	void UpdateTransformMatrix(float rotateX, float rotateY, float scaleX, float scaleY);

	void OnSurfaceCreated();

	void OnSurfaceChanged(int width, int height);

	void OnDrawFrame();

	static VkRenderContext* GetInstance();
	static void DestroyInstance();

private:
	static VkRenderContext *m_pContext;
	VkSampleBase *m_pCurSample;
    VkSampleBase *m_pNewSample;
	int m_OutputWidth;
	int m_OutputHeight;
    AAssetManager* assetManager;
    ANativeWindow* window;
};


#endif //OPENGLES_3_X_MYGLRENDERCONTEXT_H
