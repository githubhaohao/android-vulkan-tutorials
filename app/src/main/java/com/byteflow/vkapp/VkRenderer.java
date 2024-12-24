package com.byteflow.vkapp;

import static com.byteflow.vkapp.JniImpl.SAMPLE_TYPE;

import android.content.res.AssetManager;
import android.util.Log;
import android.view.Surface;

public class VkRenderer {
    private static final String TAG = "GLRenderer";
    private JniImpl mNativeRender;
    private int mSampleType;

    VkRenderer() {
        mNativeRender = new JniImpl();
    }

    public void onSurfaceCreated() {
        mNativeRender.native_OnSurfaceCreated();
    }


    public void onSurfaceChanged(int width, int height) {
        mNativeRender.native_OnSurfaceChanged(width, height);

    }
    public void onDrawFrame() {
        mNativeRender.native_OnDrawFrame();
    }

    public void init(Surface surface, AssetManager assetManager) {
        mNativeRender.native_Init(surface, assetManager);
    }

    public void unInit() {
        mNativeRender.native_UnInit();
    }

    public void setParamsInt(int paramType, int value0, int value1) {
        if (paramType == SAMPLE_TYPE) {
            mSampleType = value0;
        }
        mNativeRender.native_SetParamsInt(paramType, value0, value1);
    }

//    public void setTouchLoc(float x, float y)
//    {
//        mNativeRender.native_SetParamsFloat(SAMPLE_TYPE_SET_TOUCH_LOC, x, y);
//    }
//
//    public void setGravityXY(float x, float y) {
//        mNativeRender.native_SetParamsFloat(SAMPLE_TYPE_SET_GRAVITY_XY, x, y);
//    }

    public void setImageData(int format, int width, int height, byte[] bytes) {
        mNativeRender.native_SetImageData(format, width, height, bytes);
    }

    public void setImageDataWithIndex(int index, int format, int width, int height, byte[] bytes) {
        mNativeRender.native_SetImageDataWithIndex(index, format, width, height, bytes);
    }

    public void setAudioData(short[] audioData) {
        mNativeRender.native_SetAudioData(audioData);
    }

    public int getSampleType() {
        return mSampleType;
    }

    public void updateTransformMatrix(float rotateX, float rotateY, float scaleX, float scaleY)
    {
        Log.d(TAG, "updateTransformMatrix() called with: rotateX = [" + rotateX + "], rotateY = [" + rotateY + "], scaleX = [" + scaleX + "], scaleY = [" + scaleY + "]");
        mNativeRender.native_UpdateTransformMatrix(rotateX, rotateY, scaleX, scaleY);
    }

}
