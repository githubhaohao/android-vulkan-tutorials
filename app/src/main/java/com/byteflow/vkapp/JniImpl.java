/**
 *
 * Created by 公众号：字节流动 on 2024/3/12.
 * 技术交流、获取视频教程可以添加我的个人微信 Byte-Flow , 或关注公众号：字节流动，拉你进技术交流群
 *
 * */

package com.byteflow.vkapp;

import android.content.res.AssetManager;
import android.view.Surface;

public class JniImpl {
    public static final int SAMPLE_TYPE  =  200;

    public static final int SAMPLE_TYPE_KEY_TRIANGLE                = SAMPLE_TYPE;
    public static final int SAMPLE_TYPE_KEY_UBO                     = SAMPLE_TYPE + 1;
    public static final int SAMPLE_TYPE_KEY_TEXTURE_MAPPING         = SAMPLE_TYPE + 2;
    public static final int SAMPLE_TYPE_KEY_PIPELINES               = SAMPLE_TYPE + 3;
    public static final int SAMPLE_TYPE_KEY_PUSH_CONSTANTS          = SAMPLE_TYPE + 4;
    public static final int SAMPLE_TYPE_KEY_RENDER_NV21             = SAMPLE_TYPE + 5;
    public static final int SAMPLE_TYPE_KEY_RENDER_I420             = SAMPLE_TYPE + 6;
    public static final int SAMPLE_TYPE_KEY_RENDER_YUYV             = SAMPLE_TYPE + 7;
    public static final int SAMPLE_TYPE_KEY_RENDER_I444             = SAMPLE_TYPE + 8;
    public static final int SAMPLE_TYPE_KEY_SPECIALIZATION_INFO     = SAMPLE_TYPE + 9;
    public static final int SAMPLE_TYPE_KEY_CUBEMAP                 = SAMPLE_TYPE + 10;
    public static final int SAMPLE_TYPE_KEY_INPUT_ATTACHMENTS       = SAMPLE_TYPE + 11;
    public static final int SAMPLE_TYPE_KEY_OFFSCREEN_RENDERING     = SAMPLE_TYPE + 12;
    public static final int SAMPLE_TYPE_KEY_DEPTH_TESTING           = SAMPLE_TYPE + 13;
    public static final int SAMPLE_TYPE_KEY_STENCIL_TESTING         = SAMPLE_TYPE + 14;
    public static final int SAMPLE_TYPE_KEY_MULTISAMPLING           = SAMPLE_TYPE + 15;
    public static final int SAMPLE_TYPE_KEY_MULTITHREADING          = SAMPLE_TYPE + 16;
    public static final int SAMPLE_TYPE_KEY_INSTANCING              = SAMPLE_TYPE + 17;
    public static final int SAMPLE_TYPE_KEY_READ_PIXELS             = SAMPLE_TYPE + 18;
    public static final int SAMPLE_TYPE_KEY_COMPUTE_SHADER          = SAMPLE_TYPE + 19;

    public static final int SAMPLE_TYPE_FBO                     = SAMPLE_TYPE + 901;
    public static final int SAMPLE_TYPE_EGL                     = SAMPLE_TYPE + 900;
    public static final int SAMPLE_TYPE_COORD_SYSTEM            = SAMPLE_TYPE + 100;
    public static final int SAMPLE_TYPE_BASIC_LIGHTING          = SAMPLE_TYPE + 110;
    public static final int SAMPLE_TYPE_BLENDING                = SAMPLE_TYPE + 140;
    public static final int SAMPLE_TYPE_TRANS_FEEDBACK          = SAMPLE_TYPE + 150;
    public static final int SAMPLE_TYPE_INSTANCING              = SAMPLE_TYPE + 160;
    public static final int SAMPLE_TYPE_INSTANCING3D            = SAMPLE_TYPE + 170;
    public static final int SAMPLE_TYPE_PARTICLES               = SAMPLE_TYPE + 180;
    public static final int SAMPLE_TYPE_SKYBOX                  = SAMPLE_TYPE + 190;
    public static final int SAMPLE_TYPE_FILTER_SPLITSCREEN      = SAMPLE_TYPE + 200;
    public static final int SAMPLE_TYPE_FILTER_COLOROFFSET      = SAMPLE_TYPE + 210;
    public static final int SAMPLE_TYPE_FILTER_ROTATE           = SAMPLE_TYPE + 220;
    public static final int SAMPLE_TYPE_LUT                     = SAMPLE_TYPE + 230;
    public static final int SAMPLE_TYPE_3D_MODEL                = SAMPLE_TYPE + 240;
    public static final int SAMPLE_TYPE_PBO                     = SAMPLE_TYPE + 250;
    public static final int SAMPLE_TYPE_MRT                     = SAMPLE_TYPE + 260;
    public static final int SAMPLE_TYPE_COMPUTE_SHADER          = SAMPLE_TYPE + 270;
    public static final int SAMPLE_TYPE_TBO                     = SAMPLE_TYPE + 280;

    static {
        System.loadLibrary("native-render");
    }

    public native void native_Init(Surface surface, AssetManager assetManager);

    public native void native_UnInit();

    public native void native_SetParamsInt(int paramType, int value0, int value1);

    public native void native_SetParamsFloat(int paramType, float value0, float value1);

    public native void native_UpdateTransformMatrix(float rotateX, float rotateY, float scaleX, float scaleY);

    public native void native_SetImageData(int format, int width, int height, byte[] bytes);

    public native void native_SetImageDataWithIndex(int index, int format, int width, int height, byte[] bytes);

    public native void native_SetAudioData(short[] audioData);

    public native void native_OnSurfaceCreated();

    public native void native_OnSurfaceChanged(int width, int height);

    public native void native_OnDrawFrame();
}
