package com.byteflow.vkapp;

import android.content.Context;
import android.util.AttributeSet;
import android.util.Log;
import android.view.MotionEvent;
import android.view.ScaleGestureDetector;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

import androidx.annotation.NonNull;

public class AutoFitView extends SurfaceView implements ScaleGestureDetector.OnScaleGestureListener, SurfaceHolder.Callback {
    private static final String TAG = "GLAutoFitView";

    private final float TOUCH_SCALE_FACTOR = 180.0f / 320;

    public static final int IMAGE_FORMAT_RGBA = 0x01;
    public static final int IMAGE_FORMAT_NV21 = 0x02;
    public static final int IMAGE_FORMAT_NV12 = 0x03;
    public static final int IMAGE_FORMAT_I420 = 0x04;
    public static final int IMAGE_FORMAT_YUYV = 0x05;
    public static final int IMAGE_FORMAT_GARY = 0x06;

    public static final int IMAGE_FORMAT_I444 = 0x07;


    private float mPreviousY;
    private float mPreviousX;
    private int mXAngle;
    private int mYAngle;

    private VkRenderer mVkRenderer;

    private int mRatioWidth = 0;
    private int mRatioHeight = 0;

    private ScaleGestureDetector mScaleGestureDetector;
    private float mPreScale = 1.0f;
    private float mCurScale = 1.0f;
    private long mLastMultiTouchTime;

//    public GLAutoFitView(Context context) {
//        this(context, null);
//    }
//
//    public GLAutoFitView(Context context, AttributeSet attrs) {
//        super(context, attrs);
//        this.setEGLContextClientVersion(2);
//        mGLRender = new MyGLRender();
//        /*If no setEGLConfigChooser method is called,
//        then by default the view will choose an RGB_888 surface with a depth buffer depth of at least 16 bits.*/
//        setEGLConfigChooser(8, 8, 8, 8, 16, 8);
//        setRenderer(mGLRender);
//        setRenderMode(RENDERMODE_WHEN_DIRTY);
//        mScaleGestureDetector = new ScaleGestureDetector(context, this);
//
//    }

    public AutoFitView(Context context, VkRenderer glRender) {
        this(context, glRender, null);
    }

    public AutoFitView(Context context, VkRenderer glRender, AttributeSet attrs) {
        super(context, attrs);
        mVkRenderer = glRender;
        mScaleGestureDetector = new ScaleGestureDetector(context, this);
        getHolder().addCallback(this);
    }

    @Override
    public boolean onTouchEvent(MotionEvent e) {
        if (e.getPointerCount() == 1) {
            consumeTouchEvent(e);
            long currentTimeMillis = System.currentTimeMillis();
            if (currentTimeMillis - mLastMultiTouchTime > 200)
            {
                float y = e.getY();
                float x = e.getX();
                switch (e.getAction()) {
                    case MotionEvent.ACTION_MOVE:
                        float dy = y - mPreviousY;
                        float dx = x - mPreviousX;
                        mYAngle += dx * TOUCH_SCALE_FACTOR;
                        mXAngle += dy * TOUCH_SCALE_FACTOR;
                }
                mPreviousY = y;
                mPreviousX = x;
                mVkRenderer.updateTransformMatrix(mXAngle, mYAngle, mCurScale, mCurScale);
                requestRender();
//                switch (mVkRenderer.getSampleType()) {
//                    case SAMPLE_TYPE_TRIANGLE:
//                    case SAMPLE_TYPE_COORD_SYSTEM:
//                    case SAMPLE_TYPE_BASIC_LIGHTING:
//                    case SAMPLE_TYPE_TRANS_FEEDBACK:
//                    case SAMPLE_TYPE_DEPTH_TESTING:
//                    case SAMPLE_TYPE_INSTANCING:
//                    case SAMPLE_TYPE_STENCIL_TESTING:
//                    case SAMPLE_TYPE_PARTICLES:
//                    case SAMPLE_TYPE_SKYBOX:
//                    case SAMPLE_TYPE_3D_MODEL:
//                    case SAMPLE_TYPE_UBO:
//                    case SAMPLE_TYPE_FILTER_ROTATE:
//                        mVkRenderer.updateTransformMatrix(mXAngle, mYAngle, mCurScale, mCurScale);
//                        requestRender();
//                        break;
//                    default:
//                        break;
//                }
            }

        } else {
            mScaleGestureDetector.onTouchEvent(e);
        }

        return true;
    }

    @Override
    protected void onMeasure(int widthMeasureSpec, int heightMeasureSpec) {
        super.onMeasure(widthMeasureSpec, heightMeasureSpec);
        int width = MeasureSpec.getSize(widthMeasureSpec);
        int height = MeasureSpec.getSize(heightMeasureSpec);

        if (0 == mRatioWidth || 0 == mRatioHeight) {
            setMeasuredDimension(width, height);
        } else {
            if (width < height * mRatioWidth / mRatioHeight) {
                setMeasuredDimension(width, width * mRatioHeight / mRatioWidth);
            } else {
                setMeasuredDimension(height * mRatioWidth / mRatioHeight, height);
            }
        }
    }

    public void setAspectRatio(int width, int height) {
        Log.d(TAG, "setAspectRatio() called with: outputWidth = [" + width + "], outputHeight = [" + height + "]");
        if (width < 0 || height < 0) {
            throw new IllegalArgumentException("Size cannot be negative.");
        }

        mRatioWidth = width;
        mRatioHeight = height;
        requestLayout();
    }

    public VkRenderer getVkRenderer() {
        return mVkRenderer;
    }

    @Override
    public boolean onScale(ScaleGestureDetector detector) {
        float preSpan = detector.getPreviousSpan();
        float curSpan = detector.getCurrentSpan();
        if (curSpan < preSpan) {
            mCurScale = mPreScale - (preSpan - curSpan) / 200;
        } else {
            mCurScale = mPreScale + (curSpan - preSpan) / 200;
        }
        mCurScale = Math.max(0.05f, Math.min(mCurScale, 80.0f));
        mVkRenderer.updateTransformMatrix(mXAngle, mYAngle, mCurScale, mCurScale);
        requestRender();
//        switch (mVkRenderer.getSampleType()) {
//            case SAMPLE_TYPE_TRIANGLE:
//            case SAMPLE_TYPE_COORD_SYSTEM:
//            case SAMPLE_TYPE_BASIC_LIGHTING:
//            case SAMPLE_TYPE_INSTANCING:
//            case SAMPLE_TYPE_3D_MODEL:
//            {
//                float preSpan = detector.getPreviousSpan();
//                float curSpan = detector.getCurrentSpan();
//                if (curSpan < preSpan) {
//                    mCurScale = mPreScale - (preSpan - curSpan) / 200;
//                } else {
//                    mCurScale = mPreScale + (curSpan - preSpan) / 200;
//                }
//                mCurScale = Math.max(0.05f, Math.min(mCurScale, 80.0f));
//                mVkRenderer.updateTransformMatrix(mXAngle, mYAngle, mCurScale, mCurScale);
//                requestRender();
//            }
//                break;
//            default:
//                break;
//        }

        return false;
    }

    @Override
    public boolean onScaleBegin(ScaleGestureDetector detector) {
        return true;
    }

    @Override
    public void onScaleEnd(ScaleGestureDetector detector) {
        mPreScale = mCurScale;
        mLastMultiTouchTime = System.currentTimeMillis();

    }

    public void consumeTouchEvent(MotionEvent e) {
        dealClickEvent(e);
        float touchX = -1, touchY = -1;
        switch (e.getAction()) {
            case MotionEvent.ACTION_MOVE:
                touchX = e.getX();
                touchY = e.getY();
                break;
            case MotionEvent.ACTION_UP:
            case MotionEvent.ACTION_CANCEL:
                touchX = -1;
                touchY = -1;
                break;
            default:
                break;
        }

//        //滑动、触摸
//        switch (mGLRender.getSampleType()) {
//            case SAMPLE_TYPE_KEY_SCRATCH_CARD:
//                mGLRender.setTouchLoc(touchX, touchY);
//                requestRender();
//                break;
//            default:
//                break;
//        }

        //点击
        switch (e.getAction()) {
            case MotionEvent.ACTION_MOVE:
                //touchX = e.getX();
                //touchY = e.getY();
                break;
            case MotionEvent.ACTION_UP:
                break;
            case MotionEvent.ACTION_CANCEL:
                break;
            default:
                break;
        }

    }

    public void dealClickEvent(MotionEvent e) {
        float touchX = -1, touchY = -1;
        switch (e.getAction()) {
            case MotionEvent.ACTION_UP:
                touchX = e.getX();
                touchY = e.getY();
            {
//                //点击
//                switch (mGLRender.getSampleType()) {
//                    case SAMPLE_TYPE_KEY_SHOCK_WAVE:
//                        mGLRender.setTouchLoc(touchX, touchY);
//                        break;
//                    default:
//                        break;
//                }
            }
                break;
            default:
                break;
        }
    }

    @Override
    public void surfaceCreated(@NonNull SurfaceHolder surfaceHolder) {
        mVkRenderer.init(surfaceHolder.getSurface(), getContext().getAssets());
        mVkRenderer.onSurfaceCreated();
    }

    @Override
    public void surfaceChanged(@NonNull SurfaceHolder surfaceHolder, int i, int i1, int i2) {
        mVkRenderer.onSurfaceChanged(i1, i2);
        requestRender();
    }

    @Override
    public void surfaceDestroyed(@NonNull SurfaceHolder surfaceHolder) {
        mVkRenderer.unInit();
    }

    public void requestRender() {
        mVkRenderer.onDrawFrame();
    }
}