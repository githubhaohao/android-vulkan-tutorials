/**
 *
 * Created by 公众号：字节流动 on 2024/3/12.
 * 技术交流、获取视频教程可以添加我的个人微信 Byte-Flow , 或关注公众号：字节流动，拉你进技术交流群
 *
 * */

package com.byteflow.vkapp;

import android.Manifest;
import android.app.AlertDialog;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import androidx.annotation.*;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import android.os.Bundle;
import android.os.Environment;
import android.view.LayoutInflater;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.ViewGroup;
import android.view.ViewTreeObserver;
import android.widget.Button;
import android.widget.RelativeLayout;
import android.widget.Toast;

import com.byteflow.vkapp.adapter.MyRecyclerViewAdapter;
import com.byteflow.vkapp.audio.AudioCollector;
import com.byteflow.vkapp.egl.EGLActivity;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.Arrays;

import static com.byteflow.vkapp.GLAutoFitView.IMAGE_FORMAT_GARY;
import static com.byteflow.vkapp.GLAutoFitView.IMAGE_FORMAT_I420;
import static com.byteflow.vkapp.GLAutoFitView.IMAGE_FORMAT_I444;
import static com.byteflow.vkapp.GLAutoFitView.IMAGE_FORMAT_NV21;
import static com.byteflow.vkapp.GLAutoFitView.IMAGE_FORMAT_RGBA;
import static com.byteflow.vkapp.GLAutoFitView.IMAGE_FORMAT_YUYV;
import static com.byteflow.vkapp.JniImpl.SAMPLE_TYPE;
import static com.byteflow.vkapp.JniImpl.SAMPLE_TYPE_3D_MODEL;
import static com.byteflow.vkapp.JniImpl.SAMPLE_TYPE_BASIC_LIGHTING;
import static com.byteflow.vkapp.JniImpl.SAMPLE_TYPE_BLENDING;
import static com.byteflow.vkapp.JniImpl.SAMPLE_TYPE_COORD_SYSTEM;
import static com.byteflow.vkapp.JniImpl.SAMPLE_TYPE_KEY_COMPUTE_SHADER;
import static com.byteflow.vkapp.JniImpl.SAMPLE_TYPE_KEY_DEPTH_TESTING;
import static com.byteflow.vkapp.JniImpl.SAMPLE_TYPE_EGL;
import static com.byteflow.vkapp.JniImpl.SAMPLE_TYPE_FBO;
import static com.byteflow.vkapp.JniImpl.SAMPLE_TYPE_FILTER_COLOROFFSET;
import static com.byteflow.vkapp.JniImpl.SAMPLE_TYPE_FILTER_SPLITSCREEN;
import static com.byteflow.vkapp.JniImpl.SAMPLE_TYPE_FILTER_ROTATE;
import static com.byteflow.vkapp.JniImpl.SAMPLE_TYPE_INSTANCING;
import static com.byteflow.vkapp.JniImpl.SAMPLE_TYPE_INSTANCING3D;
import static com.byteflow.vkapp.JniImpl.SAMPLE_TYPE_COMPUTE_SHADER;
import static com.byteflow.vkapp.JniImpl.SAMPLE_TYPE_KEY_CUBEMAP;
import static com.byteflow.vkapp.JniImpl.SAMPLE_TYPE_KEY_INPUT_ATTACHMENTS;
import static com.byteflow.vkapp.JniImpl.SAMPLE_TYPE_KEY_INSTANCING;
import static com.byteflow.vkapp.JniImpl.SAMPLE_TYPE_KEY_MULTITHREADING;
import static com.byteflow.vkapp.JniImpl.SAMPLE_TYPE_KEY_MULTISAMPLING;
import static com.byteflow.vkapp.JniImpl.SAMPLE_TYPE_KEY_OFFSCREEN_RENDERING;
import static com.byteflow.vkapp.JniImpl.SAMPLE_TYPE_KEY_READ_PIXELS;
import static com.byteflow.vkapp.JniImpl.SAMPLE_TYPE_KEY_SPECIALIZATION_INFO;
import static com.byteflow.vkapp.JniImpl.SAMPLE_TYPE_KEY_PIPELINES;
import static com.byteflow.vkapp.JniImpl.SAMPLE_TYPE_KEY_PUSH_CONSTANTS;
import static com.byteflow.vkapp.JniImpl.SAMPLE_TYPE_TBO;
import static com.byteflow.vkapp.JniImpl.SAMPLE_TYPE_KEY_UBO;
import static com.byteflow.vkapp.JniImpl.SAMPLE_TYPE_LUT;
import static com.byteflow.vkapp.JniImpl.SAMPLE_TYPE_MRT;
import static com.byteflow.vkapp.JniImpl.SAMPLE_TYPE_PARTICLES;
import static com.byteflow.vkapp.JniImpl.SAMPLE_TYPE_PBO;
import static com.byteflow.vkapp.JniImpl.SAMPLE_TYPE_KEY_RENDER_I420;
import static com.byteflow.vkapp.JniImpl.SAMPLE_TYPE_KEY_RENDER_I444;
import static com.byteflow.vkapp.JniImpl.SAMPLE_TYPE_KEY_RENDER_NV21;
import static com.byteflow.vkapp.JniImpl.SAMPLE_TYPE_KEY_RENDER_YUYV;
import static com.byteflow.vkapp.JniImpl.SAMPLE_TYPE_SKYBOX;
import static com.byteflow.vkapp.JniImpl.SAMPLE_TYPE_KEY_STENCIL_TESTING;
import static com.byteflow.vkapp.JniImpl.SAMPLE_TYPE_KEY_TEXTURE_MAPPING;
import static com.byteflow.vkapp.JniImpl.SAMPLE_TYPE_TRANS_FEEDBACK;
import static com.byteflow.vkapp.JniImpl.SAMPLE_TYPE_KEY_TRIANGLE;

public class MainActivity extends AppCompatActivity implements AudioCollector.Callback, ViewTreeObserver.OnGlobalLayoutListener, SensorEventListener {
    private static final String TAG = "MainActivity";
    private static final String[] REQUEST_PERMISSIONS = {
            Manifest.permission.WRITE_EXTERNAL_STORAGE,
            Manifest.permission.RECORD_AUDIO,
    };
    private static final int PERMISSION_REQUEST_CODE = 1;
    private static final String[] SAMPLE_TITLES = {
            "VkSample01_Triangle",
            "VkSample02_Ubo",
            "VkSample03_TextureMapping",
            "VkSample04_Pipelines",
            "VkSample05_PushConstants",
            "VkSample06_RenderNV21",
            "VkSample07_RenderI420",
            "VkSample08_RenderYUYV",
            "VkSample09_RenderI444",
            "VkSample10_SpecializationInfo",
            "VkSample11_CubeMap",
            "VkSample12_InputAttachments",
            "VkSample13_OffscreenRendering",
            "VkSample14_DepthTesting",
            "VkSample15_StencilTesting",
            "VkSample16_MultiSampling",
            "VkSample17_MultiThreading",
            "VkSample18_Instancing",
            "VkSample19_ReadPixels",
            "VkSample20_ComputeShader",
    };

    private AutoFitView mAutoFitView;
    private ViewGroup mRootView;
    private int mSampleSelectedIndex = 0;
    private AudioCollector mAudioCollector;
    private VkRenderer mVkRenderer = new VkRenderer();
    private SensorManager mSensorManager;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        mRootView = (ViewGroup) findViewById(R.id.rootView);
        mRootView.getViewTreeObserver().addOnGlobalLayoutListener(this);
        mSensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);
        //mVkRenderer.init();

    }

    @Override
    public void onGlobalLayout() {
        mRootView.getViewTreeObserver().removeOnGlobalLayoutListener(this);
        RelativeLayout.LayoutParams lp = new RelativeLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT);
        lp.addRule(RelativeLayout.CENTER_IN_PARENT);
        mAutoFitView = new AutoFitView(this, mVkRenderer);
        mRootView.addView(mAutoFitView, lp);

    }

    @Override
    protected void onResume() {
        super.onResume();
        mSensorManager.registerListener(this,
                mSensorManager.getDefaultSensor(Sensor.TYPE_GRAVITY),
                SensorManager.SENSOR_DELAY_FASTEST);
        if (!hasPermissionsGranted(REQUEST_PERMISSIONS)) {
            ActivityCompat.requestPermissions(this, REQUEST_PERMISSIONS, PERMISSION_REQUEST_CODE);
        }
        ///sdcard/Android/data/com.byteflow.vkapp/files/Download
        String fileDir = getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS).getAbsolutePath();
        CommonUtils.copyAssetsDirToSDCard(MainActivity.this, "poly", fileDir + "/model");
        CommonUtils.copyAssetsDirToSDCard(MainActivity.this, "fonts", fileDir);
        CommonUtils.copyAssetsDirToSDCard(MainActivity.this, "yuv", fileDir);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if (requestCode == PERMISSION_REQUEST_CODE) {
            if (!hasPermissionsGranted(REQUEST_PERMISSIONS)) {
                Toast.makeText(this, "We need the permission: WRITE_EXTERNAL_STORAGE", Toast.LENGTH_SHORT).show();
            }
        } else {
            super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        mSensorManager.unregisterListener(this);
        if (mAudioCollector != null) {
            mAudioCollector.unInit();
            mAudioCollector = null;
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        //mVkRenderer.unInit();
        /*
        * Once the EGL context gets destroyed all the GL buffers etc will get destroyed with it,
        * so this is unnecessary.
        * */
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        int id = item.getItemId();

        //noinspection SimplifiableIfStatement
        if (id == R.id.action_change_sample) {
            showGLSampleDialog();
        }
        return true;
    }

    @Override
    public void onAudioBufferCallback(short[] buffer) {
        //Log.e(TAG, "onAudioBufferCallback() called with: buffer[0] = [" + buffer[0] + "]");
        mVkRenderer.setAudioData(buffer);
        //mGLSurfaceView.requestRender();
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        switch (event.sensor.getType()) {
            case Sensor.TYPE_GRAVITY:
                //Log.d(TAG, "onSensorChanged() called with TYPE_GRAVITY: [x,y,z] = [" + event.values[0] + ", " + event.values[1] + ", " + event.values[2] + "]");
//                if(mSampleSelectedIndex + SAMPLE_TYPE == SAMPLE_TYPE_KEY_AVATAR)
//                {
//                    mGLRender.setGravityXY(event.values[0], event.values[1]);
//                }
                break;
        }

    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {

    }

    private void showGLSampleDialog() {
        final AlertDialog.Builder builder = new AlertDialog.Builder(this);
        LayoutInflater inflater = LayoutInflater.from(this);
        final View rootView = inflater.inflate(R.layout.sample_selected_layout, null);

        final AlertDialog dialog = builder.create();

        Button confirmBtn = rootView.findViewById(R.id.confirm_btn);
        confirmBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                dialog.cancel();
            }
        });

        final RecyclerView resolutionsListView = rootView.findViewById(R.id.resolution_list_view);

        final MyRecyclerViewAdapter myPreviewSizeViewAdapter = new MyRecyclerViewAdapter(this, Arrays.asList(SAMPLE_TITLES));
        myPreviewSizeViewAdapter.setSelectIndex(mSampleSelectedIndex);
        myPreviewSizeViewAdapter.addOnItemClickListener(new MyRecyclerViewAdapter.OnItemClickListener() {
            @Override
            public void onItemClick(View view, int position) {
                mRootView.removeView(mAutoFitView);
                RelativeLayout.LayoutParams lp = new RelativeLayout.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT);
                lp.addRule(RelativeLayout.CENTER_IN_PARENT);
                mAutoFitView = new AutoFitView(MainActivity.this, mVkRenderer);
                mRootView.addView(mAutoFitView, lp);

                int selectIndex = myPreviewSizeViewAdapter.getSelectIndex();
                myPreviewSizeViewAdapter.setSelectIndex(position);
                myPreviewSizeViewAdapter.notifyItemChanged(selectIndex);
                myPreviewSizeViewAdapter.notifyItemChanged(position);
                mSampleSelectedIndex = position;
                //mAutoFitView.setRenderMode(RENDERMODE_WHEN_DIRTY);

//                if (mRootView.getWidth() != mAutoFitView.getWidth()
//                        || mRootView.getHeight() != mAutoFitView.getHeight()) {
//                    mAutoFitView.setAspectRatio(mRootView.getWidth(), mRootView.getHeight());
//                }

                mVkRenderer.setParamsInt(SAMPLE_TYPE, position + SAMPLE_TYPE, 0);

                int sampleType = position + SAMPLE_TYPE;
                Bitmap tmp;
                switch (sampleType) {
                    case SAMPLE_TYPE_KEY_TRIANGLE:
                    case SAMPLE_TYPE_KEY_TEXTURE_MAPPING:
                    case SAMPLE_TYPE_KEY_PIPELINES:
                    case SAMPLE_TYPE_KEY_PUSH_CONSTANTS:
                    case SAMPLE_TYPE_KEY_SPECIALIZATION_INFO:
                    case SAMPLE_TYPE_KEY_CUBEMAP:
                    case SAMPLE_TYPE_KEY_INPUT_ATTACHMENTS:
                    case SAMPLE_TYPE_KEY_OFFSCREEN_RENDERING:
                    case SAMPLE_TYPE_KEY_DEPTH_TESTING:
                    case SAMPLE_TYPE_KEY_STENCIL_TESTING:
                    case SAMPLE_TYPE_KEY_MULTISAMPLING:
                    case SAMPLE_TYPE_KEY_MULTITHREADING:
                    case SAMPLE_TYPE_KEY_INSTANCING:
                    case SAMPLE_TYPE_KEY_READ_PIXELS:
                    case SAMPLE_TYPE_KEY_COMPUTE_SHADER:
                        break;
                    case SAMPLE_TYPE_KEY_RENDER_NV21:
                        loadNV21Image2();
                        break;
                    case SAMPLE_TYPE_KEY_RENDER_I420:
                        loadI420Image();
                        break;
                    case SAMPLE_TYPE_KEY_RENDER_YUYV:
                        loadYUYVImage();
                        //mAutoFitView.setAspectRatio(440, 310);
                        break;
                    case SAMPLE_TYPE_KEY_RENDER_I444:
                        loadI444Image();
                        //mAutoFitView.setAspectRatio(440, 310);
                        break;
                    case SAMPLE_TYPE_FBO:
                    {
                        Bitmap bitmap = loadRGBAImage(R.drawable.lye);
                        mAutoFitView.setAspectRatio(bitmap.getWidth(), bitmap.getHeight());
                    }
                        break;
                    case SAMPLE_TYPE_EGL:
                        startActivity(new Intent(MainActivity.this, EGLActivity.class));
                        break;
//                    case SAMPLE_TYPE_FBO_LEG:
//                        loadRGBAImage(R.drawable.leg);
//                        break;

                    case SAMPLE_TYPE_COORD_SYSTEM:
                        tmp = loadRGBAImage(R.drawable.board_texture);
                        mAutoFitView.setAspectRatio(tmp.getWidth(), tmp.getHeight());
                        break;
                    case SAMPLE_TYPE_BASIC_LIGHTING:
                    case SAMPLE_TYPE_TRANS_FEEDBACK:
                    //case SAMPLE_TYPE_MULTI_LIGHTS:
                    case SAMPLE_TYPE_INSTANCING3D:
                        loadRGBAImage(R.drawable.board_texture);
                        break;
                    case SAMPLE_TYPE_INSTANCING:
                        tmp = loadRGBAImage(R.drawable.front);
                        mAutoFitView.setAspectRatio(tmp.getWidth(), tmp.getHeight());
                        break;
                    case SAMPLE_TYPE_BLENDING:
                        loadRGBAImage(R.drawable.board_texture,0);
                        loadRGBAImage(R.drawable.floor,1);
                        loadRGBAImage(R.drawable.window,2);
                        break;
                    case SAMPLE_TYPE_PARTICLES:
                        tmp = loadRGBAImage(R.drawable.front);
                        mAutoFitView.setAspectRatio(tmp.getWidth(), tmp.getHeight());
                        //mAutoFitView.setRenderMode(RENDERMODE_CONTINUOUSLY);
                        break;
                    case SAMPLE_TYPE_SKYBOX:
                        loadRGBAImage(R.drawable.right,0);
                        loadRGBAImage(R.drawable.left,1);
                        loadRGBAImage(R.drawable.top,2);
                        loadRGBAImage(R.drawable.bottom,3);
                        loadRGBAImage(R.drawable.back,4);
                        loadRGBAImage(R.drawable.front,5);
                        break;
                    case SAMPLE_TYPE_FILTER_SPLITSCREEN:
                    case SAMPLE_TYPE_FILTER_COLOROFFSET:
                    case SAMPLE_TYPE_FILTER_ROTATE:
                        tmp = loadRGBAImage(R.drawable.lye4);
                        mAutoFitView.setAspectRatio(tmp.getWidth(), tmp.getHeight());
                        break;
                    case SAMPLE_TYPE_LUT:
                        loadRGBAImage(R.drawable.lut,0);
                        tmp = loadRGBAImage(R.drawable.lye4);
                        mAutoFitView.setAspectRatio(tmp.getWidth(), tmp.getHeight());
                        break;
                    case SAMPLE_TYPE_3D_MODEL:
                        break;
                    case SAMPLE_TYPE_PBO:
                        loadRGBAImage(R.drawable.front);
                        //mAutoFitView.setRenderMode(RENDERMODE_CONTINUOUSLY);
                        break;
                    case SAMPLE_TYPE_MRT:
                    case SAMPLE_TYPE_TBO:
                    case SAMPLE_TYPE_KEY_UBO:

                        Bitmap b4 = loadRGBAImage(R.drawable.lye);
                        mAutoFitView.setAspectRatio(b4.getWidth(), b4.getHeight());
                        break;
                    case SAMPLE_TYPE_COMPUTE_SHADER:
                        break;
                    default:
                        break;
                }

                if(mAutoFitView.getVisibility() == View.VISIBLE) {
                    //mAutoFitView.requestRender();
                }
//                if(sampleType != SAMPLE_TYPE_KEY_VISUALIZE_AUDIO && mAudioCollector != null) {
//                    mAudioCollector.unInit();
//                    mAudioCollector = null;
//                }

                dialog.cancel();
            }
        });

        LinearLayoutManager manager = new LinearLayoutManager(this);
        manager.setOrientation(LinearLayoutManager.VERTICAL);
        resolutionsListView.setLayoutManager(manager);

        resolutionsListView.setAdapter(myPreviewSizeViewAdapter);
        resolutionsListView.scrollToPosition(mSampleSelectedIndex);

        dialog.show();
        dialog.getWindow().setContentView(rootView);

    }

    private Bitmap loadRGBAImage(int resId) {
        InputStream is = this.getResources().openRawResource(resId);
        Bitmap bitmap;
        try {
            bitmap = BitmapFactory.decodeStream(is);
            if (bitmap != null) {
                int bytes = bitmap.getByteCount();
                ByteBuffer buf = ByteBuffer.allocate(bytes);
                bitmap.copyPixelsToBuffer(buf);
                byte[] byteArray = buf.array();
                mVkRenderer.setImageData(IMAGE_FORMAT_RGBA, bitmap.getWidth(), bitmap.getHeight(), byteArray);
            }
        }
        finally
        {
            try
            {
                is.close();
            }
            catch(IOException e)
            {
                e.printStackTrace();
            }
        }
        return bitmap;
    }

    private Bitmap loadRGBAImage(int resId, int index) {
        InputStream is = this.getResources().openRawResource(resId);
        Bitmap bitmap;
        try {
            bitmap = BitmapFactory.decodeStream(is);
            if (bitmap != null) {
                int bytes = bitmap.getByteCount();
                ByteBuffer buf = ByteBuffer.allocate(bytes);
                bitmap.copyPixelsToBuffer(buf);
                byte[] byteArray = buf.array();
                mVkRenderer.setImageDataWithIndex(index, IMAGE_FORMAT_RGBA, bitmap.getWidth(), bitmap.getHeight(), byteArray);
            }
        }
        finally
        {
            try
            {
                is.close();
            }
            catch(IOException e)
            {
                e.printStackTrace();
            }
        }
        return bitmap;
    }

    private void loadNV21Image() {
        InputStream is = null;
        try {
            is = getAssets().open("YUV_Image_840x1074.NV21");
        } catch (IOException e) {
            e.printStackTrace();
        }

        int lenght = 0;
        try {
            lenght = is.available();
            byte[] buffer = new byte[lenght];
            is.read(buffer);
            mVkRenderer.setImageData(IMAGE_FORMAT_NV21, 840, 1074, buffer);
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try
            {
                is.close();
            }
            catch(IOException e)
            {
                e.printStackTrace();
            }
        }

    }

    private void loadNV21Image2() {
        InputStream is = null;
        try {
            is = getAssets().open("yuv/IMAGE_4406x3108.NV21");
        } catch (IOException e) {
            e.printStackTrace();
        }

        int lenght = 0;
        try {
            lenght = is.available();
            byte[] buffer = new byte[lenght];
            is.read(buffer);
            mVkRenderer.setImageData(IMAGE_FORMAT_NV21, 4406, 3108, buffer);
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try
            {
                is.close();
            }
            catch(IOException e)
            {
                e.printStackTrace();
            }
        }
    }

    private void loadI420Image() {
        InputStream is = null;
        try {
            is = getAssets().open("yuv/IMAGE_5288x3732.I420");
        } catch (IOException e) {
            e.printStackTrace();
        }

        int lenght = 0;
        try {
            lenght = is.available();
            byte[] buffer = new byte[lenght];
            is.read(buffer);
            mVkRenderer.setImageData(IMAGE_FORMAT_I420, 5288, 3732, buffer);
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try
            {
                is.close();
            }
            catch(IOException e)
            {
                e.printStackTrace();
            }
        }
    }

    private void loadI444Image() {
        InputStream is = null;
        try {
            is = getAssets().open("yuv/IMAGE_5288x3732.I444");
        } catch (IOException e) {
            e.printStackTrace();
        }

        int lenght = 0;
        try {
            lenght = is.available();
            byte[] buffer = new byte[lenght];
            is.read(buffer);
            mVkRenderer.setImageData(IMAGE_FORMAT_I444, 5288, 3732, buffer);
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try
            {
                is.close();
            }
            catch(IOException e)
            {
                e.printStackTrace();
            }
        }
    }

    private void loadYUYVImage() {
        InputStream is = null;
        try {
            is = getAssets().open("yuv/IMAGE_5288x3732.YUYV");
        } catch (IOException e) {
            e.printStackTrace();
        }

        int lenght = 0;
        try {
            lenght = is.available();
            byte[] buffer = new byte[lenght];
            is.read(buffer);
            mVkRenderer.setImageData(IMAGE_FORMAT_YUYV, 5288, 3732, buffer);
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try
            {
                is.close();
            }
            catch(IOException e)
            {
                e.printStackTrace();
            }
        }
    }

    private void loadGrayImage() {
        InputStream is = null;
        try {
            is = getAssets().open("lye_1280x800.Gray");
        } catch (IOException e) {
            e.printStackTrace();
        }

        int lenght = 0;
        try {
            lenght = is.available();
            byte[] buffer = new byte[lenght];
            is.read(buffer);
            mVkRenderer.setImageDataWithIndex(0, IMAGE_FORMAT_GARY, 1280, 800, buffer);
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try
            {
                is.close();
            }
            catch(IOException e)
            {
                e.printStackTrace();
            }
        }

    }

    protected boolean hasPermissionsGranted(String[] permissions) {
        for (String permission : permissions) {
            if (ActivityCompat.checkSelfPermission(this, permission)
                    != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

}
