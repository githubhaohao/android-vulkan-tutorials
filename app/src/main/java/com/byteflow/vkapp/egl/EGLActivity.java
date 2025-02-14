package com.byteflow.vkapp.egl;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.opengl.GLES20;
import android.opengl.GLException;
import android.os.Bundle;
import androidx.annotation.*;
import androidx.appcompat.app.AppCompatActivity;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.ImageView;

import com.byteflow.vkapp.R;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;


public class EGLActivity extends AppCompatActivity {
    private static final String TAG = "EGLActivity";
    public static final int PARAM_TYPE_SHADER_INDEX = 200;
    private ImageView mImageView;
    private NativeEglRender mNativeEglRender;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_egl);

        mImageView = (ImageView) findViewById(R.id.imageView);
        mNativeEglRender = new NativeEglRender();
        mNativeEglRender.native_EglRenderInit();

    }

    @Override
    protected void onResume() {
        super.onResume();
    }

    @Override
    protected void onPause() {
        super.onPause();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        mNativeEglRender.native_EglRenderUnInit();
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_egl, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        int id = item.getItemId();

        int shaderIndex = 0;
        switch (id) {
            case R.id.action_shader0:
                shaderIndex = 0;
                break;
            case R.id.action_shader1:
                shaderIndex = 1;
                break;

            case R.id.action_shader2:
                shaderIndex = 2;
                break;
            case R.id.action_shader3:
                shaderIndex = 3;
                break;
            case R.id.action_shader4:
                shaderIndex = 4;
                break;
            case R.id.action_shader5:
                shaderIndex = 5;
                break;
            case R.id.action_shader6:
                shaderIndex = 6;
                break;
                default:
        }

        if (mNativeEglRender != null) {
            mNativeEglRender.native_EglRenderSetIntParams(PARAM_TYPE_SHADER_INDEX, shaderIndex);
            requestDraw();
        }
        return true;
    }
    private void requestDraw() {
        Bitmap bitmap = loadRGBAImage(R.drawable.lye4, mNativeEglRender);
        mNativeEglRender.native_EglRenderDraw();
        mImageView.setImageBitmap(createBitmapFromGLSurface(0, 0, bitmap.getWidth(), bitmap.getHeight()));
    }

    private Bitmap loadRGBAImage(int resId, NativeEglRender render) {
        InputStream is = this.getResources().openRawResource(resId);
        Bitmap bitmap;
        try {
            bitmap = BitmapFactory.decodeStream(is);
            if (bitmap != null) {
                int bytes = bitmap.getByteCount();
                ByteBuffer buf = ByteBuffer.allocate(bytes);
                bitmap.copyPixelsToBuffer(buf);
                byte[] byteArray = buf.array();
                render.native_EglRenderSetImageData(byteArray, bitmap.getWidth(), bitmap.getHeight());
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

    private Bitmap createBitmapFromGLSurface(int x, int y, int w, int h) {
        int bitmapBuffer[] = new int[w * h];
        int bitmapSource[] = new int[w * h];
        IntBuffer intBuffer = IntBuffer.wrap(bitmapBuffer);
        intBuffer.position(0);
        try {
            GLES20.glReadPixels(x, y, w, h, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE,
                    intBuffer);
            int offset1, offset2;
            for (int i = 0; i < h; i++) {
                offset1 = i * w;
                offset2 = (h - i - 1) * w;
                for (int j = 0; j < w; j++) {
                    int texturePixel = bitmapBuffer[offset1 + j];
                    int blue = (texturePixel >> 16) & 0xff;
                    int red = (texturePixel << 16) & 0x00ff0000;
                    int pixel = (texturePixel & 0xff00ff00) | red | blue;
                    bitmapSource[offset2 + j] = pixel;
                }
            }
        } catch (GLException e) {
            return null;
        }
        return Bitmap.createBitmap(bitmapSource, w, h, Bitmap.Config.ARGB_8888);
    }
}
