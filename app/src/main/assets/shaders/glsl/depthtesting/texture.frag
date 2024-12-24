#version 450
layout (binding = 1) uniform sampler2D samplerColor;
layout (location = 0) in vec2 inUV;
layout (location = 1) in float iTime;
layout (location = 0) out vec4 outFragColor;

// 设置是否显示深度值
layout (constant_id = 0) const int MODEL = 0;

void main() 
{
	if(MODEL == 0) {
		outFragColor = texture(samplerColor, inUV, 0.0);
	} else {
		// 获取当前片段的深度值
		float depth = gl_FragCoord.z;

		// 将深度值作为灰度值输出（0.0 到 1.0）
		outFragColor = vec4(vec3(depth), depth);
	}
}