#version 450
layout (binding = 1) uniform sampler2D samplerColor;
layout (location = 0) in vec2 inUV;
layout (location = 1) in float iTime;
layout (location = 0) out vec4 outFragColor;

// 设置是否叠加颜色
layout (constant_id = 0) const int MODEL = 0;
// 设置分屏数量
layout (constant_id = 1) const float SPLIT = 4.0;

void main() 
{
	vec2 uv = inUV;
	uv *= SPLIT;
	uv = fract(uv);
	if(MODEL == 0) {
		outFragColor = texture(samplerColor, uv, 0.0) ;
	} else {
		vec3 lightColor = 0.5 + 0.5*cos(iTime+uv.xyx+vec3(0.0,2.0,4.0));
		outFragColor = texture(samplerColor, uv, 0.0) * vec4(lightColor, 1.0);
	}
}