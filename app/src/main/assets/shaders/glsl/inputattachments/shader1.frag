#version 450
layout (binding = 1) uniform sampler2D samplerColor;
layout (location = 0) in vec2 inUV;
layout (location = 1) in float iTime;
layout (location = 0) out vec4 outFragColor;

void main() 
{
	vec2 uv = inUV;
	float splitNum = floor(mod(iTime, 6.0)) + 1.0;
	uv *= splitNum;
	uv = fract(uv);
	outFragColor = texture(samplerColor, uv, 0.0);
}