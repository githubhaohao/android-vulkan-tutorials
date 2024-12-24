#version 450
layout (binding = 1) uniform sampler2D samplerColor;
layout (location = 0) in vec2 inUV;
layout (location = 1) in float iTime;
layout (location = 0) out vec4 outFragColor;

void main() 
{
	float offset = 1.0 / 100.0;
	vec4 color = texture(samplerColor, inUV, 0.0);
	vec4 color0 = texture(samplerColor, inUV + offset, 0.0);
	vec4 color1 = texture(samplerColor, inUV - offset, 0.0);
	outFragColor = vec4(color0.r, color.g, color1.b, color.a);
}