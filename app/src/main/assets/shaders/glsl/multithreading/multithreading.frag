#version 450
layout (binding = 0) uniform sampler2D samplerColor;
layout (location = 0) in vec2 inUV;
layout (location = 0) out vec4 outFragColor;

layout (std140, push_constant) uniform PushConsts
{
	mat4 mvp;
	vec3 color;
} pushConsts;

void main() 
{
	vec4 resultCol = texture(samplerColor, inUV, 0.0) * vec4(pushConsts.color, 1.0);
	outFragColor = resultCol;
}