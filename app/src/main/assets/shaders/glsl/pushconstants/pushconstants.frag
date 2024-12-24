#version 450
layout (binding = 1) uniform sampler2D samplerColor;
layout (location = 0) in vec2 inUV;
layout (location = 0) out vec4 outFragColor;

layout(push_constant) uniform PushConsts {
	vec4 color;
	vec4 position;
} pushConsts;

void main() 
{
	vec4 resultCol = texture(samplerColor, inUV, 0.0) * pushConsts.color;
	outFragColor = resultCol;
}