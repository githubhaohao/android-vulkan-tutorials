#version 450
layout (binding = 1) uniform sampler2D samplerColor;
layout (location = 0) in vec2 inUV;
layout (location = 1) in vec3 inCol;
layout (location = 0) out vec4 outFragColor;

void main() 
{
	vec4 resultCol = texture(samplerColor, inUV, 0.0) * vec4(inCol, 1.0);
	outFragColor = resultCol;
}