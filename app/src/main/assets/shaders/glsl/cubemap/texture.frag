#version 450
layout (binding = 1) uniform samplerCube samplerCubeMap;
layout (location = 0) in vec3 texCoord;
layout (location = 0) out vec4 outFragColor;

void main() 
{
	outFragColor = texture(samplerCubeMap, texCoord);
}