#version 450
layout (location = 0) in vec3 inPos;
layout (location = 1) in vec2 inUV;
layout (location = 2) in vec3 inNormal;

layout (binding = 0) uniform UBO
{
	mat4 projectionMatrix;
	mat4 modelMatrix;
	mat4 viewMatrix;
	float iTime;
} ubo;


out gl_PerVertex
{
	vec4 gl_Position;
};

void main() 
{
	vec4 pos = vec4(inPos.xyz * 1.2, 1.0);
	gl_Position = ubo.projectionMatrix * ubo.viewMatrix * ubo.modelMatrix * pos;
}
