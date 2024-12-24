#version 450
layout (location = 0) in vec3 inPos;
layout (location = 1) in vec2 inUV;
layout (location = 2) in vec3 inNormal;

layout (binding = 0) uniform UBO 
{
	mat4 mvpMatrix;
} ubo;

layout (location = 0) out vec2 outUV;

layout(push_constant) uniform PushConsts {
	vec4 color;
	vec4 position;
} pushConsts;

out gl_PerVertex 
{
    vec4 gl_Position;   
};

void main() 
{
	outUV = inUV;
	vec4 worldPos = vec4(inPos, 1.0) + pushConsts.position;
	gl_Position =  ubo.mvpMatrix * worldPos;
}
