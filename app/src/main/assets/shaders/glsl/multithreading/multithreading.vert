#version 450
layout (location = 0) in vec3 inPos;
layout (location = 1) in vec2 inUV;
layout (location = 2) in vec3 inNormal;


layout (location = 0) out vec2 outUV;

layout (std140, push_constant) uniform PushConsts
{
	mat4 mvp;
	vec3 color;
} pushConsts;

out gl_PerVertex 
{
    vec4 gl_Position;   
};

void main() 
{
	outUV = inUV;
	vec4 worldPos = vec4(inPos, 1.0);
	gl_Position =  pushConsts.mvp * worldPos;
}
