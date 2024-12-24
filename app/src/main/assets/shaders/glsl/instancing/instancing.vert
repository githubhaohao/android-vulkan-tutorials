#version 450
layout (location = 0) in vec3 inPos;
layout (location = 1) in vec2 inUV;
layout (location = 2) in vec3 inNormal;

// Instanced attributes
layout (location = 3) in vec3 instancePos;
layout (location = 4) in vec3 instanceRot;
layout (location = 5) in float instanceScale;
layout (location = 6) in vec3 instanceCol;

layout (location = 0) out vec2 outUV;
layout (location = 1) out vec3 outCol;

layout (binding = 0) uniform UBO
{
	mat4 mvp;
} ubo;

out gl_PerVertex 
{
    vec4 gl_Position;   
};

void main() 
{
	outUV = inUV;
	outCol = instanceCol;

	mat3 mx, my, mz;

	// rotate around x
	float s = sin(instanceRot.x);
	float c = cos(instanceRot.x);

	mx[0] = vec3(c, s, 0.0);
	mx[1] = vec3(-s, c, 0.0);
	mx[2] = vec3(0.0, 0.0, 1.0);

	// rotate around y
	s = sin(instanceRot.y);
	c = cos(instanceRot.y);

	my[0] = vec3(c, 0.0, s);
	my[1] = vec3(0.0, 1.0, 0.0);
	my[2] = vec3(-s, 0.0, c);

	// rot around z
	s = sin(instanceRot.z);
	c = cos(instanceRot.z);

	mz[0] = vec3(1.0, 0.0, 0.0);
	mz[1] = vec3(0.0, c, s);
	mz[2] = vec3(0.0, -s, c);

	mat3 rotMat = mz * my * mx;

	vec4 locPos = vec4(inPos.xyz * rotMat, 1.0);
	vec4 pos = vec4((locPos.xyz * instanceScale) + instancePos, 1.0);
	gl_Position =  ubo.mvp * pos;
}
