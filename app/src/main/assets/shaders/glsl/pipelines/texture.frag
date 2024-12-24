#version 450
layout (binding = 1) uniform sampler2D samplerColor;
layout (location = 0) in vec2 inUV;
layout (location = 1) in float iTime;
layout (location = 0) out vec4 outFragColor;

void main() 
{
	vec3 lightColor = 0.5 + 0.5*cos(iTime+inUV.xyx+vec3(0.0,2.0,4.0));
	outFragColor = texture(samplerColor, inUV, 0.0) * vec4(lightColor, 1.0);
}