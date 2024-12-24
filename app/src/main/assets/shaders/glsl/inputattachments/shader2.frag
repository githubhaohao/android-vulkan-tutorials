#version 450
layout (input_attachment_index = 0, binding = 0) uniform subpassInput inputColor;
layout (input_attachment_index = 1, binding = 1) uniform subpassInput inputDepth;

layout (location = 0) in vec2 inUV;
layout (location = 1) in float iTime;
layout (location = 0) out vec4 outFragColor;

void main() 
{
	vec4 color = subpassLoad(inputColor);
	vec3 lightColor = 0.5 + 0.5*cos(iTime+inUV.xyx+vec3(0.0,2.0,4.0));
	outFragColor = color * vec4(lightColor, 1.0);
}