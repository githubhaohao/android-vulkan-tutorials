#version 450
layout (binding = 1) uniform sampler2D samplerColor;
layout (location = 0) in vec2 inUV;
layout (location = 1) in float iTime;
layout (location = 0) out vec4 outFragColor;

void main() 
{
	vec3 lightColor = 0.5 + 0.5*cos(iTime+inUV.xyx+vec3(0.0,2.0,4.0));
	vec2 uv = inUV;
	uv *= 2.0;
	vec2 iuv = floor(uv);
	uv = fract(uv);
	vec4 resultCol = texture(samplerColor, uv, 0.0);
	float d = mod((iuv.x + iuv.y), 2.0);
	outFragColor = resultCol * (d > 0.0 ?  vec4(lightColor, 1.0) : vec4(1.0));
}