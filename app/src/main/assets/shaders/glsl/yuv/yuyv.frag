#version 450
layout (binding = 1) uniform sampler2D samplerYPlane;
layout (binding = 2) uniform sampler2D samplerUVPlane;
layout (location = 0) in vec2 inUV;
layout (location = 0) out vec4 outFragColor;

void main()
{
	float yCol = texture(samplerYPlane, inUV, 0.0).r - 0.063;
	vec4 yuyv = texture(samplerUVPlane, inUV, 0.0);
	float uCol = yuyv.g - 0.502;
	float vCol = yuyv.a - 0.502;

	vec3 rgb = mat3(1.164, 1.164, 1.164,
	0, 		 -0.392, 	2.017,
	1.596,   -0.813,    0.0) * vec3(yCol, uCol, vCol);

	outFragColor = vec4(rgb, 1.0);
}