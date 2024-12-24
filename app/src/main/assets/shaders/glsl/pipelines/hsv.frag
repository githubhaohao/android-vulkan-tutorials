#version 450
layout (binding = 1) uniform sampler2D samplerColor;
layout (location = 0) in vec2 inUV;
layout (location = 1) in float iTime;
layout (location = 0) out vec4 outFragColor;

vec3 rgb2hsv(vec3 c,float time) {
	vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
	vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
	vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
	float d = q.x - min(q.w, q.y);
	float e = 1.0e-10;
	vec3 hsvCol = vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
	hsvCol.z = 3.14159*2.0* fract(time*0.05)-hsvCol.z;;
	return hsvCol;
}

vec3 hsv2rgb(vec3 c,float time)
{
	vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
	vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
	return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() 
{
	vec4 orgCol = texture(samplerColor, inUV, 0.0);
	vec3 hsvCol = rgb2hsv(orgCol.rgb,iTime);
	hsvCol.x = hsvCol.z;
	hsvCol.y = 1.0;
	hsvCol.z = 1.0;
	vec3 resultCol = hsv2rgb(hsvCol,iTime);
	outFragColor = vec4(resultCol, orgCol.a);
}