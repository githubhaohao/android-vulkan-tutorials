#version 450
layout (binding = 2) uniform UBO
{
	mat4 projectionMatrix;
	mat4 modelMatrix;
	mat4 viewMatrix;
	float iTime;
} ubo;

layout (location = 0) out vec2 outUV;
layout (location = 1) out float iTime;

out gl_PerVertex 
{
    vec4 gl_Position;   
};

void main() 
{
	iTime = ubo.iTime;
	// 根据顶点索引计算纹理坐标
	// gl_VertexIndex 的值依次为 0, 1, 2
	outUV = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);

	// 将纹理坐标转换为标准化设备坐标
	// outUV 的值依次为 (0, 0), (2, 0), (0, 2)
	gl_Position = vec4(outUV * 2.0 - 1.0, 0.0, 1.0);
}
