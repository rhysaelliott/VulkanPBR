#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require

#include "input_structures.glsl"

layout(location =0) out vec3 outNormal;
layout(location =1) out vec3 outColor;
layout(location = 2) out vec2 outUV;
layout(location = 3) out vec3 outPos;
layout(location = 4) out vec4 outFragPosLightSpace;

struct Vertex
{
	vec3 position;
	float uv_x;
	vec3 normal;
	float uv_y;
	vec4 color;
};

layout(buffer_reference, std430) readonly buffer VertexBuffer
{
	Vertex vertices[];
};

layout(push_constant) uniform constants
{
	mat4 render_matrix;
	VertexBuffer vertexBuffer;
} PushConstants;

const mat4 biasMat = mat4( 
	0.5, 0.0, 0.0, 0.0,
	0.0, 0.5, 0.0, 0.0,
	0.0, 0.0, 1.0, 0.0,
	0.5, 0.5, 0.0, 1.0 );

void main()
{
	Vertex v = PushConstants.vertexBuffer.vertices[gl_VertexIndex];
	
	vec4 position = vec4(v.position,1.0f);

	outPos= (PushConstants.render_matrix*position).xyz;

	outNormal =	(PushConstants.render_matrix * vec4(v.normal,0.f)).xyz;
	outColor = v.color.xyz * materialData.colorFactors.xyz;
	outUV.x = v.uv_x;
	outUV.y = v.uv_y;

  	outFragPosLightSpace = ( lightData.lights[0].viewproj * PushConstants.render_matrix) * vec4(v.position, 1.0);

  	gl_Position = sceneData.viewproj * PushConstants.render_matrix * position;
}