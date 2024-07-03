#version 450
#extension GL_GOOGLE_include_directive : require
#include "input_structures.glsl"

layout (location =0) in vec3 inNormal;
layout(location=1) in vec3 inColor;
layout(location =2) in vec2 inUV;
layout(location =3) in vec3 inPos;


layout(location=0) out vec4 outFragColor;


float saturate(in float num)
{
	return clamp(num, 0.0,1.0);
}

void main()
{
	float lightValue = max(dot(inNormal, sceneData.sunlightDirection.xyz), 0.1f);

	vec3 color = inColor * texture(colorTex, inUV).xyz;
	vec3 ambient = color * sceneData.ambientColor.xyz;

	//get light position in local space

	//calculate light attenuation

	//calculate Light direction, View direction and Halfway vector vectors

	//calculate normalised normal and tbn 

	//calculate N

	//get roughness and metallic

	//mix between base and metallic for constant base specular value (0.04)

	//do cubemap stuff?

	//compute material reflectance (frensel-schlik)

	//do cook-torrance fresnel

	//do diffuse component (lambertian)

	//do diffuse 

	//calculate radiance

	//accumulate light contribution


	vec3 lightColor = vec3(0.0);
	for(int i =0; i<lightData.numLights;i++)
	{
		if(lightData.lights[i].range>= length(lightData.lights[i].position - inPos))
		{


			//multiply spec by specref and dif by difref
			lightColor += lightData.lights[i].color;	
		}
	}



	outFragColor =vec4(color*lightValue*sceneData.sunlightColor.w + ambient + lightColor, 1.0f);
}
