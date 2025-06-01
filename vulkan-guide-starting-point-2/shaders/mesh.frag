#version 450
#extension GL_GOOGLE_include_directive : require
#include "input_structures.glsl"
#define PI 3.1415926
//10000.f, 0.1f
	const float near_plane = 1.0;
	const float far_plane = 96.0;

layout (location =0) in vec3 inNormal;
layout(location=1) in vec3 inColor;
layout(location =2) in vec2 inUV;
layout(location =3) in vec3 inPos;
layout(location = 4) in vec4 inFragPosLightSpace;


layout(location=0) out vec4 outFragColor;


float saturate(in float num)
{
	return clamp(num, 0.0,1.0);
}

float phongDiffuse()
{
	return (1.0/PI);
}

float distributuionGGX(vec3 N, vec3 H, float roughness)
{
	float a = roughness*roughness;
	float a2 = a*a;
	float NdotH = max(dot(N,H), 0.0);
	float NdotH2 = NdotH * NdotH;

	float denom = (NdotH2 * (a2 - 1.0) +1.0);
	denom = PI* denom*denom;

	return a2/denom;
}

float geometrySchlickGGX(float NdotV, float roughness)
{
	float r = (roughness +1.0);
	float k = (r*r)/8.0;

	float denom = NdotV * (1.0-k) + k;

	return NdotV / denom;
}

float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
	float NdotV = max(dot(N,V), 0.0);
	float NdotL = max(dot(N,L), 0.0);
	float ggx1 = geometrySchlickGGX(NdotV, roughness);
	float ggx2 = geometrySchlickGGX(NdotL, roughness);

	return ggx1*ggx2;
}
float LinearizeDepth(float depth)
{
    float z = depth; // Back to NDC 
    return (2.0 * near_plane) / (far_plane + near_plane - z * (far_plane - near_plane));
}
float shadowCalculation(vec4 fragPosLightSpace, vec3 lightDir)
{
    // perform perspective divide
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    // transform to [0,1] range
    projCoords.xy = projCoords.xy * 0.5 + 0.5;
    // get closest depth value from light's perspective (using [0,1] range fragPosLight as coords)
    float closestDepth = LinearizeDepth(texture(shadowTex, projCoords.xy).r)/far_plane; 
    // get depth of current fragment from light's perspective
    float currentDepth = LinearizeDepth(projCoords.z)/far_plane;
    // check whether current frag pos is in shadow
    float bias = 0.000000001;  
    float shadow = currentDepth - bias > closestDepth  ? 1.0 : 0.0;  


    return shadow;
}


void main()
{
	float lightValue = max(dot(inNormal, sceneData.sunlightDirection.xyz), 0.1f);

	vec3 base = texture(colorTex, inUV).xyz;
	vec3 color = inColor * base;
	vec3 ambient = color * sceneData.ambientColor.xyz;




	vec3 lightColor = vec3(0.0);
	for (int i = 0; i < 1; i++) 
	{
    LightStruct light = lightData.lights[i];

    float distance = length(light.position - inPos);
    if (light.range >= distance) {

        float A = light.constant + light.linear * distance + light.quadratic * (distance * distance);
        A = max(A, 0.001);
        A = 1.0 / A;

        vec3 localLightPos = vec3(sceneData.view * vec4(light.position, 1.0)).xyz;
        vec3 L = normalize(localLightPos - inPos);
        vec3 V = normalize(-inPos);
        vec3 H = normalize(L + V);
        vec3 nn = normalize(inNormal);

        float metal = texture(metalRoughTex, inUV).x;
        float rough = texture(metalRoughTex, inUV).y;

        vec3 baseSpec = mix(vec3(0.04), base, metal);
        vec3 F0 = baseSpec;
        vec3 F = F0 + (1.0 - F0) * pow(1.0 - dot(H, V), 5.0);

        float NDF = distributuionGGX(nn, H, rough);
        float G = geometrySmith(nn, V, L, rough);
        vec3 numerator = NDF * G * F;
        float denom = 4.0 * max(dot(nn, V), 0.0) * max(dot(nn, L), 0.0);
        vec3 spec = numerator / max(denom, 0.001);

        vec3 kD = vec3(1.0) - F;
        kD *= 1.0 - metal;

        vec3 diffuse = kD * (color / PI);
        vec3 radiance = light.color * A * light.intensity;

        float NdotL = max(dot(nn, L), 0.0);
        vec3 contribution = (diffuse + spec) * radiance * NdotL;

        
		//outFragColor = vec4(shadow, 0,0,1);
		//return;
        if (light.lightType == 1) {
            vec3 Ldir = normalize(light.position - inPos);
            float theta = dot(normalize(-Ldir), normalize(light.direction));
            float epsilon = 0.1;
            float intensity = clamp((theta - cos(radians(light.cone))) / epsilon, 0.0, 1.0);
            contribution *= intensity;
			//float shadow = filterPCF(inFragPosLightSpace / inFragPosLightSpace.w);
            //float shadow = textureProj(inFragPosLightSpace / inFragPosLightSpace.w, vec2(0.0));
            float shadow = shadowCalculation(inFragPosLightSpace, light.direction);
            contribution *= (1.0 - shadow);
        }

        lightColor += contribution;
    }
}

	

outFragColor = vec4(color * lightValue * sceneData.sunlightColor.w + ambient + lightColor, 1.0f);
// outFragColor = vec4(vec3(shadowCalculation(inFragPosLightSpace)), 1.0);
// outFragColor = vec4(inFragPosLightSpace.xyz, 1.0);
    // vec3 projCoords = inFragPosLightSpace.xyz / inFragPosLightSpace.w;
    // // transform to [0,1] range
    // projCoords = projCoords * 0.5 + 0.5;
    // outFragColor = vec4(projCoords, 1.0);
    //     vec3 projCoords = inFragPosLightSpace.xyz / inFragPosLightSpace.w;
    // // transform to [0,1] range
    // projCoords = projCoords * 0.5 + 0.5;
    // // get closest depth value from light's perspective (using [0,1] range fragPosLight as coords)
    // float closestDepth = texture(shadowTex, projCoords.xy).r; 
    // outFragColor = vec4(vec3(1.0-LinearizeDepth(closestDepth)), 1.0);
}
