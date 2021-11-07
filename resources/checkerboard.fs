#version 300 es
precision mediump float;

in vec3 fragPosition;
in vec2 fragTexCoord;
in vec4 fragColor;
in vec3 fragNormal;

uniform vec4 colDiffuse;

out vec4 finalColor;

void main()
{
    float total = floor(fragPosition.x * 2.0f) +
                  floor(fragPosition.z * 2.0f);
                  
    finalColor = mod(total, 2.0f) == 0.0f ? 
        vec4(0.8f, 0.8f, 0.8f, 1.0f) : 
        vec4(0.85f, 0.85f, 0.85f, 1.0f);
}
