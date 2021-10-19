#version 330

in vec3 fragPosition;
in vec2 fragTexCoord;
in vec4 fragColor;
in vec3 fragNormal;

uniform vec4 colDiffuse;

out vec4 finalColor;

void main()
{
    float total = floor(fragPosition.x * 2.0) +
                  floor(fragPosition.z * 2.0);
                  
    finalColor = mod(total, 2.0) == 0.0 ? 
        vec4(0.8, 0.8, 0.8, 1.0) : 
        vec4(0.85, 0.85, 0.85, 1.0);
}
