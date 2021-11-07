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
    vec3 light_dir = normalize(vec3(0.25, -0.8, 0.1));

    float half_lambert = (dot(-light_dir, fragNormal) + 1.0) / 2.0;

    finalColor = vec4(half_lambert * colDiffuse.xyz + 0.1, 1.0);
}
