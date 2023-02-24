#version 300 es
precision mediump float;

in vec3 fragPosition;
in vec2 fragTexCoord;
in vec4 fragColor;
in vec3 fragNormal;

uniform vec4 colDiffuse;

out vec4 finalColor;

float dither_threshold(float x, float y) {
    
    const int indexMatrix8x8[64] =  int[64](
		0,  32, 8,  40, 2,  34, 10, 42,
		48, 16, 56, 24, 50, 18, 58, 26,
		12, 44, 4,  36, 14, 46, 6,  38,
		60, 28, 52, 20, 62, 30, 54, 22,
		3,  35, 11, 43, 1,  33, 9,  41,
		51, 19, 59, 27, 49, 17, 57, 25,
		15, 47, 7,  39, 13, 45, 5,  37,
		63, 31, 55, 23, 61, 29, 53, 21);
    
    int ix = int(mod(x, 8.0f));
    int iy = int(mod(y, 8.0f));
    return float(indexMatrix8x8[(ix + iy * 8)]) / 64.0f;
}


void main()
{
    if (colDiffuse.a < dither_threshold(gl_FragCoord.x, gl_FragCoord.y)) {
        discard;
    }

    vec3 light_dir = normalize(vec3(0.25, -0.8, 0.1));

    float half_lambert = (dot(-light_dir, fragNormal) + 1.0) / 2.0;

    finalColor = vec4(half_lambert * colDiffuse.xyz + 0.1, 1.0);
}
