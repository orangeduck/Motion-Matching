#pragma once

#include "common.h"

struct vec2
{
    vec2() : x(0.0f), y(0.0f) {}
    vec2(float _x, float _y) : x(_x), y(_y) {}
    
    float x, y;
};

static inline vec2 operator+(float s, vec2 v)
{
    return vec2(v.x + s, v.y + s);
}

static inline vec2 operator+(vec2 v, float s)
{
    return vec2(v.x + s, v.y + s);
}

static inline vec2 operator+(vec2 v, vec2 w)
{
    return vec2(v.x + w.x, v.y + w.y);
}

static inline vec2 operator-(float s, vec2 v)
{
    return vec2(s - v.x, s - v.y);
}

static inline vec2 operator-(vec2 v, float s)
{
    return vec2(v.x - s, v.y - s);
}

static inline vec2 operator-(vec2 v, vec2 w)
{
    return vec2(v.x - w.x, v.y - w.y);
}

static inline vec2 operator*(float s, vec2 v)
{
    return vec2(v.x * s, v.y * s);
}

static inline vec2 operator*(vec2 v, float s)
{
    return vec2(v.x * s, v.y * s);
}

static inline vec2 operator*(vec2 v, vec2 w)
{
    return vec2(v.x * w.x, v.y * w.y);
}

static inline vec2 operator/(vec2 v, float s)
{
    return vec2(v.x / s, v.y / s);
}

static inline vec2 operator/(float s, vec2 v)
{
    return vec2(s / v.x, s / v.y);
}

static inline vec2 operator/(vec2 v, vec2 w)
{
    return vec2(v.x / w.x, v.y / w.y);
}

static inline vec2 operator-(vec2 v)
{
    return vec2(-v.x, -v.y);
}

static inline float dot(vec2 v, vec2 w)
{
    return v.x*w.x + v.y*w.y;
}

static inline float length(vec2 v)
{
    return sqrtf(v.x*v.x + v.y*v.y);
}

static inline vec2 normalize(vec2 v, float eps=1e-8f)
{
    return v / (length(v) + eps);
}

static inline vec2 lerp(vec2 v, vec2 w, float alpha)
{
    return v * (1.0f - alpha) + w * alpha;
}

//--------------------------------------

struct vec3
{
    vec3() : x(0.0f), y(0.0f), z(0.0f) {}
    vec3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}
    
    float x, y, z;
};

static inline vec3 operator+(float s, vec3 v)
{
    return vec3(v.x + s, v.y + s, v.z + s);
}

static inline vec3 operator+(vec3 v, float s)
{
    return vec3(v.x + s, v.y + s, v.z + s);
}

static inline vec3 operator+(vec3 v, vec3 w)
{
    return vec3(v.x + w.x, v.y + w.y, v.z + w.z);
}

static inline vec3 operator-(float s, vec3 v)
{
    return vec3(s - v.x, s - v.y, s - v.z);
}

static inline vec3 operator-(vec3 v, float s)
{
    return vec3(v.x - s, v.y - s, v.z - s);
}

static inline vec3 operator-(vec3 v, vec3 w)
{
    return vec3(v.x - w.x, v.y - w.y, v.z - w.z);
}

static inline vec3 operator*(float s, vec3 v)
{
    return vec3(v.x * s, v.y * s, v.z * s);
}

static inline vec3 operator*(vec3 v, float s)
{
    return vec3(v.x * s, v.y * s, v.z * s);
}

static inline vec3 operator*(vec3 v, vec3 w)
{
    return vec3(v.x * w.x, v.y * w.y, v.z * w.z);
}

static inline vec3 operator/(vec3 v, float s)
{
    return vec3(v.x / s, v.y / s, v.z / s);
}

static inline vec3 operator/(float s, vec3 v)
{
    return vec3(s / v.x, s / v.y, s / v.z);
}

static inline vec3 operator/(vec3 v, vec3 w)
{
    return vec3(v.x / w.x, v.y / w.y, v.z / w.z);
}

static inline vec3 operator-(vec3 v)
{
    return vec3(-v.x, -v.y, -v.z);
}

static inline float dot(vec3 v, vec3 w)
{
    return v.x*w.x + v.y*w.y + v.z*w.z;
}

static inline vec3 cross(vec3 v, vec3 w)
{
    return vec3(
        v.y*w.z - v.z*w.y,
        v.z*w.x - v.x*w.z,
        v.x*w.y - v.y*w.x);
}

static inline float length(vec3 v)
{
    return sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
}

static inline vec3 normalize(vec3 v, float eps=1e-8f)
{
    return v / (length(v) + eps);
}

static inline vec3 lerp(vec3 v, vec3 w, float alpha)
{
    return v * (1.0f - alpha) + w * alpha;
}

static inline vec3 clamp(vec3 v, vec3 min, vec3 max)
{
    return vec3(
        clampf(v.x, min.x, max.x),
        clampf(v.y, min.y, max.y),
        clampf(v.z, min.z, max.z));
}
