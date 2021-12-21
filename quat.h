#pragma once

#include "vec.h"

struct quat
{
    quat() : w(1.0f), x(0.0f), y(0.0f), z(0.0f) {}
    quat(float _w, float _x, float _y, float _z) : w(_w), x(_x), y(_y), z(_z) {}

    float w, x, y, z;
};

static inline quat operator*(quat q, float s)
{
    return quat(q.w * s, q.x * s, q.y * s, q.z * s);
}

static inline quat operator*(float s, quat q)
{
    return quat(q.w * s, q.x * s, q.y * s, q.z * s);
}

static inline quat operator+(quat q, quat p)
{
    return quat(q.w + p.w, q.x + p.x, q.y + p.y, q.z + p.z);
}

static inline quat operator-(quat q, quat p)
{
    return quat(q.w - p.w, q.x - p.x, q.y - p.y, q.z - p.z);
}

static inline quat operator/(quat q, float s)
{
    return quat(q.w / s, q.x / s, q.y / s, q.z / s);
}

static inline quat operator-(quat q)
{
    return quat(-q.w, -q.x, -q.y, -q.z);
}

static inline float quat_length(quat q)
{
    return sqrtf(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z);
}

static inline quat quat_normalize(quat q, const float eps=1e-8f)
{
    return q / (quat_length(q) + eps);
}

static inline quat quat_inv(quat q)
{
    return quat(-q.w, q.x, q.y, q.z);
}

static inline quat quat_mul(quat q, quat p)
{
  return quat(
    p.w*q.w - p.x*q.x - p.y*q.y - p.z*q.z,
    p.w*q.x + p.x*q.w - p.y*q.z + p.z*q.y,
    p.w*q.y + p.x*q.z + p.y*q.w - p.z*q.x,
    p.w*q.z - p.x*q.y + p.y*q.x + p.z*q.w);
}

static inline quat quat_inv_mul(quat q, quat p)
{
  return quat_mul(quat_inv(q), p);
}

static inline quat quat_mul_inv(quat q, quat p)
{
  return quat_mul(q, quat_inv(p));
}

static inline vec3 quat_mul_vec3(quat q, vec3 v)
{
    vec3 t = 2.0f * cross(vec3(q.x, q.y, q.z), v);
    return v + q.w * t + cross(vec3(q.x, q.y, q.z), t);
}

static inline vec3 quat_inv_mul_vec3(quat q, vec3 v)
{
    return quat_mul_vec3(quat_inv(q), v);
}

static inline quat quat_abs(quat x)
{
    return x.w < 0.0 ? -x : x;
}

static inline quat quat_exp(vec3 v, float eps=1e-8f)
{
    float halfangle = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
	
    if (halfangle < eps)
    {
        return quat_normalize(quat(1.0f, v.x, v.y, v.z));
    }
    else
    {
        float c = cosf(halfangle);
        float s = sinf(halfangle) / halfangle;
        return quat(c, s * v.x, s * v.y, s * v.z);
    }
}

static inline vec3 quat_log(quat q, float eps=1e-8f)
{
    float length = sqrtf(q.x*q.x + q.y*q.y + q.z*q.z);
	
    if (length < eps)
    {
        return vec3(q.x, q.y, q.z);
    }
    else
    {
        float halfangle = acosf(clampf(q.w, -1.0f, 1.0f));
        return halfangle * (vec3(q.x, q.y, q.z) / length);
    }
}

static inline quat quat_from_scaled_angle_axis(vec3 v, float eps=1e-8f)
{
    return quat_exp(v / 2.0f, eps);
}

static inline vec3 quat_to_scaled_angle_axis(quat q, float eps=1e-8f)
{
    return 2.0f * quat_log(q, eps);
}

static inline vec3 quat_differentiate_angular_velocity(
    quat next, quat curr, float dt, float eps=1e-8f)
{
    return quat_to_scaled_angle_axis(
        quat_abs(quat_mul(next, quat_inv(curr))), eps) / dt; 
}

static inline quat quat_integrate_angular_velocity(
    vec3 vel, quat curr, float dt, float eps=1e-8f)
{
    return quat_mul(quat_from_scaled_angle_axis(vel * dt, eps), curr);
}

static inline quat quat_from_angle_axis(float angle, vec3 axis)
{
    float c = cosf(angle / 2.0f);
    float s = sinf(angle / 2.0f);
    return quat(c, s * axis.x, s * axis.y, s * axis.z);
}

static inline void quat_to_angle_axis(quat q, float& angle, vec3& axis, float eps=1e-8f)
{
    float length = sqrtf(q.x*q.x + q.y*q.y + q.z*q.z);

    if (length < eps)
    {
        angle = 0.0f;
        axis = vec3(1.0f, 0.0f, 0.0f);
    }
    else
    {
        angle = 2.0f * acosf(clampf(q.w, -1.0f, 1.0f));
        axis = vec3(q.x, q.y, q.z) / length;
    }
}

static inline float quat_dot(quat q, quat p)
{
    return q.w*p.w + q.x*p.x + q.y*p.y + q.z*p.z;
}

static inline quat quat_nlerp(quat q, quat p, float alpha)
{
    return quat_normalize(quat(
        lerpf(q.w, p.w, alpha),
        lerpf(q.x, p.x, alpha),
        lerpf(q.y, p.y, alpha),
        lerpf(q.z, p.z, alpha)));
}

static inline quat quat_nlerp_shortest(quat q, quat p, float alpha)
{
    if (quat_dot(q, p) < 0.0f)
    {
        p = -p;
    }
    
    return quat_nlerp(q, p, alpha);
}

static inline quat quat_slerp_shortest(quat q, quat p, float alpha, float eps=1e-5f)
{
    if (quat_dot(q, p) < 0.0f)
    {
        p = -p;
    }
    
    float dot = quat_dot(q, p);
	  float theta = acosf(clampf(dot, -1.0f, 1.0f));

    if (theta < eps)
    {
        return quat_nlerp(q, p, alpha);
    }
    
    quat r = quat_normalize(p - q*dot);

    return q * cosf(theta * alpha) + r * sinf(theta * alpha);
}

// Taken from https://zeux.io/2015/07/23/approximating-slerp/
static inline quat quat_slerp_shortest_approx(quat q, quat p, float alpha)
{
    float ca = quat_dot(q, p);
    
    if (ca < 0.0f)
    {
        p = -p;
    }
    
    float d = fabsf(ca);
    float a = 1.0904f + d * (-3.2452f + d * (3.55645f - d * 1.43519f));
    float b = 0.848013f + d * (-1.06021f + d * 0.215638f);
    float k = a * (alpha - 0.5f) * (alpha - 0.5f) + b;
    float oalpha = alpha + alpha * (alpha - 0.5f) * (alpha - 1) * k;

    return quat_nlerp(q, p, oalpha);
}

static inline float quat_angle_between(quat q, quat p)
{   
    quat diff = quat_abs(quat_mul_inv(q, p));
    return 2.0f * acosf(clampf(diff.w, -1.0f, 1.0f));
}

static inline quat quat_between(vec3 p, vec3 q)
{
    vec3 c = cross(p, q);
    
    return quat_normalize(quat(
        sqrtf(dot(p, p) * dot(q, q)) + dot(p, q),
        c.x, 
        c.y, 
        c.z));
}

static inline quat quat_from_cols(vec3 c0, vec3 c1, vec3 c2)
{
    if (c2.z < 0.0f)
    {
        if (c0.x > c1.y)
        {
            return quat_normalize(quat(
                c1.z-c2.y, 
                1.0f + c0.x - c1.y - c2.z, 
                c0.y+c1.x, 
                c2.x+c0.z));
        }
        else
        {
            return quat_normalize(quat(
                c2.x-c0.z, 
                c0.y+c1.x, 
                1.0f - c0.x + c1.y - c2.z, 
                c1.z+c2.y));
        }
    }
    else
    {
        if (c0.x < -c1.y)
        {
            return quat_normalize(quat(
                c0.y-c1.x, 
                c2.x+c0.z, 
                c1.z+c2.y, 
                1.0f - c0.x - c1.y + c2.z));
        }
        else
        {
            return quat_normalize(quat(
                1.0f + c0.x + c1.y + c2.z, 
                c1.z-c2.y, 
                c2.x-c0.z, 
                c0.y-c1.x));
        }
    }
}

static inline quat quat_from_xform_xy(vec3 x, vec3 y)
{
    vec3 c2 = normalize(cross(x, y));
    vec3 c1 = normalize(cross(c2, x));
    vec3 c0 = x;
    return quat_from_cols(c0, c1, c2);
}

