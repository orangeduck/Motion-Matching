#pragma once
#include "vec.h"

struct mat3
{
    mat3() : 
        xx(1.0f), xy(0.0f), xz(0.0f),
        yx(0.0f), yy(1.0f), yz(0.0f),
        zx(0.0f), zy(0.0f), zz(1.0f) {}
    
    mat3(
        float _xx, float _xy, float _xz, 
        float _yx, float _yy, float _yz, 
        float _zx, float _zy, float _zz) : 
        xx(_xx), xy(_xy), xz(_xz),
        yx(_yx), yy(_yy), yz(_yz),
        zx(_zx), zy(_zy), zz(_zz) {}
    
    mat3(vec3 r0, vec3 r1, vec3 r2) : 
        xx(r0.x), xy(r0.y), xz(r0.z),
        yx(r1.x), yy(r1.y), yz(r1.z),
        zx(r2.x), zy(r2.y), zz(r2.z) {}
    
    vec3 r0() const { return vec3(xx, xy, xz); }
    vec3 r1() const { return vec3(yx, yy, yz); }
    vec3 r2() const { return vec3(zx, zy, zz); }

    vec3 c0() const { return vec3(xx, yx, zx); }
    vec3 c1() const { return vec3(xy, yy, zy); }
    vec3 c2() const { return vec3(xz, yz, zz); }

    float xx, xy, xz,
          yx, yy, yz,
          zx, zy, zz;
};

static inline mat3 operator+(mat3 m, mat3 n)
{
    return mat3(
        m.xx + n.xx, m.xy + n.xy, m.xz + n.xz,
        m.yx + n.yx, m.yy + n.yy, m.yz + n.yz,
        m.zx + n.zx, m.zy + n.zy, m.zz + n.zz);
}

static inline mat3 operator-(mat3 m, mat3 n)
{
    return mat3(
        m.xx - n.xx, m.xy - n.xy, m.xz - n.xz,
        m.yx - n.yx, m.yy - n.yy, m.yz - n.yz,
        m.zx - n.zx, m.zy - n.zy, m.zz - n.zz);
}

static inline mat3 operator/(mat3 m, float v)
{
    return mat3(
        m.xx / v, m.xy / v, m.xz / v,
        m.yx / v, m.yy / v, m.yz / v,
        m.zx / v, m.zy / v, m.zz / v);
}

static inline mat3 operator*(float v, mat3 m)
{
    return mat3(
        v * m.xx, v * m.xy, v * m.xz,
        v * m.yx, v * m.yy, v * m.yz,
        v * m.zx, v * m.zy, v * m.zz);
}

static inline mat3 mat3_zero()
{
    return mat3(
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f);
}

static inline mat3 mat3_transpose(mat3 m)
{
    return mat3(
        m.xx, m.yx, m.zx,
        m.xy, m.yy, m.zy,
        m.xz, m.yz, m.zz);
}

static inline mat3 mat3_mul(mat3 m, mat3 n)
{
  return mat3(
      dot(m.r0(), n.c0()), dot(m.r0(), n.c1()), dot(m.r0(), n.c2()),
      dot(m.r1(), n.c0()), dot(m.r1(), n.c1()), dot(m.r1(), n.c2()),
      dot(m.r2(), n.c0()), dot(m.r2(), n.c1()), dot(m.r2(), n.c2()));
}

static inline mat3 mat3_transpose_mul(mat3 m, mat3 n)
{
  return mat3(
      dot(m.c0(), n.c0()), dot(m.c0(), n.c1()), dot(m.c0(), n.c2()),
      dot(m.c1(), n.c0()), dot(m.c1(), n.c1()), dot(m.c1(), n.c2()),
      dot(m.c2(), n.c0()), dot(m.c2(), n.c1()), dot(m.c2(), n.c2()));
}

static inline vec3 mat3_mul_vec3(mat3 m, vec3 v)
{
    return vec3(
        dot(m.r0(), v),
        dot(m.r1(), v),
        dot(m.r2(), v));
}

static inline vec3 mat3_transpose_mul_vec3(mat3 m, vec3 v)
{
    return vec3(
        dot(m.c0(), v),
        dot(m.c1(), v),
        dot(m.c2(), v));
}

static inline mat3 mat3_from_angle_axis(float angle, vec3 axis)
{
    float a0 = axis.x, a1 = axis.y, a2 = axis.z; 
    float c = cosf(angle), s = sinf(angle), t = 1.0f - cosf(angle);
    
    return mat3(
        c+a0*a0*t, a0*a1*t-a2*s, a0*a2*t+a1*s,
        a0*a1*t+a2*s, c+a1*a1*t, a1*a2*t-a0*s,
        a0*a2*t-a1*s, a1*a2*t+a0*s, c+a2*a2*t);
}

static inline mat3 mat3_outer(vec3 v, vec3 w)
{
    return mat3(
        v.x * w.x, v.x * w.y, v.x * w.z,
        v.y * w.x, v.y * w.y, v.y * w.z,
        v.z * w.x, v.z * w.y, v.z * w.z);
}

static inline vec3 mat3_svd_dominant_eigen(
    const mat3 A, 
    const vec3 v0,
    const int iterations, 
    const float eps)
{
    // Initial Guess at Eigen Vector & Value
    vec3 v = v0;
    float ev = (mat3_mul_vec3(A, v) / v).x;
    
    for (int i = 0; i < iterations; i++)
    {
        // Power Iteration
        vec3 Av = mat3_mul_vec3(A, v);
        
        // Next Guess at Eigen Vector & Value
        vec3 v_new = normalize(Av);
        float ev_new = (mat3_mul_vec3(A, v_new) / v_new).x;
        
        // Break if converged
        if (fabs(ev - ev_new) < eps)
        {
            break;
        }
        
        // Update best guess
        v = v_new;
        ev = ev_new;
    }
    
    return v;
}

static inline void mat3_svd_piter(
    mat3& U,
    vec3& s,
    mat3& V,
    const mat3 A, 
    const int iterations = 64,
    const float eps = 1e-5f)
{
    // First Eigen Vector
    vec3 g0 = vec3(1, 0, 0);
    mat3 B0 = A;
    vec3 u0 = mat3_svd_dominant_eigen(B0, g0, iterations, eps);
    vec3 v0_unnormalized = mat3_transpose_mul_vec3(A, u0);
    float s0 = length(v0_unnormalized);
    vec3 v0 = s0 < eps ? g0 : normalize(v0_unnormalized);

    // Second Eigen Vector
    mat3 B1 = A;
    vec3 g1 = normalize(cross(vec3(0, 0, 1), v0));
    B1 = B1 - s0 * mat3_outer(u0, v0);
    vec3 u1 = mat3_svd_dominant_eigen(B1, g1, iterations, eps);
    vec3 v1_unnormalized = mat3_transpose_mul_vec3(A, u1);
    float s1 = length(v1_unnormalized);
    vec3 v1 = s1 < eps ? g1 : normalize(v1_unnormalized);
    
    // Third Eigen Vector
    mat3 B2 = A;
    vec3 v2 = normalize(cross(v0, v1));
    B2 = B2 - s0 * mat3_outer(u0, v0);
    B2 = B2 - s1 * mat3_outer(u1, v1);
    vec3 u2 = mat3_svd_dominant_eigen(B2, v2, iterations, eps);
    float s2 = length(mat3_transpose_mul_vec3(A, u2));
    
    // Done
    U = mat3(u0, u1, u2);
    s = vec3(s0, s1, s2);
    V = mat3(v0, v1, v2);
}
