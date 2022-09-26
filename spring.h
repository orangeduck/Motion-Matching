#pragma once

#include "common.h"
#include "vec.h"
#include "quat.h"

//--------------------------------------

static inline float damper_implicit(float x, float g, float halflife, float dt, float eps=1e-5f)
{
    return lerpf(x, g, 1.0f - fast_negexpf((LN2f * dt) / (halflife + eps)));
}

static inline vec3 damper_implicit(vec3 x, vec3 g, float halflife, float dt, float eps=1e-5f)
{
    return lerp(x, g, 1.0f - fast_negexpf((LN2f * dt) / (halflife + eps)));
}

static inline quat damper_implicit(quat x, quat g, float halflife, float dt, float eps=1e-5f)
{
    return quat_slerp_shortest_approx(x, g, 1.0f - fast_negexpf((LN2f * dt) / (halflife + eps)));
}

static inline float damp_adjustment_implicit(float g, float halflife, float dt, float eps=1e-5f)
{
    return g * (1.0f - fast_negexpf((LN2f * dt) / (halflife + eps)));
}

static inline vec3 damp_adjustment_implicit(vec3 g, float halflife, float dt, float eps=1e-5f)
{
    return g * (1.0f - fast_negexpf((LN2f * dt) / (halflife + eps)));
}

static inline quat damp_adjustment_implicit(quat g, float halflife, float dt, float eps=1e-5f)
{
    return quat_slerp_shortest_approx(quat(), g, 1.0f - fast_negexpf((LN2f * dt) / (halflife + eps)));
}

//--------------------------------------

static inline float halflife_to_damping(float halflife, float eps = 1e-5f)
{
    return (4.0f * LN2f) / (halflife + eps);
}
    
static inline float damping_to_halflife(float damping, float eps = 1e-5f)
{
    return (4.0f * LN2f) / (damping + eps);
}

static inline float frequency_to_stiffness(float frequency)
{
   return squaref(2.0f * PIf * frequency);
}

static inline float stiffness_to_frequency(float stiffness)
{
    return sqrtf(stiffness) / (2.0f * PIf);
}

//--------------------------------------

static inline void simple_spring_damper_implicit(
    float& x, 
    float& v, 
    const float x_goal, 
    const float halflife, 
    const float dt)
{
    float y = halflife_to_damping(halflife) / 2.0f; 
    float j0 = x - x_goal;
    float j1 = v + j0*y;
    float eydt = fast_negexpf(y*dt);

    x = eydt*(j0 + j1*dt) + x_goal;
    v = eydt*(v - j1*y*dt);
}

static inline void simple_spring_damper_implicit(
    vec3& x, 
    vec3& v, 
    const vec3 x_goal, 
    const float halflife, 
    const float dt)
{
    float y = halflife_to_damping(halflife) / 2.0f; 
    vec3 j0 = x - x_goal;
    vec3 j1 = v + j0*y;
    float eydt = fast_negexpf(y*dt);

    x = eydt*(j0 + j1*dt) + x_goal;
    v = eydt*(v - j1*y*dt);
}

static inline void simple_spring_damper_implicit(
    quat& x, 
    vec3& v, 
    const quat x_goal, 
    const float halflife, 
    const float dt)
{
    float y = halflife_to_damping(halflife) / 2.0f; 
    
    vec3 j0 = quat_to_scaled_angle_axis(quat_abs(quat_mul(x, quat_inv(x_goal))));
    vec3 j1 = v + j0*y;
    
    float eydt = fast_negexpf(y*dt);

    x = quat_mul(quat_from_scaled_angle_axis(eydt*(j0 + j1*dt)), x_goal);
    v = eydt*(v - j1*y*dt);
}

//--------------------------------------

static inline void decay_spring_damper_implicit(
    float& x, 
    float& v, 
    const float halflife, 
    const float dt)
{
    float y = halflife_to_damping(halflife) / 2.0f; 
    float j1 = v + x*y;
    float eydt = fast_negexpf(y*dt);

    x = eydt*(x + j1*dt);
    v = eydt*(v - j1*y*dt);
}

static inline void decay_spring_damper_implicit(
    vec3& x, 
    vec3& v, 
    const float halflife, 
    const float dt)
{
    float y = halflife_to_damping(halflife) / 2.0f; 
    vec3 j1 = v + x*y;
    float eydt = fast_negexpf(y*dt);

    x = eydt*(x + j1*dt);
    v = eydt*(v - j1*y*dt);
}

static inline void decay_spring_damper_implicit(
    quat& x, 
    vec3& v, 
    const float halflife, 
    const float dt)
{
    float y = halflife_to_damping(halflife) / 2.0f; 
    
    vec3 j0 = quat_to_scaled_angle_axis(x);
    vec3 j1 = v + j0*y;
    
    float eydt = fast_negexpf(y*dt);

    x = quat_from_scaled_angle_axis(eydt*(j0 + j1*dt));
    v = eydt*(v - j1*y*dt);
}

//--------------------------------------

static inline void inertialize_transition(
    vec3& off_x, 
    vec3& off_v, 
    const vec3 src_x,
    const vec3 src_v,
    const vec3 dst_x,
    const vec3 dst_v)
{
    off_x = (src_x + off_x) - dst_x;
    off_v = (src_v + off_v) - dst_v;
}

static inline void inertialize_update(
    vec3& out_x, 
    vec3& out_v,
    vec3& off_x, 
    vec3& off_v,
    const vec3 in_x, 
    const vec3 in_v,
    const float halflife,
    const float dt)
{
    decay_spring_damper_implicit(off_x, off_v, halflife, dt);
    out_x = in_x + off_x;
    out_v = in_v + off_v;
}

static inline void inertialize_transition(
    quat& off_x, 
    vec3& off_v, 
    const quat src_x,
    const vec3 src_v,
    const quat dst_x,
    const vec3 dst_v)
{
    off_x = quat_abs(quat_mul(quat_mul(off_x, src_x), quat_inv(dst_x)));
    off_v = (off_v + src_v) - dst_v;
}

static inline void inertialize_update(
    quat& out_x, 
    vec3& out_v,
    quat& off_x, 
    vec3& off_v,
    const quat in_x, 
    const vec3 in_v,
    const float halflife,
    const float dt)
{
    decay_spring_damper_implicit(off_x, off_v, halflife, dt);
    out_x = quat_mul(off_x, in_x);
    out_v = off_v + in_v;
}

//--------------------------------------

void decayed_offset_cubic(
    vec3& out_x,
    vec3& out_v,
    const vec3 init_x,
    const vec3 init_v,
    const float blendtime, 
    const float dt,
    const float eps=1e-8f)
{
    float t = clampf(dt / (blendtime + eps), 0.0f, 1.0f);

    vec3 d = init_x;
    vec3 c = init_v * blendtime;
    vec3 b = -3.0f*d - 2.0f*c;
    vec3 a = 2.0f*d + c;
    
    out_x = a*t*t*t + b*t*t + c*t + d;
    out_v = (3.0f*a*t*t + 2.0f*b*t + c) / (blendtime + eps);
}

void decayed_offset_cubic(
    quat& out_x,
    vec3& out_v,
    const quat init_x,
    const vec3 init_v,
    const float blendtime, 
    const float dt,
    const float eps=1e-8f)
{
    float t = clampf(dt / (blendtime + eps), 0.0f, 1.0f);

    vec3 d = quat_to_scaled_angle_axis(init_x);
    vec3 c = init_v * blendtime;
    vec3 b = -3.0f*d - 2.0f*c;
    vec3 a = 2.0f*d + c;
    
    out_x = quat_from_scaled_angle_axis(a*t*t*t + b*t*t + c*t + d);
    out_v = (3.0f*a*t*t + 2.0f*b*t + c) / (blendtime + eps);
}

static inline void inertialize_cubic_transition(
    vec3& off_x, 
    vec3& off_v, 
    float& off_t,
    const vec3 src_x,
    const vec3 src_v,
    const vec3 dst_x,
    const vec3 dst_v,
    const float blendtime)
{
    vec3 dec_x, dec_v;
    decayed_offset_cubic(dec_x, dec_v, off_x, off_v, blendtime, off_t);
    
    off_x = (dec_x + src_x) - dst_x;
    off_v = (dec_v + src_v) - dst_v;
    off_t = 0.0f;
}

static inline void inertialize_cubic_update(
    vec3& out_x, 
    vec3& out_v,
    float& off_t,
    const vec3 off_x,
    const vec3 off_v,
    const vec3 in_x, 
    const vec3 in_v,
    const float blendtime,
    const float dt)
{
    off_t += dt;
    
    vec3 dec_x, dec_v;
    decayed_offset_cubic(dec_x, dec_v, off_x, off_v, blendtime, off_t);
    
    out_x = in_x + dec_x;
    out_v = in_v + dec_v;
}

static inline void inertialize_cubic_transition(
    quat& off_x, 
    vec3& off_v, 
    float& off_t,
    const quat src_x,
    const vec3 src_v,
    const quat dst_x,
    const vec3 dst_v,
    const float blendtime)
{
    quat dec_x; vec3 dec_v;
    decayed_offset_cubic(dec_x, dec_v, off_x, off_v, blendtime, off_t);
    
    off_x = quat_abs(quat_mul(quat_mul(dec_x, src_x), quat_inv(dst_x)));
    off_v = (dec_v + src_v) - dst_v;
    off_t = 0.0f;
}

static inline void inertialize_cubic_update(
    quat& out_x, 
    vec3& out_v,
    float& off_t,
    const quat off_x,
    const vec3 off_v,
    const quat in_x, 
    const vec3 in_v,
    const float blendtime,
    const float dt)
{
    off_t += dt;
    
    quat dec_x; vec3 dec_v;
    decayed_offset_cubic(dec_x, dec_v, off_x, off_v, blendtime, off_t);
    
    out_x = quat_mul(dec_x, in_x);
    out_v = dec_v + in_v;
}


