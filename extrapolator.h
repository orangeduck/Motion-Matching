#pragma once

#include "common.h"
#include "vec.h"
#include "mat.h"
#include "quat.h"
#include "array.h"
#include "nnet.h"

//--------------------------------------

void extrapolate_linear(
    slice1d<vec3> bone_positions,
    slice1d<vec3> bone_velocities,
    slice1d<quat> bone_rotations,
    slice1d<vec3> bone_angular_velocities,
    const float dt = 1.0f / 60.0f)
{
    for (int i = 0; i < bone_positions.size; i++)
    {
        bone_positions(i) = bone_positions(i) + dt * bone_velocities(i);
        bone_rotations(i) = quat_mul(quat_from_scaled_angle_axis(dt * bone_angular_velocities(i)), bone_rotations(i));
    }
}

void extrapolate_decay(
    slice1d<vec3> bone_positions,
    slice1d<vec3> bone_velocities,
    slice1d<quat> bone_rotations,
    slice1d<vec3> bone_angular_velocities,
    const float root_halflife,
    const float halflife,
    const float dt = 1.0f / 60.0f)
{
    for (int i = 0; i < bone_positions.size; i++)
    {
        bone_velocities(i) = damper_decay_exact(bone_velocities(i), i == 0 ? root_halflife : halflife, dt);
        bone_angular_velocities(i) = damper_decay_exact(bone_angular_velocities(i), i == 0 ? root_halflife : halflife, dt);
    }
  
    for (int i = 0; i < bone_positions.size; i++)
    {
        bone_positions(i) = bone_positions(i) + dt * bone_velocities(i);
        bone_rotations(i) = quat_mul(quat_from_scaled_angle_axis(dt * bone_angular_velocities(i)), bone_rotations(i));
    }
}

static inline vec3 apply_kdop_limit(
    const vec3 limit_space_rotation,
    const slice1d<float> limit_mins,
    const slice1d<float> limit_maxs,
    const vec3 limit_position,
    const mat3 limit_rotation,
    const slice1d<vec3> kdop_axes)
{   
    // Inverse transform point using position and rotation
    vec3 limit_point = mat3_transpose_mul_vec3(
        limit_rotation,
        limit_space_rotation - limit_position);
        
    for (int k = 0; k < kdop_axes.size; k++)
    {   
        // Clamp point along given axes
        vec3 t0 = limit_point - limit_mins(k) * kdop_axes(k);
        vec3 t1 = limit_point - limit_maxs(k) * kdop_axes(k);
        limit_point -= minf(dot(t0, kdop_axes(k)), 0.0f) * kdop_axes(k);
        limit_point -= maxf(dot(t1, kdop_axes(k)), 0.0f) * kdop_axes(k);
    }
    
    // Transform point using position and rotation
    return mat3_mul_vec3(limit_rotation, limit_point) + limit_position;
}

void extrapolate_clamp(
    slice1d<vec3> bone_positions,
    slice1d<vec3> bone_velocities,
    slice1d<quat> bone_rotations,
    slice1d<vec3> bone_angular_velocities,
    const float root_halflife,
    const float halflife,
    const slice1d<quat> reference_rotations,
    const slice1d<vec3> limit_positions,
    const slice1d<mat3> limit_rotations,
    const slice1d<vec3> kdop_axes,
    const slice2d<float> kdop_limit_mins,
    const slice2d<float> kdop_limit_maxs,
    const float dt = 1.0f / 60.0f)
{
    for (int i = 0; i < bone_positions.size; i++)
    {
        bone_velocities(i) = damper_decay_exact(bone_velocities(i), i == 0 ? root_halflife : halflife, dt);
        bone_angular_velocities(i) = damper_decay_exact(bone_angular_velocities(i), i == 0 ? root_halflife : halflife, dt);
    }
  
    for (int i = 0; i < bone_positions.size; i++)
    {
        bone_positions(i) = bone_positions(i) + dt * bone_velocities(i);
        bone_rotations(i) = quat_mul(quat_from_scaled_angle_axis(dt * bone_angular_velocities(i)), bone_rotations(i));
        
        if (i != 0)
        {
            vec3 limit_space_rotation = quat_to_scaled_angle_axis(
                quat_abs(quat_inv_mul(reference_rotations(i), bone_rotations(i))));
            
            limit_space_rotation = apply_kdop_limit(
                limit_space_rotation, 
                kdop_limit_mins(i),
                kdop_limit_maxs(i),
                limit_positions(i),
                limit_rotations(i),
                kdop_axes);
            
            bone_rotations(i) = quat_mul(reference_rotations(i), quat_from_scaled_angle_axis(limit_space_rotation));
        }

    }
}

void extrapolate_bounce(
    slice1d<vec3> bone_positions,
    slice1d<vec3> bone_velocities,
    slice1d<quat> bone_rotations,
    slice1d<vec3> bone_angular_velocities,
    const float root_halflife,
    const float halflife,
    const slice1d<quat> reference_rotations,
    const slice1d<vec3> limit_positions,
    const slice1d<mat3> limit_rotations,
    const slice1d<vec3> kdop_axes,
    const slice2d<float> kdop_limit_mins,
    const slice2d<float> kdop_limit_maxs,
    const float bounce_strength = 1.0f,
    const float dt = 1.0f / 60.0f,
    const float eps = 1e-5f)
{
    for (int i = 1; i < bone_rotations.size; i++)
    {
        vec3 limit_space_rotation = quat_to_scaled_angle_axis(
            quat_abs(quat_inv_mul(reference_rotations(i), bone_rotations(i))));
        
        vec3 projected_rotation = apply_kdop_limit(
            limit_space_rotation, 
            kdop_limit_mins(i),
            kdop_limit_maxs(i),
            limit_positions(i),
            limit_rotations(i),
            kdop_axes);
        
        if (length(limit_space_rotation - projected_rotation) > eps)
        {
            quat target_rotation = quat_mul(reference_rotations(i), quat_from_scaled_angle_axis(projected_rotation));
            vec3 bounce_velocity = bounce_strength * quat_to_scaled_angle_axis(quat_abs(quat_mul_inv(target_rotation, bone_rotations(i))));
            
            bone_angular_velocities(i) += bounce_velocity;
        }
    }
  
    for (int i = 0; i < bone_positions.size; i++)
    {
        bone_velocities(i) = damper_decay_exact(bone_velocities(i), i == 0 ? root_halflife : halflife, dt);        
        bone_angular_velocities(i) = damper_decay_exact(bone_angular_velocities(i), i == 0 ? root_halflife : halflife, dt);
    }
    
    for (int i = 0; i < bone_rotations.size; i++)
    {
        bone_positions(i) = bone_positions(i) + dt * bone_velocities(i);
        bone_rotations(i) = quat_mul(quat_from_scaled_angle_axis(dt * bone_angular_velocities(i)), bone_rotations(i));
    }
}

//--------------------------------------

void extrapolator_evaluate(
    slice1d<vec3> bone_positions,
    slice1d<vec3> bone_velocities,
    slice1d<quat> bone_rotations,
    slice1d<vec3> bone_angular_velocities,
    nnet_evaluation& evaluation,
    const nnet& nn,
    const float dt = 1.0f / 60.0f)
{
    slice1d<float> input_layer = evaluation.layers.front();
    slice1d<float> output_layer = evaluation.layers.back();
  
    int offset = 0;
    for (int i = 0; i < bone_positions.size - 1; i++)
    {
        input_layer(offset+i*3+0) = bone_positions(i+1).x;
        input_layer(offset+i*3+1) = bone_positions(i+1).y;
        input_layer(offset+i*3+2) = bone_positions(i+1).z;
    }
    offset += (bone_positions.size - 1) * 3;
    
    for (int i = 0; i < bone_rotations.size - 1; i++)
    {
        vec3 axis0 = quat_mul_vec3(bone_rotations(i+1), vec3(1, 0, 0));
        vec3 axis1 = quat_mul_vec3(bone_rotations(i+1), vec3(0, 1, 0));
        
        input_layer(offset+i*6+0) = axis0.x;
        input_layer(offset+i*6+1) = axis1.x;
        input_layer(offset+i*6+2) = axis0.y;
        input_layer(offset+i*6+3) = axis1.y;
        input_layer(offset+i*6+4) = axis0.z;
        input_layer(offset+i*6+5) = axis1.z;
    }
    offset += (bone_rotations.size - 1) * 6;
    
    for (int i = 0; i < bone_velocities.size - 1; i++)
    {
        input_layer(offset+i*3+0) = bone_velocities(i+1).x;
        input_layer(offset+i*3+1) = bone_velocities(i+1).y;
        input_layer(offset+i*3+2) = bone_velocities(i+1).z;
    }
    offset += (bone_velocities.size - 1) * 3;
    
    for (int i = 0; i < bone_angular_velocities.size - 1; i++)
    {
        input_layer(offset+i*3+0) = bone_angular_velocities(i+1).x;
        input_layer(offset+i*3+1) = bone_angular_velocities(i+1).y;
        input_layer(offset+i*3+2) = bone_angular_velocities(i+1).z;
    }
    offset += (bone_angular_velocities.size - 1) * 3;
    
    vec3 root_velocity = quat_inv_mul_vec3(bone_rotations(0), bone_velocities(0));
    vec3 root_angular_velocity = quat_inv_mul_vec3(bone_rotations(0), bone_angular_velocities(0));
    
    input_layer(offset+0) = root_velocity.x;
    input_layer(offset+1) = root_velocity.y;
    input_layer(offset+2) = root_velocity.z;
    offset += 3;
    
    input_layer(offset+0) = root_angular_velocity.x;
    input_layer(offset+1) = root_angular_velocity.y;
    input_layer(offset+2) = root_angular_velocity.z;
    offset += 3;
    
    assert(offset == nn.input_mean.size);
    
    // Evaluate network
    nnet_evaluate(evaluation, nn);
    
    // Update bone velocities
    offset = 0;
    for (int i = 0; i < bone_velocities.size - 1; i++)
    {
        bone_velocities(i + 1) = vec3(
            output_layer(offset+i*3+0),
            output_layer(offset+i*3+1),
            output_layer(offset+i*3+2));
    }
    offset += (bone_velocities.size - 1) * 3;
    
    // Update bone angular velocities
    for (int i = 0; i < bone_angular_velocities.size - 1; i++)
    {
        bone_angular_velocities(i + 1) = vec3(
            output_layer(offset+i*3+0),
            output_layer(offset+i*3+1),
            output_layer(offset+i*3+2));
    }
    offset += (bone_angular_velocities.size - 1) * 3;
    
    // Update Root Velocities
    
    bone_velocities(0) = quat_mul_vec3(bone_rotations(0), vec3(
        output_layer(offset+0),
        output_layer(offset+1),
        output_layer(offset+2)));
    offset += 3;

   bone_angular_velocities(0) = quat_mul_vec3(bone_rotations(0), vec3(
        output_layer(offset+0),
        output_layer(offset+1),
        output_layer(offset+2)));
    offset += 3;
    
    assert(offset == nn.output_mean.size);

    // Update Positions and Rotations
    
    for (int i = 0; i < bone_positions.size; i++)
    {
        bone_positions(i) = bone_positions(i) + dt * bone_velocities(i);
        bone_rotations(i) = quat_mul(quat_from_scaled_angle_axis(dt * bone_angular_velocities(i)), bone_rotations(i));
    }
}

//--------------------------------------

void extrapolate(
    slice1d<vec3> bone_positions,
    slice1d<vec3> bone_velocities,
    slice1d<quat> bone_rotations,
    slice1d<vec3> bone_angular_velocities,
    int extrapolation_method,
    const float root_halflife,
    const float bounce_halflife,
    const float halflife,
    const slice1d<quat> reference_rotations,
    const slice1d<vec3> limit_positions,
    const slice1d<mat3> limit_rotations,
    const slice1d<vec3> kdop_axes,
    const slice2d<float> kdop_limit_mins,
    const slice2d<float> kdop_limit_maxs,
    const float bounce_strength,
    nnet_evaluation& evaluation,
    const nnet& nn,
    const float dt = 1.0f / 60.0f,
    const float eps = 1e-5f)
{
    if (extrapolation_method == 0)
    {
      
    }
    else if (extrapolation_method == 1)
    {
        extrapolate_linear(
            bone_positions,
            bone_velocities,
            bone_rotations,
            bone_angular_velocities,
            dt);
    }
    else if (extrapolation_method == 2)
    {
        extrapolate_decay(
            bone_positions,
            bone_velocities,
            bone_rotations,
            bone_angular_velocities,
            root_halflife,
            halflife,
            dt);
    }
    else if (extrapolation_method == 3)
    {
        extrapolate_clamp(
            bone_positions,
            bone_velocities,
            bone_rotations,
            bone_angular_velocities,
            root_halflife,
            halflife,
            reference_rotations,
            limit_positions,
            limit_rotations,
            kdop_axes,
            kdop_limit_mins,
            kdop_limit_maxs,
            dt);
    }
    else if (extrapolation_method == 4)
    {
        extrapolate_bounce(
            bone_positions,
            bone_velocities,
            bone_rotations,
            bone_angular_velocities,
            root_halflife,
            bounce_halflife,
            reference_rotations,
            limit_positions,
            limit_rotations,
            kdop_axes,
            kdop_limit_mins,
            kdop_limit_maxs,
            bounce_strength,
            dt);
    }
    else if (extrapolation_method == 5)
    {
        extrapolator_evaluate(
            bone_positions,
            bone_velocities,
            bone_rotations,
            bone_angular_velocities,
            evaluation,
            nn,
            dt);
    }
}
