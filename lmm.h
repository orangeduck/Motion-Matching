#pragma once

#include "common.h"
#include "vec.h"
#include "quat.h"
#include "array.h"
#include "nnet.h"

// This function uses the decompressor network
// to generate the pose of the character. It 
// requires as input the feature values and latent 
// values as well as a current root position and 
// rotation.
void decompressor_evaluate(
    slice1d<vec3> bone_positions,
    slice1d<vec3> bone_velocities,
    slice1d<quat> bone_rotations,
    slice1d<vec3> bone_angular_velocities,
    slice1d<bool> bone_contacts,
    nnet_evaluation& evaluation,
    const slice1d<float> features,
    const slice1d<float> latent,
    const vec3 root_position,
    const quat root_rotation,
    const nnet& nn,
    const float dt = 1.0f / 60.0f)
{
    slice1d<float> input_layer = evaluation.layers.front();
    slice1d<float> output_layer = evaluation.layers.back();
  
    // First copy feature values and latent variables to 
    // the input layer of the network
  
    for (int i = 0; i < features.size; i++)
    {
        input_layer(i) = features(i);
    }
    
    for (int i = 0; i < latent.size; i++)
    {
        input_layer(features.size + i) = latent(i);
    }
    
    // Evaluate network
    nnet_evaluate(evaluation, nn);
    
    // Extract bone positions
    int offset = 0;
    for (int i = 0; i < bone_positions.size - 1; i++)
    {
        bone_positions(i + 1) = vec3(
            output_layer(offset+i*3+0),
            output_layer(offset+i*3+1),
            output_layer(offset+i*3+2));
    }
    offset += (bone_positions.size - 1) * 3;
    
    // Extract bone rotations, convert from 2-axis representation
    for (int i = 0; i < bone_rotations.size - 1; i++)
    {   
        bone_rotations(i + 1) = quat_from_xform_xy(
            vec3(output_layer(offset+i*6+0),
                 output_layer(offset+i*6+2),
                 output_layer(offset+i*6+4)),
            vec3(output_layer(offset+i*6+1),
                 output_layer(offset+i*6+3),
                 output_layer(offset+i*6+5)));
    }
    offset += (bone_rotations.size - 1) * 6;
    
    // Extract bone velocities
    for (int i = 0; i < bone_velocities.size - 1; i++)
    {
        bone_velocities(i + 1) = vec3(
            output_layer(offset+i*3+0),
            output_layer(offset+i*3+1),
            output_layer(offset+i*3+2));
    }
    offset += (bone_velocities.size - 1) * 3;
    
    // Extract bone angular velocities
    for (int i = 0; i < bone_angular_velocities.size - 1; i++)
    {
        bone_angular_velocities(i + 1) = vec3(
            output_layer(offset+i*3+0),
            output_layer(offset+i*3+1),
            output_layer(offset+i*3+2));
    }
    offset += (bone_angular_velocities.size - 1) * 3;
    
    // Extract root velocities and put in world space
    
    vec3 root_velocity = quat_mul_vec3(root_rotation, vec3(
        output_layer(offset+0),
        output_layer(offset+1),
        output_layer(offset+2)));
        
    vec3 root_angular_velocity = quat_mul_vec3(root_rotation, vec3(
        output_layer(offset+3),
        output_layer(offset+4),
        output_layer(offset+5)));
    
    offset += 6;

    // Find new root position/rotation/velocities etc.
    
    bone_positions(0) = dt * root_velocity + root_position;
    bone_rotations(0) = quat_mul(quat_from_scaled_angle_axis(root_angular_velocity * dt), root_rotation);
    bone_velocities(0) = root_velocity;
    bone_angular_velocities(0) = root_angular_velocity;    
    
    // Extract bone contacts
    if (bone_contacts.data != nullptr)
    {
        bone_contacts(0) = output_layer(offset+0) > 0.5f;
        bone_contacts(1) = output_layer(offset+1) > 0.5f;
    }

    offset += 2;
    
    // Check we got everything!
    assert(offset == nn.output_mean.size);
}

// This function updates the feature and latent values
// using the stepper network and a given dt.
void stepper_evaluate(
    slice1d<float> features,
    slice1d<float> latent,
    nnet_evaluation& evaluation,
    const nnet& nn,
    const float dt = 1.0f / 60.0f)
{
    slice1d<float> input_layer = evaluation.layers.front();
    slice1d<float> output_layer = evaluation.layers.back();
  
    // Copy features and latents to input
  
    for (int i = 0; i < features.size; i++)
    {
        input_layer(i) = features(i);
    }
    
    for (int i = 0; i < latent.size; i++)
    {
        input_layer(features.size + i) = latent(i);
    }
    
    // Evaluate network
    
    nnet_evaluate(evaluation, nn);
    
    // Update features and latents using result
    
    for (int i = 0; i < features.size; i++)
    {
        features(i) += dt * output_layer(i);
    }
    
    for (int i = 0; i < latent.size; i++)
    {
        latent(i) += dt * output_layer(features.size + i);
    }
}

// This function projects a set of feature values onto
// the nearest in the trained database, also outputting the 
// associated latent values. It also produces the matching 
// cost using the distance of the projection, and detects 
// transitions for a given transition cost by measuring the 
// distance between the projected result and the current
// feature values
void projector_evaluate(
    bool& transition,
    float& best_cost,
    slice1d<float> proj_features,
    slice1d<float> proj_latent,
    nnet_evaluation& evaluation,
    const slice1d<float> query,
    const slice1d<float> features_offset,
    const slice1d<float> features_scale,
    const slice1d<float> curr_features,
    const nnet& nn,
    const float transition_cost = 0.0f)
{
    slice1d<float> input_layer = evaluation.layers.front();
    slice1d<float> output_layer = evaluation.layers.back();
    
    // Copy query features to input
    
    for (int i = 0; i < query.size; i++)
    {
        input_layer(i) = (query(i) - features_offset(i)) / features_scale(i);      
    }
    
    // Evaluate network
    
    nnet_evaluate(evaluation, nn);
    
    // Copy projected features and latents from output
    
    for (int i = 0; i < proj_features.size; i++)
    {
        proj_features(i) = output_layer(i);
    }
    
    for (int i = 0; i < proj_latent.size; i++)
    {
        proj_latent(i) = output_layer(proj_features.size + i);
    }
    
    // Compute the distance of the projection
    
    best_cost = 0.0f;
    for (int i = 0; i < proj_features.size; i++)
    {
        best_cost += squaref(query(i) - proj_features(i));
    }
    best_cost = sqrtf(best_cost);
    
    // Compute the change in features from the current
    
    float trns_dist_squared = 0.0f;
    for (int i = 0; i < proj_features.size; i++)
    {
        trns_dist_squared += squaref(curr_features(i) - proj_features(i));
    }
    
    // If greater than the transition cost...
    if (trns_dist_squared > squaref(transition_cost))
    {
        // transition and add the transition cost
        
        transition = true;
        best_cost += transition_cost;
    }
    else
    {   
        // Don't transition and use current features as-is
        
        transition = false;
        
        for (int i = 0; i < proj_features.size; i++)
        {
            proj_features(i) = curr_features(i);
        }
        
        // Re-compute the projection cost
        
        best_cost = 0.0f;
        for (int i = 0; i < curr_features.size; i++)
        {
            best_cost += squaref(query(i) - curr_features(i));
        }
        best_cost = sqrtf(best_cost);
    }
}
