#pragma once

#include "vec.h"
#include "quat.h"
#include "array.h"

#include <assert.h>
#include <stdio.h>

//--------------------------------------

enum Bones
{
    Bone_Entity        = 0,
    Bone_Hips          = 1,
    Bone_LeftUpLeg     = 2,
    Bone_LeftLeg       = 3,
    Bone_LeftFoot      = 4,
    Bone_LeftToe       = 5,
    Bone_RightUpLeg    = 6,
    Bone_RightLeg      = 7,
    Bone_RightFoot     = 8,
    Bone_RightToe      = 9,
    Bone_Spine         = 10,
    Bone_Spine1        = 11,
    Bone_Spine2        = 12,
    Bone_Neck          = 13,
    Bone_Head          = 14,
    Bone_LeftShoulder  = 15,
    Bone_LeftArm       = 16,
    Bone_LeftForeArm   = 17,
    Bone_LeftHand      = 18,
    Bone_RightShoulder = 19,
    Bone_RightArm      = 20,
    Bone_RightForeArm  = 21,
    Bone_RightHand     = 22,
};

//--------------------------------------

struct character
{
    array1d<vec3> positions;
    array1d<vec3> normals;
    array1d<vec2> texcoords;
    array1d<unsigned short> triangles;
    
    array2d<float> bone_weights;
    array2d<unsigned short> bone_indices;
    
    array1d<vec3> bone_rest_positions;
    array1d<quat> bone_rest_rotations;
};

void character_load(character& c, const char* filename)
{
    FILE* f = fopen(filename, "rb");
    assert(f != NULL);
    
    array1d_read(c.positions, f);
    array1d_read(c.normals, f);
    array1d_read(c.texcoords, f);
    array1d_read(c.triangles, f);
    
    array2d_read(c.bone_weights, f);
    array2d_read(c.bone_indices, f);
    
    array1d_read(c.bone_rest_positions, f);
    array1d_read(c.bone_rest_rotations, f);
    
    fclose(f);
}

//--------------------------------------

void linear_blend_skinning_positions(
    slice1d<vec3> anim_positions,
    const slice1d<vec3> rest_positions,
    const slice2d<float> bone_weights,
    const slice2d<unsigned short> bone_indices,
    const slice1d<vec3> bone_rest_positions,
    const slice1d<quat> bone_rest_rotations,
    const slice1d<vec3> bone_anim_positions,
    const slice1d<quat> bone_anim_rotations)
{
    anim_positions.zero();
    
    for (int i = 0; i < anim_positions.size; i++)
    {
        for (int j = 0; j < bone_indices.cols; j++)
        {
            if (bone_weights(i, j) > 0.0f)
            {
                int b = bone_indices(i, j);
                
                vec3 position = rest_positions(i);            
                position = quat_mul_vec3(quat_inv(bone_rest_rotations(b)), position - bone_rest_positions(b));
                position = quat_mul_vec3(bone_anim_rotations(b), position) + bone_anim_positions(b);
                
                anim_positions(i) = anim_positions(i) + bone_weights(i, j) * position;
            }
        } 
    }
}

void linear_blend_skinning_normals(
    slice1d<vec3> anim_normals,
    const slice1d<vec3> rest_normals,
    const slice2d<float> bone_weights,
    const slice2d<unsigned short> bone_indices,
    const slice1d<quat> bone_rest_rotations,
    const slice1d<quat> bone_anim_rotations)
{
    anim_normals.zero();
    
    for (int i = 0; i < anim_normals.size; i++)
    { 
        for (int j = 0; j < bone_indices.cols; j++)
        {
            if (bone_weights(i, j) > 0.0f)
            {
                int b = bone_indices(i, j);
                
                vec3 normal = rest_normals(i);
                normal = quat_mul_vec3(quat_inv(bone_rest_rotations(b)), normal);
                normal = quat_mul_vec3(bone_anim_rotations(b), normal);
                
                anim_normals(i) = anim_normals(i) + bone_weights(i, j) * normal;
            }
        }
    }
    
    for (int i = 0; i < anim_normals.size; i++)
    { 
        anim_normals(i) = normalize(anim_normals(i));
    }
}

