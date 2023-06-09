extern "C"
{
#include "raylib.h"
#include "raymath.h"
#define RAYGUI_IMPLEMENTATION
#include "raygui.h"
}
#if defined(PLATFORM_WEB)
#include <emscripten/emscripten.h>
#endif

#include "common.h"
#include "vec.h"
#include "mat.h"
#include "quat.h"
#include "spring.h"
#include "array.h"
#include "character.h"
#include "database.h"
#include "nnet.h"
#include "extrapolator.h"

#include <initializer_list>
#include <functional>

/*
#include <profileapi.h>

#define PROFILE_INIT() \
    LARGE_INTEGER ____prof_freq; \
    QueryPerformanceFrequency(&____prof_freq);

#define PROFILE_BEGIN(TIMER) \
    LARGE_INTEGER ____prof_li_start_##TIMER; \
    QueryPerformanceCounter(&____prof_li_start_##TIMER);

#define PROFILE_END(TIMER) \
    LARGE_INTEGER ____prof_li_end_##TIMER; \
    QueryPerformanceCounter(&____prof_li_end_##TIMER); \
    printf("%s: %5.1fus\n", #TIMER, (double)((____prof_li_end_##TIMER.QuadPart - ____prof_li_start_##TIMER.QuadPart) * 1000000) / (double)____prof_freq.QuadPart);
*/

//--------------------------------------

static inline Vector3 to_Vector3(vec3 v)
{
    return (Vector3){ v.x, v.y, v.z };
}

//--------------------------------------

// Perform linear blend skinning and copy 
// result into mesh data. Update and upload 
// deformed vertex positions and normals to GPU
void deform_character_mesh(
  Mesh& mesh, 
  const character& c,
  const slice1d<vec3> bone_anim_positions,
  const slice1d<quat> bone_anim_rotations,
  const slice1d<int> bone_parents)
{
    linear_blend_skinning_positions(
        slice1d<vec3>(mesh.vertexCount, (vec3*)mesh.vertices),
        c.positions,
        c.bone_weights,
        c.bone_indices,
        c.bone_rest_positions,
        c.bone_rest_rotations,
        bone_anim_positions,
        bone_anim_rotations);
    
    linear_blend_skinning_normals(
        slice1d<vec3>(mesh.vertexCount, (vec3*)mesh.normals),
        c.normals,
        c.bone_weights,
        c.bone_indices,
        c.bone_rest_rotations,
        bone_anim_rotations);
    
    UpdateMeshBuffer(mesh, 0, mesh.vertices, mesh.vertexCount * 3 * sizeof(float), 0);
    UpdateMeshBuffer(mesh, 2, mesh.normals, mesh.vertexCount * 3 * sizeof(float), 0);
}

Mesh make_character_mesh(character& c)
{
    Mesh mesh = { 0 };
    
    mesh.vertexCount = c.positions.size;
    mesh.triangleCount = c.triangles.size / 3;
    mesh.vertices = (float*)MemAlloc(c.positions.size * 3 * sizeof(float));
    mesh.texcoords = (float*)MemAlloc(c.texcoords.size * 2 * sizeof(float));
    mesh.normals = (float*)MemAlloc(c.normals.size * 3 * sizeof(float));
    mesh.indices = (unsigned short*)MemAlloc(c.triangles.size * sizeof(unsigned short));
    
    memcpy(mesh.vertices, c.positions.data, c.positions.size * 3 * sizeof(float));
    memcpy(mesh.texcoords, c.texcoords.data, c.texcoords.size * 2 * sizeof(float));
    memcpy(mesh.normals, c.normals.data, c.normals.size * 3 * sizeof(float));
    memcpy(mesh.indices, c.triangles.data, c.triangles.size * sizeof(unsigned short));
    
    UploadMesh(&mesh, true);
    
    return mesh;
}

//--------------------------------------

// Basic functionality to get gamepad input including deadzone and 
// squaring of the stick location to increase sensitivity. To make 
// all the other code that uses this easier, we assume stick is 
// oriented on floor (i.e. y-axis is zero)

enum
{
    GAMEPAD_PLAYER = 0,
};

enum
{
    GAMEPAD_STICK_LEFT,
    GAMEPAD_STICK_RIGHT,
};

vec3 gamepad_get_stick(int stick, const float deadzone = 0.2f)
{
    float gamepadx = GetGamepadAxisMovement(GAMEPAD_PLAYER, stick == GAMEPAD_STICK_LEFT ? GAMEPAD_AXIS_LEFT_X : GAMEPAD_AXIS_RIGHT_X);
    float gamepady = GetGamepadAxisMovement(GAMEPAD_PLAYER, stick == GAMEPAD_STICK_LEFT ? GAMEPAD_AXIS_LEFT_Y : GAMEPAD_AXIS_RIGHT_Y);
    float gamepadmag = sqrtf(gamepadx*gamepadx + gamepady*gamepady);
    
    if (gamepadmag > deadzone)
    {
        float gamepaddirx = gamepadx / gamepadmag;
        float gamepaddiry = gamepady / gamepadmag;
        float gamepadclippedmag = gamepadmag > 1.0f ? 1.0f : gamepadmag*gamepadmag;
        gamepadx = gamepaddirx * gamepadclippedmag;
        gamepady = gamepaddiry * gamepadclippedmag;
    }
    else
    {
        gamepadx = 0.0f;
        gamepady = 0.0f;
    }
    
    return vec3(gamepadx, 0.0f, gamepady);
}

//--------------------------------------

float orbit_camera_update_azimuth(
    const float azimuth, 
    const vec3 gamepadstick_right,
    const bool desired_strafe,
    const float dt)
{
    vec3 gamepadaxis = desired_strafe ? vec3() : gamepadstick_right;
    return azimuth + 2.0f * dt * -gamepadaxis.x;
}

float orbit_camera_update_altitude(
    const float altitude, 
    const vec3 gamepadstick_right,
    const bool desired_strafe,
    const float dt)
{
    vec3 gamepadaxis = desired_strafe ? vec3() : gamepadstick_right;
    return clampf(altitude + 2.0f * dt * gamepadaxis.z, 0.0, 0.4f * PIf);
}

float orbit_camera_update_distance(
    const float distance, 
    const float dt)
{
    float gamepadzoom = 
        IsGamepadButtonDown(GAMEPAD_PLAYER, GAMEPAD_BUTTON_LEFT_TRIGGER_1)  ? +1.0f :
        IsGamepadButtonDown(GAMEPAD_PLAYER, GAMEPAD_BUTTON_RIGHT_TRIGGER_1) ? -1.0f : 0.0f;
        
    return clampf(distance +  10.0f * dt * gamepadzoom, 0.1f, 100.0f);
}

// Updates the camera using the orbit cam controls
void orbit_camera_update(
    Camera3D& cam, 
    float& camera_azimuth,
    float& camera_altitude,
    float& camera_distance,
    const vec3 target,
    const vec3 gamepadstick_right,
    const bool desired_strafe,
    const float dt)
{
    camera_azimuth = orbit_camera_update_azimuth(camera_azimuth, gamepadstick_right, desired_strafe, dt);
    camera_altitude = orbit_camera_update_altitude(camera_altitude, gamepadstick_right, desired_strafe, dt);
    camera_distance = orbit_camera_update_distance(camera_distance, dt);
    
    quat rotation_azimuth = quat_from_angle_axis(camera_azimuth, vec3(0, 1, 0));
    vec3 position = quat_mul_vec3(rotation_azimuth, vec3(0, 0, camera_distance));
    vec3 axis = normalize(cross(position, vec3(0, 1, 0)));
    
    quat rotation_altitude = quat_from_angle_axis(camera_altitude, axis);
    
    vec3 eye = target + quat_mul_vec3(rotation_altitude, position);

    cam.target = (Vector3){ target.x, target.y, target.z };
    cam.position = (Vector3){ eye.x, eye.y, eye.z };
}

//--------------------------------------

void draw_axis(const vec3 pos, const quat rot, const float scale = 1.0f)
{
    vec3 axis0 = pos + quat_mul_vec3(rot, scale * vec3(1.0f, 0.0f, 0.0f));
    vec3 axis1 = pos + quat_mul_vec3(rot, scale * vec3(0.0f, 1.0f, 0.0f));
    vec3 axis2 = pos + quat_mul_vec3(rot, scale * vec3(0.0f, 0.0f, 1.0f));
    
    DrawLine3D(to_Vector3(pos), to_Vector3(axis0), RED);
    DrawLine3D(to_Vector3(pos), to_Vector3(axis1), GREEN);
    DrawLine3D(to_Vector3(pos), to_Vector3(axis2), BLUE);
}

//--------------------------------------

void update_callback(void* args)
{
    ((std::function<void()>*)args)->operator()();
}

int main(void)
{
    //PROFILE_INIT();
  
    // Init Window
    
    const int screen_width = 1280;
    const int screen_height = 720;

    SetConfigFlags(FLAG_VSYNC_HINT);
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(screen_width, screen_height, "raylib [extrapolation]");
    SetTargetFPS(60);
    
    // Camera

    Camera3D camera = { 0 };
    camera.position = (Vector3){ 0.0f, 10.0f, 10.0f };
    camera.target = (Vector3){ 0.0f, 0.0f, 0.0f };
    camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    float camera_azimuth = 0.0f;
    float camera_altitude = 0.4f;
    float camera_distance = 4.0f;
    
    // Ground Plane
    
    Shader ground_plane_shader = LoadShader("./resources/checkerboard.vs", "./resources/checkerboard.fs");
    Mesh ground_plane_mesh = GenMeshPlane(20.0f, 20.0f, 10, 10);
    Model ground_plane_model = LoadModelFromMesh(ground_plane_mesh);
    ground_plane_model.materials[0].shader = ground_plane_shader;
    
    // Character
    
    character character_data;
    character_load(character_data, "./resources/character.bin");
    
    Shader character_shader = LoadShader("./resources/character.vs", "./resources/character.fs");
   
    Mesh curr_mesh = make_character_mesh(character_data);
    Mesh pred_mesh = make_character_mesh(character_data);
    
    Model curr_model = LoadModelFromMesh(curr_mesh);
    Model pred_model = LoadModelFromMesh(pred_mesh);
    
    curr_model.materials[0].shader = character_shader;
    pred_model.materials[0].shader = character_shader;
    
    // Load Animation Data and build Matching Database
    
    database db;
    database_load(db, "./resources/database.bin");
    
    // Pose & Extrapolation Data
    
    int frame_index = db.range_starts(2);
    
    int extrapolation_duration = 60;
    int extrapolation_method = 0;
    bool extrapolation_method_edit = false; 
    float extrapolation_root_halflife = 5.0f;
    float extrapolation_halflife = 0.3f;
    float extrapolation_bounce_halflife = 0.5f;    
    float extrapolation_bounce_strength = 10.0f;

    array1d<vec3> curr_bone_positions = db.bone_positions(frame_index);
    array1d<vec3> curr_bone_velocities = db.bone_velocities(frame_index);
    array1d<quat> curr_bone_rotations = db.bone_rotations(frame_index);
    array1d<vec3> curr_bone_angular_velocities = db.bone_angular_velocities(frame_index);

    array1d<vec3> pred_bone_positions = db.bone_positions(frame_index);
    array1d<vec3> pred_bone_velocities = db.bone_velocities(frame_index);
    array1d<quat> pred_bone_rotations = db.bone_rotations(frame_index);
    array1d<vec3> pred_bone_angular_velocities = db.bone_angular_velocities(frame_index);
    
    array1d<vec3> curr_global_bone_positions(db.nbones());
    array1d<quat> curr_global_bone_rotations(db.nbones());
    array1d<bool> curr_global_bone_computed(db.nbones());
    
    array1d<vec3> pred_global_bone_positions(db.nbones());
    array1d<quat> pred_global_bone_rotations(db.nbones());
    array1d<bool> pred_global_bone_computed(db.nbones());
    
    // Joint Limits
    
    array1d<vec3> reference_positions;
    array1d<quat> reference_rotations;
    array1d<vec3> limit_positions;
    array1d<mat3> limit_rotations;
    array1d<vec3> kdop_axes;
    array2d<float> kdop_limit_mins;
    array2d<float> kdop_limit_maxs;
    
    FILE* f = fopen("resources/limits_kdop.bin", "rb");
    assert(f != NULL);
    
    array1d_read(reference_positions, f);
    array1d_read(reference_rotations, f);
    array1d_read(limit_positions, f);
    array1d_read(limit_rotations, f);
    array1d_read(kdop_axes, f);
    array2d_read(kdop_limit_mins, f);
    array2d_read(kdop_limit_maxs, f);
    
    fclose(f);
    
    // Extrapolator
    
    nnet extrapolator;    
    nnet_load(extrapolator, "./resources/extrapolator.bin");
    
    nnet_evaluation extrapolator_evaluation;
    extrapolator_evaluation.resize(extrapolator);
    
    // Go

    float dt = 1.0f / 60.0f;

    auto update_func = [&]()
    {
        vec3 gamepadstick_right = gamepad_get_stick(GAMEPAD_STICK_RIGHT);

        // Tick frame
        frame_index++; // Assumes dt is fixed to 60fps
        
        // Look-up Next Pose
        curr_bone_positions = db.bone_positions(frame_index);
        curr_bone_velocities = db.bone_velocities(frame_index);
        curr_bone_rotations = db.bone_rotations(frame_index);
        curr_bone_angular_velocities = db.bone_angular_velocities(frame_index);
        
        if (frame_index % extrapolation_duration == 0)
        {
            pred_bone_positions = curr_bone_positions;
            pred_bone_velocities = curr_bone_velocities;
            pred_bone_rotations = curr_bone_rotations;
            pred_bone_angular_velocities = curr_bone_angular_velocities;
        }
        else
        {
            //PROFILE_BEGIN(extrapolation);
          
            extrapolate(
                pred_bone_positions,
                pred_bone_velocities,
                pred_bone_rotations,
                pred_bone_angular_velocities,
                extrapolation_method,
                extrapolation_root_halflife,
                extrapolation_bounce_halflife,
                extrapolation_halflife,
                reference_rotations,
                limit_positions,
                limit_rotations,
                kdop_axes,
                kdop_limit_mins,
                kdop_limit_maxs,
                extrapolation_bounce_strength,
                extrapolator_evaluation,
                extrapolator,
                dt);
          
            //PROFILE_END(extrapolation);
        }
        
        forward_kinematics_full(
            curr_global_bone_positions,
            curr_global_bone_rotations,
            curr_bone_positions,
            curr_bone_rotations,
            db.bone_parents);
        
        forward_kinematics_full(
            pred_global_bone_positions,
            pred_global_bone_rotations,
            pred_bone_positions,
            pred_bone_rotations,
            db.bone_parents);
        
        // Update camera
        
        orbit_camera_update(
            camera, 
            camera_azimuth,
            camera_altitude,
            camera_distance,
            curr_bone_positions(0) + vec3(0, 1, 0),
            gamepadstick_right,
            false,
            dt);

        // Render
        
        BeginDrawing();
        ClearBackground(RAYWHITE);
        
        BeginMode3D(camera);
        
        deform_character_mesh(
            curr_mesh, 
            character_data, 
            curr_global_bone_positions, 
            curr_global_bone_rotations,
            db.bone_parents);
        
        deform_character_mesh(
            pred_mesh, 
            character_data, 
            pred_global_bone_positions, 
            pred_global_bone_rotations,
            db.bone_parents);
        
        DrawModel(curr_model, (Vector3){0.0f, 0.0f, 0.0f}, 1.0f, RAYWHITE);
        DrawModel(pred_model, (Vector3){0.0f, 0.0f, 0.0f}, 1.0f, (Color){ 86, 112, 255, 200 });
        
        // Draw Ground Plane
        
        DrawModel(ground_plane_model, (Vector3){0.0f, -0.01f, 0.0f}, 1.0f, WHITE);
        DrawGrid(20, 1.0f);
        draw_axis(vec3(), quat());
        
        EndMode3D();

        // UI
        
        //---------
        
        float ui_hei = 20;
        
        GuiGroupBox((Rectangle){ 970, ui_hei, 290, 160 }, "extrapolation");

        float float_extrapolation_duration = extrapolation_duration;

        GuiSliderBar(
            (Rectangle){ 1100, ui_hei + 10, 120, 20 }, 
            "duration", 
            TextFormat("%i", extrapolation_duration), 
            &float_extrapolation_duration, 0, 120);
        
        extrapolation_duration = (int)float_extrapolation_duration;
        
        GuiSliderBar(
            (Rectangle){ 1100, ui_hei + 40, 120, 20 }, 
            "root decay halflife", 
            TextFormat("%5.3f", extrapolation_root_halflife), 
            &extrapolation_root_halflife, 0.0, 5.0);
        
        GuiSliderBar(
            (Rectangle){ 1100, ui_hei + 70, 120, 20 }, 
            "decay halflife", 
            TextFormat("%5.3f", extrapolation_halflife), 
            &extrapolation_halflife, 0.0, 1.0);
        
        GuiSliderBar(
            (Rectangle){ 1100, ui_hei + 100, 120, 20 }, 
            "bounce decay halflife", 
            TextFormat("%5.3f", extrapolation_bounce_halflife), 
            &extrapolation_bounce_halflife, 0.0, 1.0);
        
        if (GuiDropdownBox(
            (Rectangle){ 1100, ui_hei + 130, 120, 20 }, 
            "None;Linear;Decay;Clamp;Bounce;Extrapolator",
            &extrapolation_method,
            extrapolation_method_edit))
        {
            extrapolation_method_edit = !extrapolation_method_edit;
        }
        
        EndDrawing();

    };

#if defined(PLATFORM_WEB)
    std::function<void()> u{update_func};
    emscripten_set_main_loop_arg(update_callback, &u, 0, 1);
#else
    while (!WindowShouldClose())
    {
        update_func();
    }
#endif

    // Unload stuff and finish
    UnloadModel(curr_model);
    UnloadModel(pred_model);
    UnloadModel(ground_plane_model);
    UnloadShader(character_shader);
    UnloadShader(ground_plane_shader);

    CloseWindow();

    return 0;
}