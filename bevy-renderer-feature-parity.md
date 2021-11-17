# bevy pipelined-rendering Features

## STUFF

* when modular-rendering is merged:
  * make a PR for visibility branch
  * rework transparency / depth prepass branch on top of visibility
    * [DONE] add AlphaMode as a member of StandardMaterial
    * [DONE] add EntityPhaseItem trait for entity function
    * [DONE] fix da bugs
    * maybe compare reusing the main standard material bindings instead of creating alpha mask depth prepass specific ones

* when pipeline specialisation is ready:
  * MSAA
  * fix double-sided
  * vertex tangents and normal maps

* clustered forward rendering
  * [DONE] support disabling shadow mapping
    * [DONE] Refactor ViewLights into ViewLightEntities, ViewLightsUniformOffset, and ViewShadowBindings
    * [DONE] If shadow mapping is disabled, then clear visible entities
  * update cluster bounds
    * per camera
  * [WIP] allocate lights to clusters
    * for each camera
      * for each cluster
        * for each light
          if light intersects cluster
            push entity onto cluster lights
  * list of all point lights
    * extract all the point lights
    * map Entity -> index in array in uniform
  * list of points lights in clusters concatenated
    * extract cluster visible point lights
    * prepare by iterating the clusters in order and appending them with the bookkeeping for getting the offset first and writing the offset and count to the cluster offset and count uniform
  * cluster offset and count pointer
  * ~~put lights in a storage buffer~~

* mark the tracy frame as done just after the swapchain image is dropped
  * just add info!(tracy.frame_marker = true) at the end of App::update
  * need to figure out how to filter the event properly

* fit directional light projection to the meshes in the scene
* add visibility_culling bool to views (cameras, lights)

* left-handed cube mapping
* PCF
* PCSS

* update SSAO
* dynamic sky

* post-processing
  * HDR
  * bloom
  * fog

## Feature Parity with 0.5

This is a checklist of features to support on top of the pipelined-rendering branch in order for it to have feature parity with bevy 0.5. The purpose of this is to not incur feature regressions and allow people to mostly just port over and continue from where they were.

* port all examples
* update docs
* 2d
  * 
* 3d
  * DONE Choose a single perspective projection - infinite reverse rh
  * DONE gltf model support - @superdump has a pipelined-gltf branch
  * PR - MSAA
  * PR - normal maps - either implement as before with some form of shader preprocessor or support pre-calculated/on-the-fly vertex tangents
  * DONE clear color
  * wireframe
  * CART support flexible/custom vertex attributes
  * CART hot shader reloading
  * CART shader defs? or a similar pattern?
  * CART shader custom material - MeshBundle without StandardMaterial? or StarToaster suggested a way of having the StandardMaterial available but being able to write a custom shader to use it
* ui
  * No support currently
  * DONE bevy-egui ported

## Beyond Renderer Feature Parity

* HDR
* Post-processing stack - bloom, depth of field, vignette, colour grading, tone mapping, motion blur, grain, exposure, etc
  * Exponential height / depth fog - @superdump has something that needs debugging
* DONE Shadow casters / shadow receivers / enable/disable shadows on lights
* Cascade shadow maps
* mipmaps
* transparency - just implement WBOIT?
* ambient occlusion - @superdump has implemented Screen Space Ambient Occlusion
* Global Illumination - someone is working on voxel cone tracing. SDFGI like godot 4, or DDGI (ray tracing) are good options
* Reflections
* Sub-Surface Scattering
* Physically-based sky with dynamic time of day using a directional light - @superdump has something that needs porting
* Volumetric fog
* Clouds
* Weather