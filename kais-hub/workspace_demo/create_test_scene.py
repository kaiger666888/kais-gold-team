"""Blender test scene generator — creates a simple scene and renders it.

Run inside Blender container:
    blender -b --python create_test_scene.py -- --output /workspace/.done/<task_id>/frame_####.png
"""
import bpy
import sys

# Parse custom args after '--'
argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]
else:
    argv = []

output_path = "/workspace/.done/test/frame_####"
for i, arg in enumerate(argv):
    if arg == "--output" and i + 1 < len(argv):
        output_path = argv[i + 1]

# Clean default scene
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

# Add a torus knot
bpy.ops.mesh.primitive_torus_add(
    location=(0, 0, 0),
    major_radius=1.5,
    minor_radius=0.4,
)

torus = bpy.context.active_object
torus.name = "Torus"
mat = bpy.data.materials.new(name="GlowMat")
mat.use_nodes = True
bsdf = mat.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Base Color"].default_value = (0.1, 0.4, 0.9, 1.0)
bsdf.inputs["Metallic"].default_value = 0.8
bsdf.inputs["Roughness"].default_value = 0.2
torus.data.materials.append(mat)

# Add a plane (floor)
bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, -2))
floor = bpy.context.active_object
floor.name = "Floor"
floor_mat = bpy.data.materials.new(name="FloorMat")
floor_mat.use_nodes = True
floor_bsdf = floor_mat.node_tree.nodes["Principled BSDF"]
floor_bsdf.inputs["Base Color"].default_value = (0.15, 0.15, 0.15, 1.0)
floor_bsdf.inputs["Roughness"].default_value = 0.6
floor.data.materials.append(floor_mat)

# Add a point light
bpy.ops.object.light_add(type="POINT", location=(3, -2, 4))
light = bpy.context.active_object
light.data.energy = 2000
light.data.color = (1.0, 0.95, 0.85)

# Add camera
bpy.ops.object.camera_add(location=(4, -4, 3))
cam = bpy.context.active_object
cam.rotation_euler = (1.1, 0, 0.8)
bpy.context.scene.camera = cam

# Cycles render settings
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.samples = 64
scene.render.resolution_x = 960
scene.render.resolution_y = 540
scene.render.filepath = output_path
scene.render.image_settings.file_format = "PNG"

print(f"Scene ready — rendering to {output_path}")
bpy.ops.render.render(write_still=True)
print("Render complete.")
