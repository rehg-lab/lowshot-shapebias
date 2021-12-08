import bpy
import numpy as np
from mathutils import Matrix


def make_area_lamp(location, rotation, size_x=0, size_y=0, strength=10, temp=5000):
    """
    inputs:
        location  - (x,y,z) location of area light
        rotation  - (x,y,z) rotation of area light in radians
        size_x    - size in x direction of area light
        size_y    - size in y direction of area light
        strength  - strength (brightness) of area light
        temp      - color temperature in Kelvin of area light
    """

    # initialize ligth and set size
    bpy.context.view_layer.objects.active = None
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.light_add(type="AREA", location=location, rotation=rotation)

    lamp = bpy.data.lights[bpy.context.active_object.name]
    lamp.shape = "RECTANGLE"
    lamp.size = size_x
    lamp.size_y = size_y

    # create blackbody nodes for color temperature control
    lamp.use_nodes = True
    nodes = lamp.node_tree.nodes

    for node in nodes:
        nodes.remove(node)

    node_blackbody = nodes.new(type="ShaderNodeBlackbody")
    node_emission = nodes.new(type="ShaderNodeEmission")
    node_output = nodes.new(type="ShaderNodeOutputLight")

    node_output.location[1] = 400
    node_emission.location[1] = 200

    lamp.node_tree.links.new(node_blackbody.outputs[0], node_emission.inputs[0])
    lamp.node_tree.links.new(node_emission.outputs[0], node_output.inputs[0])

    node_emission.inputs[1].default_value = strength
    node_blackbody.inputs[0].default_value = temp
    lamp_obj = bpy.data.objects[lamp.name]
    lamp_obj.select_set(False)


def apply_rot(obj, axis, angle):
    """
    inputs:
        obj   - bpy.data.objects object to rotate globally
        axis  - axis along which to rotate - 'X', 'Y', or 'Z'
        angle - angle in global coordinates along axis in degrees
    """

    rot_mat = Matrix.Rotation(np.radians(angle), 4, axis)

    o_loc, o_rot, o_scl = obj.matrix_world.decompose()
    o_loc_mat = Matrix.Translation(o_loc)
    o_rot_mat = o_rot.to_matrix().to_4x4()
    o_scl_mat = (
        Matrix.Scale(o_scl[0], 4, (1, 0, 0))
        @ Matrix.Scale(o_scl[1], 4, (0, 1, 0))
        @ Matrix.Scale(o_scl[2], 4, (0, 0, 1))
    )

    # assemble the new matrix
    obj.matrix_world = o_loc_mat @ rot_mat @ o_rot_mat @ o_scl_mat


def reset_rot(obj):
    obj.rotation_euler = (0, 0, 0)


def load_obj(scn, path, dataset_type):
    """
    Loads an object for either ShapeNet, ModelNet or Toys

    Inputs:
        scn - bpy.context.scene object
        path - absolute path to load from
        dataset_type - "toys", "modelnet" or "shapenet" string
                       used to determine initial transform after loading
    """

    if dataset_type == "toys":

        with bpy.data.libraries.load(path, link=False) as (data_from, data_to):
            data_to.objects = [name for name in data_from.objects]

        for obj in data_to.objects:
            if obj is not None:
                scn.collection.objects.link(obj)

        obj = [obj for obj in bpy.data.objects if (obj.name != "Camera")][0]
        obj.name = "object"

        obj.rotation_mode = "XYZ"
        obj.rotation_euler = (np.radians(-90), 0, 0)

        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
        obj.location = (0, 0, 0)
        bpy.ops.object.transform_apply(scale=True, location=True, rotation=True)

        return obj

    if dataset_type == "modelnet":
        bpy.ops.import_scene.obj(filepath=path)

        obj = [obj for obj in bpy.data.objects if (obj.name != "Camera")][0]
        obj.name = "object"

        obj.rotation_mode = "XYZ"
        obj.rotation_euler = (np.radians(-90), np.radians(180), 0)

        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
        obj.location = (0, 0, 0)
        bpy.ops.object.transform_apply(scale=True, location=True, rotation=True)

        return obj

    if dataset_type == "shapenet":
        bpy.ops.import_scene.obj(filepath=path)

        obj = [obj for obj in bpy.data.objects if (obj.name != "Camera")][0]
        obj.name = "object"

        obj.rotation_mode = "XYZ"
        obj.rotation_euler = (0, np.radians(180), 0)

        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
        obj.location = (0, 0, 0)
        bpy.ops.object.transform_apply(scale=True, location=True, rotation=True)

        return obj


def apply_settings(scn, render_parameters):
    scn.render.resolution_x = render_parameters["resolution"]
    scn.render.resolution_y = render_parameters["resolution"]
    scn.render.resolution_percentage = render_parameters["resolution_percentage"]
    scn.view_layers["RenderLayer"].samples = render_parameters["render_samples"]
    scn.cycles.debug_use_spatial_splits = render_parameters["use_spatial_splits"]
    scn.cycles.max_bounces = render_parameters["max_bounces"]
    scn.cycles.min_bounces = render_parameters["min_bounces"]
    scn.cycles.transparent_max_bounces = render_parameters["transparent_max_bounces"]
    scn.cycles.transparent_min_bounces = render_parameters["transparent_min_bounces"]
    scn.cycles.glossy_bounces = render_parameters["glossy_bounces"]
    scn.cycles.transmission_bounces = render_parameters["transmission_bounces"]
    scn.render.use_persistent_data = render_parameters["use_persistent_data"]
    scn.render.tile_x = render_parameters["render_tile_x"]
    scn.render.tile_y = render_parameters["render_tile_y"]
    scn.cycles.caustics_refractive = render_parameters["use_caustics_refractive"]
    scn.cycles.caustics_reflective = render_parameters["use_caustics_reflective"]
    scn.cycles.device = render_parameters["rendering_device"]
    scn.render.image_settings.color_mode = render_parameters["color_mode"]
    scn.view_layers["RenderLayer"].cycles.use_denoising = render_parameters[
        "use_denoising"
    ]
    scn.view_layers["RenderLayer"].cycles.denoising_radius = render_parameters[
        "denoising_radius"
    ]
    scn.cycles.film_transparent = render_parameters["use_film_transparent"]
    scn.world.use_nodes = True
    bpy.data.worlds["World"].node_tree.nodes[1].inputs["Color"].default_value = (
        1,
        1,
        1,
        1,
    )
    bpy.data.worlds["World"].node_tree.nodes[1].inputs["Strength"].default_value = 0.25
