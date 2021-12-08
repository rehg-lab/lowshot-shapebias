import bpy
import numpy as np
import sys
import os
import json
import argparse


fpath = bpy.data.filepath
dir_path = "/".join(fpath.split("/")[:-1])
sys.path.append(dir_path)
print(dir_path)

import utils


def main():

    ### read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--param_path", type=str)
    parser.add_argument("--dataset_type", type=str)
    parser.add_argument("--gpu", type=int, help="number of views to be rendered")

    argv = sys.argv[sys.argv.index("--") + 1 :]
    args = parser.parse_args(argv)
    
    ### load datagen params
    with open(args.param_path, "r") as load_file:
        data_gen_params = json.load(load_file)

    scn = bpy.context.scene

    ### Set GPU
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "CUDA"  # or "OPENCL"
    # Set the device and feature set
    scn.cycles.device = "GPU"
    prefs.get_devices()  # get_devices() to let Blender detect GPU device
    devices = prefs.devices
    for i in np.arange(len(devices)):
        if i == int(args.gpu):
            prefs.devices[int(args.gpu)]["use"] = 1
        else:
            prefs.devices[i]["use"] = 0
    
    ### apply settings
    render_parameters = data_gen_params["render_parameters"]
    utils.apply_settings(scn, render_parameters)

    ### load object
    obj = utils.load_obj(scn, args.input_path, args.dataset_type)

    # clear normals
    bpy.ops.mesh.customdata_custom_splitnormals_clear()

    # recompute normals
    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.editmode_toggle()

    ### rescaling object to fit well in unit cube
    vertices = np.array([v.co for v in obj.data.vertices])
    obj.scale = obj.scale * 0.45 / np.max(np.abs(vertices))
    bpy.ops.object.transform_apply(scale=True, location=True, rotation=True)

    ### adding light
    light_params = data_gen_params["light_parameters"]
    utils.make_area_lamp(
        light_params["area_light_location"],
        light_params["area_light_rotation"],
        size_x=light_params["area_size_x"],
        size_y=light_params["area_size_y"],
        strength=light_params["area_strength"],
        temp=light_params["light_temperature"],
    )

    ### output paths
    tree = bpy.context.scene.node_tree
    segmentation_output = tree.nodes["Segmentation"]
    image_output = tree.nodes["Image_Output"]
    depth_output = tree.nodes["Depth_Output"]

    segmentation_output.base_path = os.path.join(args.output_path, "segmentation_output")
    image_output.base_path = os.path.join(args.output_path, "image_output")
    depth_output.base_path = os.path.join(args.output_path, "depth_output")

    ### camera settings
    bpy.data.objects["Camera"].data.sensor_width = data_gen_params["camera"]["sensor_size_mm"]
    bpy.data.objects["Camera"].data.lens = data_gen_params["camera"]["focal_length_mm"]
    bpy.data.objects["Camera"].location = (0,0, data_gen_params["camera"]["distance_units"])
    
    ## this can be used for a debug render before any 
    ## additional transformations are applied

    #scn.frame_current = 10000
    #bpy.ops.render.render()

    if data_gen_params["gen_params"]["debug"]:
        rotations = np.linspace(0, 360, num=16)
        elevations = np.ones(25)*45

    else:
        rotations = np.random.uniform(low=data_gen_params['gen_params']['azim_range'][0],
                                  high=data_gen_params['gen_params']['azim_range'][1],
                                  size=data_gen_params['gen_params']['n_points'])

        elevations = np.random.uniform(low=data_gen_params['gen_params']['elev_range'][0],
                                       high=data_gen_params['gen_params']['elev_range'][1],
                                       size=data_gen_params['gen_params']['n_points'])

    count = 0

    for i, (rot, el) in enumerate(zip(rotations, elevations)):

        utils.apply_rot(obj, "Y", rot)
        utils.apply_rot(obj, "X", el)

        dg = bpy.context.evaluated_depsgraph_get()
        dg.update()

        scn.frame_current = count
        bpy.ops.render.render()

        utils.reset_rot(obj)

        dg = bpy.context.evaluated_depsgraph_get()
        dg.update()

        count += 1


if __name__ == "__main__":
    main()
