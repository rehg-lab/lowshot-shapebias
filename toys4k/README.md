# Toys4K 3D Object Dataset
![myimage](https://user-images.githubusercontent.com/13421307/144886151-2d04660f-816b-4f40-9b7e-ed1851c1f2c6.gif)

This directory contains instructions on how to download the 3D assets for the Toys4K dataset and render them into training data.

## Dependencies
### Blender
Download Blender 2.80 for your operating system from the [official Blender release](https://download.blender.org/release/Blender2.80/).

For **MacOS** the default installation path is `/Applications/Blender/blender.app/Contents/MacOS/blender`. 

For **Linux** it involves extracting a `blender-2.80-linux-glibc217-x86_64.tar.bz2` which makes the path `<path to extracted tarball>/blender-2.80-linux-glibc217-x86_64/blender`.

Depending on your OS, add the appropriate path in the rendering Bash scripts `renderer/render_demo.sh` and `renderer/render_toys.sh` under the variable `blender_path`.
### Python
The simplest way to make sure all the right Python dependencies are in place is with [conda](https://docs.conda.io/en/latest/miniconda.html).
```bash
conda env create -f environment.yml
conda activate blender_datagen
```

## View Demo Renders - Render Your Own Data
To generate a set of demo renders to test your install and our code run the following commands
```bash
cd renderer
bash render_demo.sh
```
After this has completed, you should see the following images under `renderer/demo_output/airplane/airplane_010/image_output`
![image](https://user-images.githubusercontent.com/13421307/144892070-2ded709d-f380-488b-9e35-4d805fb234e3.png)

## Downloading Toys4K
To download the dataset, please fill out the [following form](https://forms.gle/w7Zf82umwaKxr9L7A). You will get access to:
```
toys4k_blend_files.zip - blend files to use for rendering
toys4k_obj_files.zip - extracted obj files without materials to use for 3D geometry processing-type tasks
toys4k_point_clouds.zip - extracted surface point clouds and normals
toys_blend_sample_renders.zip - sample renders from the blend files
toys_obj_sample_renders.zip - sample renders from the obj files
```

## Creating Training Data
### Toys 4K
1. Download the Toys `.blend` files archive as specified in the instructions above and unzip it.
2. Set the appropriate paths in `renderer/render_toys.sh` and run `bash render_toys.sh`
### ModelNet40
1. Download the manually aligned modelnet and unit 
```
wget https://lmb.informatik.uni-freiburg.de/resources/datasets/ORION/modelnet40_manually_aligned.tar
```
2. Convert to `.obj` files (required for Blender rendering). Expect this to take a few hours to complete.
```
python util_scripts/convert_to_obj.py --src_path=<path to untarred directory> --dest_path=<path to output the converted files>
```
3. Set the appropriate paths in `renderer/render_modelnet.sh` and run `bash render_modelnet.sh`

### ShapeNetCore.v2
1. Download ShapeNetCore.v2 from the [official webpage](https://shapenet.org/).
2. Set the appropriate paths in `renderer/render_shapenet.sh` and run `bash render_shapenet.sh`

## Notes
* Some objects have been removed from the inital set of renders/point clouds to allow for this release.
* You may need to explicitly enable CUDA and GPUs at Edit/Preferences/System in Blender in order to do GPU-based rendering.
