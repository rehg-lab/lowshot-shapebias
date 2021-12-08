import os
import time
import argparse
import json

def get_id_info(path, dataset_type):
    if dataset_type == "modelnet":
        category = path.split("/")[-3]
        obj_id = path.split("/")[-1][:-4]
        return category, obj_id

    if dataset_type == "shapenet":
        category = path.split("/")[-4]
        obj_id = path.split("/")[-3]
        return category, obj_id

    if dataset_type == "toys":
        category = path.split("/")[-3]
        obj_id = path.split("/")[-2]
        return category, obj_id

def main():
    
    # set paths
    blender_script_path = os.path.abspath("generate.py")
    blendfile_path = os.path.abspath("empty_scene.blend")
    
    # load arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, help="start point")
    parser.add_argument("--end", type=int, help="end point")
    parser.add_argument("--gpu_idx", type=int, help="end point")
    parser.add_argument("--demo", type=int, default=0, help="whether to do demo render or not")
    parser.add_argument("--input_path", type=str, help="dataset input path")
    parser.add_argument("--param_path", type=str, help="datagen param json path")
    parser.add_argument("--output_path", type=str, help="render output path")
    parser.add_argument("--blender_path", type=str, help="path to blender")
    parser.add_argument(
        "--dataset_type", type=str, help="either <modelnet>, <shapenet> or <toys>"
    )

    args = parser.parse_args()

    print("Start index: {}".format(args.start))
    print("End index: {}".format(args.end))
    print("GPU index: {}".format(args.gpu_idx))
    print("Input path: {}".format(args.input_path))
    print("Output path: {}".format(args.output_path))
    print("Param path: {}".format(args.param_path))
    print("Dataset type: {}".format(args.dataset_type))

    data_json_path = os.path.join("jsons", "{}_dict.json".format(args.dataset_type))

    with open(data_json_path, "r") as f:
        data_dict = json.load(f)
    
    # collect paths
    paths = []
    for category, object_paths in data_dict.items():
        object_paths = [
            os.path.join(args.input_path, category, x) for x in object_paths
        ]
        paths.extend(object_paths)
    
    ## demo conditional statement
    if args.demo == 1:
        paths = [x for x in paths if 'airplane_010' in x]
    
    i = 0
    global_time = time.time()
    gpu_idx = args.gpu_idx
    
    # render each object
    for idx, fpath in enumerate(paths[args.start : args.end]):
        start_time = time.time()
        
        render_output_path = os.path.join(
                os.path.abspath(args.output_path), 
                *get_id_info(fpath, args.dataset_type)
            )
            
        cmd = (
            f"{args.blender_path} -noaudio --background {blendfile_path} --python {blender_script_path} -- "
            f"--dataset_type {args.dataset_type} "
            f"--input_path {fpath} "
            f"--output_path {render_output_path} "
            f"--param_path {args.param_path} "
            f"--gpu {args.gpu_idx}" + "1>/tmp/out.txt")

        os.system(cmd)

        i += 1
        print(
            "--- {:.2f} seconds for obj {} [{}/{}] ---".format(
                time.time() - start_time, fpath, args.start + i, args.end
            )
        )
        
        if i % 50 == 0:
            print(
                "Total time since start is {:.2f} minutes".format(
                    (time.time() - global_time) / 60
                )
            )

    total_time = time.time() - global_time
    print("Total time {:.2f}".format(total_time / 60))


if __name__ == "__main__":
    main()
