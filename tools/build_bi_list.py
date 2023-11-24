#!/usr/bin/env python3

import os
import sys
import json

def error_exit(msg):
    print(msg)
    sys.exit(1)

def check_file_exists(file):
    if not os.path.isfile(file):
        error_exit(f"File not found: {file}")

def check_file_readable(file):
    check_file_exists(file)
    if not os.access(file, os.R_OK):
        error_exit(f"File not readable: {file}")

def check_dir_exists(dir):
    if not os.path.isdir(dir):
        error_exit(f"Directory not found: {dir}")

def isBlank(string):
    return not (string and string.strip())

def load_bi(file):
    check_file_readable(file)
    lines = []
    with open(file, 'r') as f:
        lines = f.readlines()

    # Process each non empty line and split the = separated key value pairs
    bi = {}
    for line in lines:
        # ignore empty lines
        if line is not isBlank(line):
            if line.strip():
                key, value = line.split('=')
                bi[key.strip()] = value.strip()

    if 'CTPO_BUILD' not in bi:
        error_exit(f"CTPO_BUILD not found in {file}")

    return bi['CTPO_BUILD'], bi

def extract_file_details(file):
    # BuildDetails/20230704/cuda_tensorflow_pytorch_opencv-11.8.0_2.12.0_2.0.1_4.7.0-20230704/BuildInfo.txt

    # Split the file path into components
    path_split = file.split('/')
    if len(path_split) != 4:
        error_exit(f"Invalid file path: {file} [path: {path_split}]]")
    
    release = path_split [1]
    build_info = path_split [2]

    path_value = os.path.join(path_split[0], path_split[1], path_split[2])

    components = build_info.split('-')
    headers = components[0].split('_')
    versions = components[1].split('_')
    release_check = components[2]

    if len(headers) != len(versions):
        error_exit(f"Invalid build info: {file}")

    # Split the build info into components
    cuda_version = None
    if 'cuda' in build_info:
        cuda_location = headers.index('cuda')
        cuda_version = versions[cuda_location]

    tensorflow_version = None
    if 'tensorflow' in build_info:
        tensorflow_location = headers.index('tensorflow')
        tensorflow_version = versions[tensorflow_location]

    pytorch_version = None
    if 'pytorch' in build_info:
        pytorch_location = headers.index('pytorch')
        pytorch_version = versions[pytorch_location]

    opencv_version = None
    if 'opencv' in build_info:
        opencv_location = headers.index('opencv')
        opencv_version = versions[opencv_location]

    if release_check != release:
        error_exit(f"Release mismatch: {file}")

    return path_value, components[0], components[1], release, cuda_version, tensorflow_version, pytorch_version, opencv_version

def get_dir_list(dir):
    dir_list = []
    check_dir_exists(dir)
    dir_list = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
#    print(f" -- Found {len(dir_list)} directories in {dir}")
    return dir_list

def process_BuildDetails(dir):
    print(f" -- Getting release list from {dir}")
    release_list = get_dir_list(dir)

    all_build_info = {}
    for release in release_list:
        print(f" -- Getting build info list for release: {release}")
        build_info_list = get_dir_list(os.path.join(dir, release))
        for build_info_dir in build_info_list:
            print(f" |  -- Getting build info for build: {build_info_dir}")
            build_info_file = os.path.join(dir, release, build_info_dir, 'BuildInfo.txt')
            if os.path.isfile(build_info_file):
                path_value, build_comp, version_comp, release_version, cuda_version, tensorflow_version, pytorch_version, opencv_version = extract_file_details(build_info_file)
                build_type, bi = load_bi(build_info_file)
                if build_type not in all_build_info:
                    all_build_info[build_type] = {}
                if build_comp not in all_build_info[build_type]:
                    all_build_info[build_type][build_comp] = {}
                if release_version not in all_build_info[build_type][build_comp]:
                    all_build_info[build_type][build_comp][release_version] = {}
                if version_comp not in all_build_info[build_type][build_comp][release_version]:
                    all_build_info[build_type][build_comp][release_version][version_comp] = {}
                all_build_info[build_type][build_comp][release_version][version_comp] = { 'path': path_value, 'cuda_version': cuda_version, 'tensorflow_version': tensorflow_version, 'pytorch_version': pytorch_version, 'opencv_version': opencv_version, 'bi': bi }
            else:
                print(f" |!!!! Build info file NOT found: {build_info_file}   !!!!!!!!!!!")
    
    return all_build_info

def get_wanted_columns(build_comp):
# Bi structure:
# CTPO_FROM: nvidia/cuda:11.8.0-devel-ubuntu22.04
# CTPO_BUILD=GPU
# CTPO_TENSORFLOW_VERSION=2.12.0
# CTPO_PYTORCH_VERSION=None
# CTPO_CUDA_VERSION=11.8.0
# CTPO_OPENCV_VERSION=4.7.0
# CTPO_RELEASE=20230704
# FOUND_UBUNTU=22.04
# OpenCV_Built=4.7.0
# CUDA_found=11.8.89
# cuDNN_found=8.6.0
# TensorFlow_Built=2.12.0
# FFmpeg_Built=5.1.2
# Torch_Built=2.0.0a0+gite9ebda2
# TorchVision_Built=0.15.2a0+fa99a53
# TorchAudio_Built=2.0.2+31de77d
# TorchData_Built=0.6.1+e1feeb2
# TorchText_Built=0.15.2a0+4571036
 
    if build_comp == 'tensorflow_opencv':
        return ['TensorFlow', 'OpenCV', 'FFmpeg', 'Ubuntu'], { 'TensorFlow': 'TensorFlow_Built', 'OpenCV': 'OpenCV_Built', 'FFmpeg': 'FFmpeg_Built', 'Ubuntu': 'FOUND_UBUNTU' }
    elif build_comp == 'pytorch_opencv':
        return ['PyTorch', 'OpenCV', 'FFmpeg', 'Ubuntu'], { 'PyTorch': 'Torch_Built', 'OpenCV': 'OpenCV_Built', 'FFmpeg': 'FFmpeg_Built', 'Ubuntu': 'FOUND_UBUNTU' }
    elif build_comp == 'tensorflow_pytorch_opencv':
        return ['TensorFlow', 'PyTorch', 'OpenCV', 'FFmpeg', 'Ubuntu'], { 'TensorFlow': 'TensorFlow_Built', 'PyTorch': 'Torch_Built', 'OpenCV': 'OpenCV_Built', 'FFmpeg': 'FFmpeg_Built', 'Ubuntu': 'FOUND_UBUNTU' }
    elif build_comp == 'cuda_tensorflow_opencv':
        return ['CUDA', 'cuDNN', 'TensorFlow', 'OpenCV', 'FFmpeg', 'Ubuntu'], { 'CUDA': 'CUDA_found', 'cuDNN': 'cuDNN_found', 'TensorFlow': 'TensorFlow_Built', 'OpenCV': 'OpenCV_Built', 'FFmpeg': 'FFmpeg_Built', 'Ubuntu': 'FOUND_UBUNTU' }
    elif build_comp == 'cuda_pytorch_opencv':
        return ['CUDA', 'cuDNN', 'PyTorch', 'OpenCV', 'FFmpeg', 'Ubuntu'], { 'CUDA': 'CUDA_found', 'cuDNN': 'cuDNN_found', 'PyTorch': 'Torch_Built', 'OpenCV': 'OpenCV_Built', 'FFmpeg': 'FFmpeg_Built', 'Ubuntu': 'FOUND_UBUNTU' }
    elif build_comp == 'cuda_tensorflow_pytorch_opencv':
        return ['CUDA', 'cuDNN', 'TensorFlow', 'PyTorch', 'OpenCV', 'FFmpeg', 'Ubuntu'], { 'CUDA': 'CUDA_found', 'cuDNN': 'cuDNN_found', 'TensorFlow': 'TensorFlow_Built', 'PyTorch': 'Torch_Built', 'OpenCV': 'OpenCV_Built', 'FFmpeg': 'FFmpeg_Built', 'Ubuntu': 'FOUND_UBUNTU' }
    else:
        error_exit(f" Unknown build type: {build_comp}")

def generate_markdown(abi):
    print(" -- Generating markdown")

    title = "# Available Builds\n"
    toc = ""
    body = ""

    for build_type in sorted(abi.keys()):
        body += f"## {build_type}\n"
        toc += f"  - [{build_type}](#{build_type})\n"
        for build_comp in sorted(abi[build_type].keys()):
            body += f"### {build_comp}\n"
            toc += f"    - [{build_comp}](#{build_comp})\n"
            wanted_cols, wanted_match = get_wanted_columns(build_comp)
            wanted = ['Docker tag'] + wanted_cols
            body += f"| {' | '.join(wanted)} |\n"
            body += f"| {' | '.join(['---' for i in range(len(wanted))])} |\n"
            for release_version in sorted(abi[build_type][build_comp].keys(), reverse=True):
                files = {}
                for version_comp in sorted(abi[build_type][build_comp][release_version].keys()):
                    print(f"  - {build_type} {build_comp} {release_version} {version_comp}")
                    path = abi[build_type][build_comp][release_version][version_comp]['path']

                    dockerfile = os.path.join(path, 'Dockerfile')
                    check_file_readable(dockerfile)

                    ffmpeg_txt = os.path.join(path, 'FFmpeg--Details.txt')
                    check_file_readable(ffmpeg_txt)
                    files['FFmpeg'] = ffmpeg_txt

                    opencv_txt = os.path.join(path, 'OpenCV--Details.txt')
                    check_file_readable(opencv_txt)
                    files['OpenCV'] = opencv_txt

                    torch_txt = os.path.join(path, 'PyTorch--Details.txt')
                    check_file_readable(torch_txt)
                    files['PyTorch'] = torch_txt

                    system_txt = os.path.join(path, 'System--Details.txt')
                    check_file_readable(system_txt)
                    files['Ubuntu'] = system_txt

                    tensorflow_txt = os.path.join(path, 'TensorFlow--Details.txt')
                    check_file_readable(tensorflow_txt)
                    files['TensorFlow'] = tensorflow_txt

                    bi = abi[build_type][build_comp][release_version][version_comp]['bi']

                    line_cols = []
                    for wanted_col in wanted:
                        if wanted_col == wanted[0]:
                            line_cols.append(f"[{version_comp}-{release_version}]({dockerfile})")
                        elif wanted_match[wanted_col] not in bi:
                            error_exit(f"Column {wanted_col} not found in {bi}")
                        else:
                            if wanted_col not in files:
                                line_cols.append(bi[wanted_match[wanted_col]])
                            else:
                                line_cols.append(f"[{bi[wanted_match[wanted_col]]}]({files[wanted_col]})")

                    body += f"| {' | '.join(line_cols)} |\n"
            body += "\n"

    return title + toc + body

def main():
    if len(sys.argv) != 3:
        error_exit("Usage: build_bi_list.py <BuildDetails dir> <output.md>")
    
    dir = sys.argv[1]
    mdfile = sys.argv[2]

    abi = process_BuildDetails(dir)
# Structure of abi: 
#     abi[build_type][build_comp][release_version][version_comp] = { 'cuda_version': cuda_version, 'tensorflow_version': tensorflow_version, 'pytorch_version': pytorch_version, 'opencv_version': opencv_version, 'bi': bi }
# ex: abi[CPU][tensorflow_pytorch_opencv][20230704][2.12.0_2.0.1_4.7.0] = { "cuda_version": null, "tensorflow_version": "2.12.0", "pytorch_version": "2.0.1", "opencv_version": "4.7.0", "bi": { "CTPO_FROM": "ubuntu:22.04", ...
#    print(json.dumps(abi, indent=2))

    md = generate_markdown(abi)

    with open(mdfile, 'w') as f:
        f.write(md)
        print(f" -- Markdown written to {mdfile}")
    check_file_readable(mdfile)

    print("Done")
    sys.exit(0)

if __name__ == "__main__":
    main()
