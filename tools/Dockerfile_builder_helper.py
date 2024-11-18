#!/usr/bin/env python3

import sys
import os
import os.path
import shutil

import argparse

def error_exit(txt):
    print("[ERROR] " + txt)
    sys.exit(1)

def isBlank (myString):
    return not (myString and myString.strip())

def isNotBlank (myString):
    return bool(myString and myString.strip())

#####

def parse_build(build):
    tensorflow = False
    if 'tensorflow' in build:
        tensorflow = True
    
    pytorch = False
    if 'pytorch' in build:
        pytorch = True
    
    cuda = False
    if 'cuda' in build:
        cuda = True
    
    return tensorflow, pytorch, cuda

def parse_tag(tag, tensorflow, pytorch, cuda, cuda_arch):
    # Tag order: [cuda_version] [tensorflow_version] [pytorch_version] opencv_version
    parts = tag.split('_')

    check = len(parts)
    count = 0

    cuda_version = None
    if cuda is True:
        cuda_version = parts.pop(0)
        count += 1

    dnn_used = None
    if cuda_version is not None:
        if cuda_version not in cuda_arch:
            error_exit(f"Error: Unable to find a match for cuda version {cuda_version} in CUDA versions {cuda_arch}")
        dnn_used = cuda_arch[cuda_version]

    tensorflow_version = None
    if tensorflow is True:
        tensorflow_version = parts.pop(0)
        count += 1

    pytorch_version = None
    if pytorch is True:
        pytorch_version = parts.pop(0)
        count += 1

    opencv_version = parts.pop(0)
    count += 1

    if count != check:
        error_exit(f"Error: Invalid tag ({tag}) -- should have contained {check} parts, but found {count}")

    return tensorflow_version, pytorch_version, cuda_version, opencv_version, dnn_used

#####

def slurp_file(filename):
    txt = None
    with open(filename, 'r') as f:
        txt = f.read()

    if txt is None:
        error_exit(f"Error: Could not read input file ({filename})")

    return txt

##

def find_line_core(text, search):
    lines = text.splitlines()
    for i in range(len(lines)):
        if search in lines[i]:
            return lines[i], "\n".join(lines[:i]), "\n".join(lines[i+1:])
    return None, None, None

##

def find_line(text, search):
    (line, before, after) = find_line_core(text, search)
    if line is None:
        error_exit(f"Error: Could not find {search} in Dockerfile")

    return line, before, after

##

def replace_line(text, search, replace):
    (line, before, after) = find_line(text, search)
    return f"{before}\n{replace}\n{after}"

############################################################

def return_FROM(cuda_version, cudnn_install):
    if cuda_version is None:
        return "ubuntu:22.04"
    
    if cudnn_install is None:
        if cuda_version.startswith("12.3."):
            return f"nvidia/cuda:{cuda_version}-cudnn9-devel-ubuntu22.04"
        return f"nvidia/cuda:{cuda_version}-cudnn8-devel-ubuntu22.04"

    return f"nvidia/cuda:{cuda_version}-devel-ubuntu22.04"

##

def return_BOOTSTRAP_NVIDIA(cuda_version, cudnn_install, indir):
    if cudnn_install is None:
        return slurp_file(f"{indir}/BOOTSTRAP_NVIDIA.False")
    
    tmp = slurp_file(f"{indir}/BOOTSTRAP_NVIDIA.GPU")
    tmp = replace_line(tmp, "ENV NV_CUDNN_VERSION", f"ENV NV_CUDNN_VERSION={cudnn_install}-1")
    if cudnn_install.startswith("8."):
        tmp = replace_line(tmp, "ENV NV_CUDNN_PACKAGE_NAME", f"ENV NV_CUDNN_PACKAGE_NAME=\"libcudnn8\"")
    elif cudnn_install.startswith("9."):
        tmp = replace_line(tmp, "ENV NV_CUDNN_PACKAGE_BASENAME", f"ENV NV_CUDNN_PACKAGE_BASENAME=\"libcudnn9\"")

    if cuda_version.startswith("11.7"):
        tmp = replace_line(tmp, "ENV NV_CUDA_ADD=", f"ENV NV_CUDA_ADD=cuda11.7")
    elif cuda_version.startswith("11.8"):
        tmp = replace_line(tmp, "ENV NV_CUDA_ADD=", f"ENV NV_CUDA_ADD=cuda11.8")
    elif cuda_version.startswith("12.2"):
        tmp = replace_line(tmp, "ENV NV_CUDA_ADD=", f"ENV NV_CUDA_ADD=cuda12.2")
    elif cuda_version.startswith("12.3"):
        if cudnn_install.startswith("9."):
            tmp = replace_line(tmp, "ENV NV_CUDA_ADD=", f"ENV NV_CUDA_ADD=cuda12.3")
        else: # latest is 8.9.7 for 12.2
            tmp = replace_line(tmp, "ENV NV_CUDA_ADD=", f"ENV NV_CUDA_ADD=cuda12.2")
    elif cuda_version.startswith("12.4.1"):
        tmp = replace_line(tmp, "ENV NV_CUDA_ADD=", f"ENV NV_CUDA_ADD=cuda-12")
    elif cuda_version.startswith("12.5.1"):
        tmp = replace_line(tmp, "ENV NV_CUDA_ADD=", f"ENV NV_CUDA_ADD=cuda-12")
    elif cuda_version.startswith("13.0"):
        tmp = replace_line(tmp, "ENV NV_CUDA_ADD=", f"ENV NV_CUDA_ADD=cuda12.4")

    return tmp

## 

def return_APT_TORCH(pytorch_version, indir):
    if pytorch_version is None:
        return slurp_file(f"{indir}/APT_TORCH.False")
    
    return slurp_file(f"{indir}/APT_TORCH.True")

##

def return_CTPO_CUDA_APT(cuda_version, cudnn_install, indir):
    if cuda_version is None:
        return slurp_file(f"{indir}/APT_CUDA.False")

# Disabled until some of those packages are needed
    return slurp_file(f"{indir}/APT_CUDA.False")
#    tmp = slurp_file(f"{indir}/APT_CUDA.True")
#    # 22.04: https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/
#    repl = None
#    if cuda_version == "11.7.1":
#        repl = "11-7"
#    elif cuda_version == "11.8.0":
#        repl = "11-8"
#    elif cuda_version == "12.2.2":
#        repl = "12-2"
#    elif cuda_version == "12.3.0":
#        repl = "12-3"
#    else:
#        error_exit(f"Error: cuda version {cuda_version} is not supported")
#
#    repl_v=f"cuda-command-line-tools-{repl} cuda-cudart-dev-{repl} cuda-nvcc-{repl} cuda-cupti-{repl} cuda-nvprune-{repl} cuda-libraries-{repl} cuda-nvrtc-{repl} libcufft-{repl} libcurand-{repl} libcusolver-{repl} libcusparse-{repl} libcublas-{repl}"
##    repl_v = f"cuda-libraries-{repl} cuda-libraries-dev-{repl} cuda-tools-{repl} cuda-toolkit-{repl} libcublas-{repl} libcublas-dev-{repl} libcufft-{repl} libcufft-dev-{repl} libnpp-{repl} libnpp-dev-{repl}"
## Removed to avoid held packages issue
##    if cudnn_install is not None:
##        repl_v += f" libnccl2 libnccl-dev"
#    tmp = replace_line(tmp, "ENV CTPO_CUDA_APT", f"ENV CTPO_CUDA_APT=\"{repl_v}\"")
#
#    return tmp

##

def return_INSTALL_CLANG(clang_version, indir):
    if clang_version is None:
        return slurp_file(f"{indir}/INSTALL_CLANG.False")
    
    tmp = slurp_file(f"{indir}/INSTALL_CLANG.True")
    tmp = replace_line(tmp, "ENV CTPO_CLANG_VERSION", f"ENV CTPO_CLANG_VERSION={clang_version}")
    return tmp


##

def return_FIX_Misc(cuda_version, indir):
    if cuda_version is None:
        return slurp_file(f"{indir}/FIX_Misc.CPU")
    
    return slurp_file(f"{indir}/FIX_Misc.GPU")

##

def return_PIP_TORCH(pytorch_version, indir):
    if pytorch_version is None:
        return slurp_file(f"{indir}/PIP_TORCH.False")
    
    return slurp_file(f"{indir}/PIP_TORCH.True")

##

def return_PIP_KERAS(tensorflow_version, indir):
    if tensorflow_version is None:
        return slurp_file(f"{indir}/PIP_KERAS.False")
    
    return slurp_file(f"{indir}/PIP_KERAS.True")

##

def return_BUILD_TensorFlow(tensorflow_version, cuda_version, dnn_used, indir, args):
    if tensorflow_version is None:
        return slurp_file(f"{indir}/BUILD_TensorFlow.False")

    infile = f"{indir}/BUILD_TensorFlow.CPU"
    if cuda_version is not None:
        infile = f"{indir}/BUILD_TensorFlow.GPU"
    testfile = f"{indir}/BUILD_TensorFlow.{tensorflow_version}"
    if os.path.isfile(testfile):
        infile = testfile

    tmp = slurp_file(infile)
    if cuda_version is not None:
        tmp = replace_line(tmp, "ENV CTPO_TF_CONFIG", f"ENV CTPO_TF_CONFIG=\"--config=cuda\"")
        tmp = replace_line(tmp, "ENV TF_NEED_CUDA", f"ENV TF_NEED_CUDA=1")
        tmp = replace_line(tmp, "ENV TF_CUDA_COMPUTE_CAPABILITIES", f"ENV TF_CUDA_COMPUTE_CAPABILITIES={dnn_used}")
        if args.clang_version is not None:
            tmp = replace_line(tmp, "ENV TF_CUDA_CLANG", f"ENV TF_CUDA_CLANG=1")

    tmp = replace_line(tmp, "ENV LATEST_BAZELISK", f"ENV LATEST_BAZELISK={args.latest_bazelisk}")
    tmp = replace_line(tmp, "ENV CTPO_TENSORFLOW_VERSION", f"ENV CTPO_TENSORFLOW_VERSION={tensorflow_version}")

    if args.clang_version is not None:
        tmp = replace_line(tmp, "ENV TF_NEED_CLANG", f"ENV TF_NEED_CLANG=1")
    else:
        tmp = replace_line(tmp, "ENV TF_NEED_CLANG", f"ENV TF_NEED_CLANG=0")

    return tmp

## 
 
def return_BUILD_FFMPEG(cuda_version, indir, args):
    tmp = None
    if cuda_version is None:
        tmp = slurp_file(f"{indir}/BUILD_FFMPEG.CPU")
    else:
        tmp = slurp_file(f"{indir}/BUILD_FFMPEG.GPU")
        tmp = replace_line(tmp, "ENV CTPO_FFMPEG_NVCODEC=", f"ENV CTPO_FFMPEG_NVCODEC={args.ffmpeg_nvcodec}")

    tmp = replace_line(tmp, "ENV CTPO_FFMPEG_VERSION", f"ENV CTPO_FFMPEG_VERSION={args.ffmpeg_version}")

    if "unredistributable" in args.nonfree:
        tmp = replace_line(tmp, "ENV CTPO_FFMPEG_NONFREE=", f"ENV CTPO_FFMPEG_NONFREE=\"--enable-nonfree --enable-libnpp\"")

    return tmp

##

def return_BUILD_OPENCV(opencv_version, cuda_version, indir, args):
    tmp = slurp_file(f"{indir}/BUILD_OPENCV")

    tmp = replace_line(tmp, "ENV CTPO_OPENCV_VERSION", f"ENV CTPO_OPENCV_VERSION={opencv_version}")

    if cuda_version is not None:
        tmp = replace_line(tmp, "ENV CTPO_OPENCV_CUDA=", f"ENV CTPO_OPENCV_CUDA=\"-D WITH_CUDA=ON -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda -D CMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs -D CUDA_FAST_MATH=1 -D WITH_CUBLAS=1 -D WITH_NVCUVID=1 -DCUDA_ARCH_BIN={args.dnn_arch} -DCUDA_ARCH_PTX={args.dnn_arch}\"")

    if "unredistributable" in args.nonfree:
        tmp = replace_line(tmp, "ENV CTPO_OPENCV_NONFREE=", f"ENV CTPO_OPENCV_NONFREE=\"-DOPENCV_ENABLE_NONFREE=1\"")

    return tmp

##

def return_BUILD_MAGMA(cuda_version, pytorch_version, indir, args):
    if cuda_version is None:
        return slurp_file(f"{indir}/BUILD_MAGMA.False")
    if pytorch_version is None:
        return slurp_file(f"{indir}/BUILD_MAGMA.False")
    
    tmp = slurp_file(f"{indir}/BUILD_MAGMA.GPU")
    tmp = replace_line(tmp, "ENV CTPO_MAGMA=", f"ENV CTPO_MAGMA={args.magma_version}")
    tmp = replace_line(tmp, "ENV CTPO_MAGMA_ARCH", f"ENV CTPO_MAGMA_ARCH=\"{args.magma_arch}\"")

    return tmp

##

def return_BUILD_TORCH(cuda_version, pytorch_version, indir, args):
    if pytorch_version is None:
        return(slurp_file(f"{indir}/BUILD_TORCH.False"))

    tmp = None
    mode = "CPU"
    if cuda_version is not None:
        mode = "GPU"

    tmp = slurp_file(f"{indir}/BUILD_TORCH.{mode}")
    tmp_file = f"{indir}/BUILD_TORCH.{mode}.{pytorch_version}"
    if os.path.isfile(tmp_file):
        tmp = slurp_file(tmp_file)
    if mode == "GPU":
        tmp = replace_line(tmp, "ENV CTPO_TORCH_CUDA_ARCH", f"ENV CTPO_TORCH_CUDA_ARCH=\"{args.torch_arch}\"")

    # To obtain diff files from GH: wget https://github.com/pytorch/audio/pull/3811.diff
    tmp = replace_line(tmp, "ENV CTPO_TORCH=", f"ENV CTPO_TORCH={pytorch_version}")
    tmp = replace_line(tmp, "ENV CTPO_TORCHVISION=", f"ENV CTPO_TORCHVISION={args.torchvision_version}")
    patch = f"{indir}/PATCH_TORCHVISION.{args.torchvision_version}"
    if os.path.isfile(patch):
        dfile = f"{args.destdir}/torchvision.patch"
        shutil.copy(patch, f"{args.destdir}/torchvision.patch")
        # replace / with ___
        dfile = dfile.replace("/", "___")
        shutil.copy(patch, f"{dfile}.temp")
        tmp = replace_line(tmp, "COPY torchvision.patch", f"COPY {dfile}.temp /tmp/torchvision.patch")
    tmp = replace_line(tmp, "ENV CTPO_TORCHAUDIO=",  f"ENV CTPO_TORCHAUDIO={args.torchaudio_version}")
    patch = f"{indir}/PATCH_TORCHAUDIO.{args.torchaudio_version}"
    if os.path.isfile(patch):
        dfile = f"{args.destdir}/torchaudio.patch"
        shutil.copy(patch, f"{args.destdir}/torchaudio.patch")
        # replace / with ___
        dfile = dfile.replace("/", "___")
        shutil.copy(patch, f"{dfile}.temp")
        tmp = replace_line(tmp, "COPY torchaudio.patch", f"COPY {dfile}.temp /tmp/torchaudio.patch")
    tmp = replace_line(tmp, "ENV CTPO_TORCHDATA=", f"ENV CTPO_TORCHDATA={args.torchdata_version}")
#    tmp = replace_line(tmp, "ENV CTPO_TORCHTEXT=",  f"ENV CTPO_TORCHTEXT={args.torchtext_version}")
    
    return tmp

############################################################

def build_dockerfile(input, indir, release, tensorflow_version, pytorch_version, cuda_version, dnn_used, cudnn_install, opencv_version, verbose, args):
    dockertxt = slurp_file(input)
    env = ""

    #==FROM==#
    repl_v = return_FROM(cuda_version, cudnn_install)
    repl = f"ARG CTPO_FROM={repl_v}"
    # There can be no ENV before a FROM, so ARG is used instead
    dockertxt = replace_line(dockertxt, "#==FROM==#", repl)
    if verbose:
        print(f"  FROM content replaced with: {repl}")
    env += f"CTPO_FROM={repl_v}\n"

    #==BOOTSTRAP_NVIDIA==#
    repl = return_BOOTSTRAP_NVIDIA(cuda_version, cudnn_install, indir)
    dockertxt = replace_line(dockertxt, "#==BOOTSTRAP_NVIDIA==#", repl)

    #==CTPO_NUMPROC==#
    repl = f"ARG CTPO_NUMPROC={args.numproc}"
    dockertxt = replace_line(dockertxt, "#==CTPO_NUMPROC==#", repl)

    #==CTPO_INSTALL_CLANG==#
    repl = return_INSTALL_CLANG(args.clang_version, indir)
    dockertxt = replace_line(dockertxt, "#==CTPO_INSTALL_CLANG==#", repl)

    #==APT_TORCH==#
    repl = return_APT_TORCH(pytorch_version, indir)
    dockertxt = replace_line(dockertxt, "#==APT_TORCH==#", repl)

    #==CTPO_CUDA_APT==#
    repl = return_CTPO_CUDA_APT(cuda_version, cudnn_install, indir)
    dockertxt = replace_line(dockertxt, "#==CTPO_CUDA_APT==#", repl)

    #==CTPO_BUILD==#
    repl_v = None
    if cuda_version is None:
        repl_v = "CPU"
    else:
        repl_v = "GPU"
    repl = f"ENV CTPO_BUILD={repl_v}"
    dockertxt = replace_line(dockertxt, "#==CTPO_BUILD==#", repl)
    env += f"CTPO_BUILD={repl_v}\n"

    #==FIX_Misc==#
    repl = return_FIX_Misc(cuda_version, indir)
    dockertxt = replace_line(dockertxt, "#==FIX_Misc==#", repl)

    #==PIP_TORCH==#
    repl = return_PIP_TORCH(pytorch_version, indir)
    dockertxt = replace_line(dockertxt, "#==PIP_TORCH==#", repl)

    #==PIP_KERAS==#
    repl = return_PIP_KERAS(tensorflow_version, indir)
    dockertxt = replace_line(dockertxt, "#==PIP_KERAS==#", repl)

    #==BUILD_TensorFlow==#
    repl = return_BUILD_TensorFlow(tensorflow_version, cuda_version, dnn_used, indir, args)
    dockertxt = replace_line(dockertxt, "#==BUILD_TensorFlow==#", repl)

    #==BUILD_FFMPEG==#
    repl = return_BUILD_FFMPEG(cuda_version, indir, args)
    dockertxt = replace_line(dockertxt, "#==BUILD_FFMPEG==#", repl)

    #==BUILD_OPENCV==#
    repl = return_BUILD_OPENCV(opencv_version, cuda_version, indir, args)
    dockertxt = replace_line(dockertxt, "#==BUILD_OPENCV==#", repl)

    #==BUILD_MAGMA==#
    repl = return_BUILD_MAGMA(cuda_version, pytorch_version, indir, args)
    dockertxt = replace_line(dockertxt, "#==BUILD_MAGMA==#", repl)

    #==BUILD_TORCH==#
    repl = return_BUILD_TORCH(cuda_version, pytorch_version, indir, args)
    dockertxt = replace_line(dockertxt, "#==BUILD_TORCH==#", repl)

    # Final sanity check
    line, before, after = find_line_core(dockertxt, "#==")
    if line is not None:
        error_exit(f"Error: Could still find potential replacement block: {line}")

    # Populate the env file
    env += f"CTPO_TENSORFLOW_VERSION={tensorflow_version}\n"
    env += f"CTPO_PYTORCH_VERSION={pytorch_version}\n"
    env += f"CTPO_CUDA_VERSION={cuda_version}\n"
    env += f"CTPO_OPENCV_VERSION={opencv_version}\n"
    env += f"CTPO_RELEASE={release}\n"
    
    return dockertxt, env

##########

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action='count', help="Increase verbosity")
    parser.add_argument("--build", help="Container image name", required=True)
    parser.add_argument("--tag", help="Container image tag", required=True)
    parser.add_argument("--release", help="Release version", required=True)
    parser.add_argument("--destdir", help="Destination directory", required=True)
    parser.add_argument("--indir", help="Input directory for some replacement snippets", default="ubuntu22.04/Snippets")
    parser.add_argument("--input", help="Input Dockerfile", default="ubuntu22.04/Dockerfile.base")
    parser.add_argument("--cuda_ver", help="CUDA version(s) list (GPU only, | separated)", default="")
    parser.add_argument("--dnn_arch", help="DNN architecture build(s) list (GPU only, | separated)", default="")
    parser.add_argument("--cudnn_ver", help="CUDNN version (GPU only, if specified force pull of cuddn from the internet versus using a FROM with cudnn)", default="")
    parser.add_argument("--latest_bazelisk", help="Specify latest Bazelisk", default="")
    parser.add_argument("--numproc", help="Number of concurrent processes to use for build", default="4")
    parser.add_argument("--nonfree", help="Include non-free packages", choices=["free", "unredistributable"], default="free")
    parser.add_argument("--overwrite", help="Overwrite existing Dockerfile", action="store_true")
    parser.add_argument("--skip", help="Skip generation if Dockerfile already exists", action="store_true")
    parser.add_argument("--ffmpeg_version", help="FFmpeg version", default="")
    parser.add_argument("--ffmpeg_nvcodec", help="FFmpeg NVcodec version", default="")
    parser.add_argument("--magma_version", help="Magma version", default="")
    parser.add_argument("--magma_arch", help="Magma GPU architecture", default="Pascal Volta Ampere")
    parser.add_argument("--torch_arch", help="PyTorch GPU architecture", default="")
    parser.add_argument("--torchvision_version", help="TorchVision version", default="")
    parser.add_argument("--torchaudio_version",  help="TorchAudio version",  default="")
    parser.add_argument("--torchdata_version",  help="TorchData version",  default="")
#    parser.add_argument("--torchtext_version",  help="TorchText version",  default="")
    parser.add_argument("--clang_version", help="Clang version", default="")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        error_exit(f"Error: Input Dockerfile {args.input} does not exist")
    if not os.access(args.input, os.R_OK):
        error_exit(f"Error: Input Dockerfile {args.input} is not readable")

    if not os.path.isdir(args.indir):
        error_exit(f"Error: Input directory {args.indir} does not exist")
    if not os.access(args.indir, os.R_OK):
        error_exit(f"Error: Input directory {args.indir} is not readable")

    if not os.path.isdir(args.destdir):
        error_exit(f"Error: Destination directory {args.destdir} does not exist")
    if not os.access(args.destdir, os.W_OK):
        error_exit(f"Error: Destination directory {args.destdir} is not writable")

    cuda_arch = {}
    if args.cuda_ver:
        if args.dnn_arch == "":
            error_exit(f"Error: Cannot specify an emtpy dnn_arch when cuda_ver is used")
        cuda_ver = [ args.cuda_ver ]
        dnn_arch = [ args.dnn_arch ]
        if '|' in args.cuda_ver:
            cuda_ver = args.cuda_ver.split("|")
            dnn_arch = args.dnn_arch.split("|")
        if len(cuda_ver) != len(dnn_arch):
            error_exit(f"Error: cuda_ver and dnn_arch must have the same number of elements")
        for i in range(len(cuda_ver)):
            cuda_arch[cuda_ver[i]] = dnn_arch[i]
    print(f"  CUDA architectures: {cuda_arch}")

    dest_df = os.path.join(args.destdir, "Dockerfile")
    if os.path.isfile(dest_df):
        if args.skip:
            print(f"Destination Dockerfile {dest_df} already exists, skipped requested in such cases, exiting now")
            sys.exit(0)
        elif not args.overwrite:
            error_exit(f"Error: Destination Dockerfile {dest_df} already exists, refusing to overwrite")
        else:
            print(f"Warning: Destination Dockerfile {dest_df} already exists, overwriting requested")

    (tensorflow, pytorch, cuda) = parse_build(args.build)
    if args.verbose:
        print(f"  Build: tensorflow={tensorflow}, pytorch={pytorch}, cuda={cuda}")
    (tensorflow_version, pytorch_version, cuda_version, opencv_version, dnn_used) = parse_tag(args.tag, tensorflow, pytorch, cuda, cuda_arch)
    if args.verbose:
        print(f"  Tag: tensorflow_version={tensorflow_version}, pytorch_version={pytorch_version}, cuda_version={cuda_version}, dnn_arch={dnn_used}, opencv_version={opencv_version}")

    if cuda_version is not None:
        if isBlank(args.dnn_arch):
            error_exit(f"Error: dnn_arch list required when GPU build requested")
        if isBlank(args.ffmpeg_nvcodec):
            error_exit(f"Error: ffmpeg_nvcodec required when GPU build requested")
        if pytorch_version is not None:
            if isBlank(args.magma_version):
                error_exit(f"Error: magma_version required when PyTorch GPU build requested")
            if isBlank(args.magma_arch):
                error_exit(f"Error: magma_arch required when PyTorch GPU build requested")

    if pytorch_version is not None:
        if cuda_version is None:
            if isBlank(args.torch_arch):
                error_exit(f"Error: torch_arch required when PyTorch build requested")
        if isBlank(args.torchvision_version):
            error_exit(f"Error: torchvision_version required when PyTorch build requested")
        if isBlank(args.torchaudio_version):
            error_exit(f"Error: torchaudio_version required when PyTorch build requested")
        if isBlank(args.torchdata_version):
            error_exit(f"Error: torchdata_version required when PyTorch build requested")
#        if isBlank(args.torchtext_version):
#            error_exit(f"Error: torchtext_version required when PyTorch build requested")

    if tensorflow_version is not None:
        if isBlank(args.latest_bazelisk):
            error_exit(f"Error: latest_bazelisk required when TF build requested")

    if isBlank(args.clang_version):
        args.clang_version = None

    if isBlank(args.ffmpeg_version):
        error_exit(f"Error: ffmpeg_version required")

    cudnn_install = None
    if cuda_version is not None:
        if args.cudnn_ver:
            if isNotBlank(args.cudnn_ver):
                cudnn_install = args.cudnn_ver

    (dockertxt, env) = build_dockerfile(args.input, args.indir, args.release, tensorflow_version, pytorch_version, cuda_version, dnn_used, cudnn_install, opencv_version, args.verbose, args)

    with open(os.path.join(args.destdir, "Dockerfile"), 'w') as f:
        f.write(dockertxt)
    with open(os.path.join(args.destdir, "env.txt"), 'w') as f:
        f.write(env)

    # Force sync to disk
    os.sync()

if __name__ == "__main__":
    main()