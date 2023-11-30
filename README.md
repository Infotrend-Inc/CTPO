# CTPO: CUDA + TensorFlow + PyTorch + OpenCV Docker containers

Latest revision: 20231120

<!-- vscode-markdown-toc -->
* 1. [Builds and Notes](#BuildsandNotes)
	* 1.1. [Tag naming conventions](#Tagnamingconventions)
	* 1.2. [Building](#Building)
	* 1.3. [Dockerfile](#Dockerfile)
	* 1.4. [Available builds on DockerHub](#AvailablebuildsonDockerHub)
	* 1.5. [Build Details](#BuildDetails)
	* 1.6. [Jupyter build](#Jupyterbuild)
	* 1.7. [Unraid build](#Unraidbuild)
* 2. [Usage and more](#Usageandmore)
	* 2.1. [A note on supported GPU in the Docker Hub builds](#AnoteonsupportedGPUintheDockerHubbuilds)
	* 2.2. [Using the container images](#Usingthecontainerimages)
* 3. [Version History](#VersionHistory)

<!-- vscode-markdown-toc-config
	numbering=true
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->

`Dockerfile`s to build containers with support for CPU and GPU (NVIDIA CUDA) containers with support for TensorFlow, PyTorch and OpenCV (or combinations of), based on Ubuntu 22.04 container images.

The tool's purpose is to enable developers, ML and CV enthusiasts to build and test solutions `FROM` a docker container, allowing fast prototyping and release of code to the community.

Building each container independently is made possible by the `Dockerfile` store in the `BuildDetails/<release>/<container_tag>` directories.
Building each container takes resources and time (counted in many cores, memory and hours).

Pre-built containers are available from Infotrend Inc.'s Docker account at https://hub.docker.com/r/infotrend/
Details on the available container and build are discussed in this document.

A Jupyter Lab and Unraid version of this WebUI-enabled version are also available on our Docker Hub.

Note: this tool was built earlier in 2023, iterations of its Jupyter Lab were made available to our data scientists, and we are releasing it to help the developer community.

##  1. <a name='BuildsandNotes'></a>Builds and Notes

The base OS for those container images is pulled from Dockerhub's official `ubuntu:22.04` or `nvidia/cuda:[...]-devel-ubuntu22.04` images. 
More details on the Nvidia base images are available at https://hub.docker.com/r/nvidia/cuda/ . 
In particular, please note that "By downloading these images, you agree to the terms of the license agreements for NVIDIA software included in the images"; with further details on DockerHub version from https://docs.nvidia.com/cuda/eula/index.html#attachment-a

For GPU-optimized versions, you will need to build the `cuda_` versions on a host with the final hardware.
When using GPU and building the container, you need to install the NVIDIA Container Toolkit found at https://github.com/NVIDIA/nvidia-container-toolkit
We note that your NVIDIA video driver needs to support the version of CUDA that you are trying to build

For CPU builds, you can simply build the non-`cuda_` versions.

Pre-built images are available for download on Infotrend's DockerHub. Those are built using the same method provided by the `Makefile`. The corresponding `Dockerfile` used is stored in the `BuildDetails` directory matching the container image.

###  1.1. <a name='Tagnamingconventions'></a>Tag naming conventions

The tag naming convention follows the `_`-components split after the base name of `infotrend/ctpo-` followed by the "release" tag.
Any `infotrend/ctpo-cuda_` build is a `GPU` build while all non-`cuda_` ones are `CPU` only. 
Note: Docker tags are always lowercase.

For example, for `infotrend/ctpo-tensorflow_pytorch_opencv:2.12.0_2.0.1_4.7.0-20231120`, this means: `"base name"-"component1"_"compoment2"_"component3":"component1_version"_"component2_version"_"component3_version"-"release tag"` with:
- `base name`=`infotrend/ctop-`
- `component1` + `component1_version` = `tensorflow` `2.12.0`
- `component2` + `component2_version` = `pytorch` `2.0.1`
- `component3` + `component3_version` = `opencv` `4.7.0`
As such, this was "Infotrend's CTPO release 20231120 with TensorFlow 2.12.0, PyTorch 2.0.1, and OpenCV 4.7.0 without any CUDA support." (Since no `cuda_` was part of the name, this is a `CPU` build)

Similarly, `infotrend/ctpo-cuda_pytorch_opencv:11.8.0_2.0.1_4.7.0-20231120` can be read as:
- `component1` + `component1_version` = `cuda` `11.8.0`
- `component2` + `component2_version` = `pytorch` `2.0.1`
- `component3` + `component3_version` = `opencv` `4.7.0`
As such, this was "Infotrend's CTPO release 20231120 with PyTorch 2.0.1, OpenCV 4.7.0 and CUDA support."

There can be more or less than three components per name (ex: `tensorflow_opencv` or `cuda_tensorflow_pytorch_opencv`). It is left to the end user to follow the naming convention.

###  1.2. <a name='Building'></a>Building

Type `make` to get the list of targets and some details of the possible builds.
Below you will see the result of this command for the `20231120` release:

```
**** Docker Image tag ending: 20231120
**** Docker Runtime: GPU
  To switch between GPU/CPU: add/remove "default-runtime": "nvidia" in /etc/docker/daemon.json then run: sudo systemctl restart docker

*** Available Docker images to be built (make targets):
  build_tpo (requires CPU Docker runtime):
    tensorflow_opencv OR pytorch_opencv OR tensorflow_pytorch_opencv (aka TPO, for CPU):
      tensorflow_opencv-2.12.0_4.7.0
      pytorch_opencv-2.0.1_4.7.0
      tensorflow_pytorch_opencv-2.12.0_2.0.1_4.7.0
  build_ctpo (requires GPU Docker runtime):
    cuda_tensorflow_opencv OR cuda_pytorch_opencv OR cuda_tensorflow_pytorch_opencv (aka CTPO, for NVIDIA GPU):
      cuda_tensorflow_opencv-11.8.0_2.12.0_4.7.0
      cuda_pytorch_opencv-11.8.0_2.0.1_4.7.0
      cuda_tensorflow_pytorch_opencv-11.8.0_2.12.0_2.0.1_4.7.0

*** Jupyter Labs ready containers (requires the base TPO & CTPO container to either be built locally or docker will attempt to pull otherwise)
  jupyter_tpo:
      jupyter-tensorflow_pytorch_opencv-2.12.0_2.0.1_4.7.0
  jupyter_ctpo:
      jupyter-cuda_tensorflow_pytorch_opencv-11.8.0_2.12.0_2.0.1_4.7.0
```

In this printout appears multiple sections:
- The `Docker Image tag ending` matches the software release tag.
- The `Docker Runtime` explains the current default runtime. For `GPU` (CTPO) builds it is recommended to add `"default-runtime": "nvidia"` in the `/etc/docker/daemon.json` file and restart the docker daemon. Similarly, for `CPU` (TPO) builds, that `"default-runtime"` should be removed (or commented.) You can check the current status of your runtime by running: `docker info | grep "Default Runtime"`
- The `Available Docker images to be built` section allows you to select the possible build targets. For `GPU`, the `cuda_` variants. For `CPU` the non `cuda_` variants. Naming conventions and tags follow the guidelines specified in the "Tag naming conventions" section.
- The `Jupyter Labs ready containers` are based on the containers built in the "Available Docker images[...]" and adding a running "Jupyter Labs" following the specific `Dockerfile` in the `Jupyter_build` directory. The list of built containers is limited to the most components per `CPU` and `GPU` to simplify distribution.

###  1.3. <a name='Dockerfile'></a>Dockerfile

Each time you request a specific `make` target a dedicated `Dockerfile` is built in the `BuildDetails/<release>/<target>` directory.

That `Dockerfile` contains `ARG` and `ENV` values that match the specific build parameters.
For example in release `20231120`, when building the `tensorflow_opencv-2.12.0_4.7.0` target, the `BuildDetails/20231120/tensorflow_opencv-2.12.0_4.7.0-20231120/Dockerfile` will be created and used to build the `tensorflow_opencv:2.12.0_4.7.0-20231120` container image.
In that file, you will see content such as:
```
ARG CTPO_FROM=ubuntu:22.04
FROM ${CTPO_FROM}
[...]
## Download & Building TensorFlow from source in same RUN
ENV LATEST_BAZELISK=1.17.0
ENV CTPO_TENSORFLOW_VERSION=2.12.0
ENV CTPO_TF_CONFIG=""
ENV TF_CUDA_COMPUTE_CAPABILITIES=""
[...]
# No Magma (PyTorch GPU only)

# No PyTorch, Torch Audio or Torch Video
RUN echo "No PyTorch built" > /tmp/torch_config.txt \
  && echo "No TorchVision built" > /tmp/torchvision_config.txt \
  && echo "No TorchAudio built" > /tmp/torchaudio_config.txt \
  && echo "No TorchData built" > /tmp/torchdata_config.txt \
  && echo "No TorchText built" > /tmp/torchtext_config.txt
```
, which is specific to the CPU build of TensorFlow and OpenCV (without PyTorch).

That `Dockerfile` should enable developers to integrate their modifications to build a specific feature.

When the maintainers upload this image to Dockerhub, that image will be preceded by `infotrend/ctpo-`.

If you choose to build the image for your hardware, please be patient, building any of those images might take a long time (counted in hours).
To build it this way, find the corresponding `Dockerfile` and `docker build -f <directory>/Dockerfile .` from the location of this `README.md`.
The build process will require some of the script in the `tools` directory to complete.

For example, to build the `BuildDetails/20231120/tensorflow_opencv-2.12.0_4.7.0-20231120/Dockerfile` and tag it as `to:test` from the directory where this `README.md` is located, run:
```
% docker build -f ./BuildDetails/20231120/tensorflow_opencv-2.12.0_4.7.0-20231120/Dockerfile --tag to:test .
```

###  1.4. <a name='AvailablebuildsonDockerHub'></a>Available builds on DockerHub

The `Dockerfile` used for a Dockerhub pushed built is shared in the `BuildDetails` directory (see the [Dockerfile](#Dockerfile) section above)

We will publish releases into [Infotrend Inc](https://hub.docker.com/r/infotrend/)'s Docker Hub account, as well as other tools.

The tag naming reflects the [Tag naming conventions](#Tagnamingconventions) section above.
`latest` is used to point to the most recent release.

The different base container images that can be found there are:
- CPU builds:
  - https://hub.docker.com/r/infotrend/ctpo-tensorflow_opencv
  - https://hub.docker.com/r/infotrend/ctpo-pytorch_opencv
  - https://hub.docker.com/r/infotrend/ctpo-tensorflow_pytorch_opencv
- GPU builds:
  - https://hub.docker.com/r/infotrend/ctpo-cuda_tensorflow_opencv
  - https://hub.docker.com/r/infotrend/ctpo-cuda_pytorch_opencv
  - https://hub.docker.com/r/infotrend/ctpo-cuda_tensorflow_pytorch_opencv
- Jupyter Lab CPU & GPU builds:
  - https://hub.docker.com/r/infotrend/ctpo-jupyter-tensorflow_pytorch_opencv
  - https://hub.docker.com/r/infotrend/ctpo-jupyter-cuda_tensorflow_pytorch_opencv
- Unraid enabled Jupyter Lab, CPU & GPU builds:
  - https://hub.docker.com/r/infotrend/ctpo-jupyter-tensorflow_pytorch_opencv-unraid
  - https://hub.docker.com/r/infotrend/ctpo-jupyter-cuda_tensorflow_pytorch_opencv-unraid


###  1.5. <a name='BuildDetails'></a>Build Details

The [`README-BuildDetails.md`](README-BuildDetails.md) file is built automatically from the content of the `BuildDetails` directory and contains link to different files stored in each sub-directory.

It reflects each build's detailed information, such as (where relevant), the Docker tag, version of CUDA, cuDNN, TensorFlow, PyTorch, OpenCV, FFmpeg and Ubuntu. Most content also links to sub-files that contain further insight into the system package, enabled build parameters, etc.

###  1.6. <a name='Jupyterbuild'></a>Jupyter build

Jupyter Lab containers are built `FROM` the `tensorflow_pytorch_opencv` or `cuda_tensorflow_pytorch_opencv` containers.

A "user" version (current user's UID and GID are passed to the internal user) can be built using `make JN_MODE="-user" jupyter_tpo jupyter_ctpo`.

The specific details of such builds are available in the `Jupyter_build` directory, in the `Dockerfile` and `Dockerfile-user` files.

In particular, the Notebook default Jupyter lab password (`iti`) is stored in the `Dockerfile` and can be modified by the builder by replacing the `--IdentityProvider.token='iti'` command line option.

When using the Jupyter-specific container, it is also important to remember to expose the port used by the tool (here 8888), as such in your `docker run` command, make sure to add something akin to `-p 8888:8888` to the command line.

Pre-built containers are available, see the [Available builds on DockerHub](#AvailablebuildsonDockerHub) section above.

###  1.7. <a name='Unraidbuild'></a>Unraid build

Those are specializations of the Jupyter Lab's builds, and container images with a `sudo`-capable `jupyter` user using unraid's specific `uid` and `gid` and the same default `iti` Jupyter lab's default password.

The unraid version can be built using `make JN_MODE="-unraid" jupyter_tpo jupyter_ctpo`.

The build details are available in the `Jupyter_build/Dockerfile-unraid` file.

Pre-built containers are available, see the [Available builds on DockerHub](#AvailablebuildsonDockerHub) section above.

##  2. <a name='Usageandmore'></a>Usage and more

###  2.1. <a name='AnoteonsupportedGPUintheDockerHubbuilds'></a>A note on supported GPU in the Docker Hub builds

In some cases, a minimum Nvidia driver version is needed to run specific version of CUDA, [Table 1: CUDA Toolkit and Compatible Driver Versions](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#binary-compatibility__table-toolkit-driver) and [Table 2: CUDA Toolkit and Minimum Compatible Driver Versions](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html) as well as the `nvidia-smi` command on your host will help you determine if a specific version of CUDA will be supported.

It is important to note that not all GPUs are supported in the Docker Hub builds. The containers are built for "compute capability (version)" (as defined in the [GPU supported](https://en.wikipedia.org/wiki/CUDA#GPUs_supported) Wikipedia page) of 6.0 and above (ie Pascal and above). 

If you need a different GPU compute capability, please edit the `Makefile` and alter the various `DNN_ARCH_` matching the one that you need to build and add your architecture. Then type `make` to see the entire list of containers that the release you have obtained can build and use the exact tag that you want to build to build it locally (on Ubuntu, you will need `docker` and `build-essential` installed --at least-- to do this). Building a container image takes a lot of CPU and can take multiple hours, so we recommend you build only the target you need.


###  2.2. <a name='Usingthecontainerimages'></a>Using the container images

Build or obtain the container image you require from DockerHub.

We understand the image names are verbose. This is to avoid confusion between the different builds.
It is recommended to `tag` containers with short names for easy `docker run`.

The `WORKDIR` for the containers is set as `/iti`, as such, should you want to map the current working directory within your container and test functions, you can `-v` as `/iti`.

When using a GPU image, make sure to add `--gpus all` to the `docker run` command line.

For example to run the GPU-Jupyter container and expose the WebUI to port 8765, one could:
```
% docker run --rm -v `pwd`:/iti --gpus all -p 8765:8888 infotrend/ctpo-jupyter-cuda_tensorflow_pytorch_opencv:11.8.0_2.12.0_2.0.1_4.7.0-20231120
```
By going to http://localhost:8765 you will be shown the Jupyter `Log in` page. As a reminder, the default token is `iti`.
When you login, you will see the Jupyter lab interface and the list of files mounted in `/iti` on the left.
From that WebUI, when you `File->Shutdown`, the container will exit.

The non-Jupyter containers are set to provide the end users with a `bash`. If the `/iti` directory is mounted in a directory where the developer has some come for testing with one of the provided tools, this can be done. For example to run some of the content of the `test` directory on CPU (in the directory where this `README.md` is located):
```
% docker run --rm -it -v `pwd`:/iti infotrend/ctpo-tensorflow_opencv:2.12.0_4.7.0-20231120

root@b859b8aced9c:/iti# python3 ./test/tf_test.py
Tensorflow test: CPU only



On CPU:
tf.Tensor(
[[22. 28.]
 [49. 64.]], shape=(2, 2), dtype=float32)



Time (s) to convolve 32x7x7x3 filter over random 100x100x100x3 images (batch x height x width x channel). Sum of ten runs.
CPU (s): 0.483618629979901
Tensorflow test: Done
```

Note that the base container runs as root, if you want to run it as a non-root user, add `-u $(id -u):$(id -g)` to the `docker` command line but ensure that you have access to the directories you will work in.

##  3. <a name='VersionHistory'></a>Version History

- 20231120: Initial Release, with support for CUDA 11.8.0, TensorFlow 2.12.0, PyTorch 2.0.1 and OpenCV 4.7.0.
- November 2023: Preparation for public release.
- June 2023: engineered to support clean `Dockerfile` generation, supporting the same versions as 20231120 releases.