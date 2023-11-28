
### Tag naming conventions

The tag naming convention follows the `_`-components split after the base name of `infotrend/ctpo-` followed by the "release" tag.
Any `infotrend/ctpo-cuda_` build is a `GPU` build while all non `cuda_` ones are `CPU` only. Note: Docker tags are always lowercase.


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


### Building

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

### Dockerfile

### Available builds on DockerHub

The `Dockerfile` used for a Dockerhub pushed built is shared in the `BuildInfo` directory

### BuildInfo

The `README-BuildDetails.md` file 