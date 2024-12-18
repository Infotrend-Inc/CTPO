# Needed SHELL since I'm using zsh
SHELL := /bin/bash
.PHONY: all build_all actual_build build_prep pre_build post_build
.NOTPARALLEL:

# Release to match data of Dockerfile and follow YYYYMMDD pattern
CTPO_RELEASE=202412wip

# The default is not to build OpenCV non-free or build FFmpeg with libnpp, as those would make the images unredistributable 
# Replace "free" by "unredistributable" if you need to use those for a personal build
CTPO_ENABLE_NONFREE="free"
#CTPO_ENABLE_NONFREE="unredistributable"

# Maximize build speed
CTPO_NUMPROC := $(shell nproc --all)

# docker build extra parameters 
DOCKER_BUILD_ARGS=
#DOCKER_BUILD_ARGS="--no-cache"

# Use "yes" below before a multi build to have docker pull the base images using "make build_all" 
DOCKERPULL=""
#DOCKERPULL="yes"

# Use "--overwrite" below to force a generation of the Dockerfile
# Because the Dockerfile should be the same (from a git perspective) when overwritten, this should not be a problem; and if different, we want to know 
# a skip can be requested with "--skip"
#OVERWRITE_DOCKERFILE=""
#OVERWRITE_DOCKERFILE="--skip"
OVERWRITE_DOCKERFILE="--overwrite"

# Use "yes" below to force some tools check post build (recommended)
# this will use docker run [...] --gpus all and extend the TF build log
CKTK_CHECK="yes"

# Table below shows driver/CUDA support; for example the 10.2 container needs at least driver 440.33
# https://docs.nvidia.com/deploy/cuda-compatibility/index.html#binary-compatibility__table-toolkit-driver
#
# According to https://hub.docker.com/r/nvidia/cuda/
# https://hub.docker.com/r/nvidia/cuda/tags?page=1&name=24.04
#
# Note: CUDA11 minimum version has to match the one used by PyTorch
# From PyTorch: Deprecation of CUDA 11.6 and Python 3.7 Support
# (from within running container) to get the list of cudnn version available: apt-cache madison cudnn9-cuda-12-4
#
# TF will likely not work unless we follow the recongized versions https://www.tensorflow.org/install/source#gpu
STABLE_CUDA=12.5.1
STABLE_CUDNN=9.3.0.75

# TensortRT:
# check available version: apt-cache madison  tensorrt-dev
TENSORRT="-TensorRT"

# CUDNN needs 5.3 at minimum, extending list from https://en.wikipedia.org/wiki/CUDA#GPUs_supported 
# Skipping Tegra, Jetson, ... (ie not desktop/server GPUs) from this list
# Keeping from Pascal and above
DNN_ARCH_CUDA=6.0,6.1,7.0,7.5,8.0,8.6,8.9,9.0
# Torch note on PTX: https://pytorch.org/docs/stable/cpp_extension.html
DNN_ARCH_TORCH=6.0 6.1 7.0 7.5 8.0 8.6 8.9 9.0+PTX

# According to https://opencv.org/releases/
STABLE_OPENCV4=4.10.0

# FFmpeg
# Release list: https://ffmpeg.org/download.html
# Note: GPU extensions are added directly in the Dockerfile
CTPO_FFMPEG_VERSION=7.1
# https://github.com/FFmpeg/nv-codec-headers/releases
CTPO_FFMPEG_NVCODEC="12.2.72.0"

# TF2 CUDA & CUDNN
# According to https://github.com/tensorflow/tensorflow/tags
# Known working CUDA & CUDNN base version https://www.tensorflow.org/install/source#gpu
# Find OS specific libcudnn file from https://developer.download.nvidia.com/compute/redist/cudnn/
STABLE_TF2=2.18.0

CLANG_VERSION=17

## Information for build
# https://github.com/bazelbuild/bazelisk
LATEST_BAZELISK=1.22.1

# Magma
# Release page: https://icl.utk.edu/magma/
# Note: GPU targets (ie ARCH) are needed
CTPO_MAGMA=2.8.0
# Get ARCHs from https://bitbucket.org/icl/magma/src/master/Makefile
CTPO_MAGMA_ARCH=Pascal Volta Turing Ampere Hopper

## PyTorch (with FFmpeg + OpenCV & Magma if available) https://pytorch.org/
# Note: same as FFmpeg and Magma, GPU specific selection (including ARCH) are in the Dockerfile
# Use release branch https://github.com/pytorch/pytorch
# https://pytorch.org/get-started/locally/
# https://pytorch.org/get-started/pytorch-2.0/#getting-started
# https://github.com/pytorch/pytorch/releases/tag/v2.0.1
STABLE_TORCH=2.5.1
# Use release branch https://github.com/pytorch/vision
CTPO_TORCHVISION="0.20.0"
# then use released branch at https://github.com/pytorch/audio
CTPO_TORCHAUDIO="2.5.0"
# check compatibility from https://github.com/pytorch/data and the release tags
CTPO_TORCHDATA="0.9.0"
# TorchText: "development is stopped"

## Docker builder helper script & BuildDetails directory
DFBH=./tools/Dockerfile_builder_helper.py
BuildDetails=BuildDetails

# Tag base for the docker image release only
TAG_RELEASE="infotrend/ctpo-"

##########

##### CUDA [ _ Tensorflow ]  [ _ PyTorch ] _ OpenCV (aka CTPO)
CTPO_BUILDALL_T  =cuda_tensorflow_opencv-${STABLE_CUDA}_${STABLE_TF2}_${STABLE_OPENCV4}

CTPO_BUILDALL_P  =cuda_pytorch_opencv-${STABLE_CUDA}_${STABLE_TORCH}_${STABLE_OPENCV4}

CTPO_BUILDALL_TP =cuda_tensorflow_pytorch_opencv-${STABLE_CUDA}_${STABLE_TF2}_${STABLE_TORCH}_${STABLE_OPENCV4}

CTPO_BUILDALL=${CTPO_BUILDALL_T} ${CTPO_BUILDALL_P} ${CTPO_BUILDALL_TP}

##### [ Tensorflow | PyTorch ] _ OpenCV (aka TPO)
TPO_BUILDALL_T =tensorflow_opencv-${STABLE_TF2}_${STABLE_OPENCV4}

TPO_BUILDALL_P =pytorch_opencv-${STABLE_TORCH}_${STABLE_OPENCV4}

TPO_BUILDALL_TP=tensorflow_pytorch_opencv-${STABLE_TF2}_${STABLE_TORCH}_${STABLE_OPENCV4}

TPO_BUILDALL=${TPO_BUILDALL_T} ${TPO_BUILDALL_P} ${TPO_BUILDALL_TP}

# For CPU builds (if GPU mode is enabled), we will use NVIDIA_VISIBLE_DEVICES=void as detailed in
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html

##### Jupyter Notebook ready based on TPO & CTPO
TPO_JUP=jupyter-tensorflow_pytorch_opencv-${STABLE_TF2}_${STABLE_TORCH}_${STABLE_OPENCV4}
CTPO_JUP=jupyter-cuda_tensorflow_pytorch_opencv-${STABLE_CUDA}_${STABLE_TF2}_${STABLE_TORCH}_${STABLE_OPENCV4}

## By default, provide the list of build targets
all:
	@$(eval CHECKED_DOCKER_RUNTIME=$(shell docker info | grep "Default Runtime" | cut -d : -f 2 | tr  -d " "))
	@$(eval CHECK_DOCKER_RUNTIME=$(shell if [ "A${CHECKED_DOCKER_RUNTIME}" == "Anvidia" ]; then echo "GPU"; else echo "CPU"; fi))
	@echo "**** Docker Image tag ending: ${CTPO_RELEASE}"
	@echo "**** Docker Runtime: ${CHECK_DOCKER_RUNTIME}"
	@echo "  To switch between GPU/CPU: add/remove "'"default-runtime": "nvidia"'" in /etc/docker/daemon.json then run: sudo systemctl restart docker"
	@echo ""
	@echo "*** Available Docker images to be built (make targets):"
	@echo "  build_tpo (requires CPU Docker runtime):"
	@echo "    tensorflow_opencv OR pytorch_opencv OR tensorflow_pytorch_opencv (aka TPO, for CPU): "; echo -n "      "; echo ${TPO_BUILDALL} | sed -e 's/ /\n      /g'
	@echo "  build_ctpo (requires GPU Docker runtime):"
	@echo "    cuda_tensorflow_opencv OR cuda_pytorch_opencv OR cuda_tensorflow_pytorch_opencv (aka CTPO, for NVIDIA GPU): "; echo -n "      "; echo ${CTPO_BUILDALL} | sed -e 's/ /\n      /g'
	@echo "  build_ctpo_tensorrt (requires GPU Docker runtime):"
	@echo "    cuda_tensorflow_pytorch_opencv (aka CTPO, for NVIDIA GPU with TensorRT): same as cuda_tensorflow_pytorch_opencv but installing TensorRT libraries"; echo "      Note:TensorRT is not supported by TensorFlow since 2.18.0"
	@echo ""
	@echo "*** Jupyter Labs ready containers (requires the base TPO & CTPO container to either be built locally or docker will attempt to pull otherwise)"
	@echo "  jupyter_tpo: "; echo -n "      "; echo ${TPO_JUP}
	@echo "  jupyter_ctpo: "; echo -n "      "; echo ${CTPO_JUP}
	@echo ""

## special command to build all targets
build_all: ${TPO_BUILDALL} ${CTPO_BUILDALL} build_ctpo_tensorrt

tensorflow_opencv: ${TPO_BUILDALL_T}

pytorch_opencv: ${TPO_BUILDALL_P}

tensorflow_pytorch_opencv: ${TPO_BUILDALL_TP}

cuda_tensorflow_opencv: ${CTPO_BUILDALL_T}

cuda_pytorch_opencv: ${CTPO_BUILDALL_P}

cuda_tensorflow_pytorch_opencv: ${CTPO_BUILDALL_TP}

build_tpo: ${TPO_BUILDALL}

build_ctpo:	${CTPO_BUILDALL}

build_ctpo_tensorrt:
	@BTARG="$@" USE_TENSORRT="${TENSORRT}" make build_prep

${TPO_BUILDALL} ${CTPO_BUILDALL}:
	@BTARG="$@" USE_TENSORRT="" make build_prep

build_prep:
	@$(eval CTPO_NAME=$(shell echo ${BTARG} | cut -d- -f 1))
	@$(eval CTPO_TAG=$(shell echo ${BTARG} | cut -d- -f 2))
	@$(eval CTPO_FULLTAG=${CTPO_TAG}-${CTPO_RELEASE}${USE_TENSORRT})
	@$(eval CTPO_FULLNAME=${CTPO_NAME}-${CTPO_FULLTAG})
	@echo ""; echo ""; echo "[*****] Build: ${CTPO_NAME}:${CTPO_FULLTAG}";
	@if [ ! -f ${DFBH} ]; then echo "ERROR: ${DFBH} does not exist"; exit 1; fi
	@if [ ! -x ${DFBH} ]; then echo "ERROR: ${DFBH} is not executable"; exit 1; fi
	@if [ ! -d ${BuildDetails} ]; then mkdir ${BuildDetails}; fi
	@$(eval BUILD_DESTDIR=${BuildDetails}/${CTPO_RELEASE}/${CTPO_FULLNAME})
	@if [ ! -d ${BUILD_DESTDIR} ]; then mkdir -p ${BUILD_DESTDIR}; fi
	@if [ ! -d ${BUILD_DESTDIR} ]; then echo "ERROR: ${BUILD_DESTDIR} directory could not be created"; exit 1; fi

	@${DFBH} --verbose ${OVERWRITE_DOCKERFILE} --numproc ${CTPO_NUMPROC} \
		--build ${CTPO_NAME} --tag ${CTPO_TAG} --release ${CTPO_RELEASE} --destdir ${BUILD_DESTDIR} --nonfree "${CTPO_ENABLE_NONFREE}" \
		--cuda_ver "${STABLE_CUDA}" --dnn_arch "${DNN_ARCH_CUDA}" \
		--cudnn_ver "${STABLE_CUDNN}" --latest_bazelisk "${LATEST_BAZELISK}" \
		--ffmpeg_version "${CTPO_FFMPEG_VERSION}" --ffmpeg_nvcodec "${CTPO_FFMPEG_NVCODEC}" \
		--magma_version ${CTPO_MAGMA} --magma_arch "${CTPO_MAGMA_ARCH}" \
		--torch_arch="${DNN_ARCH_TORCH}" --torchaudio_version=${CTPO_TORCHAUDIO} \
		--torchvision_version=${CTPO_TORCHVISION} \
		--torchdata_version=${CTPO_TORCHDATA} \
		--clang_version=${CLANG_VERSION} \
		--copyfile=tools/withincontainer_checker.sh \
		--TensorRT="${USE_TENSORRT}" \
	&& sync

	@while [ ! -f ${BUILD_DESTDIR}/env.txt ]; do sleep 1; done

	@CTPO_NAME=${CTPO_NAME} CTPO_TAG=${CTPO_TAG} CTPO_FULLTAG=${CTPO_FULLTAG} BUILD_DESTDIR=${BUILD_DESTDIR} CTPO_FULLNAME=${CTPO_FULLNAME} make pre_build

pre_build:
	@$(eval CTPO_FROM=${shell cat ${BUILD_DESTDIR}/env.txt | grep CTPO_FROM | cut -d= -f 2})
	@$(eval CTPO_BUILD=$(shell cat ${BUILD_DESTDIR}/env.txt | grep CTPO_BUILD | cut -d= -f 2))

	@if [ "A${DOCKERPULL}" == "Ayes" ]; then \
		echo "** Base image: ${CTPO_FROM}"; docker pull ${CTPO_FROM}; echo ""; \
	else \
		if [ -f ./${CTPO_FULLNAME}.log ]; then \
			echo "  !! Log file (${CTPO_FULLNAME}.log) exists, skipping rebuild (remove to force)"; echo ""; \
		else \
			CTPO_NAME=${CTPO_NAME} CTPO_TAG=${CTPO_TAG} CTPO_FULLTAG=${CTPO_FULLTAG} CTPO_FROM=${CTPO_FROM} BUILD_DESTDIR=${BUILD_DESTDIR} CTPO_FULLNAME=${CTPO_FULLNAME} CTPO_BUILD="${CTPO_BUILD}" make actual_build; \
		fi; \
	fi

actual_build:
# Build prep
	@if [ ! -f ${BUILD_DESTDIR}/env.txt ]; then echo "ERROR: ${BUILD_DESTDIR}/env.txt does not exist, aborting build"; echo ""; exit 1; fi
	@if [ ! -f ${BUILD_DESTDIR}/Dockerfile ]; then echo "ERROR: ${BUILD_DESTDIR}/Dockerfile does not exist, aborting build"; echo ""; exit 1; fi
	@if [ "A${CTPO_BUILD}" == "A" ]; then echo "Missing value for CTPO_BUILD, aborting"; exit 1; fi
	@$(eval CHECKED_DOCKER_RUNTIME=$(shell docker info | grep "Default Runtime" | cut -d : -f 2 | tr  -d " "))
	@$(eval CHECK_DOCKER_RUNTIME=$(shell if [ "A${CHECKED_DOCKER_RUNTIME}" == "Anvidia" ]; then echo "GPU"; else echo "CPU"; fi))
# GPU docker + CPU build okay using NVIDIA_VISIBLE_DEVICES=void 
	@$(eval DOCKER_PRE=$(shell if [ "A${CHECK_DOCKER_RUNTIME}" == "AGPU" ]; then if [ "A${CTPO_BUILD}" == "ACPU" ]; then echo "NVIDIA_VISIBLE_DEVICES=void"; else echo ""; fi; fi))
	@if [ "A${CTPO_BUILD}" != "A${CHECK_DOCKER_RUNTIME}" ]; then if [ "A${DOCKER_PRE}" == "" ]; then echo "ERROR: Unable to build, default runtime is ${CHECK_DOCKER_RUNTIME} and build requires ${CTPO_BUILD}. Either add or remove "'"default-runtime": "nvidia"'" in /etc/docker/daemon.json before running: sudo systemctl restart docker"; echo ""; echo ""; exit 1; else echo "Note: GPU docker + CPU build => using ${DOCKER_PRE}"; fi; fi
	@$(eval VAR_NT="${CTPO_FULLNAME}")
	@$(eval VAR_DD="${BUILD_DESTDIR}")
	@$(eval VAR_PY="${BUILD_DESTDIR}/System--Details.txt")
	@$(eval VAR_CV="${BUILD_DESTDIR}/OpenCV--Details.txt")
	@$(eval VAR_TF="${BUILD_DESTDIR}/TensorFlow--Details.txt")
	@$(eval VAR_FF="${BUILD_DESTDIR}/FFmpeg--Details.txt")
	@$(eval VAR_PT="${BUILD_DESTDIR}/PyTorch--Details.txt")
	@${eval CTPO_DESTIMAGE="${CTPO_NAME}:${CTPO_FULLTAG}"}
	@mkdir -p ${VAR_DD}
	@echo ""
	@echo "  CTPO_FROM               : ${CTPO_FROM}" | tee ${VAR_CV} | tee ${VAR_TF} | tee ${VAR_FF} | tee ${VAR_PT} | tee ${VAR_PY}
	@echo ""
	@echo -n "  Built with Docker"; docker info | grep "Default Runtime"
	@echo "  DOCKER_PRE: ${DOCKER_PRE}"
	@echo "  Docker runtime: ${CHECK_DOCKER_RUNTIME} / Build requirements: ${CTPO_BUILD}"
	@echo ""
	@echo "-- Docker command to be run:"
	@echo "cd ${BUILD_DESTDIR};" > ${VAR_NT}.cmd
	@echo "docker buildx create --use --driver-opt env.BUILDKIT_STEP_LOG_MAX_SIZE=256000000 \\ " >> ${VAR_NT}.cmd
	@echo "BUILDX_EXPERIMENTAL=1 ${DOCKER_PRE} docker buildx debug --on=error build --progress plain --platform linux/amd64 ${DOCKER_BUILD_ARGS} \\" >> ${VAR_NT}.cmd
	@echo "  --build-arg CTPO_NUMPROC=\"$(CTPO_NUMPROC)\" \\" >> ${VAR_NT}.cmd
	@echo "  --tag=\"${CTPO_DESTIMAGE}\" \\" >> ${VAR_NT}.cmd
	@echo "  -f Dockerfile \\" >> ${VAR_NT}.cmd
	@echo "  ." >> ${VAR_NT}.cmd
	@cat ${VAR_NT}.cmd | tee ${VAR_NT}.log.temp | tee -a ${VAR_CV} | tee -a ${VAR_TF} | tee -a ${VAR_FF} | tee -a ${VAR_PT} | tee -a ${VAR_PY}
	@echo "" | tee -a ${VAR_NT}.log.temp
	@echo "Press Ctl+c within 5 seconds to cancel"
	@for i in 5 4 3 2 1; do echo -n "$$i "; sleep 1; done; echo ""
# Actual build
	@chmod +x ./${VAR_NT}.cmd
	@script -a -e -c ./${VAR_NT}.cmd ${VAR_NT}.log.temp; exit "$${PIPESTATUS[0]}"
	@CTPO_DESTIMAGE="${CTPO_DESTIMAGE}" VAR_DD="${VAR_DD}" VAR_NT="${VAR_NT}" VAR_CV="${VAR_CV}" VAR_TF="${VAR_TF}" VAR_FF="${VAR_FF}" VAR_PT="${VAR_PT}" VAR_PY="${VAR_PY}" DOCKER_PRE="${DOCKER_PRE}" make post_build

post_build:
	@${eval tmp_id=$(shell docker create ${CTPO_DESTIMAGE})}
	@printf "\n\n***** OpenCV configuration:\n" >> ${VAR_CV}; docker cp ${tmp_id}:/tmp/opencv_info.txt /tmp/ctpo; cat /tmp/ctpo >> ${VAR_CV}
	@printf "\n\n***** TensorFlow configuration:\n" >> ${VAR_TF}; docker cp ${tmp_id}:/tmp/tf_env.dump /tmp/ctpo; cat /tmp/ctpo >> ${VAR_TF}
	@printf "\n\n***** FFmpeg configuration:\n" >> ${VAR_FF}; docker cp ${tmp_id}:/tmp/ffmpeg_config.txt /tmp/ctpo; cat /tmp/ctpo >> ${VAR_FF}
	@printf "\n\n***** PyTorch configuration:\n" >> ${VAR_PT}; docker cp ${tmp_id}:/tmp/torch_config.txt /tmp/ctpo; cat /tmp/ctpo >> ${VAR_PT}
	@printf "\n\n***** TorchVision configuration:\n" >> ${VAR_PT}; docker cp ${tmp_id}:/tmp/torchvision_config.txt /tmp/ctpo; cat /tmp/ctpo >> ${VAR_PT}
	@printf "\n\n***** TorchAudio configuration:\n" >> ${VAR_PT}; docker cp ${tmp_id}:/tmp/torchaudio_config.txt /tmp/ctpo; cat /tmp/ctpo >> ${VAR_PT}
	@printf "\n\n***** TorchData configuration:\n" >> ${VAR_PT}; docker cp ${tmp_id}:/tmp/torchdata_config.txt /tmp/ctpo; cat /tmp/ctpo >> ${VAR_PT}
	@printf "\n\n***** Python configuration:\n" >> ${VAR_PY}; docker cp ${tmp_id}:/tmp/python_info.txt /tmp/ctpo; cat /tmp/ctpo >> ${VAR_PY}
	@docker rm -v ${tmp_id}

	@./tools/quick_bi.sh ${VAR_DD} && sync
	@while [ ! -f ${VAR_DD}/BuildInfo.txt ]; do sleep 1; done

	@CTPO_DESTIMAGE="${CTPO_DESTIMAGE}" VAR_DD="${VAR_DD}" VAR_NT="${VAR_NT}" VAR_CV="${VAR_CV}" VAR_TF="${VAR_TF}" VAR_FF="${VAR_FF}" VAR_PT="${VAR_PT}" VAR_PY="${VAR_PY}" DOCKER_PRE="${DOCKER_PRE}" make post_build_check

	@mv ${VAR_NT}.log.temp ${VAR_NT}.log
	@rm -f ./${VAR_NT}.cmd
	@rm -f ${VAR_DD}/env.txt

	@echo ""; echo ""; echo "***** Build completed *****"; echo ""; echo "Content of ${VAR_DD}/BuildInfo.txt"; echo ""; cat ${VAR_DD}/BuildInfo.txt; echo ""; echo ""

post_build_check:
	@$(eval TF_BUILT=$(shell grep -q "TensorFlow_Built" ${VAR_DD}/BuildInfo.txt && echo "yes" || echo "no"))
	@$(eval PT_BUILT=$(shell grep -q "Torch_Built" ${VAR_DD}/BuildInfo.txt && echo "yes" || echo "no"))
	@if [ "A${CKTK_CHECK}" == "Ayes" ]; then if [ "A${TF_BUILT}" == "Ayes" ]; then CTPO_DESTIMAGE="${CTPO_DESTIMAGE}" VAR_TF=${VAR_TF} VAR_NT="${VAR_NT}" DOCKER_PRE="${DOCKER_PRE}" make force_tf_check; fi; fi
	@if [ "A${CKTK_CHECK}" == "Ayes" ]; then if [ "A${PT_BUILT}" == "Ayes" ]; then CTPO_DESTIMAGE="${CTPO_DESTIMAGE}" VAR_PT=${VAR_PT} VAR_NT="${VAR_NT}" DOCKER_PRE="${DOCKER_PRE}" make force_pt_check; fi; fi
	@CTPO_DESTIMAGE="${CTPO_DESTIMAGE}" VAR_CV=${VAR_CV} VAR_NT="${VAR_NT}" DOCKER_PRE="${DOCKER_PRE}" make force_cv_check

##### Force Toolkit checks

## TensorFlow
# might be useful https://stackoverflow.com/questions/44232898/memoryerror-in-tensorflow-and-successful-numa-node-read-from-sysfs-had-negativ/44233285#44233285
force_tf_check:
	@echo "test: tf_det"
	@${DOCKER_PRE} docker run --rm -v `pwd`:/iti -v `pwd`/tools/skip_disclaimer.sh:/opt/nvidia/nvidia_entrypoint.sh --gpus all ${CTPO_DESTIMAGE} python3 /iti/test/tf_det.py | tee -a ${VAR_TF} | tee -a ${VAR_NT}.testlog; exit "$${PIPESTATUS[0]}"
	@echo "test: tf_hw"
	@${DOCKER_PRE} docker run --rm -v `pwd`:/iti -v `pwd`/tools/skip_disclaimer.sh:/opt/nvidia/nvidia_entrypoint.sh --gpus all ${CTPO_DESTIMAGE} python3 /iti/test/tf_hw.py | tee -a ${VAR_NT}.testlog; exit "$${PIPESTATUS[0]}"
	@echo "test: tf_test"
	@${DOCKER_PRE} docker run --rm -v `pwd`:/iti -v `pwd`/tools/skip_disclaimer.sh:/opt/nvidia/nvidia_entrypoint.sh --gpus all ${CTPO_DESTIMAGE} python3 /iti/test/tf_test.py | tee -a ${VAR_NT}.testlog; exit "$${PIPESTATUS[0]}"

## PyTorch
force_pt_check:
	@echo "pt_det"
	@${DOCKER_PRE} docker run --rm -v `pwd`:/iti -v `pwd`/tools/skip_disclaimer.sh:/opt/nvidia/nvidia_entrypoint.sh --gpus all ${CTPO_DESTIMAGE} python3 /iti/test/pt_det.py | tee -a ${VAR_PT} | tee -a ${VAR_NT}.testlog; exit "$${PIPESTATUS[0]}"
	@echo "pt_hw"
	@${DOCKER_PRE} docker run --rm -v `pwd`:/iti -v `pwd`/tools/skip_disclaimer.sh:/opt/nvidia/nvidia_entrypoint.sh --gpus all ${CTPO_DESTIMAGE} python3 /iti/test/pt_hw.py | tee -a ${VAR_NT}.testlog; exit "$${PIPESTATUS[0]}"
	@echo "pt_test"
	@${DOCKER_PRE} docker run --rm -v `pwd`:/iti -v `pwd`/tools/skip_disclaimer.sh:/opt/nvidia/nvidia_entrypoint.sh --gpus all ${CTPO_DESTIMAGE} python3 /iti/test/pt_test.py  | tee -a ${VAR_NT}.testlog; exit "$${PIPESTATUS[0]}"

## OpenCV
force_cv_check:
	@echo "cv_hw"
	@${DOCKER_PRE} docker run --rm -v `pwd`:/iti -v `pwd`/tools/skip_disclaimer.sh:/opt/nvidia/nvidia_entrypoint.sh --gpus all ${CTPO_DESTIMAGE} python3 /iti/test/cv_hw.py | tee -a ${VAR_NT}.testlog; exit "$${PIPESTATUS[0]}"

##########
##### Build Details
dump_builddetails:
	@./tools/build_bi_list.py BuildDetails README-BuildDetails.md

##########
##### Docker tag: tags the regular images with the TAG_RELEASE details
## Needed to be run before for the Jupiter Notebook build, unless you want to pull the image from docker hub
docker_tag:
	PTARG="${TPO_BUILDALL} ${CTPO_BUILDALL}" TAG_PRE="${TAG_RELEASE}" CTPO_RELEASE="${CTPO_RELEASE}" DO_UPLOAD="no" make docker_tag_push_core

##########
##### Jupyter Notebook
# Requires the base TPO & CTPO container to either be built locally or docker will attempt to pull otherwise
# make JN_MODE="-user" jupyter-cuda_tensorflow_pytorch_opencv-11.8.0_2.12.0_2.0.1_4.7.0
JN_MODE=""
JN_UID=$(shell id -u)
JN_GID=$(shell id -g)

jupyter_tpo: ${TPO_JUP}

jupyter_ctpo: ${CTPO_JUP}

jupyter_build_all: jupyter_tpo jupyter_ctpo

${TPO_JUP} ${CTPO_JUP}:
	@BTARG="$@" TAG_PRE="${TAG_RELEASE}" CTPO_RELEASE="${CTPO_RELEASE}" make jupyter_build

# Do not call directly, call jupter_build_all or jupyter_tpo or jupyter_ctpo
jupyter_build:
# BTARG: jupyter-tensorflow_opencv-2.12... / split: JX: jupyter, JB: tens...opencv, JT: 2.12...
	@$(eval JX=$(shell echo ${BTARG} | cut -d- -f 1))
	@$(eval JB=$(shell echo ${BTARG} | cut -d- -f 2))
	@$(eval JT=$(shell echo ${BTARG} | cut -d- -f 3))
	@echo "JX: ${JX} | JB: ${JB} | JT: ${JT}"
	@if [ "A${JX}" == "A" ]; then echo "ERROR: Invalid target: ${BTARG}"; exit 1; fi
	@if [ "A${JB}" == "A" ]; then echo "ERROR: Invalid target: ${BTARG}"; exit 1; fi
	@if [ "A${JT}" == "A" ]; then echo "ERROR: Invalid target: ${BTARG}"; exit 1; fi
	@$(eval JUP_FROM_IMAGE="${TAG_RELEASE}${JB}:${JT}-${CTPO_RELEASE}")
	@$(eval JUP_DEST_IMAGE="${JX}-${JB}${JN_MODE}:${JT}-${CTPO_RELEASE}")
	@echo "JUP_FROM_IMAGE: ${JUP_FROM_IMAGE}"
	@echo "JUP_DEST_IMAGE: ${JUP_DEST_IMAGE}"
	@echo "JN_MODE: ${JN_MODE} / JN_UID: ${JN_UID} / JN_GID: ${JN_GID}"
	@TEST_IMAGE="${JUP_FROM_IMAGE}" make check_image_exists_then_pull
	@cd Jupyter_build; docker build --build-arg JUPBC="${JUP_FROM_IMAGE}" --build-arg JUID=${JN_UID} --build-arg JGID=${JN_GID} -f Dockerfile${JN_MODE} --tag="${JUP_DEST_IMAGE}" .
	@if [ "A${DO_UPLOAD}" == "Ayes" ]; then \
		JUP_FINAL_DEST_IMAGE="${TAG_RELEASE}${JUP_DEST_IMAGE}"; \
		echo "Tagging and uploading image: $${JUP_FINAL_DEST_IMAGE}"; \
		echo "Press Ctl+c within 5 seconds to cancel"; \
		for i in 5 4 3 2 1; do echo -n "$$i "; sleep 1; done; echo ""; \
		docker tag ${JUP_DEST_IMAGE} $${JUP_FINAL_DEST_IMAGE}; \
		docker push $${JUP_FINAL_DEST_IMAGE}; \
		tl="$$(echo $${JUP_FINAL_DEST_IMAGE} | perl -pe 's%\:([^\:]+)$$%:latest%')"; \
		docker tag $${JUP_FINAL_DEST_IMAGE} $${tl}; \
		docker push $${tl}; \
	fi


check_image_exists_then_pull:
	@echo "Checking for image: ${TEST_IMAGE}"
	@tmp=$$(docker inspect --type=image --format="Found image" ${TEST_IMAGE} 2> /dev/null); \
	if [ "A$${tmp}" == "A" ]; then \
		echo "Missing image: ${TEST_IMAGE} | Downloading it"; \
		echo "Press Ctl+c within 5 seconds to cancel"; \
		for i in 5 4 3 2 1; do echo -n "$$i "; sleep 1; done; echo ""; \
		docker pull ${TEST_IMAGE}; \
		if [ $$? -ne 0 ]; then \
			echo "ERROR: Unable to pull image: ${TEST_IMAGE}"; \
			exit 1; \
		fi; \
	fi

##### Various cleanup
clean:
	rm -f *.log.temp *.patch.temp

allclean:
	@make clean
	rm -f *.log *.testlog

buildclean:
	@echo "***** Removing ${BuildDetails}/${CTPO_RELEASE} *****"
	@echo "Press Ctl+c within 5 seconds to cancel"
	@for i in 5 4 3 2 1; do echo -n "$$i "; sleep 1; done; echo ""
	rm -rf ${BuildDetails}/${CTPO_RELEASE}

##### For Maintainers only (ie those with write access to the docker hub)
docker_push:
	PTARG="${TPO_BUILDALL} ${CTPO_BUILDALL}" TAG_PRE="${TAG_RELEASE}" CTPO_RELEASE="${CTPO_RELEASE}" DO_UPLOAD="yes" make docker_tag_push_core

docker_push_jup:
	@BTARG="${TPO_JUP}" TAG_PRE="${TAG_RELEASE}" CTPO_RELEASE="${CTPO_RELEASE}" DO_UPLOAD="yes" make jupyter_build
	@BTARG="${CTPO_JUP}" TAG_PRE="${TAG_RELEASE}" CTPO_RELEASE="${CTPO_RELEASE}" DO_UPLOAD="yes" make jupyter_build

docker_tag_push_core:
	@array=(); \
	for t in ${PTARG}; do \
        tag="$$(echo $$t | perl -pe 's%\-([^\-]+)$$%\:$$1%')-$${CTPO_RELEASE}"; \
		echo "** Checking for required image: $${tag}"; \
		tmp=$$(docker inspect --type=image --format="Found image" $${tag} 2> /dev/null); \
		if [ "A$${tmp}" == "A" ]; then \
			echo "Missing image: $${tag}"; \
			exit 1; \
		fi; \
		array+=($${tag}); \
	done; \
	echo "== Found images: $${array[@]}"; \
	echo "== TAG_PRE: $${TAG_PRE}"; \
	echo ""; \
	if [ "A${DO_UPLOAD}" == "Ayes" ]; then \
		echo "++ Tagging then uploading tags to docker hub (no build) -- Press Ctl+c within 5 seconds to cancel -- will only work for maintainers"; \
		for i in 5 4 3 2 1; do echo -n "$$i "; sleep 1; done; echo ""; \
	else \
		echo "++ Tagging only"; \
	fi; \
	for t in $${array[@]}; do \
		echo "Tagging image: $${t}"; \
		tr="$${TAG_PRE}$${t}"; \
		tl="$$(echo $${tr} | perl -pe 's%\:([^\:]+)$$%:latest%')"; \
		docker tag $${t} $${tr}; \
		docker tag $${t} $${tl}; \
		if [ "A${DO_UPLOAD}" == "Ayes" ]; then \
			echo "Uploading image: $${tr}"; \
			docker push $${tr}; \
			docker push $${tl}; \
		fi; \
	done

## Maintainers:
# - Create a new branch on GitHub that match the expected release tag, pull and checkout that branch
# - In the Makefile, update the CTPO_RELEASE variable to match the expected release tag,
#   and make appropriate changes as needed to support the build (ie CUDA version, PyTorch version, ...)
#   At the end, we will tag and make a release for that "release tag" on GitHub
# - Build ALL the CTPO images
#  % make build_ctpo
# - Build ALL the TPO images
#  % make build_tpo
# - Manually check that all the *.testlog contain valid information
#  % less *.testlog
# - Build the README-BuildDetails.md file
#  % make dump_builddetails
# - Add TAG_RELEASE tag to all the built images
#  % make docker_tag
# -Built the Jupyter Lab images
#  % make jupyter_build_all
# - Test the latest Jupyter Lab image, using network port 8765. REPLACE the tag to match the current one.
#   We are mounting pwd to /iti so if you run this from the directoy of this file, you will see the test directory, 
#   so you can create a new Python Notebook and copy the code to test: cw_hw.py, tf_test.py, pt_test.py
#   remember to delete any "extra" files created by this process
#  % docker run --rm -it -v `pwd`:/iti -p 8765:8888 --gpus all jupyter-cuda_tensorflow_pytorch_opencv:REPLACE
# - Build the Unraid images
#  % make JN_MODE="-unraid" jupyter_build_all
# - Upload the images to docker hub
#  % make docker_push
#  % make docker_push_jup
#  % make JN_MODE="-unraid" docker_push_jup
# - Update the README.md file with the new release tag + version history
# - Commit and push the changes to GitHub (in the branch created at the beginning)
# - On Github, "Open a pull request", 
#   use the value of CTPO_RELEASE for the release name (ie the YYYYMMDD value)
#   add PR modifications as a summry of the content of the commits,
#   create the PR, add a self-approve message, merge and delete the branch
# - on the build system, checkout main and pull the changes
#  % git checkout main
#  % git pull
# - delete the temporary branch (named after the CTPO_RELEASE value)
#  % git branch -d YYYYMMDD
# - Tag the release on GitHub
#  % git tag YYYYMMDD
#  % git push origin YYYYMMDD
# - Create a release on GitHub using the YYYYMMDD tag, add the release notes, and publish
