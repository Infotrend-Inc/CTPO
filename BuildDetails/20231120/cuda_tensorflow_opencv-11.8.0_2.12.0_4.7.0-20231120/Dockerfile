##### Splitting installation into controlled parts:
## - Base container setup (all apt-get + system + ldd extension)
## - Python pip3 preparation + First batch of python tools installs
## - Tensorflow build and install (optional)
## - FFmpeg build and install
## - OpenCV build and install (using FFmpeg)
## - Additional python tools installs
## - Magma build and install (GPU only, optional)
## - PyTorch (+ vison + audio) build and install (using FFmpeg + OpenCV, optional)
## - Final steps: /iti setup, ...

# The idea is to have a base Dockerfile that will be tailored to specific builds 
# so that each build has their own final Dockerfile available for inspection

ARG CTPO_FROM=nvidia/cuda:11.8.0-devel-ubuntu22.04
FROM ${CTPO_FROM}

# Adapted from https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/11.8.0/ubuntu2204/devel/cudnn8/Dockerfile
ENV NV_CUDNN_VERSION 8.6.0.163
ENV NV_CUDNN_PACKAGE_NAME "libcudnn8"
ENV NV_CUDA_ADD=cuda11.8
ENV NV_CUDNN_PACKAGE "$NV_CUDNN_PACKAGE_NAME=$NV_CUDNN_VERSION-1+$NV_CUDA_ADD"
ENV NV_CUDNN_PACKAGE_DEV "$NV_CUDNN_PACKAGE_NAME-dev=$NV_CUDNN_VERSION-1+$NV_CUDA_ADD"
LABEL com.nvidia.cudnn.version="${NV_CUDNN_VERSION}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ${NV_CUDNN_PACKAGE} \
    ${NV_CUDNN_PACKAGE_DEV} \
    && apt-mark hold ${NV_CUDNN_PACKAGE_NAME} \
    && rm -rf /var/lib/apt/lists/*

ARG CTPO_NUMPROC=32

##### Base

# Install system packages
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y --fix-missing\
  && apt-get install -y \
    apt-utils \
    locales \
    wget \
    ca-certificates \
  && apt-get clean

# UTF-8
RUN localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8
ENV LANG en_US.utf8

# Install system packages
RUN apt-get install -y \
  # Base
    build-essential \
    checkinstall \
    cmake \
    curl \
    g++ \
    gcc \
    git \
    perl \
    pkg-config \
    protobuf-compiler \
    python3-dev \
    rsync \
    time \
    unzip \
    wget \
    zip \
    zlib1g \
    zlib1g-dev \
  # OpenCV
    doxygen \
    file \
    gfortran \
    gnupg \
    gstreamer1.0-plugins-good \
    imagemagick \
    libatk-adaptor \
    libatlas-base-dev \
    libboost-all-dev \
    libcanberra-gtk-module \
    libdc1394-dev \
    libeigen3-dev \
    libfaac-dev \
    libfreetype6-dev \
    libgflags-dev \
    libglew-dev \
    libglu1-mesa \
    libglu1-mesa-dev \
    libgoogle-glog-dev \
    libgphoto2-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-bad1.0-0 \
    libgstreamer-plugins-base1.0-dev \
    libgtk2.0-dev \
    libgtk-3-dev \
    libhdf5-dev \
    libhdf5-serial-dev \
    libjpeg-dev \
    liblapack-dev \
    libmp3lame-dev \
    libopenblas-dev \
    libopencore-amrnb-dev \
    libopencore-amrwb-dev \
    libopenjp2-7 \
    libopenjp2-7-dev \
    libopenjp2-tools \
    libopenjpip-server \
    libpng-dev \
    libpostproc-dev \
    libprotobuf-dev \
    libtbb2 \
    libtbb-dev \
    libtheora-dev \
    libtiff5-dev \
    libv4l-dev \
    libvorbis-dev \
    libwebp-dev \
    libx264-dev \
    libx265-dev \
    libxi-dev \
    libxine2-dev \
    libxmu-dev \
    libxvidcore-dev \
    libzmq3-dev \
    v4l-utils \
    x11-apps \
    x264 \
    yasm \
  # No adds for torch
  # FFMpeg (source install, do not install packages: libavcodec-dev libavformat-dev libavresample-dev libavutil-dev libswscale-dev)
    libass-dev \
    libc6 \
    libc6-dev \
    libnuma1 \
    libnuma-dev \
    libopus-dev \
    libtool \
    libvpx-dev \
  && apt-get clean



ENV CTPO_BUILD=GPU
RUN touch /tmp/.${CTPO_BUILD}_build

# - Avoid "RPC failed; curl 56 GnuTLS" issue on some pulls, while keeping the system installed git
# - Some tools expect a "python" binary
# - Prepare ldconfig
RUN git config --global http.postBuffer 1048576000 \
  && mkdir -p /usr/local/bin && ln -s $(which python3) /usr/local/bin/python \
  && mkdir -p /usr/local/lib && sh -c 'echo "/usr/local/lib" >> /etc/ld.so.conf.d/usrlocallib.conf' && ldconfig

ENV LD_LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH

# Misc GPU fixes
RUN cd /usr/local \
  && if [ -e cuda ]; then if [ ! -e nvidia ]; then ln -s cuda nvidia; fi; fi \
  && tmp="/usr/local/cuda/extras/CUPTI/lib64" \
  && if [ -d $tmp ]; then \ 
       echo $tmp >> /etc/ld.so.conf.d/nvidia-cupti.conf; \
       ldconfig; \
       echo "***** CUPTI added to LD path"; \
     fi


# Setup pip
RUN wget -q -O /tmp/get-pip.py --no-check-certificate https://bootstrap.pypa.io/get-pip.py \
  && python3 /tmp/get-pip.py \
  && pip3 install -U pip \
  && rm /tmp/get-pip.py

# Install Python tools (for buiding) 
RUN pip3 install -U \
  # Base
    cmake \
    future \
    mkl \
    mkl-include \
    mock \
    numpy \
    opt_einsum \
    packaging \
    pyyaml \
    requests \
    setuptools \
    six \
    wheel \
  # OpenCV
    lxml \
    Pillow \
  # No Torch addons
  # Extra
    scikit-image \
    scikit-learn \
    autovizwidget \
    ipython \
    jupyter \
    matplotlib \
    moviepy \
    notebook \
    pandas \
  && pip3 install -U keras_preprocessing --no-deps \
  && rm -rf /root/.cache/pip

##### TensorFlow
# https://www.tensorflow.org/install/source

## Download & Building TensorFlow from source in same RUN
ENV LATEST_BAZELISK=1.17.0
ENV CTPO_TENSORFLOW_VERSION=2.12.0
ENV CTPO_TF_CONFIG="--config=cuda"
ENV TF_CUDA_COMPUTE_CAPABILITIES=6.0,6.1,7.0,7.5,8.0,8.6,8.9,9.0

ENV TF_NEED_CUDA=1
ENV TF_CUDA_CLANG=0
ENV TF_NEED_TENSORRT=0 
ENV TF_NEED_AWS=0 
ENV TF_NEED_COMPUTECPP=0 
ENV TF_NEED_GCP=0 
ENV TF_NEED_GDR=0
ENV TF_NEED_HDFS=0 
ENV TF_NEED_JEMALLOC=0
ENV TF_NEED_KAFKA=0
ENV TF_NEED_MKL=0
ENV TF_NEED_MPI=0
ENV TF_NEED_OPENCL=0
ENV TF_NEED_OPENCL_SYCL=0
ENV TF_NEED_ROCM=0
ENV TF_NEED_S3=0
ENV TF_NEED_VERBS=0
ENV TF_SET_ANDROID_WORKSPACE=0

RUN curl -s -Lo /usr/local/bin/bazel https://github.com/bazelbuild/bazelisk/releases/download/v${LATEST_BAZELISK}/bazelisk-linux-amd64 \
  && chmod +x /usr/local/bin/bazel \
  && mkdir -p /usr/local/src/tensorflow \
  && cd /usr/local/src \
  && wget -q --no-check-certificate -c https://github.com/tensorflow/tensorflow/archive/v${CTPO_TENSORFLOW_VERSION}.tar.gz -O - | tar --strip-components=1 -xz -C /usr/local/src/tensorflow \
  && cd /usr/local/src/tensorflow \
  && bazel version \
  && ./configure \
  && time bazel build --verbose_failures --config=opt --config=v2 ${CTPO_TF_CONFIG} //tensorflow/tools/pip_package:build_pip_package \
  && ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg \
  && time pip3 install /tmp/tensorflow_pkg/tensorflow-*.whl \
  && rm -rf /usr/local/src/tensorflow /tmp/tensorflow_pkg /tmp/bazel_check.pl /tmp/tf_build.sh /tmp/hsperfdata_root /root/.cache/bazel /root/.cache/pip /root/.cache/bazelisk

RUN python3 -c 'import tensorflow as tf; print(f"{tf.__version__}")' > /tmp/.tensorflow_built

RUN echo "--- Tensorflow Build: Environment variables set --- " > /tmp/tf_env.dump \
  && env | grep TF_ | grep -v CTPO_ | sort >> /tmp/tf_env.dump

##### FFMPEG

ENV CTPO_FFMPEG_VERSION=5.1.2
ENV CTPO_FFMPEG_NVCODEC=11.1.5.2
ENV CTPO_FFMPEG_NONFREE=""
RUN mkdir -p /usr/local/src/builder \
  && cd /usr/local/src \
  && wget -q --no-check-certificate https://github.com/FFmpeg/nv-codec-headers/archive/refs/tags/n${CTPO_FFMPEG_NVCODEC}.tar.gz -O - | tar --strip-components=1 -xz -C /usr/local/src/builder \
  && cd /usr/local/src/builder \
  && make install \
  && rm -rf /usr/local/src/builder
RUN mkdir -p /usr/local/src/builder \
  && cd /usr/local/src \
  && wget -q --no-check-certificate https://ffmpeg.org/releases/ffmpeg-${CTPO_FFMPEG_VERSION}.tar.gz -O - | tar --strip-components=1 -xz -C /usr/local/src/builder \
  && cd /usr/local/src/builder \
  && time ./configure --enable-cuda --enable-cuvid --enable-nvdec --enable-nvenc ${CTPO_FFMPEG_NONFREE} --extra-cflags="-I/usr/local/cuda/include/ -fPIC" --extra-ldflags="-L/usr/local/cuda/lib64/ -Wl,-Bsymbolic" --enable-shared --disable-static --enable-gpl --enable-libv4l2 --enable-libvorbis --enable-libvpx --enable-libwebp --enable-libxvid --enable-libopus --enable-pic --enable-libass --enable-libx264 --enable-libx265 | tee /tmp/ffmpeg_config.txt \
  && make -j${CTPO_NUMPROC} install \
  && ldconfig \
  && rm -rf /usr/local/src/builder
# From https://docs.nvidia.com/video-technologies/video-codec-sdk/ffmpeg-with-nvidia-gpu/#basic-testing
# GPU Testing: ffmpeg -y -vsync 0 -hwaccel cuda -hwaccel_output_format cuda -i input.mp4 -c:a copy -c:v h264_nvenc -b:v 5M output.mp4
# GPU Testing: ffmpeg -y -vsync 0 -hwaccel cuvid -c:v h264_cuvid -i in.mp4 -c:v hevc_nvenc out.mkv

RUN echo "${CTPO_FFMPEG_VERSION}" > /tmp/.ffmpeg_built

#### OpenCV

# Download & Build OpenCV in same RUN
## FYI: We are removing the OpenCV source and build directory in /usr/local/src to attempt to save additional disk space
# Comment the last line if you want to rerun cmake with additional/modified options. For example:
# cd /usr/local/src/opencv/build
# cmake -DOPENCV_ENABLE_NONFREE=ON -DBUILD_EXAMPLES=ON -DBUILD_DOCS=ON -DBUILD_TESTS=ON -DBUILD_PERF_TESTS=ON .. && make install
ENV CTPO_OPENCV_VERSION=4.7.0
ENV CTPO_OPENCV_CUDA="-D WITH_CUDA=ON -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda -D CMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs -D CUDA_FAST_MATH=1 -D WITH_CUBLAS=1 -D WITH_NVCUVID=1 -DCUDA_ARCH_BIN=6.0,6.1,7.0,7.5,8.0,8.6,8.9,9.0 -DCUDA_ARCH_PTX=6.0,6.1,7.0,7.5,8.0,8.6,8.9,9.0"
ENV CTPO_OPENCV_NONFREE=""
RUN mkdir -p /usr/local/src/opencv/build /usr/local/src/opencv_contrib \
  && cd /usr/local/src \
  && wget -q --no-check-certificate https://github.com/opencv/opencv/archive/${CTPO_OPENCV_VERSION}.tar.gz -O - | tar --strip-components=1 -xz -C /usr/local/src/opencv \
  && wget -q --no-check-certificate https://github.com/opencv/opencv_contrib/archive/${CTPO_OPENCV_VERSION}.tar.gz -O - | tar --strip-components=1 -xz -C /usr/local/src/opencv_contrib \
  && cd /usr/local/src/opencv/build \
  && time cmake \
    -DBUILD_DOCS=0 \
    -DBUILD_EXAMPLES=0 \
    -DBUILD_PERF_TESTS=0 \
    -DBUILD_TESTS=0 \
    -DBUILD_opencv_python2=0 \
    -DBUILD_opencv_python3=1 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local/ \
    -DCMAKE_INSTALL_TYPE=Release \
    -DENABLE_FAST_MATH=1 \
    -DFORCE_VTK=1 \
    -DINSTALL_C_EXAMPLES=0 \
    -DINSTALL_PYTHON_EXAMPLES=0 \
    -DOPENCV_EXTRA_MODULES_PATH=/usr/local/src/opencv_contrib/modules \
    -DOPENCV_GENERATE_PKGCONFIG=1 \
    -DOPENCV_PC_FILE_NAME=opencv.pc \
    -DOPENCV_PYTHON3_INSTALL_PATH=/usr/lib/python3/dist-packages \
    -DWITH_CSTRIPES=1 \
    -DWITH_EIGEN=1 \
    -DWITH_GDAL=1 \
    -DWITH_GSTREAMER=1 \
    -DWITH_GSTREAMER_0_10=OFF \
    -DWITH_GTK=1 \
    -DWITH_IPP=1 \
    -DWITH_OPENCL=1 \
    -DWITH_OPENMP=1 \
    -DWITH_TBB=1 \
    -DWITH_V4L=1 \
    -DWITH_WEBP=1 \
    -DWITH_XINE=1 \
    ${CTPO_OPENCV_CUDA} \
    ${CTPO_OPENCV_NONFREE} \
    .. \
  && time make -j${CTPO_NUMPROC} install \
  && ldconfig \
  && rm -rf /usr/local/src/opencv /usr/local/src/opencv_contrib

RUN python3 -c 'import cv2; print(f"{cv2.__version__}")' > /tmp/.opencv_built

# No Magma (PyTorch GPU only)

# No PyTorch, Torch Audio or Torch Video
RUN echo "No PyTorch built" > /tmp/torch_config.txt \
  && echo "No TorchVision built" > /tmp/torchvision_config.txt \
  && echo "No TorchAudio built" > /tmp/torchaudio_config.txt \
  && echo "No TorchData built" > /tmp/torchdata_config.txt \
  && echo "No TorchText built" > /tmp/torchtext_config.txt


##### Final steps

# Add lab & dataframe display widget
RUN apt-get install -y python3-venv \
  && pip3 install -U \
      jupyterlab
# https://jupyterlab.readthedocs.io/en/stable/getting_started/changelog.html
# with v4 extensions are installed using the extension manager, see https://jupyterlab.readthedocs.io/en/stable/user/extensions.html#id11

RUN touch /.within_container
COPY tools/withincontainer_checker.sh /tmp/withincontainer_checker.sh
RUN chmod +x /tmp/withincontainer_checker.sh \
  && /tmp/withincontainer_checker.sh

# Setting up working directory 
RUN mkdir /iti
WORKDIR /iti

# Uncomment as needed 
#ENV NVIDIA_VISIBLE_DEVICES all
#ENV NVIDIA_DRIVER_CAPABILITIES video,compute,utility

CMD /bin/bash