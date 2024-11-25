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

ARG CTPO_FROM=ubuntu:22.04
FROM ${CTPO_FROM}

# No CUDA, or no cuDNN bootstrap needed

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
ENV LANG=en_US.utf8

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
    patchelf \
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
  # Torch
    libomp-dev \
    libsox-dev \
    libsox-fmt-all \
    libsphinxbase-dev \
    sphinxbase-utils \
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

# No Additional CUDA apt installs



ENV CTPO_CLANG_VERSION=17
RUN apt-get install -y lsb-release software-properties-common gnupg \
  && mkdir /tmp/clang \
  && cd /tmp/clang \
  && wget https://apt.llvm.org/llvm.sh \
  && chmod +x llvm.sh \
  && ./llvm.sh ${CTPO_CLANG_VERSION} \
  && cd / \
  && rm -rf /tmp/clang
RUN which clang-${CTPO_CLANG_VERSION} \
  && which clang++-${CTPO_CLANG_VERSION} \
  && clang-${CTPO_CLANG_VERSION} --version \
  && clang++-${CTPO_CLANG_VERSION} --version

ENV CTPO_BUILD=CPU
RUN touch /tmp/.${CTPO_BUILD}_build

# - Avoid "RPC failed; curl 56 GnuTLS" issue on some pulls, while keeping the system installed git
# - Some tools expect a "python" binary
# - Prepare ldconfig
RUN git config --global http.postBuffer 1048576000 \
  && mkdir -p /usr/local/bin && ln -s $(which python3) /usr/local/bin/python \
  && mkdir -p /usr/local/lib && sh -c 'echo "/usr/local/lib" >> /etc/ld.so.conf.d/usrlocallib.conf' && ldconfig

ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Not a GPU build, no path fix to test


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
  # Torch
    cffi \
    typing \
    ninja \
  # Extra
    scikit-image \
    scikit-learn \
    matplotlib \
    moviepy \
    pandas \
  # No keras
    mkl-static mkl-include \
  && rm -rf /root/.cache/pip

# No TensorFlow build
RUN echo "No TensorFlow built" > /tmp/tf_env.dump


##### FFMPEG

ENV CTPO_FFMPEG_VERSION=7.1
ENV CTPO_FFMPEG_NONFREE=""
RUN mkdir -p /usr/local/src/builder \
  && cd /usr/local/src \
  && wget -q https://ffmpeg.org/releases/ffmpeg-${CTPO_FFMPEG_VERSION}.tar.gz -O - | tar --strip-components=1 -xz -C /usr/local/src/builder \
  && cd /usr/local/src/builder \
  && time ./configure --enable-shared --disable-static --enable-gpl --enable-libv4l2 --enable-libvorbis --enable-libvpx --enable-libwebp --enable-libxvid --enable-libopus --enable-pic --enable-libass --enable-libx264 --enable-libx265 | tee /tmp/ffmpeg_config.txt \
  && make -j${CTPO_NUMPROC} install \
  && ldconfig \
  && rm -rf /usr/local/src/builder

RUN echo "${CTPO_FFMPEG_VERSION}" > /tmp/.ffmpeg_built

#### OpenCV

# Download & Build OpenCV in same RUN
## FYI: We are removing the OpenCV source and build directory in /usr/local/src to attempt to save additional disk space
# Comment the last line if you want to rerun cmake with additional/modified options. For example:
# cd /usr/local/src/opencv/build
# cmake -DOPENCV_ENABLE_NONFREE=ON -DBUILD_EXAMPLES=ON -DBUILD_DOCS=ON -DBUILD_TESTS=ON -DBUILD_PERF_TESTS=ON .. && make install
ENV CTPO_OPENCV_VERSION=4.10.0
ENV CTPO_OPENCV_CUDA=""
ENV CTPO_OPENCV_NONFREE=""
RUN mkdir -p /usr/local/src/opencv/build /usr/local/src/opencv_contrib \
  && cd /usr/local/src \
  && wget -q https://github.com/opencv/opencv/archive/${CTPO_OPENCV_VERSION}.tar.gz -O - | tar --strip-components=1 -xz -C /usr/local/src/opencv \
  && wget -q https://github.com/opencv/opencv_contrib/archive/${CTPO_OPENCV_VERSION}.tar.gz -O - | tar --strip-components=1 -xz -C /usr/local/src/opencv_contrib \
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

##### Torch (using FFMpeg + OpenCV)

ENV CTPO_TORCH=2.5.1
RUN mkdir -p /usr/local/src \
  && cd /usr/local/src \
  && git clone --depth 1 --recurse-submodule --branch v${CTPO_TORCH} https://github.com/pytorch/pytorch.git  \
  && cd /usr/local/src/pytorch \
  && pip3 install -r requirements.txt \
  && time env PYTORCH_BUILD_VERSION=${CTPO_TORCH} PYTORCH_BUILD_NUMBER=0 USE_CUDA=0 USE_CUDNN=0 USE_MKLDNN=1 USE_FFMPEG=1 USE_OPENCV=1 python3 ./setup.py bdist_wheel | tee /tmp/torch_config.txt \
  && time pip3 install /usr/local/src/pytorch/dist/*.whl \
  && cd /tmp \
  && perl -i.bak -pe 'exit if m%^-- Configuring done%' torch_config.txt \
  && sh -c "cmp --silent torch_config.txt torch_config.txt.bak && exit 1 || rm torch_config.txt.bak" \
  && ldconfig \
  && rm -rf /root/.cache/pip /usr/local/src/pytorch

RUN python3 -c 'import torch; print(f"{torch.__version__}")' > /tmp/.torch_built

ENV CTPO_TORCHVISION=0.20.0
# setup.py options from https://github.com/pytorch/vision/blob/main/setup.py
RUN mkdir -p /usr/local/src \
  && cd /usr/local/src \
  && git clone  --depth 1 --recurse-submodule --branch v${CTPO_TORCHVISION} https://github.com/pytorch/vision.git  \
  && cd /usr/local/src/vision \
  && time env BUILD_VERSION=${CTPO_TORCHVISION} FORCE_CUDA=0 TORCHVISION_USE_FFMPEG=1 python3 ./setup.py bdist_wheel | tee /tmp/torchvision_config.txt \
  && time pip3 install /usr/local/src/vision/dist/*.whl \
  && cd /tmp \
  && perl -i.bak -pe 'exit if m%^running bdist_wheel%' torchvision_config.txt \
  && sh -c "cmp --silent torchvision_config.txt torchvision_config.txt.bak && exit 1 || rm torchvision_config.txt.bak" \
  && ldconfig \
  && rm -rf /root/.cache/pip /usr/local/src/vision

RUN python3 -c 'import torchvision; print(f"{torchvision.__version__}")' > /tmp/.torchvision_built

ENV CTPO_TORCHAUDIO=2.5.0
# setup.py options from https://github.com/pytorch/audio/blob/main/tools/setup_helpers/extension.py
RUN mkdir -p /usr/local/src \
  && cd /usr/local/src \
  && git clone --depth 1 --recurse-submodule --branch v${CTPO_TORCHAUDIO} https://github.com/pytorch/audio.git \
  && cd /usr/local/src/audio \
  && time env BUILD_VERSION=${CTPO_TORCHAUDIO} USE_CUDA=0 USE_FFMPEG=1 python3 ./setup.py bdist_wheel | tee /tmp/torchaudio_config.txt \
  && time pip3 install /usr/local/src/audio/dist/*.whl \
  && cd /tmp \
  && perl -i.bak -pe 'exit if m%^-- Configuring done%' torchaudio_config.txt \
  && sh -c "cmp --silent torchaudio_config.txt torchaudio_config.txt.bak && exit 1 || rm torchaudio_config.txt.bak" \
  && ldconfig \
  && rm -rf /root/.cache/pip /usr/local/src/audio

RUN python3 -c 'import torchaudio; print(f"{torchaudio.__version__}")' > /tmp/.torchaudio_built

ENV CTPO_TORCHDATA=0.9.0
RUN mkdir -p /usr/local/src \
  && cd /usr/local/src \
  && git clone --depth 1 --recurse-submodule --branch v${CTPO_TORCHDATA} https://github.com/pytorch/data.git  \
  && cd /usr/local/src/data \
  && BUILD_VERSION=${CTPO_TORCHDATA} pip3 install . | tee /tmp/torchdata_config.txt \
  && rm -rf /root/.cache/pip /usr/local/src/data

RUN python3 -c 'import torchdata; print(f"{torchdata.__version__}")' > /tmp/.torchdata_built

##### Final steps

# Add Jupyter lab & venv requirements
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

CMD ["/bin/bash"]