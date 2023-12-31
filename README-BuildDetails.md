# Available Builds
  - [CPU](#CPU)
    - [pytorch_opencv](#pytorch_opencv)
    - [tensorflow_opencv](#tensorflow_opencv)
    - [tensorflow_pytorch_opencv](#tensorflow_pytorch_opencv)
  - [GPU](#GPU)
    - [cuda_pytorch_opencv](#cuda_pytorch_opencv)
    - [cuda_tensorflow_opencv](#cuda_tensorflow_opencv)
    - [cuda_tensorflow_pytorch_opencv](#cuda_tensorflow_pytorch_opencv)
## CPU
### pytorch_opencv
| Docker tag | PyTorch | OpenCV | FFmpeg | Ubuntu |
| --- | --- | --- | --- | --- |
| [2.1.1_4.8.0-20231201](BuildDetails/20231201/pytorch_opencv-2.1.1_4.8.0-20231201/Dockerfile) | [2.1.0a0+git4c55dc5](BuildDetails/20231201/pytorch_opencv-2.1.1_4.8.0-20231201/PyTorch--Details.txt) | [4.8.0](BuildDetails/20231201/pytorch_opencv-2.1.1_4.8.0-20231201/OpenCV--Details.txt) | [5.1.4](BuildDetails/20231201/pytorch_opencv-2.1.1_4.8.0-20231201/FFmpeg--Details.txt) | [22.04](BuildDetails/20231201/pytorch_opencv-2.1.1_4.8.0-20231201/System--Details.txt) |
| [2.0.1_4.7.0-20231120](BuildDetails/20231120/pytorch_opencv-2.0.1_4.7.0-20231120/Dockerfile) | [2.0.0a0+gite9ebda2](BuildDetails/20231120/pytorch_opencv-2.0.1_4.7.0-20231120/PyTorch--Details.txt) | [4.7.0](BuildDetails/20231120/pytorch_opencv-2.0.1_4.7.0-20231120/OpenCV--Details.txt) | [5.1.2](BuildDetails/20231120/pytorch_opencv-2.0.1_4.7.0-20231120/FFmpeg--Details.txt) | [22.04](BuildDetails/20231120/pytorch_opencv-2.0.1_4.7.0-20231120/System--Details.txt) |

### tensorflow_opencv
| Docker tag | TensorFlow | OpenCV | FFmpeg | Ubuntu |
| --- | --- | --- | --- | --- |
| [2.14.1_4.8.0-20231201](BuildDetails/20231201/tensorflow_opencv-2.14.1_4.8.0-20231201/Dockerfile) | [2.14.1](BuildDetails/20231201/tensorflow_opencv-2.14.1_4.8.0-20231201/TensorFlow--Details.txt) | [4.8.0](BuildDetails/20231201/tensorflow_opencv-2.14.1_4.8.0-20231201/OpenCV--Details.txt) | [5.1.4](BuildDetails/20231201/tensorflow_opencv-2.14.1_4.8.0-20231201/FFmpeg--Details.txt) | [22.04](BuildDetails/20231201/tensorflow_opencv-2.14.1_4.8.0-20231201/System--Details.txt) |
| [2.12.0_4.7.0-20231120](BuildDetails/20231120/tensorflow_opencv-2.12.0_4.7.0-20231120/Dockerfile) | [2.12.0](BuildDetails/20231120/tensorflow_opencv-2.12.0_4.7.0-20231120/TensorFlow--Details.txt) | [4.7.0](BuildDetails/20231120/tensorflow_opencv-2.12.0_4.7.0-20231120/OpenCV--Details.txt) | [5.1.2](BuildDetails/20231120/tensorflow_opencv-2.12.0_4.7.0-20231120/FFmpeg--Details.txt) | [22.04](BuildDetails/20231120/tensorflow_opencv-2.12.0_4.7.0-20231120/System--Details.txt) |

### tensorflow_pytorch_opencv
| Docker tag | TensorFlow | PyTorch | OpenCV | FFmpeg | Ubuntu |
| --- | --- | --- | --- | --- | --- |
| [2.14.1_2.1.1_4.8.0-20231201](BuildDetails/20231201/tensorflow_pytorch_opencv-2.14.1_2.1.1_4.8.0-20231201/Dockerfile) | [2.14.1](BuildDetails/20231201/tensorflow_pytorch_opencv-2.14.1_2.1.1_4.8.0-20231201/TensorFlow--Details.txt) | [2.1.0a0+git4c55dc5](BuildDetails/20231201/tensorflow_pytorch_opencv-2.14.1_2.1.1_4.8.0-20231201/PyTorch--Details.txt) | [4.8.0](BuildDetails/20231201/tensorflow_pytorch_opencv-2.14.1_2.1.1_4.8.0-20231201/OpenCV--Details.txt) | [5.1.4](BuildDetails/20231201/tensorflow_pytorch_opencv-2.14.1_2.1.1_4.8.0-20231201/FFmpeg--Details.txt) | [22.04](BuildDetails/20231201/tensorflow_pytorch_opencv-2.14.1_2.1.1_4.8.0-20231201/System--Details.txt) |
| [2.12.0_2.0.1_4.7.0-20231120](BuildDetails/20231120/tensorflow_pytorch_opencv-2.12.0_2.0.1_4.7.0-20231120/Dockerfile) | [2.12.0](BuildDetails/20231120/tensorflow_pytorch_opencv-2.12.0_2.0.1_4.7.0-20231120/TensorFlow--Details.txt) | [2.0.0a0+gite9ebda2](BuildDetails/20231120/tensorflow_pytorch_opencv-2.12.0_2.0.1_4.7.0-20231120/PyTorch--Details.txt) | [4.7.0](BuildDetails/20231120/tensorflow_pytorch_opencv-2.12.0_2.0.1_4.7.0-20231120/OpenCV--Details.txt) | [5.1.2](BuildDetails/20231120/tensorflow_pytorch_opencv-2.12.0_2.0.1_4.7.0-20231120/FFmpeg--Details.txt) | [22.04](BuildDetails/20231120/tensorflow_pytorch_opencv-2.12.0_2.0.1_4.7.0-20231120/System--Details.txt) |

## GPU
### cuda_pytorch_opencv
| Docker tag | CUDA | cuDNN | PyTorch | OpenCV | FFmpeg | Ubuntu |
| --- | --- | --- | --- | --- | --- | --- |
| [11.8.0_2.1.1_4.8.0-20231201](BuildDetails/20231201/cuda_pytorch_opencv-11.8.0_2.1.1_4.8.0-20231201/Dockerfile) | 11.8.89 | 8.9.6 | [2.1.0a0+git4c55dc5](BuildDetails/20231201/cuda_pytorch_opencv-11.8.0_2.1.1_4.8.0-20231201/PyTorch--Details.txt) | [4.8.0](BuildDetails/20231201/cuda_pytorch_opencv-11.8.0_2.1.1_4.8.0-20231201/OpenCV--Details.txt) | [5.1.4](BuildDetails/20231201/cuda_pytorch_opencv-11.8.0_2.1.1_4.8.0-20231201/FFmpeg--Details.txt) | [22.04](BuildDetails/20231201/cuda_pytorch_opencv-11.8.0_2.1.1_4.8.0-20231201/System--Details.txt) |
| [11.8.0_2.0.1_4.7.0-20231120](BuildDetails/20231120/cuda_pytorch_opencv-11.8.0_2.0.1_4.7.0-20231120/Dockerfile) | 11.8.89 | 8.9.6 | [2.0.0a0+gite9ebda2](BuildDetails/20231120/cuda_pytorch_opencv-11.8.0_2.0.1_4.7.0-20231120/PyTorch--Details.txt) | [4.7.0](BuildDetails/20231120/cuda_pytorch_opencv-11.8.0_2.0.1_4.7.0-20231120/OpenCV--Details.txt) | [5.1.2](BuildDetails/20231120/cuda_pytorch_opencv-11.8.0_2.0.1_4.7.0-20231120/FFmpeg--Details.txt) | [22.04](BuildDetails/20231120/cuda_pytorch_opencv-11.8.0_2.0.1_4.7.0-20231120/System--Details.txt) |

### cuda_tensorflow_opencv
| Docker tag | CUDA | cuDNN | TensorFlow | OpenCV | FFmpeg | Ubuntu |
| --- | --- | --- | --- | --- | --- | --- |
| [11.8.0_2.14.1_4.8.0-20231201](BuildDetails/20231201/cuda_tensorflow_opencv-11.8.0_2.14.1_4.8.0-20231201/Dockerfile) | 11.8.89 | 8.7.0 | [2.14.1](BuildDetails/20231201/cuda_tensorflow_opencv-11.8.0_2.14.1_4.8.0-20231201/TensorFlow--Details.txt) | [4.8.0](BuildDetails/20231201/cuda_tensorflow_opencv-11.8.0_2.14.1_4.8.0-20231201/OpenCV--Details.txt) | [5.1.4](BuildDetails/20231201/cuda_tensorflow_opencv-11.8.0_2.14.1_4.8.0-20231201/FFmpeg--Details.txt) | [22.04](BuildDetails/20231201/cuda_tensorflow_opencv-11.8.0_2.14.1_4.8.0-20231201/System--Details.txt) |
| [11.8.0_2.12.0_4.7.0-20231120](BuildDetails/20231120/cuda_tensorflow_opencv-11.8.0_2.12.0_4.7.0-20231120/Dockerfile) | 11.8.89 | 8.6.0 | [2.12.0](BuildDetails/20231120/cuda_tensorflow_opencv-11.8.0_2.12.0_4.7.0-20231120/TensorFlow--Details.txt) | [4.7.0](BuildDetails/20231120/cuda_tensorflow_opencv-11.8.0_2.12.0_4.7.0-20231120/OpenCV--Details.txt) | [5.1.2](BuildDetails/20231120/cuda_tensorflow_opencv-11.8.0_2.12.0_4.7.0-20231120/FFmpeg--Details.txt) | [22.04](BuildDetails/20231120/cuda_tensorflow_opencv-11.8.0_2.12.0_4.7.0-20231120/System--Details.txt) |

### cuda_tensorflow_pytorch_opencv
| Docker tag | CUDA | cuDNN | TensorFlow | PyTorch | OpenCV | FFmpeg | Ubuntu |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [11.8.0_2.14.1_2.1.1_4.8.0-20231201](BuildDetails/20231201/cuda_tensorflow_pytorch_opencv-11.8.0_2.14.1_2.1.1_4.8.0-20231201/Dockerfile) | 11.8.89 | 8.7.0 | [2.14.1](BuildDetails/20231201/cuda_tensorflow_pytorch_opencv-11.8.0_2.14.1_2.1.1_4.8.0-20231201/TensorFlow--Details.txt) | [2.1.0a0+git4c55dc5](BuildDetails/20231201/cuda_tensorflow_pytorch_opencv-11.8.0_2.14.1_2.1.1_4.8.0-20231201/PyTorch--Details.txt) | [4.8.0](BuildDetails/20231201/cuda_tensorflow_pytorch_opencv-11.8.0_2.14.1_2.1.1_4.8.0-20231201/OpenCV--Details.txt) | [5.1.4](BuildDetails/20231201/cuda_tensorflow_pytorch_opencv-11.8.0_2.14.1_2.1.1_4.8.0-20231201/FFmpeg--Details.txt) | [22.04](BuildDetails/20231201/cuda_tensorflow_pytorch_opencv-11.8.0_2.14.1_2.1.1_4.8.0-20231201/System--Details.txt) |
| [11.8.0_2.12.0_2.0.1_4.7.0-20231120](BuildDetails/20231120/cuda_tensorflow_pytorch_opencv-11.8.0_2.12.0_2.0.1_4.7.0-20231120/Dockerfile) | 11.8.89 | 8.6.0 | [2.12.0](BuildDetails/20231120/cuda_tensorflow_pytorch_opencv-11.8.0_2.12.0_2.0.1_4.7.0-20231120/TensorFlow--Details.txt) | [2.0.0a0+gite9ebda2](BuildDetails/20231120/cuda_tensorflow_pytorch_opencv-11.8.0_2.12.0_2.0.1_4.7.0-20231120/PyTorch--Details.txt) | [4.7.0](BuildDetails/20231120/cuda_tensorflow_pytorch_opencv-11.8.0_2.12.0_2.0.1_4.7.0-20231120/OpenCV--Details.txt) | [5.1.2](BuildDetails/20231120/cuda_tensorflow_pytorch_opencv-11.8.0_2.12.0_2.0.1_4.7.0-20231120/FFmpeg--Details.txt) | [22.04](BuildDetails/20231120/cuda_tensorflow_pytorch_opencv-11.8.0_2.12.0_2.0.1_4.7.0-20231120/System--Details.txt) |

