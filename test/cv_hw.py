import cv2

def check_cuda():
    try:
        count = cv2.cuda.getCudaEnabledDeviceCount()
        if count > 0:
            return "gpu"
        else:
            return "cpu"
    except:
        return "cpu"

have_gpu = False
if check_cuda() == 'gpu':
  have_gpu = True

if have_gpu:
  print("OpenCV test: GPU found")
else:
  print("OpenCV test: CPU only")
