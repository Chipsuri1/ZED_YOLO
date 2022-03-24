import cv2
import pyzed.sl as sl

# Create a ZED camera object
zed = sl.Camera()

# Set configuration parameters
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD1080
init_params.camera_fps = 30

# Open the camera
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    exit(-1)

codec = cv2.VideoWriter_fourcc('M','P','E','G')

# out = cv2.VideoWriter('./processed.avi', codec, 20.0, (720,1280))
writer= cv2.VideoWriter('./basicvideo.mp4', codec, 20, (1920,1080))
# Create an RGBA sl.Mat object
image_zed = sl.Mat(zed.get_camera_information().camera_resolution.width, zed.get_camera_information().camera_resolution.height, sl.MAT_TYPE.U8_C4)

while True:
    image_ocv = image_zed.get_data()

    writer.write(image_ocv)

    cv2.imshow('frame', image_ocv)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# cap.release()
writer.release()
cv2.destroyAllWindows()