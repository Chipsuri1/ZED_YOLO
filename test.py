import cv2
import pyzed.sl as sl

# Create a ZED camera object
zed = sl.Camera()

# Set configuration parameters
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.camera_fps = 30

# Open the camera
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    exit(-1)

codec = cv2.VideoWriter_fourcc('M','P','E','G')

# out = cv2.VideoWriter('./processed.avi', codec, 20.0, (720,1280))
# writer= cv2.VideoWriter('./basicvideo.avi', codec, 20, (1280,720))
writer= cv2.VideoWriter('./basicvideo.avi', codec, 20, (720,1280))
# Create an RGBA sl.Mat object
image_zed = sl.Mat(zed.get_camera_information().camera_resolution.width, zed.get_camera_information().camera_resolution.height, sl.MAT_TYPE.U8_C4)

runtime_parameters = sl.RuntimeParameters()
runtime_parameters.sensing_mode = sl.SENSING_MODE.STANDARD
mat = sl.Mat()
point_cloud_mat = sl.Mat()

while True:
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        # get image from camera for yolo detection
        zed.retrieve_image(mat, sl.VIEW.LEFT)
        image = mat.get_data()

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # image = cv2.resize(image, (720, 1280))
    writer.write(image)

    cv2.imshow('frame', image)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# cap.release()
writer.release()
cv2.destroyAllWindows()