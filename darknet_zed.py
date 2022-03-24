#! /usr/bin/env python3.6
"""
Python 3 wrapper for identifying persons with ZED2 Camera using YoloV4

Requires DLL compilation

This file extends the existing file from: https://github.com/stereolabs/zed-yolo/blob/master/zed_python_sample/darknet_zed.py
Author: Philip Kahn, Aymeric Dujardin

@author: Luca Thomaier
@date: 20211202
"""
# pylint: disable=R, W0401, W0614, W0703
import os
import sys
import time
import logging
import random
from random import randint
import math
import statistics
import getopt
from ctypes import *
import numpy as np
import cv2
import pyzed.sl as sl

# Get the top-level logger object
#log = logging.getLogger(__name__)
#logging.basicConfig(level=logging.INFO)

logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
log = logging.getLogger()

fileHandler = logging.FileHandler("{0}/{1}.log".format("", "info.log"))
fileHandler.setFormatter(logFormatter)
log.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
log.addHandler(consoleHandler)

# filename = 'video.avi'
# frames_per_second = 24.0
# res = '720p'
#
# # Set resolution for the video capture
# # Function adapted from https://kirr.co/0l6qmh
# def change_res(cap, width, height):
#     cap.set(3, width)
#     cap.set(4, height)
#
# # Standard Video Dimensions Sizes
# STD_DIMENSIONS =  {
#     "480p": (640, 480),
#     "720p": (1280, 720),
#     "1080p": (1920, 1080),
#     "4k": (3840, 2160),
# }
#
#
# # grab resolution dimensions and set video capture to it.
# def get_dims(cap, res='1080p'):
#     width, height = STD_DIMENSIONS["480p"]
#     if res in STD_DIMENSIONS:
#         width,height = STD_DIMENSIONS[res]
#     ## change the current caputre device
#     ## to the resulting resolution
#     change_res(cap, width, height)
#     return width, height
#
# # Video Encoding, might require additional installs
# # Types of Codes: http://www.fourcc.org/codecs.php
# VIDEO_TYPE = {
#     'avi': cv2.VideoWriter_fourcc(*'XVID'),
#     #'mp4': cv2.VideoWriter_fourcc(*'H264'),
#     'mp4': cv2.VideoWriter_fourcc(*'XVID'),
# }
#
# def get_video_type(filename):
#     filename, ext = os.path.splitext(filename)
#     if ext in VIDEO_TYPE:
#       return  VIDEO_TYPE[ext]
#     return VIDEO_TYPE['avi']

def sample(probs):
    s = sum(probs)
    probs = [a / s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs) - 1


def c_array(ctype, values):
    arr = (ctype * len(values))()
    arr[:] = values
    return arr


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int),
                ("uc", POINTER(c_float)),
                ("points", c_int),
                ("embeddings", POINTER(c_float)),
                ("embedding_size", c_int),
                ("sim", c_float),
                ("track_id", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


# lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
# lib = CDLL("darknet.so", RTLD_GLOBAL)
hasGPU = True
if os.name == "nt":
    cwd = os.path.dirname(__file__)
    os.environ['PATH'] = cwd + ';' + os.environ['PATH']
    winGPUdll = os.path.join(cwd, "yolo_cpp_dll.dll")
    winNoGPUdll = os.path.join(cwd, "yolo_cpp_dll_nogpu.dll")
    envKeys = list()
    for k, v in os.environ.items():
        envKeys.append(k)
    try:
        try:
            tmp = os.environ["FORCE_CPU"].lower()
            if tmp in ["1", "true", "yes", "on"]:
                raise ValueError("ForceCPU")
            else:
                log.info("Flag value '" + tmp + "' not forcing CPU mode")
        except KeyError:
            # We never set the flag
            if 'CUDA_VISIBLE_DEVICES' in envKeys:
                if int(os.environ['CUDA_VISIBLE_DEVICES']) < 0:
                    raise ValueError("ForceCPU")
            try:
                global DARKNET_FORCE_CPU
                if DARKNET_FORCE_CPU:
                    raise ValueError("ForceCPU")
            except NameError:
                pass
            # log.info(os.environ.keys())
            # log.warning("FORCE_CPU flag undefined, proceeding with GPU")
        if not os.path.exists(winGPUdll):
            raise ValueError("NoDLL")
        lib = CDLL(winGPUdll, RTLD_GLOBAL)
    except (KeyError, ValueError):
        hasGPU = False
        if os.path.exists(winNoGPUdll):
            lib = CDLL(winNoGPUdll, RTLD_GLOBAL)
            log.warning("Notice: CPU-only mode")
        else:
            # Try the other way, in case no_gpu was
            # compile but not renamed
            lib = CDLL(winNoGPUdll, RTLD_GLOBAL)
            log.warning("Environment variables indicated a CPU run, but we didn't find `" +
                        winNoGPUdll + "`. Trying a GPU run anyway.")
else:
    lib = CDLL("./darknet/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

if hasGPU:
    set_gpu = lib.cuda_set_device
    set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(
    c_int), c_int, POINTER(c_int), c_int]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

load_net_custom = lib.load_network_custom
load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
load_net_custom.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)


def array_to_image(arr):
    import numpy as np
    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2, 0, 1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w, h, c, data)
    return im, arr


def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        if altNames is None:
            name_tag = meta.names[i]
        else:
            name_tag = altNames[i]
        res.append((name_tag, out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res


def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45, debug=False):
    """
    Performs the detection
    """
    custom_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    custom_image = cv2.resize(custom_image, (lib.network_width(
        net), lib.network_height(net)), interpolation=cv2.INTER_LINEAR)
    im, arr = array_to_image(custom_image)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(
        net, image.shape[1], image.shape[0], thresh, hier_thresh, None, 0, pnum, 0)
    num = pnum[0]
    if nms:
        do_nms_sort(dets, num, meta.classes, nms)
    res = []
    if debug:
        log.debug("about to range")
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                if altNames is None:
                    name_tag = meta.names[i]
                else:
                    name_tag = altNames[i]
                res.append((name_tag, dets[j].prob[i], (b.x, b.y, b.w, b.h), i))
    res = sorted(res, key=lambda x: -x[1])
    free_detections(dets, num)
    return res


netMain = None
metaMain = None
altNames = None


def get_object(zed, detection_parameters_rt):
    objects = sl.Objects()

    # if zed.grab() == sl.ERROR_CODE.SUCCESS:
    zed.retrieve_objects(objects, detection_parameters_rt)

    return objects


def enable_object_detection(zed):
    # Set initialization parameters
    detection_parameters = sl.ObjectDetectionParameters()

    # allows objects to be tracked across frames and keep the same ID as long as possible.
    # Positional tracking must be active in order to track objects movements independently from camera motion.
    detection_parameters.enable_tracking = True

    # Set runtime parameters
    detection_parameters_rt = sl.ObjectDetectionRuntimeParameters()
    detection_parameters_rt.detection_confidence_threshold = 70

    # run detection for every Camera grab
    # determines if object detection runs for each frame or asynchronously in a separate thread.
    detection_parameters.image_sync = True

    # outputs 2D masks over detected objects. Since it requires additional processing, disable this option if not used.
    # detection_parameters.enable_mask_output = True

    # camera_infos = zed.get_camera_information()

    if detection_parameters.enable_tracking:
        # Set positional tracking parameters
        positional_tracking_param = sl.PositionalTrackingParameters()
        # positional_tracking_param.set_as_static = True
        positional_tracking_param.set_floor_as_origin = True
        # Enable positional tracking
        zed.enable_positional_tracking(positional_tracking_param)

    zed.enable_object_detection(detection_parameters)

    return detection_parameters_rt


def get_object_depth(depth, bounds):
    '''
    Calculates the median x, y, z position of top slice(area_div) of point cloud
    in camera frame.
    Arguments:
        depth: Point cloud data of whole frame.
        bounds: Bounding box for object in pixels.
            bounds[0]: x-center
            bounds[1]: y-center
            bounds[2]: width of bounding box.
            bounds[3]: height of bounding box.

    Return:
        x, y, z: Location of object in meters.
    '''
    area_div = 2

    x_vect = []
    y_vect = []
    z_vect = []

    for j in range(int(bounds[0] - area_div), int(bounds[0] + area_div)):
        for i in range(int(bounds[1] - area_div), int(bounds[1] + area_div)):
            z = depth[i, j, 2]
            if not np.isnan(z) and not np.isinf(z):
                x_vect.append(depth[i, j, 0])
                y_vect.append(depth[i, j, 1])
                z_vect.append(z)
    try:
        x_median = statistics.median(x_vect)
        y_median = statistics.median(y_vect)
        z_median = statistics.median(z_vect)
    except Exception:
        x_median = -1
        y_median = -1
        z_median = -1
        pass

    return x_median, y_median, z_median


def generate_color(meta_path):
    '''
    Generate random colors for the number of classes mentioned in data file.
    Arguments:
    meta_path: Path to .data file.

    Return:
    color_array: RGB color codes for each class.
    '''
    random.seed(42)
    with open(meta_path, 'r') as f:
        content = f.readlines()
    class_num = int(content[0].split("=")[1])
    color_array = []
    for x in range(0, class_num):
        color_array.append((randint(0, 255), randint(0, 255), randint(0, 255)))
    return color_array


def main(argv):
    thresh = 0.25
    darknet_path = "./darknet/"
    config_path = darknet_path + "cfg/yolov4-tiny.cfg"
    weight_path = darknet_path + "yolov4-tiny.weights"
    meta_path = darknet_path + "cfg/coco.data"
    svo_path = None
    zed_id = 0

    help_str = 'darknet_zed.py -c <config> -w <weight> -m <meta> -t <threshold> -s <svo_file> -z <zed_id>'
    try:
        opts, args = getopt.getopt(
            argv, "hc:w:m:t:s:z:", ["config=", "weight=", "meta=", "threshold=", "svo_file=", "zed_id="])
    except getopt.GetoptError:
        log.exception(help_str)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            log.info(help_str)
            sys.exit()
        elif opt in ("-c", "--config"):
            config_path = arg
        elif opt in ("-w", "--weight"):
            weight_path = arg
        elif opt in ("-m", "--meta"):
            meta_path = arg
        elif opt in ("-t", "--threshold"):
            thresh = float(arg)
        elif opt in ("-s", "--svo_file"):
            svo_path = arg
        elif opt in ("-z", "--zed_id"):
            zed_id = int(arg)

    input_type = sl.InputType()
    if svo_path is not None:
        log.info("SVO file : " + svo_path)
        input_type.set_from_svo_file(svo_path)
    else:
        # Launch camera by id
        input_type.set_from_camera_id(zed_id)

    zed = sl.Camera()

    # init_parameters = sl.InitParameters(input_t=input_type)
    init_parameters = sl.InitParameters()
    init_parameters.coordinate_units = sl.UNIT.METER
    init_parameters.camera_resolution = sl.RESOLUTION.HD720
    init_parameters.camera_fps = 30

    if not zed.is_opened():
        log.info("Opening ZED Camera...")
    status = zed.open(init_parameters)
    if status != sl.ERROR_CODE.SUCCESS:
        log.error(repr(status))
        exit()

    # Use STANDARD sensing mode
    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.sensing_mode = sl.SENSING_MODE.STANDARD
    mat = sl.Mat()
    point_cloud_mat = sl.Mat()

    # enable object detection
    detection_parameters_rt = enable_object_detection(zed)

    # Import the global variables. This lets us instance Darknet once,
    # then just call performDetect() again without instancing again
    global metaMain, netMain, altNames  # pylint: disable=W0603
    assert 0 < thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(config_path):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(config_path) + "`")
    if not os.path.exists(weight_path):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weight_path) + "`")
    if not os.path.exists(meta_path):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(meta_path) + "`")
    if netMain is None:
        netMain = load_net_custom(config_path.encode(
            "ascii"), weight_path.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = load_meta(meta_path.encode("ascii"))
    if altNames is None:
        # In thon 3, the metafile default access craps out on Windows (but not Linux)
        # Read the names file and create a list to feed to detect
        try:
            with open(meta_path) as meta_fh:
                meta_contents = meta_fh.read()
                import re
                match = re.search("names *= *(.*)$", meta_contents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as names_fh:
                            names_list = names_fh.read().strip().split("\n")
                            altNames = [x.strip() for x in names_list]
                except TypeError:
                    pass
        except Exception:
            pass

    color_array = generate_color(meta_path)

    log.info("Running...")
    # cap = cv2.VideoCapture(0)
    # out = cv2.VideoWriter(filename, get_video_type(filename), 25, get_dims(cap, res))

    key = ''
    while key != 113:  # for 'q' key
        start_time = time.time()  # start time of the loop
        # err = zed.grab(runtime_parameters)
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # get image from camera for yolo detection
            zed.retrieve_image(mat, sl.VIEW.LEFT)
            image = mat.get_data()

            # out.write(image)

            # get depth information of camera
            zed.retrieve_measure(point_cloud_mat, sl.MEASURE.XYZRGBA)
            depth = point_cloud_mat.get_data()

            # Do the detection
            detections_yolo = detect(netMain, metaMain, image, thresh)
            detections_zed = get_object(zed, detection_parameters_rt)

            # log.info(chr(27) + "[2J"+"**** " + str(len(detections)) + " Results ****")
            for i in range(len(detections_yolo)):
                detection_yolo = detections_yolo[i]
                label = detection_yolo[0]
                confidence = detection_yolo[1]
                # pstring = label + ": " + str(np.rint(100 * confidence)) + "%"
                # log.info(pstring)
                bounds = detection_yolo[2]
                y_extent = int(bounds[3])
                x_extent = int(bounds[2])
                # Coordinates are around the center
                x_coord = int(bounds[0] - bounds[2] / 2)
                y_coord = int(bounds[1] - bounds[3] / 2)
                thickness = 1
                x, y, z = get_object_depth(depth, bounds)
                distance = math.sqrt(x * x + y * y + z * z)
                distance = "{:.2f}".format(distance)

                # print("lenZED: " + str(len(detections_zed.object_list)))
                # print("lenYolo: " + str(len(detections_yolo)))

                if len(detections_zed.object_list) >= len(detections_yolo):
                    detection_zed = detections_zed.object_list[i]
                    # detection_zed = sl.ObjectData()
                    # detections_zed.get_object_data_from_id(detection_zed, i)  # Get the object with ID = i

                    object_id = detection_zed.id  # Get the object id
                    object_label = detection_zed.label  # Get the object label
                    object_height = detection_zed.dimensions[1]  # Get the object dimensions
                    object_position_x, object_position_y, object_position_z = detection_zed.position  # Get the object position
                    object_velocity_x, object_velocity_y, object_velocity_z = detection_zed.velocity  # Get the object velocity
                    object_tracking_state = detection_zed.tracking_state  # Get the object tracking state
                    object_action_state = detection_zed.action_state  # Get the object action state

                    object_velocity = math.sqrt(
                        object_velocity_x * object_velocity_x + object_velocity_y * object_velocity_y + object_velocity_z * object_velocity_z)

                    log.info("ObjectID: " + str(object_id))

                else:
                    object_height = 0
                    object_position_x = 0
                    object_position_y = 0
                    object_position_z = 0
                    object_action_state = 0
                    object_velocity = 0

                cv2.rectangle(image, (x_coord - thickness, y_coord - thickness),
                              (x_coord + x_extent + thickness, y_coord + (-120 + thickness * 2)),
                              color_array[detection_yolo[3]], -1)
                cv2.putText(image, label + " | distance: " + (str(distance) + " m ") + " | height: " + str(
                    round(object_height, 2)) + " m",
                            (x_coord + (thickness * 4), y_coord + (-100 + thickness * 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(image, "status: " + str(object_action_state) + " | velocity: " + str(
                    round(object_velocity, 2)) + " m/s",
                            (x_coord + (thickness * 4), y_coord + (-60 + thickness * 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(image, "positionXYZ: " + str(round(object_position_x, 2)) + " " + str(
                    round(object_position_y, 2)) + " " + str(round(object_position_z, 2)),
                            (x_coord + (thickness * 4), y_coord + (-20 + thickness * 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.rectangle(image, (x_coord - thickness, y_coord - thickness),
                              (x_coord + x_extent + thickness, y_coord + y_extent + thickness),
                              color_array[detection_yolo[3]], int(thickness * 2))

            cv2.imshow("ZED", image)
            key = cv2.waitKey(5)
            if detections_zed:
                log.info("ZED: " + str(len(detections_zed.object_list)) + " YOLO:" + str(len(detections_yolo)))
            log.info("FPS: {}".format(1.0 / (time.time() - start_time)))
        else:
            key = cv2.waitKey(5)
    cv2.destroyAllWindows()

    zed.close()
    log.info("\nFINISH")


if __name__ == "__main__":
    main(sys.argv[1:])
