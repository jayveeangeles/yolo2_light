from ctypes import *
from enum import Enum
import math
import random

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
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
                ("sort_class", c_int)]

# class CtypesEnum(IntEnum):
#     """A ctypes-compatible IntEnum superclass."""
#     @classmethod
#     def from_param(cls, obj):
#         return int(obj)

# class LEARNING_RATE_POLICY(Enum):
#     CONSTANT = 0
#     STEP = 1
#     EXP = 2 
#     POLY = 3 
#     STEPS = 4 
#     SIG = 5 
#     RANDOM = 6

# class LEARNING_RATE_POLICY(Enum):
#     CONSTANT, STEP, EXP, POLY, STEPS, SIG, RANDOM = range(7)

(CONSTANT, STEP, EXP, POLY, STEPS, SIG, RANDOM) = range(7)

class TREE(Structure):
    _fields_ = [("leaf", POINTER(c_int)),
                ("n", c_int),
                ("parent", POINTER(c_int)),
                ("group", POINTER(c_int)),
                ("name", POINTER(c_char_p)),
                ("groups", c_int),
                ("group_size", POINTER(c_int)),
                ("group_offset", POINTER(c_int))]

class NETWORK(Structure):
    _fields_ = [("quantized", c_int),
                ("workspace", POINTER(c_float)),
                ("n", c_int),
                ("batch", c_int),
                ("input_calibration", POINTER(c_float)),
                ("input_calibration_size", c_int),
                ("seen", POINTER(c_uint64)),
                ("epoch", c_float),
                ("subdivisions", c_int),
                ("momentum", c_float),
                ("decay", c_float),
                ("layers", c_void_p),
                ("outputs", c_int),
                ("output", POINTER(c_float)),
                ("policy", c_int),
                ("learning_rate", c_float),
                ("gamma", c_float),
                ("scale", c_float),
                ("power", c_float),
                ("time_steps", c_int),
                ("step", c_int),
                ("mini_batches", c_int),
                ("scales", POINTER(c_float)),
                ("steps", POINTER(c_int)),
                ("num_steps", c_int),
                ("burn_in", c_int),
                ("adam", c_int),
                ("B1", c_float),
                ("B2", c_float),
                ("eps", c_float),
                ("inputs", c_int),
                ("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("max_crop", c_int),
                ("min_crop", c_int),
                ("angle", c_float),
                ("aspect", c_float),
                ("exposure", c_float),
                ("saturation", c_float),
                ("hue", c_float),
                ("gpu_index", c_int),
                ("hierarchy", POINTER(TREE)),
                ("do_input_calibration", c_int),
                ("input_state_gpu", POINTER(c_float)),
                ("input_gpu", POINTER(POINTER(c_float))),
                ("truth_gpu", POINTER(POINTER(c_float)))]

class IMAGE(Structure):
    _fields_ = [("h", c_int),
                ("w", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class DETECTION_WITH_CLASS(Structure):
    _fields_ = [("det", DETECTION),
                ("best_class", c_int)]


#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("/root/Yolo/yolo2_light/libdarknet.so", RTLD_GLOBAL)

cuda_set_device = lib.cuda_set_device
cuda_set_device.argtpyes = [c_int]

parse_network_cfg = lib.parse_network_cfg
parse_network_cfg.argtypes = [c_char_p, c_int, c_int]
parse_network_cfg.restype = NETWORK

calculate_binary_weights = lib.calculate_binary_weights
calculate_binary_weights.argtypes = [NETWORK]

load_weights_upto_cpu = lib.load_weights_upto_cpu
load_weights_upto_cpu.argtypes = [c_void_p, c_char_p, c_int]

yolov2_fuse_conv_batchnorm = lib.yolov2_fuse_conv_batchnorm
yolov2_fuse_conv_batchnorm.argtypes = [NETWORK]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

get_actual_detections = lib.get_actual_detections
get_actual_detections.argtypes = [POINTER(DETECTION), c_int, c_float, POINTER(c_int)]
get_actual_detections.restype = POINTER(DETECTION_WITH_CLASS)

quantinization_and_get_multipliers = lib.quantinization_and_get_multipliers
quantinization_and_get_multipliers.argtpyes = [NETWORK]

load_image = lib.load_image
load_image.argtypes = [c_char_p, c_int, c_int, c_int]
load_image.restype = IMAGE

resize_image = lib.resize_image
resize_image.argtypes = [IMAGE, c_int, c_int]
resize_image.restype = IMAGE

network_predict_gpu_cudnn_quantized = lib.network_predict_gpu_cudnn_quantized
network_predict_gpu_cudnn_quantized.argtypes = [NETWORK, POINTER(c_float)]
network_predict_gpu_cudnn_quantized.restype = POINTER(c_float)

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
get_network_boxes.restype = POINTER(DETECTION)

free_image = lib.free_image
free_image.argtypes = [IMAGE]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

save_image_png = lib.save_image_png
save_image_png.argtypes = [IMAGE, c_char_p]

def detect(im, cfgfile, weightfile, names, thresh=.24, hier_thresh=.5, nms=.4):
    net = parse_network_cfg(cfgfile, 1, 1)
    load_weights_upto_cpu(byref(net), weightfile, net.n)

    yolov2_fuse_conv_batchnorm(net)
    calculate_binary_weights(net)
    quantinization_and_get_multipliers(net)

    sized = resize_image(im, net.w, net.h)

    network_predict_gpu_cudnn_quantized(net, sized.data)

    nboxes = c_int(0)
    letterbox = 0
    selected_detections_num = c_int(0)
    dets = get_network_boxes(byref(net), im.w, im.h, thresh, hier_thresh, None, 1, byref(nboxes), letterbox)

    do_nms_sort(dets, nboxes, 80, nms) # no. of classes
    selected_detections = get_actual_detections(dets, nboxes, thresh, byref(selected_detections_num))

    print "class: %s, prob: %f" % (names[selected_detections[0].best_class], selected_detections[0].det.prob[selected_detections[0].best_class])
    print "class: %s, prob: %f" % (names[selected_detections[1].best_class], selected_detections[1].det.prob[selected_detections[1].best_class])
    print "class: %s, prob: %f" % (names[selected_detections[2].best_class], selected_detections[2].det.prob[selected_detections[2].best_class])

    free_image(im)
    free_image(sized)
    return selected_detections
    
if __name__ == "__main__":
    #net = load_net("cfg/densenet201.cfg", "/home/pjreddie/trained/densenet201.weights", 0)
    #im = load_image("data/wolf.jpg", 0, 0)
    #meta = load_meta("cfg/imagenet1k.data")
    #r = classify(net, meta, im)
    #print r[:10]

    random.seed(2222222)
    
    cuda_set_device(0)
    with open('../bin/coco.names') as f:
        names = f.read().splitlines()

    im = load_image( "/root/Yolo/yolo2_light/bin/dog.jpg", 0, 0, 3)
    print "width: %d, height: %d\n" % (im.w, im.h)

    r = detect(im, "/root/Yolo/yolo2_light/bin/yolov3-tiny.cfg", "/root/Yolo/yolo2_light/bin/yolov3-tiny.weights", names)
    
    print "left_x: %4.0f   top_y: %4.0f   width: %4.0f   height: %4.0f\n" %  ((r[0].det.bbox.x - r[0].det.bbox.w / 2)*im.w, (r[0].det.bbox.y - r[0].det.bbox.h / 2)*im.h,r[0].det.bbox.w*im.w, r[0].det.bbox.h*im.h)
    print "left_x: %4.0f   top_y: %4.0f   width: %4.0f   height: %4.0f\n" %  ((r[1].det.bbox.x - r[1].det.bbox.w / 2)*im.w, (r[1].det.bbox.y - r[1].det.bbox.h / 2)*im.h,r[1].det.bbox.w*im.w, r[1].det.bbox.h*im.h)
    print "left_x: %4.0f   top_y: %4.0f   width: %4.0f   height: %4.0f\n" %  ((r[2].det.bbox.x - r[2].det.bbox.w / 2)*im.w, (r[2].det.bbox.y - r[2].det.bbox.h / 2)*im.h,r[2].det.bbox.w*im.w, r[2].det.bbox.h*im.h)