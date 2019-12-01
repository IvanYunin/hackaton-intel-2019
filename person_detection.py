import cv2
import numpy as np
from copy import copy
from time import time
from openvino.inference_engine import IENetwork, IECore

#----------------------------------------------------------------------------------------------------------

input_layer_name = "data"

def add_extension(iecore: IECore, path_to_extension: str, device: str):
    if path_to_extension:
        if device == 'GPU':
            iecore.set_config({'CONFIG_FILE': path_to_extension}, device)
        if device == 'CPU' or device == 'MYRIAD':
            iecore.add_extension(path_to_extension, device)

def set_config(iecore: IECore, device: str, nthreads: int, nstreams: int):
    config = {}
    if device == 'CPU':
        if nthreads:
            config.update({'CPU_THREADS_NUM': str(nthreads)})
        cpu_throughput = {'CPU_THROUGHPUT_STREAMS': 'CPU_THROUGHPUT_AUTO'}
        if nstreams:
            cpu_throughput['CPU_THROUGHPUT_STREAMS'] = str(nstreams)
        config.update(cpu_throughput)
    if device == 'GPU':
        gpu_throughput = {'GPU_THROUGHPUT_STREAMS': 'GPU_THROUGHPUT_AUTO'}
        if nstreams:
            gpu_throughput['GPU_THROUGHPUT_STREAMS'] = str(nstreams)
        config.update(gpu_throughput)
    if device == 'MYRIAD':
        config.update({'LOG_LEVEL': 'LOG_INFO', 'VPU_LOG_LEVEL': 'LOG_WARNING'})
    iecore.set_config(config, device)

def create_ie_core(device: str, path_to_extension: str, nthreads: int, nstreams: int):
    ie = IECore()
    add_extension(ie, path_to_extension, device)
    set_config(ie, device, nthreads, nstreams)
    return ie

def create_network(model_xml: str, model_bin: str):
    return IENetwork(model=model_xml, weights=model_bin)

def get_input_shape(model: IENetwork):
    return model.inputs[input_layer_name].shape

def get_capture_shape(capture):
    _, frame = capture.read()
    return frame.shape

def get_next_batch(capture, batch_size: int, network_shape: list):
    _, c, h, w = network_shape
    next_batch = np.ndarray(shape=(batch_size, c, h, w))
    frames = []
    for i in range(batch_size):
        code, frame = capture.read()
        frames.append(frame)
        next_batch[i] = cv2.resize(frame, (w, h)).transpose((2, 0, 1))
    return next_batch, frames

def create_executable_network(iecore: IECore, network: IENetwork, device: str):
    return iecore.load_network(network=network, device_name=device)

def inference_sync(executable_network, input: dict):
    start_inference = time()
    output = executable_network.infer(inputs=input)
    end_inference = time()
    inference_time = end_inference - start_inference
    return output, inference_time

def detection_output(network_shape: list, frames: list, output: dict, threshold: float):
    output_layer = "detection_out"
    out = output[output_layer]
    h, w  = frames[0].shape[:2]
    output_points = [[] for i in range(len(frames))]
    output_frames = frames.copy()
    for string in out[0][0]:
        if string[2] == -1:
            break
        if string[2] > threshold:
            frame_number = int(string[0])
            x_min = int(string[3] * w)
            y_min = int(string[4] * h)
            x_max = int(string[5] * w)
            y_max = int(string[6] * h)
            x_mid = int((x_min + x_max) / 2)
            output_frames[frame_number] = cv2.rectangle(output_frames[frame_number], (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
            if y_max < h - 5:
                output_points[frame_number].append((x_mid, y_max))
    return output_points, output_frames

#----------------------------------------------------------------------------------------------------------

realD = 432
realW = 768
realH = 100

floor_bottom_left = (0, realD - 1)
floor_bottom_right = (realW - 1, realD - 1)
floor_top_left = (90, 210)
floor_top_right = (630, 210)

def prepare_transform_floor():
    curr_points = np.float32([floor_top_left, floor_top_right, floor_bottom_left, floor_bottom_right])
    dest_points = np.float32([[0, 0],[realW - 1, 0],[0, realD - 1], [realW - 1, realD - 1]])
    return cv2.getPerspectiveTransform(curr_points, dest_points)

def prepare_colormap_floor():
    return np.zeros(shape=(realD, realW, 1))

def prepare_heatmap():
    return np.zeros(shape=(realD, realW, 3))

def leg2floor(transform, x: int, y: int):
    input = np.zeros(shape=(3,))
    input[0] = x
    input[1] = y
    input[2] = 1
    output = transform.dot(input)
    output /= output[2]
    return int(output[0]), int(output[1])

def locality(cl, x, y, l):
    if (0 <= x < realW and 0 <= y < realD):
       y1 = y - l if y - l > 0 else 0
       y2 = y + l if y + l < realD else (realD - 1)
       x1 = x - l if x - l > 0 else 0
       x2 = x + l if x + l < realW else (realW - 1)
       for iy in range(y1, y2):
           for ix in range(x1, x2):
                if((x-ix)**2 + (y-iy)**2 < l**2):
                    cl[iy, ix] += 1 - ((x-ix)**2 + (y-iy)**2) / l**2
    return cl

def update_heatmap(heatmap, cl, transform, points: list):
    for point in points:
        x, y = leg2floor(transform, point[0], point[1])
        cl = locality(cl, x, y, 40)
        heatmap = cv2.circle(heatmap, (x, y), 7, (0, 0, 255), -1)
    return heatmap, cl

video_name = "people-detection.mp4"
device = "CPU"
path_to_model = "person-detection-retail-0013/FP32/person-detection-retail-0013"
model_xml = path_to_model + ".xml"
model_bin = path_to_model + ".bin"
path_to_extension = "/opt/intel/openvino_2019.3.334/deployment_tools/inference_engine/lib/intel64/libcpu_extension_avx2.so"
nthreads = None
nstreams = 1
batch_size = 1

def main():
    # try:
    capture = cv2.VideoCapture(0 or video_name)
    iecore = create_ie_core(device, path_to_extension, nthreads, nstreams)
    network = create_network(model_xml, model_bin)
    input_shape = get_input_shape(network)
    network.batch_size = batch_size
    executable_network = create_executable_network(iecore, network, device)
    input = dict.fromkeys([input_layer_name])
    capture_shape = get_capture_shape(capture)
    heatmap = prepare_heatmap()
    cl = prepare_colormap_floor()
    transform = prepare_transform_floor()
    frame_counter = 500
    capture.set(cv2.CAP_PROP_POS_FRAMES, 500)
    while True:
        input[input_layer_name], frames = get_next_batch(capture, batch_size, input_shape)
        output, inference_time = inference_sync(executable_network, input)
        fps = batch_size / inference_time
        output_points, output_frames = detection_output(input_shape, frames, output, 0.5)
        for i in range(len(output_points)):
            heatmap, cl = update_heatmap(heatmap, cl, transform, output_points[i])
            cl_norm = cl
            cl_norm = cv2.normalize(cl, cl_norm, alpha=0, beta=255, 
                                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            clMap = cv2.applyColorMap(cl_norm , cv2.COLORMAP_JET)
            frame_counter += 1
            if frame_counter == capture.get(cv2.CAP_PROP_FRAME_COUNT) - 1:
                frame_counter = 0 #Or whatever as long as it is the same as next line
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame = cv2.putText(output_frames[i], "FPS: {0:0.3f}".format(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX,  
                            1, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imshow('Colormap', clMap)
            cv2.imshow("Output", frame)
            # cv2.imshow("Heatmap", heatmap)
            ch = cv2.waitKey(1)
        if ch & 255 == 27:
            break
    capture.release()
    cv2.destroyAllWindows()
    # except Exception as error:
    #     print("error: " + str(error))

if __name__ == "__main__":
    main()