import cv2
import numpy as np
from copy import copy
from openvino.inference_engine import IENetwork, IECore

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
        # cpu_throughput = {'CPU_THROUGHPUT_STREAMS': 'CPU_THROUGHPUT_AUTO'}
        # if nstreams:
        #     cpu_throughput['CPU_THROUGHPUT_STREAMS'] = str(nstreams)
        # config.update(cpu_throughput)
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
    return executable_network.infer(inputs=input)

def human_pose_output(network_shape: list, frames: list, output: dict, threshold: float):
    pairs = output["Mconv7_stage2_L1"]
    points = output["Mconv7_stage2_L2"]
    frameH = frames[0].shape[0]
    frameW = frames[0].shape[1]
    inputH = network_shape[2]
    inputW = network_shape[3]
    output_points = []
    for number_frame in range(len(frames)):
        pts = []
        for i in range(18):
            probMap = points[number_frame, i, :, :]
            _, prob, _, point = cv2.minMaxLoc(probMap)
            x = (frameW * point[0]) / 57
            y = (frameH * point[1]) / 32
            if prob > threshold:
                pts.append((i, int(x), int(y))) # class, x, y
        output_points.append(pts)
    return output_points

def prepare_transform_floor():
    curr_points = np.float32([[90, 210],[630, 210],[0,451], [767, 431]])
    dest_points = np.float32([[0, 0],[767, 0],[0,451], [767, 431]])
    return cv2.getPerspectiveTransform(curr_points, dest_points)

def prepare_heatmap():
    return np.zeros(shape=(432, 738, 3))

def leg2floor(transform, x: int, y: int):
    input = np.zeros(shape=(3,))
    input[0] = x
    input[1] = y
    input[2] = 1
    output = transform.dot(input)
    output /= output[2]
    return int(output[0]), int(output[1])

def update_heatmap(heatmap, transform, points: list):
    for point in points:
        number_class = point[0]
        if number_class == 10 or number_class == 13:
            x, y = leg2floor(transform, point[1], point[2])
            heatmap = cv2.circle(heatmap, (x, y), 7, (0, 0, 255), -1)
    return heatmap

video_name = "people-detection.mp4"
device = "CPU"
path_to_model = "/opt/intel/openvino/deployment_tools/tools/model_downloader/intel/human-pose-estimation-0001/INT8/human-pose-estimation-0001"
model_xml = path_to_model + ".xml"
model_bin = path_to_model + ".bin"
path_to_extension = None
nthreads = None
nstreams = None
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
    transform = prepare_transform_floor()
    while True:
        input[input_layer_name], frames = get_next_batch(capture, batch_size, input_shape)
        output = inference_sync(executable_network, input)
        output_points = human_pose_output(input_shape, frames, output, 0.5)
        for i in range(len(output_points)):
            heatmap = update_heatmap(heatmap, transform, output_points[i])
            cv2.imshow("Output", frames[i])
            cv2.imshow("Heatmap", heatmap)
            ch = cv2.waitKey(1)
        if ch & 255 == 27:
            break
    capture.release()
    cv2.destroyAllWindows()
    # except Exception as error:
    #     print("error: " + str(error))

if __name__ == "__main__":
    main()