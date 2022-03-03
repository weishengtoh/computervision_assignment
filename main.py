from genericpath import exists
import os
import logging

import cv2
import numpy as np

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort.generate_detections import create_box_encoder
from deep_sort import preprocessing

import torch

import hydra
from omegaconf import DictConfig, OmegaConf

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


@hydra.main(config_path='configs', config_name="config")
def run(config: DictConfig):

    # Obtain the parameters from config file as dictionary
    config = OmegaConf.to_container(config)
    params = config['parameters']

    # Set debugging on/off
    debug = config['main']['verbose']

    # This is required to obtain cwd when using hydra
    cwd_path = hydra.utils.get_original_cwd()

    logger.info('Initialise tracker')
    # Initialise the DeepSORT tracker
    model_filename = params['model_filename']
    model_path = os.path.join(cwd_path, model_filename)
    encoder = create_box_encoder(
        model_path,
        batch_size=1)  # maybe should explore training own feature extractor
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", params['max_cosine_distance'],
        None if params['nn_budget'] == 'None' else params['nn_budget'])
    tracker = Tracker(metric)

    # Initialise the object detector
    logger.info('Initialise object detector')
    object_detector = get_object_detector()

    # Obtain the file paths from config file
    input_path = os.path.join(cwd_path, params['input_path'])
    output_path = os.path.join(cwd_path, params['output_path'])

    os.makedirs(os.path.dirname(os.path.realpath(output_path)), exist_ok=True)

    v_in = cv2.VideoCapture(input_path)

    # Check if video can be read
    if (v_in.isOpened() == False):
        logger.error('Error opening video file')

    logger.info('Successfully read input video file')

    # Obtain the video properties from input video
    fps = int(v_in.get(cv2.CAP_PROP_FPS))
    frame_width = int(v_in.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(v_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(v_in.get(cv2.CAP_PROP_FOURCC))

    # Initialise the videowriter for video output
    v_out = cv2.VideoWriter(output_path, fourcc, fps,
                            (frame_width, frame_height))

    # Initialise the counter for total vehicles count and frame count
    unique_vehicles = {}
    frame_no = 1

    while True:

        success, frame = v_in.read()

        # Terminate loop at end of video file
        if not success:
            logger.info('No frame from stream - exiting')
            break

        # Log info every 10 frames
        frame_no += 1
        if frame_no % 10 == 0:
            logger.info(f'Processing frame number: {frame_no}')

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run the object detector and obtain bounding boxes
        annotated_img, metadata = annotate_image(object_detector, frame_rgb)
        annotated_img_bgr = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)

        # Extract bonding boxes for 'truck' and 'car' classes
        bboxes = []
        confidences = []
        labels = []

        for i in range(len(metadata.name)):

            if (metadata.name[i] in params['labels']):

                top_x = int(metadata.xmin[i])
                top_y = int(metadata.ymin[i])
                width = int(metadata.xmax[i]) - int(metadata.xmin[i])
                height = int(metadata.ymax[i]) - int(metadata.ymin[i])
                bbox = [top_x, top_y, width, height]

                bboxes.append(bbox)
                confidences.append(metadata.confidence[i])
                labels.append(metadata.name[i])

        bboxes = np.array(bboxes)
        confidences = np.array(confidences)
        labels = np.array(labels)
        features = np.array(encoder(annotated_img_bgr, bboxes))
        detections = [
            Detection(tlwh, conf, feature,
                      label) for tlwh, conf, feature, label in zip(
                          bboxes, confidences, features, labels)
        ]

        # Apply nonmax suppression to the detections
        nms_indices = preprocessing.non_max_suppression(
            bboxes, params['nms_max_overlap'], confidences)
        detections = [detections[i] for i in nms_indices]

        # Update the tracker and obtain information from tracker.tracks
        tracker.predict()
        tracker.update(detections)

        results = {}
        for track in tracker.tracks:

            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            track_id = track.track_id
            label = track.label

            results[track_id] = (bbox, label)

        # Draw bounding box around detections
        tracked_frame = draw_bboxes(frame, results)

        # Vehicles in current frame can be obtained by retrieving numbers of current tracks
        frame_vehicles_count = len(results.keys())

        # Store the vehicles ids used and the closest bbox to center
        # The idea is that the ideal image to save should be the image closest to center of frame
        frame_center = np.array(
            (int(frame.shape[1] / 2), int(frame.shape[0] / 2)))

        if debug:
            cv2.circle(tracked_frame, frame_center, 2, (255, 0, 0), 2)

        for id, value in results.items():
            bbox, label = value
            min_x, min_y, max_x, max_y = bbox

            # Compute the euclidean distance
            x = int(min_x + 0.5 * (max_x - min_x))
            y = int(min_y + 0.5 * (max_y - min_y))
            bbox_center = np.array((x, y))
            dist = np.linalg.norm(frame_center - bbox_center)

            if debug:
                cv2.line(tracked_frame, frame_center, bbox_center, (255, 0, 0),
                         1)
                cv2.putText(tracked_frame, f'{dist:0.2f}', bbox_center,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1,
                            cv2.LINE_AA)

            if dist < unique_vehicles.get(id, np.inf):
                unique_vehicles[id] = dist

        # Save vehicles images
        save_image(frame, results, unique_vehicles, params['images_folder'])

        # Total vehicles counted equals to unique vehicles ids observed
        total_vehicles_count = len(unique_vehicles.keys())

        # Display number of vehicles in current frame and total vehicles counted
        text_frame_vehicles = f'Vehicles in current frame: {frame_vehicles_count}'
        text_total_vehicles = f'Total vehicles counted: {total_vehicles_count}'

        cv2.putText(tracked_frame, text_frame_vehicles, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, cv2.LINE_AA)

        cv2.putText(tracked_frame, text_total_vehicles, (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, cv2.LINE_AA)

        # Write image to output
        v_out.write(tracked_frame)

    logger.info('Releasing Video Writer objects')
    v_out.release()
    v_in.release()


def draw_bboxes(frame, results):

    output = frame.copy()
    for id, value in results.items():
        bbox, label = value
        min_x, min_y, max_x, max_y = bbox

        # Check that the bounding box does not exceed frame
        min_x, min_y, = max(int(min_x), 0), max(int(min_y), 0)
        max_x, max_y, = min(int(max_x),
                            frame.shape[1]), min(int(max_y), frame.shape[0])

        frame = cv2.rectangle(output, (min_x, min_y), (max_x, max_y),
                              (0, 255, 0), 2)

        position = (int(min_x), int(min_y - 5))
        text_output = f'{label} ID: {id}'
        cv2.putText(frame, text_output, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2, cv2.LINE_AA)

    return output


def save_image(frame, results, unique_vehicles, images_folder):

    cwd_path = hydra.utils.get_original_cwd()
    pic_folder = os.path.join(cwd_path, images_folder)

    os.makedirs(pic_folder, exist_ok=True)

    frame_center = np.array((int(frame.shape[1] / 2), int(frame.shape[0] / 2)))
    for id, value in results.items():
        bbox, label = value
        min_x, min_y, max_x, max_y = bbox

        # Compute the euclidean distance
        x = int(min_x + 0.5 * (max_x - min_x))
        y = int(min_y + 0.5 * (max_y - min_y))
        bbox_center = np.array((x, y))
        dist = np.linalg.norm(bbox_center - frame_center)

        if dist > unique_vehicles.get(id, np.inf):
            continue

        # Check that the bounding box does not exceed frame when obtaining ROI
        min_x, min_y, = max(int(min_x), 0), max(int(min_y), 0)
        max_x, max_y, = min(int(max_x),
                            frame.shape[1]), min(int(max_y), frame.shape[0])

        img = frame[min_y:max_y, min_x:max_x]
        filepath = os.path.join(pic_folder, f'{label}_{id}.png')
        print(f'{id}.png')
        cv2.imwrite(filepath, img)


def get_object_detector():

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    return model


def annotate_image(model, image):

    # perform precidtion
    pred = model(image)

    # 'render' the predictions
    pred.render()

    # extract the first annotated image
    annotated_image = pred.imgs[0]

    return annotated_image, pred.pandas().xyxy[0]


if __name__ == '__main__':
    run()
