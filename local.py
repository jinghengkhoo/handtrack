import cv2
import numpy as np
import pygame
import pygame.freetype

from canvas import Airbrush, Calligraphy, Clear, FlowerBrush, LineTool
from utils import detector_utils as detector_utils


class BoundingBox:
    def __init__(self, confidence, x1, x2, y1, y2, image_width, image_height):
        self.confidence = confidence
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.u1 = x1 / image_width
        self.u2 = x2 / image_width
        self.v1 = y1 / image_height
        self.v2 = y2 / image_height

    def box(self):
        return (self.x1, self.y1, self.x2, self.y2)

    def width(self):
        return self.x2 - self.x1

    def height(self):
        return self.y2 - self.y1

    def center_absolute(self):
        return (0.5 * (self.x1 + self.x2), 0.5 * (self.y1 + self.y2))

    def center_normalized(self):
        return (0.5 * (self.u1 + self.u2), 0.5 * (self.v1 + self.v2))

    def size_absolute(self):
        return (self.x2 - self.x1, self.y2 - self.y1)

    def size_normalized(self):
        return (self.u2 - self.u1, self.v2 - self.v1)

def _nms_boxes(detections, nms_threshold):
    """Apply the Non-Maximum Suppression (NMS) algorithm on the bounding
    boxes with their confidence scores and return an array with the
    indexes of the bounding boxes we want to keep.
    # Args
        detections: Nx7 numpy arrays of
                    [[x, y, w, h, box_confidence, class_id, class_prob],
                     ......]
    """
    x_coord = detections[:, 0]
    y_coord = detections[:, 1]
    width = detections[:, 2]
    height = detections[:, 3]
    box_confidences = detections[:, 4] * detections[:, 6]

    areas = width * height
    ordered = box_confidences.argsort()[::-1]

    keep = list()
    while ordered.size > 0:
        # Index of the current element:
        i = ordered[0]
        keep.append(i)
        xx1 = np.maximum(x_coord[i], x_coord[ordered[1:]])
        yy1 = np.maximum(y_coord[i], y_coord[ordered[1:]])
        xx2 = np.minimum(x_coord[i] + width[i], x_coord[ordered[1:]] + width[ordered[1:]])
        yy2 = np.minimum(y_coord[i] + height[i], y_coord[ordered[1:]] + height[ordered[1:]])

        width1 = np.maximum(0.0, xx2 - xx1 + 1)
        height1 = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = width1 * height1
        union = (areas[i] + areas[ordered[1:]] - intersection)
        iou = intersection / union
        indexes = np.where(iou <= nms_threshold)[0]
        ordered = ordered[indexes + 1]

    keep = np.array(keep)
    return keep

def postprocessing(output, img_w, img_h, nms_threshold=0.4):
    """Postprocess TensorRT outputs.
    # Args
        output: list of detections with schema [x, y, w, h, box_confidence, class_id, class_prob]
        conf_th: confidence threshold
        letter_box: boolean, referring to _preprocess_yolo()
    # Returns
        list of bounding boxes with all detections above threshold and after nms, see class BoundingBox
    """
    # filter low-conf detections
    detections = output.reshape((-1, 7))
    # detections = detections[detections[:, 4] * detections[:, 6] >= conf_th]

    if len(detections) == 0:
        boxes = np.zeros((0, 4), dtype=np.int)
        scores = np.zeros((0,), dtype=np.float32)
    else:

        # scale x, y, w, h from [0, 1] to pixel values
        old_h, old_w = img_h, img_w
        detections[:, 0:4] *= np.array(
            [old_w, old_h, old_w, old_h], dtype=np.float32)

        # NMS
        nms_detections = np.zeros((0, 7), dtype=detections.dtype)
        for class_id in set(detections[:, 5]):
            idxs = np.where(detections[:, 5] == class_id)
            cls_detections = detections[idxs]
            keep = _nms_boxes(cls_detections, nms_threshold)
            nms_detections = np.concatenate(
                [nms_detections, cls_detections[keep]], axis=0)

        xx = nms_detections[:, 0].reshape(-1, 1)
        yy = nms_detections[:, 1].reshape(-1, 1)
        ww = nms_detections[:, 2].reshape(-1, 1)
        hh = nms_detections[:, 3].reshape(-1, 1)
        boxes = np.concatenate([xx, yy, xx+ww, yy+hh], axis=1) + 0.5
        boxes = boxes.astype(np.int)
        scores = nms_detections[:, 4] * nms_detections[:, 6]
    detected_objects = []
    for box, score in zip(boxes, scores):
        detected_objects.append(BoundingBox(score, box[0], box[2], box[1], box[3], img_h, img_w))
    return detected_objects


def main():
    detection_graph, sess = detector_utils.load_inference_graph()

    vid = cv2.VideoCapture(0)

    score_thresh = 0.3

    ret, image_np = vid.read()

    im_height, im_width, _ = image_np.shape
    canvas_height, canvas_width = 900, 1600

    pygame.init()
    screen = pygame.display.set_mode((1600, 900))
    sprites = pygame.sprite.Group()
    clock = pygame.time.Clock()

    font = pygame.freetype.SysFont(None, 26)
    offset = 0, 50
    canvas = pygame.Surface((1600, 850))
    canvas.set_colorkey((1,1,1))
    canvas.fill((1,1,1))
    tmpcanvas = canvas.copy()

    x=10
    for tool in (Calligraphy, Airbrush, FlowerBrush, LineTool, Clear):
        tool((x, 10), font, canvas, tmpcanvas, sprites, offset)
        x+= 40
    while(True):
        ret, image_np = vid.read()

        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        boxes, scores = detector_utils.detect_objects(image_np,
                                                      detection_graph, sess)

        #[x, y, w, h, box_confidence, class_id, class_prob]
        output = []
        for idx, score in enumerate(scores):
            if score > score_thresh:
                output.append([boxes[idx][1], boxes[idx][0], boxes[idx][3] - boxes[idx][1], boxes[idx][2] - boxes[idx][0], score, 1, score])
        detected_objects = postprocessing(np.array(output), canvas_width, canvas_height, nms_threshold=0.5)

        for box in detected_objects:
            cv2.rectangle(image_np, (int(box.x1 / canvas_width * im_width), int(box.y1 / canvas_height * im_height)), (int(box.x2 / canvas_width * im_width), int(box.y2 / canvas_height * im_height)), (77, 255, 9), 3, 1)

        cv2.imshow('Single-Threaded Detection', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

        events = pygame.event.get()
        for e in events:
            if e.type == pygame.QUIT:
                return
        tmpcanvas.fill((1, 1, 1))

        if detected_objects:
            sprites.update(events, detected_objects[0].center_absolute())
        else:
            sprites.update(events)

        screen.fill((30, 30, 30))
        screen.blit(canvas, offset)
        screen.blit(tmpcanvas, offset)
        sprites.draw(screen)
        pygame.display.update()
        clock.tick(60)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    main()