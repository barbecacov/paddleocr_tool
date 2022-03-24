import argparse
from abc import ABCMeta, abstractmethod
from pathlib import Path
import pdb
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax
from tqdm import tqdm
from shapely.geometry import Polygon
import pyclipper
import sys


class PaddleOcrABC(metaclass=ABCMeta):
    def __init__(self, limit_side_len=960, limit_type='max', det_thresh=0.3, box_thresh=0.6,
                 unclip_ratio=1.5, use_dilation=False, score_mode='fast'):
        self.limit_side_len = limit_side_len
        self.limit_type = limit_type
        self.det_thresh = det_thresh
        self.box_thresh = box_thresh
        self.unclip_ratio = unclip_ratio
        self.use_dilation = use_dilation
        self.score_mode = score_mode
        self.std = [0.229, 0.224, 0.225]
        self.mean = [0.485, 0.456, 0.406]

    def resize_image(self, img):
        h, w, _ = img.shape
        if self.limit_type == 'max':
            if max(h, w) > self.limit_side_len:
                if h > w:
                    ratio = float(self.limit_side_len) / h
                else:
                    ratio = float(self.limit_side_len) / w
            else:
                ratio = 1.
        else:
            if min(h, w) < self.limit_side_len:
                if h < w:
                    ratio = float(self.limit_side_len) / h
                else:
                    ratio = float(self.limit_side_len) / w
            else:
                ratio = 1.

        resize_h = int(h * ratio)
        resize_w = int(w * ratio)

        resize_h = int(round(resize_h / 32) * 32)
        resize_w = int(round(resize_w / 32) * 32)

        try:
            if int(resize_w) <= 0 or int(resize_h) <= 0:
                return None, (None, None)
            img = cv2.resize(img, (int(resize_w), int(resize_h)))
        except:
            print(img.shape, resize_w, resize_h)
            sys.exit(0)

        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        # return img, np.array([h, w])
        return img, [ratio_h, ratio_w]

    def preprocess(self, img):
        im_h, im_w, _ = img.shape
        img_resize, ratio_hw = self.resize_image(img)

        img_input = img_resize.astype(np.float32) / 255.
        img_mean = np.array(self.mean, dtype=np.float32).reshape(1, 1, 3)
        img_std = np.array(self.std, dtype=np.float32).reshape(1, 1, 3)
        img_input = (img_input - img_mean) / img_std
        img_input = np.transpose(img_input, [2, 0, 1])
        img_input = np.expand_dims(img_input, axis=0)
        return img_input, ratio_hw

    @abstractmethod
    def infer_image(self, img_input):
        pass

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        min_size = 3
        bitmap = _bitmap
        height, width = bitmap.shape
        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 3:
            img, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        num_contours = min(len(contours), 1000)
        boxes = []
        scores = []
        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < min_size:
                continue
            points = np.array(points)
            if self.score_mode == "fast":
                score = self.box_score_fast(pred, points.reshape(-1, 2))
            else:
                score = self.box_score_slow(pred, contour)
            if self.box_thresh > score:
                continue

            box = self.unclip(points).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < min_size + 2:
                continue
            box = np.array(box)

            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.astype(np.int16))
            scores.append(score)
        return np.array(boxes, dtype=np.int16), scores

    def unclip(self, box):
        unclip_ratio = self.unclip_ratio
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [
            points[index_1], points[index_2], points[index_3], points[index_4]
        ]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        '''
        box_score_fast: use bbox mean score as the mean score
        '''
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def box_score_slow(self, bitmap, contour):
        '''
        box_score_slow: use polyon mean score as the mean score
        '''
        h, w = bitmap.shape[:2]
        contour = contour.copy()
        contour = np.reshape(contour, (-1, 2))

        xmin = np.clip(np.min(contour[:, 0]), 0, w - 1)
        xmax = np.clip(np.max(contour[:, 0]), 0, w - 1)
        ymin = np.clip(np.min(contour[:, 1]), 0, h - 1)
        ymax = np.clip(np.max(contour[:, 1]), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)

        contour[:, 0] = contour[:, 0] - xmin
        contour[:, 1] = contour[:, 1] - ymin

        cv2.fillPoly(mask, contour.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def postprocess(self, pred, shape_list):
        dilation_kernel = None if not self.use_dilation else np.array([[1, 1], [1, 1]])
        # pred = pred[0].numpy()
        pred = pred[:, 0, :, :]
        segmentation = pred > self.det_thresh
        src_h, src_w, ratio_h, ratio_w = shape_list
        if dilation_kernel is not None:
            mask = cv2.dilate(np.array(segmentation[0]).astype(np.uint8), dilation_kernel)
        else:
            mask = segmentation[0]

        boxes, scores = self.boxes_from_bitmap(pred[0], mask, src_w, src_h)
        return boxes

    def order_points_clockwise(self, pts):
        """
        reference from: https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
        # sort the points based on their x-coordinates
        """
        xSorted = pts[np.argsort(pts[:, 0]), :]

        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        (tr, br) = rightMost

        rect = np.array([tl, tr, br, bl], dtype="float32")
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def detect(self, img):
        src_h, src_w, _ = img.shape
        img_input, ratio_hw = self.preprocess(img)
        scores = self.infer_image(img_input)
        boxes = self.postprocess(scores, [src_h, src_w, ratio_hw[0], ratio_hw[1]])
        return boxes

    def draw_box(self, dt_boxes, src_im):
        for box in dt_boxes:
            box = np.array(box).astype(np.int32).reshape(-1, 2)
            cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
        return src_im


class PaddleOcrONNX(PaddleOcrABC):
    import onnxruntime as ort

    def __init__(self, model_path, *args, **kwargs):
        super(PaddleOcrONNX, self).__init__(*args, **kwargs)
        print("Using ONNX as inference backend")
        print(f"Using weight: {model_path}")

        # load model
        self.model_path = model_path
        self.ort_session = self.ort.InferenceSession(self.model_path)
        self.input_name = self.ort_session.get_inputs()[0].name

    def infer_image(self, img_input):
        inference_results = self.ort_session.run(None,
                                                 {self.input_name: img_input})

        return inference_results[0]


def test_one():
    detector = PaddleOcrONNX("./en_det.onnx")
    img = cv2.imread("./t18.jpg")
    bbox = detector.detect(img)
    res_img = detector.draw_box(bbox, img)
    cv2.imshow("ee", res_img)
    cv2.imwrite("res.jpg", res_img)
    cv2.waitKey(0)


test_one()
