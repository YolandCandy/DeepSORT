from pathlib import Path
import argparse
import csv
import math
import os

# os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2

_YOLO_CONFIG_DIR = Path(__file__).resolve().parent / ".ultralytics_config"
_YOLO_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_YOLO_CONFIG_DIR))

from ultralytics import YOLO


VEHICLE_CLASSES = [0]  # COCO: car=2, bus=5
REAL_BOX_COLOR = (255, 0, 0)
DEFAULT_CONF = 0.05
DEFAULT_IOU = 0.8
DEFAULT_STITCH_MAX_GAP = 15
DEFAULT_STITCH_IOU = 0.25
DEFAULT_STITCH_CENTER_DIST = 0.6


def clip_box(box, width, height):
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(round(x1)), width - 1))
    y1 = max(0, min(int(round(y1)), height - 1))
    x2 = max(x1 + 1, min(int(round(x2)), width - 1))
    y2 = max(y1 + 1, min(int(round(y2)), height - 1))
    return [x1, y1, x2, y2]


def draw_labeled_box(frame, box, label, color):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = clip_box(box, w, h)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    label_h = text_h + baseline + 8
    label_w = text_w + 10

    label_y1 = max(0, y1 - label_h)
    label_y2 = label_y1 + label_h
    label_x2 = min(w - 1, x1 + label_w)
    cv2.rectangle(frame, (x1, label_y1), (label_x2, label_y2), color, -1)
    cv2.putText(frame, label, (x1 + 5, label_y2 - baseline - 4), font, font_scale, (255, 255, 255), thickness)


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    intersection = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def predict_box(box, velocity, frame_gap):
    return [box[index] + velocity[index] * frame_gap for index in range(4)]


def center_distance_ratio(box_a, box_b):
    ax = (box_a[0] + box_a[2]) / 2
    ay = (box_a[1] + box_a[3]) / 2
    bx = (box_b[0] + box_b[2]) / 2
    by = (box_b[1] + box_b[3]) / 2
    diagonal = math.hypot(box_a[2] - box_a[0], box_a[3] - box_a[1])
    if diagonal <= 0:
        return float("inf")
    return math.hypot(bx - ax, by - ay) / diagonal


class StableIdStitcher:
    def __init__(self, enabled=True, max_gap=DEFAULT_STITCH_MAX_GAP, iou_threshold=DEFAULT_STITCH_IOU, center_dist_threshold=DEFAULT_STITCH_CENTER_DIST):
        self.enabled = enabled
        self.max_gap = max_gap
        self.iou_threshold = iou_threshold
        self.center_dist_threshold = center_dist_threshold
        self.raw_to_stable = {}
        self.raw_ids = set()
        self.stable_tracks = {}
        self.stitched_fragments = 0

    def resolve(self, raw_track_id, box, class_id, frame_num, blocked_stable_ids=None):
        raw_track_id = int(raw_track_id)
        class_id = int(class_id)
        box = [float(value) for value in box]
        self.raw_ids.add(raw_track_id)

        if not self.enabled:
            return raw_track_id

        if raw_track_id in self.raw_to_stable:
            stable_id = self.raw_to_stable[raw_track_id]
        else:
            stable_id = self._find_match(box, class_id, frame_num, blocked_stable_ids or set())
            if stable_id is None:
                stable_id = raw_track_id
            else:
                self.stitched_fragments += 1
            self.raw_to_stable[raw_track_id] = stable_id

        self._update_track(stable_id, box, class_id, frame_num)
        return stable_id

    def _find_match(self, box, class_id, frame_num, blocked_stable_ids):
        best_stable_id = None
        best_rank = None
        for stable_id, track in self.stable_tracks.items():
            if stable_id in blocked_stable_ids or track["class_id"] != class_id:
                continue

            frame_gap = frame_num - track["last_frame"]
            if frame_gap <= 0 or frame_gap > self.max_gap:
                continue

            candidate_box = predict_box(track["last_box"], track["velocity"], frame_gap)
            iou_score = box_iou(candidate_box, box)
            center_ratio = center_distance_ratio(candidate_box, box)
            if iou_score < self.iou_threshold and center_ratio > self.center_dist_threshold:
                continue

            rank = (iou_score >= self.iou_threshold, iou_score, -center_ratio, -frame_gap)
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_stable_id = stable_id

        return best_stable_id

    def _update_track(self, stable_id, box, class_id, frame_num):
        previous = self.stable_tracks.get(stable_id)
        velocity = [0.0, 0.0, 0.0, 0.0]
        if previous is not None:
            frame_gap = frame_num - previous["last_frame"]
            if frame_gap > 0:
                velocity = [(box[index] - previous["last_box"][index]) / frame_gap for index in range(4)]
            else:
                velocity = previous["velocity"]

        self.stable_tracks[stable_id] = {
            "class_id": class_id,
            "last_box": box,
            "last_frame": frame_num,
            "velocity": velocity,
        }

    @property
    def raw_count(self):
        return len(self.raw_ids)

    @property
    def stable_count(self):
        if not self.enabled:
            return len(self.raw_ids)
        return len(set(self.raw_to_stable.values()))


class ByteTrack:
    def __init__(self, device=None):
        base_dir = Path(__file__).resolve().parent
        weight_path = base_dir / "yolov11n_last.pt"
        self.video_path = "UA_Detrac_test.mp4"
        self.bytetrack_yaml = base_dir / "bytetrack.yaml"
        self.device = device

        if not weight_path.exists():
            raise FileNotFoundError(f"Model weight not found: {weight_path}")
        # if not self.video_path.exists():
        #     raise FileNotFoundError(f"Video not found: {self.video_path}")
        if not self.bytetrack_yaml.exists():
            raise FileNotFoundError(f"ByteTrack config not found: {self.bytetrack_yaml}")

        self.model = YOLO(weight_path)

    def detect(self):
        predict_args = dict(
            source=self.video_path, 
            show=True, 
            line_width=2, 
            conf=DEFAULT_CONF, 
            iou=DEFAULT_IOU,
            classes=VEHICLE_CLASSES,
            save=False,
        )
        if self.device:
            predict_args["device"] = self.device
        return self.model.predict(**predict_args)

    def track(
        self,
        output_path=None,
        show=False,
        conf=DEFAULT_CONF,
        iou=DEFAULT_IOU,
        image_scale=1,
        save_pred_csv=None,
        save_eval_csv=None,
        max_frames=None,
        stitch_ids=True,
        stitch_max_gap=DEFAULT_STITCH_MAX_GAP,
        stitch_iou=DEFAULT_STITCH_IOU,
        stitch_center_dist=DEFAULT_STITCH_CENTER_DIST,
    ):
        output_path = Path(output_path) if output_path else Path(__file__).resolve().parent / "MVI_39031.mp4"
        if save_pred_csv is None:
            save_pred_csv = output_path.with_name(f"{output_path.stem}_prediction.csv")
        else:
            save_pred_csv = Path(save_pred_csv)
        if save_eval_csv is None:
            save_eval_csv = output_path.with_name(f"{output_path.stem}_evaluation.csv")
        else:
            save_eval_csv = Path(save_eval_csv)

        frame_count = 0
        pred_rows = []
        eval_rows = []
        sequence_name = self.video_path.stem
        cap = cv2.VideoCapture(str(self.video_path))
        writer = None
        stitcher = StableIdStitcher(
            enabled=stitch_ids,
            max_gap=stitch_max_gap,
            iou_threshold=stitch_iou,
            center_dist_threshold=stitch_center_dist,
        )

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if max_frames is not None and frame_count > max_frames:
                break

            h, w = frame.shape[:2]
            new_h = round(h / image_scale)
            new_w = round(w / image_scale)
            frame = cv2.resize(frame, (new_w, new_h))
            scale_x = w / new_w
            scale_y = h / new_h

            if writer is None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output_path), fourcc, fps, (new_w, new_h))
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot create output video: {output_path}")

            track_args = dict(
                source=frame, 
                persist=True, 
                tracker=str(self.bytetrack_yaml),
                conf=conf,
                iou=iou,
                classes=VEHICLE_CLASSES,
                verbose=False,
            )
            if self.device:
                track_args["device"] = self.device

            results = self.model.track(**track_args)
            boxes = results[0].boxes
            if boxes is not None and boxes.id is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                ids = boxes.id.cpu().numpy().astype(int)
                confidences = boxes.conf.cpu().numpy()
                class_ids = boxes.cls.cpu().numpy().astype(int)
                current_raw_ids = {int(track_id) for track_id in ids}
                blocked_stable_ids = {
                    stitcher.raw_to_stable[raw_id]
                    for raw_id in current_raw_ids
                    if raw_id in stitcher.raw_to_stable
                }
                for box, raw_track_id, confidence, class_id in zip(xyxy, ids, confidences, class_ids):
                    raw_track_id = int(raw_track_id)
                    box = [float(value) for value in box]
                    stable_track_id = stitcher.resolve(
                        raw_track_id,
                        box,
                        class_id,
                        frame_count,
                        blocked_stable_ids=blocked_stable_ids,
                    )
                    blocked_stable_ids.add(stable_track_id)
                    label = f"ID {stable_track_id}"
                    draw_labeled_box(frame, box, label, REAL_BOX_COLOR)
                    if save_pred_csv or save_eval_csv:
                        original_box = [
                            box[0] * scale_x,
                            box[1] * scale_y,
                            box[2] * scale_x,
                            box[3] * scale_y,
                        ]
                        x1, y1, x2, y2 = original_box
                        base_row = {
                            "sequence": sequence_name,
                            "frame_num": frame_count,
                            "track_id": stable_track_id,
                            "confidence": round(float(confidence), 6),
                            "class_id": int(class_id),
                        }
                        if save_pred_csv:
                            pred_rows.append(
                                {
                                    **base_row,
                                    "x1": round(x1, 3),
                                    "y1": round(y1, 3),
                                    "x2": round(x2, 3),
                                    "y2": round(y2, 3),
                                }
                            )
                        if save_eval_csv:
                            eval_rows.append(
                                {
                                    **base_row,
                                    "x": round(x1, 3),
                                    "y": round(y1, 3),
                                    "w": round(x2 - x1, 3),
                                    "h": round(y2 - y1, 3),
                                }
                            )

            writer.write(frame)

            if frame_count % 50 == 0:
                print(f"Processed {frame_count} frames")

            if show:
                cv2.imshow("frame", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

        if save_pred_csv:
            save_pred_csv.parent.mkdir(parents=True, exist_ok=True)
            with save_pred_csv.open("w", newline="", encoding="utf-8") as handle:
                fieldnames = ["sequence", "frame_num", "track_id", "x1", "y1", "x2", "y2", "confidence", "class_id"]
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(pred_rows)
            print(f"Saved prediction CSV to {save_pred_csv}")

        if save_eval_csv:
            save_eval_csv.parent.mkdir(parents=True, exist_ok=True)
            with save_eval_csv.open("w", newline="", encoding="utf-8") as handle:
                fieldnames = ["sequence", "frame_num", "track_id", "x", "y", "w", "h", "confidence", "class_id"]
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(eval_rows)
            print(f"Saved evaluation CSV to {save_eval_csv}")

        print(f"Raw IDs: {stitcher.raw_count}")
        print(f"Stable IDs: {stitcher.stable_count}")
        print(f"Stitched ID fragments: {stitcher.stitched_fragments}")
        print(f"Saved tracking video to {output_path}")
        return output_path


def run_detection():
    ot = ByteTrack()
    ot.detect()


def run_tracking(
    output_path=None,
    show=False,
    device=None,
    conf=DEFAULT_CONF,
    iou=DEFAULT_IOU,
    image_scale=1,
    save_pred_csv=None,
    save_eval_csv=None,
    max_frames=None,
    stitch_ids=True,
    stitch_max_gap=DEFAULT_STITCH_MAX_GAP,
    stitch_iou=DEFAULT_STITCH_IOU,
    stitch_center_dist=DEFAULT_STITCH_CENTER_DIST,
):
    ot = ByteTrack(device=device)
    return ot.track(
        output_path=output_path,
        show=show,
        conf=conf,
        iou=iou,
        image_scale=image_scale,
        save_pred_csv=save_pred_csv,
        save_eval_csv=save_eval_csv,
        max_frames=max_frames,
        stitch_ids=stitch_ids,
        stitch_max_gap=stitch_max_gap,
        stitch_iou=stitch_iou,
        stitch_center_dist=stitch_center_dist,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO + ByteTrack and save an annotated video.")
    parser.add_argument("--output", "-o", default="bytetrack_output.mp4", help="Output video path.")
    parser.add_argument("--device", default=None, help="Torch device, for example cuda:0 or cpu.")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="Minimum detection confidence passed into tracker.")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU, help="NMS IoU threshold.")
    parser.add_argument("--scale", type=float, default=1.0, help="Resize factor divisor. 2 means half size.")
    parser.add_argument("--save-pred-csv", default=None, help="Optional path to save x1,y1,x2,y2 prediction CSV.")
    parser.add_argument("--save-eval-csv", default=None, help="Optional path to save x,y,w,h CSV for MOTA evaluation.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional limit for quick tests.")
    parser.add_argument("--no-stitch-ids", action="store_true", help="Disable stable ID stitching after ByteTrack.")
    parser.add_argument("--stitch-max-gap", type=int, default=DEFAULT_STITCH_MAX_GAP, help="Max missing frames for stable ID stitching.")
    parser.add_argument("--stitch-iou", type=float, default=DEFAULT_STITCH_IOU, help="Minimum IoU for stable ID stitching.")
    parser.add_argument("--stitch-center-dist", type=float, default=DEFAULT_STITCH_CENTER_DIST, help="Max center distance ratio for stable ID stitching.")
    parser.add_argument("--show", action="store_true", help="Show the video while processing.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_tracking(
        output_path=args.output,
        show=args.show,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        image_scale=args.scale,
        save_pred_csv=args.save_pred_csv,
        save_eval_csv=args.save_eval_csv,
        max_frames=args.max_frames,
        stitch_ids=not args.no_stitch_ids,
        stitch_max_gap=args.stitch_max_gap,
        stitch_iou=args.stitch_iou,
        stitch_center_dist=args.stitch_center_dist,
    )
