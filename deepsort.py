import os
import cv2
import argparse
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Car Tracking with YOLO26 and DeepSORT")
    parser.add_argument("--video", type=str, required=True, help="Path to input video (e.g., UA-DETRAC sequence)")
    parser.add_argument("--output", type=str, default="output.mp4", help="Path to save output video")
    parser.add_argument("--trackeval-output", type=str, default=None, help="Path to save TrackEval-formatted tracking results (.txt)")
    parser.add_argument("--model", type=str, default="yolo/yolov11n_last.pt", help="Path to YOLO26 model weights")
    # Argument to toggle processed output video saving
    parser.add_argument("--save-video", action="store_true", help="Save the processed output video (default: False)")
    # Argument to toggle TrackEval text file saving (enabled by default)
    parser.add_argument("--no-save-txt", action="store_false", dest="save_txt", default=True, help="Do not save TrackEval tracking results (default: False, i.e., save by default)")
    # Argument to toggle real-time video window visualization
    parser.add_argument("--show-video", action="store_true", help="Show video visualization in real-time (default: False)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize YOLO26 model
    # YOLO26 is natively supported by the ultralytics package in this environment
    print(f"Loading YOLO model: {args.model}")
    model = YOLO(args.model)
    
    # Initialize DeepSORT Tracker
    # We use the pre-trained embedder by default.
    # To use a custom Re-ID model, you can pass embedder="mobilenet" or a custom path
    print("Initializing DeepSORT...")
    tracker = DeepSort(
        max_age=30,
        n_init=3,
        nms_max_overlap=1.0,
        max_cosine_distance=0.2,
        nn_budget=None,
        override_track_class=None,
        embedder="mobilenet",
        half=True,
        bgr=True
    )
    
    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video}")
        return
        
    # Get video properties for output
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Initialize VideoWriter only if save_video is set to True
    out = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"Writing output video to: {args.output}")

    # Initialize TrackEval output file only if save_txt is set to True
    trackeval_file = None
    trackeval_output = None
    if args.save_txt:
        trackeval_output = args.trackeval_output
        if trackeval_output is None:
            trackeval_output = os.path.splitext(args.output)[0] + ".txt"
        trackeval_file = open(trackeval_output, "w")
        print(f"Writing TrackEval formatted output to: {trackeval_output}")
    
    print("Starting tracking...")
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Run YOLO26 inference
        # only class vehicle
        results = model(frame, verbose=False)
        
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                w = x2 - x1
                h = y2 - y1
                
                # Confidence
                conf = float(box.conf[0].cpu().numpy())
                
                # Class
                cls_id = int(box.cls[0].cpu().numpy())
                class_name = model.names[cls_id]
                
                # Format for deep_sort_realtime: [ [left,top,w,h], confidence, detection_class ]
                detections.append(([x1, y1, w, h], conf, class_name))
                
        # Update tracks
        # The tracker expects a BGR image by default
        tracks = tracker.update_tracks(detections, frame=frame)
        
        # Draw bounding boxes and track IDs
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            confidence = track.get_det_conf()
            try:
                if track_id is None:
                    continue
                track_id = int(track_id)
            except (ValueError, TypeError):
                continue
            if track_id < 0 or confidence is None:
                continue

            ltrb = track.to_ltrb(orig=True)
            if ltrb is None:
                ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            w = x2 - x1
            h = y2 - y1
            score = float(confidence)

            # Use a common color for all tracks
            box_color = (255, 0, 0)  # blue

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            # Draw label
            label = f"ID: {track_id}"
            cv2.putText(frame, label, (x1, max(10, y1 - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

            # Only write tracking result to file if save_txt is enabled
            if args.save_txt and trackeval_file is not None:
                # TrackEval format: frame, id, x, y, w, h, score, class, visibility, -1
                track_class_id = 0
                visibility = 1.0
                trackeval_file.write(
                    f"{frame_count},{track_id},{x1},{y1},{w},{h},{score:.6f},{track_class_id},-1,-1\n"
                )
                        
        # Save to output video if save_video is enabled
        if args.save_video and out is not None:
            out.write(frame)
        
        # Display in real-time if show_video is enabled
        if args.show_video:
            cv2.imshow("Car Tracking (YOLOv11 + DeepSORT)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    cap.release()
    
    # Release VideoWriter if it was initialized
    if args.save_video and out is not None:
        out.release()
        print(f"Tracking complete. Output saved to {args.output}")
    
    # Close TrackEval output file if it was initialized
    if args.save_txt and trackeval_file is not None:
        trackeval_file.close()
        print(f"TrackEval results saved to {trackeval_output}")
        
    # Close cv2 visualization windows if opened
    if args.show_video:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()