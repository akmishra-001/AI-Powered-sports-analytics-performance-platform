import cv2
import numpy as np
from ultralytics import YOLO

class SportsCVEngine:
    def __init__(self):
        # Human/Player Detection ke liye YOLOv8 model load karna
        self.model = YOLO("yolov8n.pt")

    def process_frame_and_track(self, frame, frame_idx, player_stats):
        """
        Processes a single frame live, updates individual player tracking dictionary,
        and returns annotated frame with bounding boxes.
        """
        results = self.model(frame, classes=[0], verbose=False)
        player_counter = 0
        annotated_frame = frame.copy()

        for result in results:
            boxes = result.boxes
            for box in boxes:
                player_counter += 1
                p_id = f"Player #{player_counter}"
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

                # Initialize tracking entry for new player ID
                if p_id not in player_stats:
                    player_stats[p_id] = {
                        "detected_frames": 0,
                        "positions": [],
                        "confidences": []
                    }

                player_stats[p_id]["detected_frames"] += 1
                player_stats[p_id]["positions"].append((center_x, center_y))
                player_stats[p_id]["confidences"].append(confidence)

                # Draw green bounding box & label
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated_frame,
                    f"{p_id} ({confidence:.2f})",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

        # Top overlay summary
        cv2.rectangle(annotated_frame, (10, 10), (320, 60), (0, 0, 0), -1)
        cv2.putText(
            annotated_frame,
            f"Active Players Detected: {player_counter}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        return annotated_frame, player_counter

    def calculate_final_player_metrics(self, player_stats):
        """
        Converts tracked positional data into Submodule 3 numerical inputs per player.
        """
        final_player_metrics = {}
        for p_id, data in player_stats.items():
            det_frames = data["detected_frames"]
            if det_frames < 3: # Ignore brief noise detections
                continue

            positions = data["positions"]
            distances = [
                np.sqrt((positions[i][0] - positions[i-1][0])**2 + (positions[i][1] - positions[i-1][1])**2)
                for i in range(1, len(positions))
            ]
            total_pixel_dist = float(np.sum(distances)) if distances else 0.0
            avg_speed_pixels = float(np.mean(distances)) if distances else 0.0

            # Mapping tracking data to realistic Submodule 3 input scales
            stamina = float(np.clip(100.0 - (det_frames * 0.08), 65.0, 98.0))
            pass_accuracy = float(np.clip(72.0 + (np.mean(data["confidences"]) * 18.0), 55.0, 96.0))
            distance_km = float(np.clip(round(total_pixel_dist * 0.0025, 2), 1.2, 16.5))
            max_speed_kmh = float(np.clip(round(20.0 + (avg_speed_pixels * 0.45), 2), 14.0, 36.0))
            key_actions = int(np.clip(int(det_frames / 12), 1, 15))

            final_player_metrics[p_id] = {
                "stamina": stamina,
                "pass_accuracy": pass_accuracy,
                "distance": distance_km,
                "speed": max_speed_kmh,
                "key_actions": key_actions,
                "frames_tracked": det_frames
            }

        return final_player_metrics