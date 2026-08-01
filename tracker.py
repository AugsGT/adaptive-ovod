import numpy as np
import supervision as sv

try:
    # New tracker package (recommended)
    from trackers import ByteTrackTracker

    NEW_TRACKER = True

except ImportError:

    # Fallback for current implementation
    NEW_TRACKER = False


class ByteTrackerWrapper:

    def __init__(self):

        if NEW_TRACKER:
            self.tracker = ByteTrackTracker()
        else:
            self.tracker = sv.ByteTrack()

    def update(self, detections):

        if len(detections) == 0:
            return []

        xyxy = np.array(
            [d["bbox"] for d in detections],
            dtype=np.float32,
        )

        confidence = np.array(
            [d["confidence"] for d in detections],
            dtype=np.float32,
        )

        class_id = np.zeros(
            len(detections),
            dtype=np.int32,
        )

        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
        )

        if NEW_TRACKER:
            tracked = self.tracker.update(
                sv_detections
            )
        else:
            tracked = self.tracker.update_with_detections(
                sv_detections
            )

        trajectories = []

        if len(tracked) == 0:
            return trajectories

        for i in range(len(tracked)):

            trajectories.append(
                {
                    "track_id": int(tracked.tracker_id[i]),
                    "bbox": tracked.xyxy[i].tolist(),
                    "confidence": float(tracked.confidence[i]),
                    "label": detections[i]["label"],
                }
            )

        return trajectories