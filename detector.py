from PIL import Image
import torch
import cv2
import numpy as np

from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
)


class GroundingDINODetector:
    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-base",
        device: str | None = None,
    ):
        # Automatically use GPU if available, otherwise fall back to CPU
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        print(f"Loading model on {self.device}...")

        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_id)

        # Load model
        self.model = (
            AutoModelForZeroShotObjectDetection
            .from_pretrained(model_id)
            .to(self.device)
        )

        self.model.eval()

        print("Grounding DINO loaded.")

    def detect(
        self,
        image_path: str,
        text_prompt: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ):
        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(
            images=image,
            text=text_prompt,
            return_tensors="pt",
        )

        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
        }

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=inputs["input_ids"],
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
        )

        hf = results[0]

        labels = hf.get("text_labels", hf["labels"])

        detections = []

        for score, label, box in zip(
            hf["scores"],
            labels,
            hf["boxes"],
        ):
            detections.append(
                {
                    "bbox": box.tolist(),
                    "confidence": float(score),
                    "label": str(label),
                }
            )

        return image, detections

    def draw_detections(
        self,
        image,
        detections,
        output_path="output.jpg",
    ):
        image = np.array(image)

        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR,
        )

        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            label = f'{det["label"]} {det["confidence"]:.2f}'

            cv2.putText(
                image,
                label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        cv2.imwrite(output_path, image)


if __name__ == "__main__":

    detector = GroundingDINODetector()

    IMAGE = "11.jpg"
    PROMPT = "forklift"

    image, detections = detector.detect(
        IMAGE,
        PROMPT,
    )

    print("\nDetections:\n")

    for det in detections:
        print(
            f'{det["label"]:20s}'
            f'{det["confidence"]:.3f} '
            f'{det["bbox"]}'
        )

    detector.draw_detections(
        image,
        detections,
        "output.jpg",
    )

    print("\nSaved annotated image as output.jpg")