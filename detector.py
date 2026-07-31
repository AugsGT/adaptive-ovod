from pathlib import Path
from PIL import Image
import torch

from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection
)


class GroundingDINODetector:
    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-base",
        device: str | None = None,
    ):
        # If else code block to use CUDA when GPU is available  else fallback to CPU
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu" #if else condition to switch to cpu

        # Store the selected compute device so all future model inputs and outputs are moved there
        self.device = device

        print(f"Loading model on {self.device}...")

        # The processor converts images and text prompts into the tensor format expected by Grounding DINO
        self.processor = AutoProcessor.from_pretrained(model_id)

        # Load the pretrained zero-shot object detection model and move it to the selected device
        self.model = (
            AutoModelForZeroShotObjectDetection
            .from_pretrained(model_id)
            .to(self.device)
        )

        # Put the model in evaluation mode so batch normalization and dropout behave consistently for inference
        self.model.eval()

        print("Grounding DINO loaded.")

    def detect(
        self,
        image_path: str,
        text_prompt: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ):

        # Open the input image and convert it to RGB so the model receives a consistent 3-channel image format
        image = Image.open(image_path).convert("RGB")

        # Run the image + text prompt through the processor to create model-ready tensors
        inputs = self.processor(
            images=image,
            text=text_prompt,
            return_tensors="pt",
        )

        # Move every tensor in the processed input dictionary to the same device as the model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Disable gradient tracking because we are only doing inference, not training
        with torch.no_grad():
            # Run the model on the prepared input tensors and get raw detection predictions
            outputs = self.model(**inputs)

        # Convert the model's raw outputs into human-readable detection results, including boxes and labels,
        # while filtering out weak predictions based on the threshold values
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=inputs["input_ids"],
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
)
            
        # Return the original image alongside the first result set for the supplied image
        return image, results[0]


if __name__ == "__main__":

    # Create a detector instance using the default model and automatic device selection
    detector = GroundingDINODetector()

    # Path to the image we want to analyze.
    IMAGE = "11.jpg"

    # Text query used to guide object detection; the model will look for matches to this description
    PROMPT = "forklift"

    # Run detection for the chosen image and text query
    image, detections = detector.detect(
        IMAGE,
        PROMPT,
    )

    # Print a header before listing all detected objects
    print("\nDetections:\n")

    # Iterate through the scores, labels, and bounding boxes returned for each detection
    labels = detections.get("text_labels", detections["labels"])

for score, label, box in zip(
    detections["scores"],
    labels,
    detections["boxes"],
):
        # Print each detection as: label, confidence score, and the bounding box coordinates
        print(
            f"{label:20s}"
            f"{float(score):.3f} "
            f"{box.tolist()}"
        )