import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import random
from logger import Logger
from utils import read_yaml
from models.model_ENet import ENet

logger = Logger()

# Read configuration
config_path = "configs/config.yaml"
try:
    config = read_yaml(config_path)
    logger.log_debug(f"Read configuration from {config_path}")
except Exception as e:
    logger.log_error(f"Failed to read configuration from {config_path}: {e}")
    raise


class LaneDetectionEvaluator:
    def __init__(
        self, model_path, device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = torch.device(device)
        self.model = ENet(num_classes=1).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def evaluate(self, image, mask):
        with torch.no_grad():
            # Prepare the image
            image_tensor = torch.from_numpy(image).unsqueeze(0).to(self.device).float()

            # Get the model output
            output = self.model(image_tensor)
            # output = torch.sigmoid(output)
            # output = (output >= 0.5).float()

            # Convert tensors to numpy arrays
            image_np = image.transpose(1, 2, 0)  # Convert to HWC
            mask_np = mask.squeeze()  # Remove channel dimension if exists
            output_np = output.squeeze().cpu().numpy()  # Squeeze and move to CPU

            # Plot the images
            self.plot_results(image_np, mask_np, output_np)

    def plot_results(self, image, mask, output):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        axs[0].imshow(image)
        axs[0].set_title("Original Image")
        axs[0].axis("off")

        axs[1].imshow(mask, cmap="gray")
        axs[1].set_title("Mask Ground Truth")
        axs[1].axis("off")

        axs[2].imshow(output, cmap="gray")
        axs[2].set_title("Model Output")
        axs[2].axis("off")

        plt.show()


if __name__ == "__main__":
    # Example usage
    evaluator = LaneDetectionEvaluator(model_path="models/Lane_Model_ENet.pth")

    val_image_folder = config["validation"]["val_images_path"]
    val_mask_folder = config["validation"]["val_labels_path"]

    random_image = random.choice(os.listdir(val_image_folder))
    img_path = os.path.join(val_image_folder, random_image)

    mask_name = f"{os.path.splitext(random_image)[0]}.png"
    mask_path = os.path.join(val_mask_folder, mask_name)

    print(img_path)
    print(mask_path)

    # Open image and mask using PIL
    image = Image.open(img_path).convert("RGB")  # Ensure image is in RGB mode
    mask = Image.open(mask_path).convert("L")  # Convert mask to grayscale

    # evaluator.evaluate(image, mask)
