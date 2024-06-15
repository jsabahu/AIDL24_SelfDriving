import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import random
from logger import Logger
from utils import read_yaml
from models.model_ENet import ENet
from torchvision import transforms
from torchviz import make_dot
import os
import torch
from models.LaneNet.LaneNet import LaneNet
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2


logger = Logger()

# Read configuration
config_path = "configs/config.yaml"
try:
    config = read_yaml(config_path)
    logger.log_debug(f"Read configuration from {config_path}")
except Exception as e:
    logger.log_error(f"Failed to read configuration from {config_path}: {e}")
    raise


class LaneNetEvaluator:
    def __init__(
        self, model_path, device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = torch.device(device)
        self.model = LaneNet()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.model.to(self.device)

    def load_test_data(self, img_path, transform):
        img = Image.open(img_path)
        img = transform(img)
        return img

    def test(self):
        if os.path.exists("test_output") == False:
            os.mkdir("test_output")

        test_image_folder = config["dataset"]["tusimple"]["test"]["images_path"]
        random_image = random.choice(os.listdir(test_image_folder))
        img_path = os.path.join(test_image_folder, random_image)
        resize_height = int(config["main"]["resize_height"])
        resize_width = int(config["main"]["resize_width"])

        data_transform = transforms.Compose(
            [
                transforms.Resize((resize_height, resize_width)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        dummy_input = self.load_test_data(img_path, data_transform).to(self.device)
        dummy_input = torch.unsqueeze(dummy_input, dim=0)
        outputs = self.model(dummy_input)

        input = Image.open(img_path)
        input = input.resize((resize_width, resize_height))
        input = np.array(input)

        instance_pred = (
            torch.squeeze(outputs["instance_seg_logits"].detach().to("cpu")).numpy()
            * 255
        )
        binary_pred = torch.squeeze(outputs["binary_seg_pred"]).to("cpu").numpy() * 255

        cv2.imwrite(os.path.join("test_output", random_image), input)
        cv2.imwrite(
            os.path.join(
                "test_output",
                os.path.splitext(random_image)[0] + "_instance_output.jpg",
            ),
            instance_pred.transpose((1, 2, 0)),
        )
        cv2.imwrite(
            os.path.join(
                "test_output", os.path.splitext(random_image)[0] + "_binary_output.jpg"
            ),
            binary_pred,
        )


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

            my_transforms = transforms.Compose(
                [
                    transforms.Resize((360, 640), antialias=True),
                    transforms.ToTensor(),
                ]
            )

            image = my_transforms(image).unsqueeze(0).to(self.device)
            mask = my_transforms(mask).to(self.device)
            mask = (mask == 1).type(torch.int)

            # Get the model output
            output = self.model(image)

            make_dot(output, params=dict(list(self.model.named_parameters()))).render(
                "rnn_torchviz", format="png"
            )

            output = torch.sigmoid(output)
            output = (output >= 0.5).float()
            print(output)

            # Convert tensors to numpy arrays
            image_np = (
                image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            )  # Convert to HWC and move to CPU
            mask_np = (
                mask.squeeze().cpu().numpy()
            )  # Remove channel dimension and move to CPU
            output_np = (
                torch.sigmoid(output).squeeze().cpu().numpy()
            )  # Apply sigmoid, squeeze, and move to CPU

            # Binarize the output
            output_np = (output_np >= 0.5).astype(np.float32)

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

    model_name = "LaneNet"

    if model_name == "ENet":
        # Example usage
        evaluator = LaneDetectionEvaluator(model_path="models/Lane_Model_ENet.pth")

        val_image_folder = config["val"]["images_path"]
        val_mask_folder = config["val"]["labels_path"]

        random_image = random.choice(os.listdir(val_image_folder))
        img_path = os.path.join(val_image_folder, random_image)

        mask_name = f"{os.path.splitext(random_image)[0]}.png"
        mask_path = os.path.join(val_mask_folder, mask_name)

        print(img_path)
        print(mask_path)

        # Open image and mask using PIL
        image = Image.open(img_path).convert("RGB")  # Ensure image is in RGB mode
        mask = Image.open(mask_path).convert("L")  # Convert mask to grayscale

        evaluator.evaluate(image, mask)
    elif model_name == "LaneNet":
        evaluator = LaneNetEvaluator(model_path="models/Lane_Model_ENet.pth")
        for i in range(10):
            evaluator.test()
