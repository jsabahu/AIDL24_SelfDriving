from torchvision.io.image import read_image
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
import torch.nn.functional as F
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image, pil_to_tensor

from torchvision.io.image import read_image
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.transforms.functional import to_pil_image

from torchvision.models.detection import (
    maskrcnn_resnet50_fpn_v2,
    MaskRCNN_ResNet50_FPN_V2_Weights,
)

import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from models.model_mask_R_CNN import LaneDetectionModel
from logger import Logger
from utils import read_yaml
from models.LaneNet.LaneNet import LaneNet
from torchvision import transforms
import numpy as np
import cv2
from utils import generate_full_image_rois
import cv2
import imageio


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
        self,
        model_path,
        checkpoint,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = torch.device(device)
        self.model = LaneNet()
        if model_path == "" and checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.model.to(self.device)

    def load_test_data(self, img_path, transform):
        img = Image.open(img_path)
        img = transform(img)
        return img

    def output_img(self, img):
        # Assuming img is a PIL Image, we need to transform it first
        resize_height = int(config["main"]["resize_height"])
        resize_width = int(config["main"]["resize_width"])

        data_transform = transforms.Compose(
            [
                transforms.Resize((resize_height, resize_width)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        img_transformed = data_transform(img).to(self.device)
        img_transformed = torch.unsqueeze(img_transformed, dim=0)

        # Perform inference
        with torch.no_grad():
            outputs = self.model(img_transformed)

        # Get the binary mask
        binary_pred = torch.squeeze(outputs["binary_seg_pred"]).to("cpu").numpy() * 255
        binary_pred = binary_pred.astype(np.uint8)

        return binary_pred


def inference_fastRCNN(img):
    # Step 1: Initialize model with the best available weights
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
    model.eval()

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()

    # Step 3: Apply inference preprocessing transforms
    batch = [preprocess(img)]

    # Step 4: Use the model and visualize the prediction
    prediction = model(batch)[0]
    labels = [weights.meta["categories"][i] for i in prediction["labels"]]
    if isinstance(img, Image.Image):
        img = pil_to_tensor(img)
    box = draw_bounding_boxes(
        img,
        boxes=prediction["boxes"],
        labels=labels,
        colors="red",
        width=4,
        font_size=30,
    )
    im = to_pil_image(box.detach())

    return im


def inference_LaneNet(img):
    # Initialize the LaneNetEvaluator
    checkpoint_path = "models/checkpoint_30.pth"
    checkpoint = None
    if not torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    else:
        checkpoint = torch.load(checkpoint_path)

    evaluator = LaneNetEvaluator(model_path="", checkpoint=checkpoint)

    if not isinstance(img, Image.Image):
        # Convert torch tensor to PIL Image
        img_pil = to_pil_image(img)
    else:
        img_pil = img

    im = evaluator.output_img(img_pil)

    # Apply a colormap only to the white pixels of the binary mask
    colored_mask = cv2.applyColorMap(im, cv2.COLORMAP_AUTUMN)

    # Create a mask where the binary mask is white
    white_mask = im == 255

    # Apply the white mask to the colored mask
    masked_colored_mask = np.zeros_like(colored_mask)
    masked_colored_mask[white_mask] = colored_mask[white_mask]

    # Convert the original image to a numpy array
    input_img = np.array(img_pil.resize((im.shape[1], im.shape[0])))

    # Blend the original image and the masked colored mask
    blended_image = cv2.addWeighted(input_img, 0.7, masked_colored_mask, 0.3, 0)

    # Convert to PIL image for consistent output format
    blended_image_pil = Image.fromarray(blended_image)

    return blended_image_pil


def inference_LaneDetectionModel(img):
    # Initialize the LaneDetectionModel
    model_path = "models/train_mask_rCNN.pth"
    # Device configuration
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    target_size1 = (180, 320)

    # Prepare Mask R-CNN model
    model1 = LaneDetectionModel().to(DEVICE)
    model1.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model1.eval()

    # Transform for Mask R-CNN
    transform1 = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize(target_size1, antialias=True)]
    )

    if isinstance(img, torch.Tensor):
        image1 = img.to(torch.float32).unsqueeze(0).to(DEVICE)
    else:
        image1 = transform1(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output1 = model1(image1, generate_full_image_rois(1, target_size1, 0))
        output1 = F.interpolate(
            output1, size=target_size1, mode="bilinear", align_corners=False
        )
        output1 = (output1 - output1.min()) / (output1.max() - output1.min())
        output1 = (output1 > 0.01).type(torch.int)
        output1_np = output1.detach().cpu().numpy()[0, 0]  # Squeeze the last dimension

    return output1_np


def inference_MaskRCNN(img):
    # Step 1: Initialize model with the best available weights
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn_v2(weights=weights)
    model.eval()

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()

    # Step 3: Apply inference preprocessing transforms
    batch = preprocess(img).unsqueeze(0)

    # Step 4: Use the model and visualize the prediction
    with torch.no_grad():
        predictions = model(batch)

    # Step 5: Extract the masks and visualize
    masks = predictions[0]["masks"]
    labels = predictions[0]["labels"]

    # Define colors for each category
    category_colors = {
        "car": (255, 0, 0),  # Red
        "motorcycle": (0, 255, 0),  # Green
        "bus": (0, 0, 255),  # Blue
        "truck": (255, 255, 0),  # Yellow
        "traffic light": (255, 0, 255),  # Magenta
    }

    # Categories of interest
    categories_of_interest = [
        "car",
    ]

    # Convert the original image to a numpy array
    if not isinstance(img, Image.Image):
        input_img = np.array(to_pil_image(img))
    else:
        input_img = np.array(img)

    # Resize the input image if necessary
    input_img_resized = cv2.resize(input_img, (input_img.shape[1], input_img.shape[0]))

    # Create a blank mask for the combined masks
    combined_mask = np.zeros_like(input_img_resized)

    # Find the masks corresponding to the specified categories and apply colors
    for i, label in enumerate(labels):
        category = weights.meta["categories"][label.item()]
        if category in categories_of_interest:
            mask = masks[i, 0].cpu().numpy()
            mask = (mask > 0.99).astype(np.uint8) * 255
            color = category_colors[category]
            colored_mask = np.zeros_like(input_img_resized)
            colored_mask[mask == 255] = color
            combined_mask = cv2.addWeighted(combined_mask, 1.0, colored_mask, 0.5, 0)

    # Blend the original image and the combined mask
    blended_image = cv2.addWeighted(input_img_resized, 0.7, combined_mask, 0.3, 0)

    # Convert to PIL image for consistent output format
    blended_image_pil = Image.fromarray(blended_image)

    return blended_image_pil


def video_to_frames(video_path):
    # Open the video file
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    frames = []

    while success:
        # Convert the frame from BGR (OpenCV format) to RGB (PIL format)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Convert the frame to a PIL Image
        pil_image = Image.fromarray(image_rgb)
        frames.append(pil_image)
        success, image = vidcap.read()

    vidcap.release()
    return frames


def frames_to_gif(frames, gif_path, duration=0.1):
    """
    Converts a list of PIL images to a GIF.

    Parameters:
        frames (list): List of PIL Image frames.
        gif_path (str): Path where the GIF will be saved.
        duration (float): Duration of each frame in seconds.
    """
    # Convert frames to numpy arrays
    frames_np = [np.array(frame) for frame in frames]

    # Save the frames as a GIF
    imageio.mimsave(gif_path, frames_np, duration=duration)


if __name__ == "__main__":
    video_path = "data/selfmade/youtube_dashcam.mp4"
    # img = read_image("data/bdd100ktrans/images/100k/train/fe16aa77-810c2e1f.jpg")ç
    frames = video_to_frames(video_path=video_path)
    processed_frames_LaneNet = []
    processed_frames_FasterRCNN = []
    processed_frames_MaskRCNN = []
    count = 0
    for img in frames:
        print(f"Processing image --> {count}/{len(frames)}")
        processed_frames_LaneNet.append(inference_LaneNet(img))
        processed_frames_FasterRCNN.append(inference_fastRCNN(img))
        processed_frames_MaskRCNN.append(inference_MaskRCNN(img))
        count += 1

    frames_to_gif(
        processed_frames_LaneNet, "results/youtube_dashcam_processed_LaneNet.gif"
    )
    frames_to_gif(
        processed_frames_FasterRCNN, "results/youtube_dashcam_processed_FasterRCNN.gif"
    )
    frames_to_gif(
        processed_frames_MaskRCNN, "results/youtube_dashcam_processed_MaskRCNN.gif"
    )
