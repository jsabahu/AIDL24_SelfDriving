import torch
import os
import zipfile
import tarfile
import shutil
from pathlib import Path
import yaml
from typing import List, Optional, Any
from logger import Logger
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2
import numpy as np

logger = Logger()


class Rescale:
    """Rescale the image in a sample to a given size.

    Args:
        output_size (width, height) (tuple): Desired output size (width, height). Output is
            matched to output_size.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (tuple))
        self.output_size = output_size

    def __call__(self, sample):
        sample = cv2.resize(
            sample, dsize=self.output_size, interpolation=cv2.INTER_NEAREST
        )

        return sample


def binary_accuracy_with_logits(labels: torch.Tensor, outputs: torch.Tensor) -> float:
    """
    Calculate the binary accuracy with logits.

    Parameters:
    labels (torch.Tensor): Ground truth labels.
    outputs (torch.Tensor): Model outputs.

    Returns:
    float: Accuracy value.
    """
    preds = torch.sigmoid(outputs).round()
    # acc = (preds == labels.view_as(preds)).float().mean().item()
    acc = (preds == labels.view_as(preds)).float().detach().numpy().mean()
    return acc


def list_files_in_folder(folder_path: str) -> Optional[List[str]]:
    """
    List all .jpg files in a folder.

    Parameters:
    folder_path (str): Path to the folder.

    Returns:
    Optional[List[str]]: List of .jpg files in the folder, or None if an error occurs.
    """
    try:
        folder = Path(folder_path)
        if not folder.exists():
            logger.log_error("Folder does not exist.")
            return []
        return [file.name for file in folder.glob("*.jpg")]
    except Exception as e:
        logger.log_error("Error listing files in folder:", e)
        return None


def unzip_folder(file_path: str, folder_inside: str, destination_path: str):
    """
    Unzip a folder from inside a ZIP or TAR/TGZ file to a destination.

    Parameters:
    file_path (str): Path to the ZIP or TAR/TGZ file.
    folder_inside (str): Folder inside the archive to extract.
    destination_path (str): Destination path to extract files to.
    """
    if os.path.exists(destination_path + folder_inside):
        logger.log_error("Folder: " + destination_path + " exists")
    else:
        if file_path.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(file_path, "r") as zip_ref:
                    logger.log_error("Unzipping folder..." + folder_inside)
                    all_members = zip_ref.getmembers()
                    folder_members = [
                        member
                        for member in all_members
                        if member.name.startswith(folder_inside)
                    ]
                    zip_ref.extractall(path=destination_path, members=folder_members)
                    # Relocate the images to the destination_path
                    for member in folder_members:
                        if os.path.dirname(member.name):
                            src = os.path.join(destination_path, member.name)
                            dst = os.path.join(
                                destination_path, os.path.basename(member.name)
                            )
                            os.rename(src, dst)
                    # Delete empty folder
                    DeleteFolder = os.path.split(Path(folder_inside))
                    shutil.rmtree(os.path.join(destination_path, DeleteFolder[0]))
                    logger.log_error("Unzipped in..." + destination_path)

            except Exception as e:
                logger.log_error("Error extracting from ZIP file:", e)

        # Unzipp a folder from inside a TAR/TGZ file to a destination
        elif file_path.lower().endswith(".tar") or file_path.lower().endswith(".tgz"):
            try:
                with tarfile.open(file_path, "r:gz") as tar_ref:
                    logger.log_error("Unzipping folder..." + folder_inside)
                    all_members = tar_ref.getmembers()
                    folder_members = [
                        member
                        for member in all_members
                        if member.name.startswith(folder_inside)
                    ]
                    tar_ref.extractall(path=destination_path, members=folder_members)
                    # Relocate the images to the destination_path
                    for member in folder_members:
                        if os.path.dirname(member.name):
                            src = os.path.join(destination_path, member.name)
                            dst = os.path.join(
                                destination_path, os.path.basename(member.name)
                            )
                            os.rename(src, dst)
                    # Delete empty folder
                    DeleteFolder = os.path.split(Path(folder_inside))
                    shutil.rmtree(os.path.join(destination_path, DeleteFolder[0]))
                    logger.log_error("Unzipped in..." + destination_path)

            except Exception as e:
                logger.log_error("Error extracting from TAR/TGZ file:", e)


def read_yaml(file_path: str) -> Optional[dict]:
    """
    Read a YAML file.

    Parameters:
    file_path (str): Path to the YAML file.

    Returns:
    Optional[dict]: Parsed YAML data, or None if an error occurs.
    """
    try:
        with open(file_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.log_error(f"Error: The file {file_path} was not found.")
    except yaml.YAMLError as exc:
        logger.log_error(f"Error parsing YAML file: {exc}")
    except Exception as exc:
        logger.log_error(f"An unexpected error occurred: {exc}")
    return None


def accuracy(labels: torch.Tensor, outputs: torch.Tensor) -> float:
    """
    Calculate the accuracy.

    Parameters:
    labels (torch.Tensor): Ground truth labels.
    outputs (torch.Tensor): Model outputs.

    Returns:
    float: Accuracy value.
    """
    preds = outputs.argmax(dim=-1)
    acc = (preds == labels.view_as(preds)).float().mean().item()
    return acc


def save_model(model: torch.nn.Module, model_name: str):
    """
    Save the model's state dictionary.

    Parameters:
    model (torch.nn.Module): The model to save.
    model_name (str): The name of the model file.
    """
    save_folder = Path("models")
    save_folder.mkdir(parents=True, exist_ok=True)
    filepath = save_folder / model_name
    torch.save(model.state_dict(), filepath)
    logger.log_debug(f"Model saved at {filepath}")


def generate_full_image_rois(batch_size, target_size, mode):
    rois = []
    # Mode 0, full image
    # Mode 1, 2/3 of image
    # Mode 2, 3/4 of image
    # Mode 3, 1/2 of image
    # Mode 4, fixed 180x320 centered horizontal bottom of image
    for i in range(batch_size):
        # RoI format: [batch_index, x1, y1, x2, y2]
        if (mode == 0) or (mode > 4):
            rois.append([i, 0, 0, target_size[0], target_size[1]])
        if mode == 1:
            rois.append([i, 0, 0, target_size[0]*0.66, target_size[1]])
        if mode == 2:
            rois.append([i, 0, 0, target_size[0]*0.75, target_size[1]])
        if mode == 3:
            rois.append([i, 0, 0, target_size[0]*0.5, target_size[1]])
        if mode == 4:
            rois.append([i, (target_size[0]-180)/2, 0, (target_size[0]-180)/2+180, 320])
    return torch.tensor(rois, dtype=torch.float32)

def show_sample(model,image_path,mask_path,target_size,device):

    # Read Image & Mask Example
    image = Image.open(image_path).convert("RGB")  # Ensure image is in RGB mode
    mask = Image.open(mask_path).convert("L")  # Convert mask to grayscale

    # Prepare image
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(target_size, antialias=True)])
    images = transform(image)
    images = images.unsqueeze(0)

    # Get output from model
    output = model(images, generate_full_image_rois(1,target_size,0))
    # Apply weights
    weights = torch.tensor([0.2989, 0.5870, 0.1140], device=device).view(1, 3, 1, 1)
    output = (output * weights).sum(dim=1, keepdim=True)
    # Scale to target_size
    output = F.interpolate(output, size=target_size, mode="bilinear", align_corners=False)
    # Normalize to 0-1 range
    output = (output - output.min()) / (output.max() - output.min()) 
    # Apply threshold to eliminate noise
    output = (output > 0.001).type(torch.int)

    # Prepare mask
    transform_mask = transforms.Compose([transforms.ToTensor(), transforms.Resize(target_size, antialias=True)])
    masks = transform_mask(mask)
    masks = (masks == 1).type(torch.int)
    masks = masks.unsqueeze(0)

    # Convert the tensors to numpy arrays for plotting
    image_np = (images[0].numpy().transpose(1, 2, 0))  # Convert to HWC format for plotting
    mask_np = masks[0].numpy().transpose(1, 2, 0)  # Convert to HWC format for plotting
    output_np = output.detach().cpu().numpy()[0].transpose(1, 2, 0) 

    # Plot the image and mask
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    axes[0].imshow(image_np)
    axes[0].set_title("Image")
    axes[0].axis("off")
    axes[1].imshow(mask_np, cmap='gray')  # Display mask in grayscale
    axes[1].set_title("Mask")
    axes[1].axis("off")
    axes[2].imshow(output_np, cmap='gray')  # Display predicted mask in grayscale
    axes[2].set_title("Predicted Mask")
    axes[2].axis("off")
    plt.show()
    return

def calculate_rotation_difference(img1, img2, type=0):
 
    def find_centroid(img):
        M = cv2.moments(img)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
        return cx, cy
    
    # Applied on mask
    if type == 0:
        # Find centroids of the binary image
        cx1, cy1 = find_centroid(img1.astype(np.uint8))
        cx2, cy2 = find_centroid(img2.astype(np.uint8))

        # Calculate the angle between the centroids
        angle1 = np.arctan2(cy1, cx1)
        angle2 = np.arctan2(cy2, cx2)
    
        # Calculate the rotation difference
        rotation_difference = np.degrees(angle2 - angle1)
    
        # Normalize the angle to the range [-180, 180]
        rotation_difference = (rotation_difference + 180) % 360 - 180

    # Applied on image
    if type == 1:

        # Convert images to numpy array
        img1 = np.array(img1)
        img2 = np.array(img2)
        
        # Convert images to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Initialize ORB detector
        orb = cv2.ORB_create()

        # Find keypoints and descriptors
        keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

        # Initialize matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors
        matches = bf.match(descriptors1, descriptors2)

        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Use the best 10 matches to estimate rotation
        if len(matches) > 10:
            matches = matches[:10]

        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Estimate affine transformation using RANSAC
        M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)

        if M is None:
            return 0.0  # No transformation estimated

        # Calculate rotation angle
        rotation_angle_rad = -np.arctan2(M[0, 1], M[0, 0])
        rotation_difference = np.degrees(rotation_angle_rad)

    return rotation_difference