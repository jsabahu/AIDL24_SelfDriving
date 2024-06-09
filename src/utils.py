import torch
import os
import zipfile
import tarfile
import shutil
from pathlib import Path
import yaml
from typing import List, Optional, Any
from logger import Logger

logger = Logger()


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


def generate_full_image_rois(batch_size, image_width, image_height ):
    rois = []
    for i in range(batch_size):
        # RoI format: [batch_index, x1, y1, x2, y2]
        rois.append([i, 0, 0, image_width, image_height])
    return torch.tensor(rois, dtype=torch.float32)
