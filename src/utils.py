import torch
import os
import zipfile
import tarfile
import shutil
from pathlib import Path
import yaml


# Function use in train
def binary_accuracy_with_logits(labels, outputs):
    preds = torch.sigmoid(outputs).round()
    acc = (preds == labels.view_as(preds)).float().detach().numpy().mean()
    return acc


def list_files_in_folder(folder_path):
    try:
        if not os.path.exists(folder_path):
            print("Folder does not exist.")
            return []
        # Get a list of all files and directories in the folder
        files_in_folder = os.listdir(folder_path)
        # Show only jpg files
        files = [file for file in files_in_folder if file.lower().endswith(".jpg")]
        return files
    except Exception as e:
        print("Error listing files in folder:", e)
        return None


def UnZipFolder(file_path, folder_inside, destination_path):
    # Unzipp a folder from inside a ZIP file to a destination
    if os.path.exists(destination_path + folder_inside):
        print("Folder: " + destination_path + " exists")
    else:
        if file_path.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(file_path, "r") as zip_ref:
                    print("Unzipping folder..." + folder_inside)
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
                    print("Unzipped in..." + destination_path)

            except Exception as e:
                print("Error extracting from ZIP file:", e)

        # Unzipp a folder from inside a TAR/TGZ file to a destination
        elif file_path.lower().endswith(".tar") or file_path.lower().endswith(".tgz"):
            try:
                with tarfile.open(file_path, "r:gz") as tar_ref:
                    print("Unzipping folder..." + folder_inside)
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
                    print("Unzipped in..." + destination_path)

            except Exception as e:
                print("Error extracting from TAR/TGZ file:", e)


def read_yaml(file_path):
    try:
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)
            return data
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file: {exc}")
    except Exception as exc:
        print(f"An unexpected error occurred: {exc}")


def accuracy(labels, outputs):
    preds = outputs.argmax(-1)
    acc = (preds == labels.view_as(preds)).float().detach().numpy().mean()
    return acc


def save_model(model, model_name):
    save_folder = "models"
    filepath = os.path.join(save_folder, model_name)
    torch.save(model.state_dict(), filepath)
