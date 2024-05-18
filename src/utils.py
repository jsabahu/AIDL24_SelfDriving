import os
import zipfile
import tarfile
import shutil
from pathlib import Path
import yaml

def list_files_in_folder(folder_path):
    try:
        # Get a list of all files and directories in the folder
        files_in_folder = os.listdir(folder_path)
        # Show only jpg files
        files = [file for file in files_in_folder if file.lower().endswith('.jpg')]
        return files
    except Exception as e:
        print("Error listing files in folder:", e)
        return None

def UnZipFolder(file_path, folder_inside, destination_path):
    # Unzipp a folder from inside a ZIP file to a destination
    if os.path.exists(destination_path+folder_inside):
        print("Folder: "+destination_path+" exists")
    else:
        if file_path.lower().endswith('.zip'):
            try:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    print("Unzipping folder..." + folder_inside)
                    all_members = zip_ref.getmembers()
                    folder_members = [member for member in all_members 
                                      if member.name.startswith(folder_inside)]
                    zip_ref.extractall(path=destination_path, members=folder_members)
                    #Relocate the images to the destination_path
                    for member in folder_members:
                        if os.path.dirname(member.name):
                            os.rename(os.path.join(destination_path,member.name),
                                      os.path.join(destination_path,os.path.basename(member.name)))
                    #Delete empty folder
                    DeleteFolder = os.path.split(Path(folder_inside))
                    shutil.rmtree(os.path.join("data/images",DeleteFolder[0]))
                    print("Unzipped in..." + destination_path)

            except Exception as e:
                print("Error extracting from ZIP file:", e)

        # Unzipp a folder from inside a TAR/TGZ file to a destination
        elif file_path.lower().endswith('.tar') or file_path.lower().endswith('.tgz'):
            try:
                with tarfile.open(file_path, 'r:gz') as tar_ref:
                    print("Unzipping folder..." + folder_inside)
                    all_members = tar_ref.getmembers()
                    folder_members = [member for member in all_members 
                                      if member.name.startswith(folder_inside)]
                    tar_ref.extractall(path=destination_path, members=folder_members)
                    #Relocate the images to the destination_path
                    for member in folder_members:
                        if os.path.dirname(member.name):
                            os.rename(os.path.join(destination_path,member.name),
                                      os.path.join(destination_path,os.path.basename(member.name)))
                    #Delete empty folder
                    DeleteFolder = os.path.split(Path(folder_inside))
                    shutil.rmtree(os.path.join("data/images",DeleteFolder[0]))
                    print("Unzipped in..." + destination_path)

            except Exception as e:
                print("Error extracting from TAR/TGZ file:", e)

def read_yaml(file_path):
    """
    Reads a YAML file and returns its contents as a dictionary.
    
    :param file_path: Path to the YAML file.
    :return: Contents of the YAML file as a dictionary.
    """
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
            return data
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file: {exc}")
    except Exception as exc:
        print(f"An unexpected error occurred: {exc}")