import os
import zipfile
import tarfile
import shutil
from pathlib import Path

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