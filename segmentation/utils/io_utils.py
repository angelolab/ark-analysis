import os


def list_folders(input_dir):
    files = os.listdir(input_dir)
    folders = [file for file in files if os.path.isdir(os.path.join(input_dir, file))]

    return folders
