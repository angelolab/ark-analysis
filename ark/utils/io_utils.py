import os
import pathlib
import warnings


def validate_paths(paths):
    """Verifys that paths exist and don't leave Docker's scope

    Args:
        paths (str or list):
            paths to verify.

    Raises:
        ValueError:
            Raised if any directory is out of scope or non-existent
    """

    # if given a single path, convert to list
    if not isinstance(paths, list):
        paths = [paths]

    for path in paths:
        if not os.path.exists(path):
            if str(path).startswith('../data'):
                for parent in reversed(pathlib.Path(path).parents):
                    if not os.path.exists(parent):
                        raise ValueError(
                            f'A bad path, {path}, was provided.\n'
                            f'The folder, {parent.name}, could not be found...')
                raise ValueError(
                    f'The file/path, {pathlib.Path(path).name}, could not be found...')
            else:
                raise ValueError(
                    f'The path, {path}, is not prefixed with \'../data\'.\n'
                    f'Be sure to add all images/files/data to the \'data\' folder, '
                    f'and to reference as \'../data/path_to_data/myfile.tif\'')


def list_files(dir_name, substrs=None):
    """ List all files in a directory containing at least one given substring

    Args:
        dir_name (str):
            Parent directory for files of interest
        substrs (str or list):
            Substring matching criteria, defaults to None (all files)

    Returns:
        list:
            List of files containing at least one of the substrings

    """

    files = os.listdir(dir_name)
    files = [file for file in files if not os.path.isdir(os.path.join(dir_name, file))]

    # default to return all files
    if substrs is None:
        return files

    # handle case where substrs is a single string (not wrapped in list)
    if type(substrs) is not list:
        substrs = [substrs]

    matches = [file
               for file in files
               if any([
                   substr in file
                   for substr in substrs
               ])]

    return matches


def remove_file_extensions(files):
    """Removes file extensions and drops preceding path from a list of files

    Args:
        files (list):
            List of files to remove file extensions from.
            Any element that doesn't have an extension is left unchanged

    Returns:
        list:
            List of files without file extensions
    """

    # make sure we don't try to split on an undefined list of files
    if files is None:
        return

    # only get the file name and not the directory path leading up to it
    names = [os.path.split(name)[1] for name in files]

    # remove anything past and including the first '.' in each entry
    names = [name.split('.')[0] for name in names]

    return names


def extract_delimited_names(names, delimiter='_', delimiter_optional=True, remove_exts=True):
    """For a given list of names, extract the delimited prefix

    Examples (if delimiter='_'):
        - 'fov1' becomes 'fov1'
        - 'fov2_part1' becomes 'fov2'
        - 'fov3_part1_part2' becomes 'fov3'

    Args:
        files (list):
            List of files to extract names from (if paths, just uses the last file/folder).
            Make sure to call remove_file_extensions first if you need to drop file extensions.
        delimiter (str):
            Character separator used to determine filename prefix. Defaults to '_'.
        delimiter_optional (bool):
            If False, function will return None if any of the files don't contain the delimiter.
            Defaults to True.
        remove_exts (bool):
            Whether to remove file extensions on the files list as well

    Raises:
        UserWarning:
            Raised if delimiter_optional=False and no delimiter is present in any of the files

    Returns:
        list:
            List of extracted names. Indicies should match that of files

    """

    # make sure we don't try to split on a non-existent list
    if names is None:
        return

    # check for bad files/folders
    if not delimiter_optional:
        no_delim = [
            delimiter not in name
            for name in names
        ]
        if any(no_delim):
            warnings.warn(f"The following files do not have the mandatory delimiter, "
                          f"'{delimiter}'...\n"
                          f"{[name for indx,name in enumerate(names) if no_delim[indx]]}")
            return None

    # now split on the delimiter as well
    names = [name.split(delimiter)[0] for name in names]

    return names


def list_folders(dir_name, substrs=None):
    """ List all folders in a directory containing at least one given substring

    Args:
        dir_name (str):
            Parent directory for folders of interest
        substrs (str or list):
            Substring matching criteria, defaults to None (all folders)

    Returns:
        list:
            List of folders containing at least one of the substrings
    """

    files = os.listdir(dir_name)
    folders = [file for file in files if os.path.isdir(os.path.join(dir_name, file))]

    # default to return all files
    if substrs is None:
        return folders

    # handle case where substrs is a single string (not wrapped in list)
    if type(substrs) is not list:
        substrs = [substrs]

    matches = [folder
               for folder in folders
               if any([
                   substr in folder
                   for substr in substrs
               ])]

    return matches
