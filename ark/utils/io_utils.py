from argparse import Namespace
import os
import pathlib
import warnings


def validate_paths(paths, data_prefix=True):
    """Verifys that paths exist and don't leave Docker's scope

    Args:
        paths (str or list):
            paths to verify.
        data_prefix (bool):
            if True, checks that directory starts with /data, necessary when inside the docker

    Raises:
        ValueError:
            Raised if any directory is out of scope or non-existent
    """

    # if given a single path, convert to list
    if not isinstance(paths, list):
        paths = [paths]

    for path in paths:
        if not os.path.exists(path):
            if str(path).startswith('../data') or not data_prefix:
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


def list_files(dir_name, substrs=None, exact_match=False, ignore_hidden=True):
    """ List all files in a directory containing at least one given substring

    Args:
        dir_name (str):
            Parent directory for files of interest
        substrs (str or list):
            Substring matching criteria, defaults to None (all files)
        exact_match (bool):
            If True, will match exact file names (so 'C' will match only 'C.tif')
            If False, will match substr pattern in file (so 'C' will match 'C.tif' and 'CD30.tif')
        ignore_hidden (bool):
            If True, will ignore hidden files. If False, will allow hidden files to be
            matched against the search substring.

    Returns:
        list:
            List of files containing at least one of the substrings
    """
    files = os.listdir(dir_name)
    files = [file for file in files if not os.path.isdir(os.path.join(dir_name, file))]

    # Filter out hidden files
    if ignore_hidden:
        files = [file for file in files if not file.startswith('.')]

    # default to return all files
    if substrs is None:
        return files

    # handle case where substrs is a single string (not wrapped in list)
    if type(substrs) is not list:
        substrs = [substrs]

    if exact_match:
        matches = [file
                   for file in files
                   if any([
                       substr == os.path.splitext(file)[0]
                       for substr in substrs
                   ])]
    else:
        matches = [file
                   for file in files
                   if any([
                       substr in file
                       for substr in substrs
                   ])]

    return matches


def remove_file_extensions(files):
    """Removes file extensions from a list of files

    Args:
        files (list):
            List of files to remove file extensions from.
            Any element that doesn't have an extension is left unchanged

    Raises:
        UserWarning:
            Some of the processed file names still contain a period

    Returns:
        list:
            List of files without file extensions
    """

    # make sure we don't try to split on a non-existent list
    if files is None:
        return

    # remove the file extension
    names = [os.path.splitext(name) for name in files]
    names_corrected = []
    extension_types = ["tiff", "tif", "png", "jpg", "jpeg", "tar", "gz", "csv", "feather"]
    for name in names:
        # We want everything after the "." for the extension
        ext = name[-1][1:]
        if (ext in extension_types) or (len(ext) == 0):
            # If it is one of the extension types, only keep the filename.
            # Or there is no extension and the names are similar to ["fov1", "fov2", "fov3", ...]
            names_corrected.append(name[:-1][0])
        else:
            # If `ext` not one of the specified file types, keep the value after the "."
            names_corrected.append(name[:-1][0] + "." + name[-1][1])

    # identify names with '.' in them: these may not be processed correctly.
    bad_names = [name for name in names_corrected if '.' in name]
    if len(bad_names) > 0:
        warnings.warn(f"These files still have \".\" in them after file extension removal: "
                      f"{','.join(bad_names)}, "
                      f"please double check that these are the correct names")

    return names_corrected


def extract_delimited_names(names, delimiter='_', delimiter_optional=True):
    """For a given list of names, extract the delimited prefix

    Examples (if delimiter='_'):

    - 'fov1' becomes 'fov1'
    - 'fov2_part1' becomes 'fov2'
    - 'fov3_part1_part2' becomes 'fov3'

    Args:
        names (list):
            List of names to split by delimiter.
            Make sure to call remove_file_extensions first if you need to drop file extensions.
        delimiter (str):
            Character separator used to determine filename prefix. Defaults to '_'.
        delimiter_optional (bool):
            If False, function will return None if any of the files don't contain the delimiter.
            Defaults to True. Ignored if delimiter is None.

    Raises:
        UserWarning:
            Raised if delimiter_optional=False and no delimiter is present in any of the files

    Returns:
        list:
            List of extracted names. Indicies should match that of files
    """

    if names is None:
        return

    # check for bad files/folders
    if delimiter is not None and not delimiter_optional:
        no_delim = [
            delimiter not in name
            for name in names
        ]
        if any(no_delim):
            print(f"The following files do not have the mandatory delimiter, "
                  f"'{delimiter}': "
                  f"{','.join([name for indx,name in enumerate(names) if no_delim[indx]])}")
            warnings.warn("files without mandatory delimiter")

            return None

    # now split on the delimiter as well
    names = [name.split(delimiter)[0] for name in names]

    return names


def list_folders(dir_name, substrs=None, exact_match=False, ignore_hidden=True):
    """ List all folders in a directory containing at least one given substring

    Args:
        dir_name (str):
            Parent directory for folders of interest
        substrs (str or list):
            Substring matching criteria, defaults to None (all folders)
        exact_match (bool):
            If True, will match exact folder names (so 'C' will match only 'C/').
            If False, will match substr pattern in folder (so 'C' will match 'C/' & 'C_DIREC/').
        ignore_hidden (bool):
            If True, will ignore hidden directories. If False, will allow hidden directories to
            be matched against the search substring.

    Returns:
        list:
            List of folders containing at least one of the substrings
    """
    files = os.listdir(dir_name)
    folders = [file for file in files if os.path.isdir(os.path.join(dir_name, file))]

    # Filter out hidden directories
    if ignore_hidden:
        folders = [folder for folder in folders if not folder.startswith('.')]

    # default to return all files
    if substrs is None:
        return folders

    # handle case where substrs is a single string (not wrapped in list)
    if type(substrs) is not list:
        substrs = [substrs]

    # Exact match case
    if exact_match:
        matches = [folder
                   for folder in folders
                   if any([
                       substr == os.path.splitext(folder)[0]
                       for substr in substrs
                   ])]
    else:
        matches = [folder
                   for folder in folders
                   if any([
                       substr in folder
                       for substr in substrs
                   ])]

    return matches
