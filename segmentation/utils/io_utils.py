import os
import warnings


def list_files(dir_name, substrs=None):
    """ List all files in a directory containing at least one given substring

    Args:
        dir_name (str):         Parent directory for files of interest
        substrs (str or list):  Substring matching criteria, defaults to None (all files)

    Returns:
        matches (list):         List of files containing at least one of the substrings

    """

    files = os.listdir(dir_name)
    files = [file for file in files if not os.path.isdir(os.path.join(dir_name, file))]

    # default to return all files
    if substrs is None:
        return files

    # handle case where substrs is a single string (not wrapped in list)
    if type(substrs) is not list:
        substrs = [substrs]

    matches = []
    for substr in substrs:
        matches.extend([
            file
            for file in files
            if substr in file
        ])

    return list(set(matches))


def extract_delimited_names(files, delimiter='_', delimiter_optional=True):
    """ Create a matched-index list of fov/point/etc names from a list of files/folders

    Extracts a delimited prefix for every file in a given list of files

    e.g.
        'Point2_restofthefilename.tiff' becomes 'Point2'

    Args:
        files (list):               List of files to extract names from (if paths,
                                    just uses the last file/folder)
        delimiter (str):            Character separator used to determine filename prefix.
                                    Defaults to '_'.
        delimiter_optional (bool):  If False, function will return None if any of the files
                                    don't contain the delimiter.  Defaults to True.

    Returns:
        names (list):   List of extracted names. Indicies should match that of files

    Raises:
        UserWarning:    Raised if delimiter_optional=False and no delimiter is present in any of
                        the files

    """
    if files is None:
        return

    names = [
        os.path.split(name)[1]
        for name in files
    ]

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

    # do filtering
    names = [
        name.split('.')[0].split(delimiter)[0]
        for name in names
    ]

    return names


def list_folders(dir_name, substrs=None):
    """ List all folders in a directory containing at least one given substring

    Args:
        dir_name (str):         Parent directory for folders of interest
        substrs (str or list):  Substring matching criteria, defaults to None (all folders)

    Returns:
        matches (list):         List of folders containing at least one of the substrings

    """

    files = os.listdir(dir_name)
    folders = [file for file in files if os.path.isdir(os.path.join(dir_name, file))]

    # default to return all files
    if substrs is None:
        return folders

    # handle case where substrs is a single string (not wrapped in list)
    if type(substrs) is not list:
        substrs = [substrs]

    matches = []
    for substr in substrs:
        matches.extend([
            folder
            for folder in folders
            if substr in folder
        ])

    return list(set(matches))
