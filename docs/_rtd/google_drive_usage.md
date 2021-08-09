## Using Google Drive

### Motivation

Image data can be very large, and it can be difficult to keep all of it on your local hard drive.
Many people use Google Drive to store their data off of their local hard drive, but this can create
logistic issues when processing it.

To mitigate these logistic issues, `ark-analysis` has a built in API for working with data stored
in Google Drive, via `GoogleDrivePaths`.  This API allows you to manipulate files just as you would
on your local hard drive, with minimal syntax differences.


### Setting Up

For now, using the `GoogleDrivePath` API requires a universal passcode.  Email
`ackagel@stanford.edu` to request access for this features.

Once you have this passcode, you can make all of your notebooks `GoogleDrivePath`-compatible by
running this code cell at the top

```
from ark.utils import google_drive_utils

google_drive_utils.init_google_drive_api("passcode")
```

This will open a web page requesting you to sign in to your Google account.  Currently, this
feature isn't verified by Google, so it will warn you about giving access to an unverified
application.  To use the `ark-analysis` Google Drive interoperability feature, you will need to
ignore Google's warning, and proceed to give `ark-analysis` access to Google Drive.

Once you have given `ark-analysis` access, you should be redirected to a 'success' screen.  If
otherwise, close the browser tab/window, interupt the jupyter kernel, and re-run the
`init_google_drive_api("passcode")` cell.  This step can occasionally take a few tries to get
running properly.

Once you have been redirected to a 'success' screen on the browser, you are now free to use
`GoogleDrivePaths`.

### Defining Paths

Template notebooks in `ark-analysis` typically have a code cell where the majority of one's paths
are defined.  For example, here is the path definition cell of the `Segment_Image_Data` notebook:

```
# existing paths
base_dir = "../data/example_dataset"
input_dir = os.path.join(base_dir, "input_data")
tiff_dir = os.path.join(input_dir, "single_channel_inputs/")

# paths for new processed data
deepcell_input_dir = os.path.join(input_dir, "deepcell_input/")
deepcell_output_dir = os.path.join(base_dir, 'deepcell_output')
single_cell_dir = os.path.join(base_dir, "single_cell_output")
viz_dir = os.path.join(base_dir, "deepcell_visualization")
```

We can incrementally change these to be `GoogleDrivePaths`.  For example, suppose all of your
tiff data is on Google Drive, in the following folder `/mibi_data/runYYYYMMDD/denoised`.
Graphically:

```
...
other_folder
mibi_data
    | ...
    | runYYYYMMDD_other
    | runYYYYMMDD
        | ...
        | extracted
        | denoised
            | ...
            | panel.csv
            | TIFs
                | ...
                | HH3.tif
                | Au.tif
another_folder
...
```

So, to configure our filepaths as `GoogleDrivePath`s, in the style of our template notebook, 
we would write:

```
base_dir = google_drive_utils.GoogleDrivePath('/mibi_data')
input_dir = google_drive_utils.path_join(base_dir, 'runYYYYMMDD')
tiff_dir = google_drive_utils.path_join(input_dir, 'denoised')

deepcell_input_dir = google_drive_utils.path_join(input_dir, 'deepcell_input/')
deepcell_output_dir = google_drive_utils.path_join(base_dir, "deepcell_output")
single_cell_dir = google_drive_utils.path_join(base_dir, "single_cell_output")
viz_dir = google_drive_utils.path_join(base_dir, "deepcell_visualization")
```

Since some of these directories don't exist, they'll have to be created as follows:

```
for directory in [deepcell_input_dir, deepcell_output_dir, single_cell_dir, viz_dir]:
    if type(directory) is GoogleDrivePath:
        directory.mkdir()
    elif not os.path.exists(directory):
        os.makedirs(directory)
```

Key take aways: 🔑
 1. Our base directory must be called as a `GoogleDrivePath('...')`, while others can be infered
 2. We don't use `os.path.join`, and instead use `google_drive_utils.path_join`, to combine paths
 3. We need to create new directories if they do not already exist

### Running notebooks

Now that your paths are formatted, most `ark-analysis` data-processing functions should work
out of the box, as if your `GoogleDrivePath`s were actual local files.  If you come across an issue
using a `GoogleDrivePath` within an `ark-analysis` function, please submit an issue [here](https://github.com/angelolab/ark-analysis/issues).

### Writing Data Out

Let's consider the last code cell of the `Segment_Image_Data` notebook:

```
# save extracted data as csv for downstream analysis
cell_table_size_normalized.to_csv(os.path.join(single_cell_dir, 'cell_table_size_normalized.csv'),
                                 index=False)
cell_table_arcsinh_transformed.to_csv(os.path.join(single_cell_dir, 'cell_table_arcsinh_transformed.csv'),
                                     index=False)
```

we can generalize both of these lines as:

```
df.to_csv(some_path, index=False)
```

The variable `cell_table_size_normalized` is a `pandas.DataFrame` which is calling a method
`to_csv()`, in order to save the dataframe as a csv file.  Because `pandas.DataFrame.to_csv()` is
not an `ark-analysis` function, it does not know what a `GoogleDrivePath` is, and will be upset if
it is directly passed one.

To deal with cases like these, we utilize the `google_drive_utils.drive_write_out` wrapper
function. We pass the `GoogleDrivePath`, as well as a lambda function, to accomplish the write out.

For the generalized example above, we can make it `GoogleDrivePath` compatable via:
```
google_drive_utils.drive_write_out( some_path, lambda x: df.to_csv(x, index=False) )
```

## Troubleshooting

### Slow Upload/Download Speeds?

In the case of slow upload or download speed, we may not want all of our paths to be
`GoogleDrivePath`s. For example, in the context of the `Segment_Image_Data` notebook,
`deepcell_input_dir` only stores the Mesmer-compatable formatted data, while `deepcell_output_dir`
stores the results of the Mesmer segmentation.  

We probably don't need to upload/download this re-formatted data, and wouldn't mind if it's stored
locally since it's very small relative to the size of the dataset.  On the other hand, we
may want to immediately upload our segmentation results to Google Drive.  In that case,
`deepcell_input_dir` should be a local folder, while `deepcell_output_dir` should be a
`GoogleDrivePath`:

```
# paths for new processed data

## local
base_local_dir = "../data/tmp"
deepcell_input_dir = os.path.join(base_local_dir, "deepcell_input/")

## google drive
deepcell_output_dir = google_drive_utils.path_join(base_dir, "deepcell_output")
single_cell_dir = google_drive_utils.path_join(base_dir, "single_cell_output")
viz_dir = google_drive_utils.path_join(base_dir, "deepcell_visualization")
```

Key take aways: 🔑

 1. We don't necesarily want all paths to be `GoogleDrivePath`s
 2. We should always create new folders regardless of if they're local or Google Drive folders

    **NOTE:** This process will require you to keep track of local paths vs `GoogleDrivePaths`.  Since that
can be a bit of a headache, please only use this solution in the case of slow upload/download
speeds