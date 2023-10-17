from ark.segmentation import marker_quantification


testfiles = io_utils.list_files(segmentation_dir, substrs=fov_name)

#
# for each label given in the argument. read in that mask for the fov, and proced with label and table appending

testfiles = io_utils.list_files(segmentation_dir, substrs=fov_name)

# Function to strip prefixes from list A, strip '.tiff' suffix from list B,
# and remove underscore prefixes, returning unique values
def process_lists(listA, listB):
    stripped_listA = [itemA for itemA in listA]
    stripped_listB = [itemB.replace('.tiff', '') for itemB in listB]

    result = []
    for itemB in stripped_listB:
        for prefix in stripped_listA:
            if itemB.startswith(prefix):
                result.append(itemB[len(prefix):])
                break  # Break the inner loop once a matching prefix is found

    # Remove underscore prefixes and return unique values
    cleaned_result = [item.lstrip('_') for item in result]
    unique_result = list(set(cleaned_result))
    return unique_result

testfiles = io_utils.list_files(segmentation_dir, substrs=fov_name)
result_list = process_lists(listA=fovs, listB=testfiles)
print(result_list)