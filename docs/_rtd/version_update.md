## Updating the Version

When a new version of ARK is ready for release, please follow these steps: 

1. Make a separate PR to just update the version number.
2. In `setup.py`, update the `VERSION` constant to the desired `x.y.z` value.
3. If the release includes the addition of new libraries that do not already exist in the `install_requires` argument of the `setup` function call, please add them.
4. Merge the PR into `master` with a tag set with the vane `v{x.y.z}`, where `{x.y.z}` is the version number we wish to change to. This tag is important because our source code is only released to PyPI with a tag.
5. Create a new release on the `ark-analysis` repo. Make sure to add a description which clearly explains the changes you made.
