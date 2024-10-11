## Windows configuration

Our repo works runs best on Linux-based systems (which includes Mac OS). If you need to run on Windows, you'll have some extra considerations to take into account. Note that this documentation applies for Windows 10 and Windows 11 machines.

### WSL 2

**We find it best to download and configure WSL 2 before anything else**. We recommend using the [Command Prompt](https://en.wikipedia.org/wiki/Cmd.exe) to run these commands (and later ones). You can find it by typing `Command Prompt` in the Windows search bar. 

The simplest way to install WSL 2 is to run `wsl --install`. If this works, great. Otherwise, you'll have to run these steps: 

* Open `Windows Features`, you can do this by typing this in the Windows search bar
* Make sure the checkbox is selected for both `Virtual Machine Platform` and `Windows Subsystem for Linux`
* Restart your machine

Now run the following steps: 

* Download the Linux kernel update package. You can find it here: [Linux kernel update package](https://docs.microsoft.com/en-us/windows/wsl/install-manual#step-4---download-the-linux-kernel-update-package).
* Set the version to WSL 2: `wsl --set-default-version 2`
* Type just `wsl` on the command line. You'll likely get an error indicating no installed distributions, meaning you'll need to associate one with your WSL backend. We recommend installing Ubuntu 20.04 LTS in the Microsoft Store.
* Once Ubuntu is installed, open it (it will open the Ubuntu terminal) and set a username and password to finalize installation.
* Run `sudo apt-get update` in the Ubuntu terminal to view all the necessary updates (alternatively, you can run `wsl sudo apt-get update` in the Windows Command Prompt).
* Run `sudo apt-get upgrade` to download and install the necessary updates (alternatively, you can run `wsl sudo apt-get upgrade`).

Note that if using a VM or the Hyper-V backend, nested virtualization needs to be turned on. This process varies depending on which virtualization platform you use: you'll have to consult their documentation for the how-tos. 

### Download git

Go to [git Windows install](https://git-scm.com/download/win) to download the Windows installer for `git`. 

After finishing the installer guidelines, go to your Command Prompt and type `git`. If you get a list of commands you can use with `git`, the installation was successful. 

### Setting up the repo

In your Command Prompt, follow similar steps to clone the `ark-analysis` repo and build the package.:

* Run `git clone https://github.com/angelolab/ark-analysis.git` to clone the `ark-analysis` repo
* Run `cd ark-analysis` to enter the cloned repo

To run the script, you have to use `bash start_jupyter.sh`. If you run into issues with invalid carriage returns (`\r`), please run the following before trying again:

* Run `wsl sudo apt-get install dos2unix`
* Run the following commands:
  > `wsl dos2unix start_jupyter.sh`

  > `wsl dos2unix update_notebooks.sh`


### Mounting Issues with External Drives

If Jupyter cannot locate the path to a certain external drive, that likely means WSL failed to mount it correctly. You'll need to restart WSL to force remount all external drives.

To do so, run the following steps:

* Run `wsl --shutdown`
* After a few seconds, run `wsl -l -v`.


### If you run into more Windows-specific issues

Please open an [issue](https://github.com/angelolab/ark-analysis/issues) on our GitHub page. Note that our codebase has not been extensively tested on Windows.
