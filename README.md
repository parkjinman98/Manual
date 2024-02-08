# User Manual Writer's Guide

## Prerequisites
### Environment Setup
- OS: Windows 10 or 11 with the latest updates
- Git
- TortoiseGit(Recommended)
### Installation

#### Method 1: [Direct Download](https://testdrive-profiling-master.github.io/download/TestDrive_Profiling_Master.exe)

#### Method 2: Installable Package by Git **(Recommended)**
1. Install from repositories
```bash
cd YourInstallationFolder
mkdir Project
cd Project
git clone https://github.com/testdrive-profiling-master/release TestDrive
git clone https://github.com/testdrive-profiling-master/profiles Profile
```

2. Program environment registration 
(To ensure that the visual studio redistributable package is installed, execute `VC_redist.x64.exe` in the `TestDrive/preinstallation` folder.)
```bash
cd TestDrive
TestDrive.exe
```

#### Method 3: Mobilint® Software Development Kit (SDK)
Available at https://dl.mobilint.com/view.php . Download after signing up for an account.

* **Issue**: On the installation process, the installer requests the user to log in https://repo.mobilint.com/user/login/openid. 

### Post-Installation
0. Just in case, reboot the system before installing the external tools, therefor the environment variables are updated.
1. Install external tools
```bash
cd YourInstallationFolder
cd ./Project/Profiles/Meitner
project.profile
```

2. Confirmation of the installation

Open the command prompt and check whether the following command is available.
```bash
ls
gcc
make
```
In addition, execute the following commands
```bash
cd YourInstallationFolder
cd ./Project/TestDrive/bin/
./ecilpse/eclipse.exe
./notepad/notepad++.exe
```
## Writing a User Manual
### General Information
To execute DocGen to generate the user manual, the following command is used.
```bash
docgen [--help] [-t template] input_file [output_file]
```
### Command Line Options
- `--help`: Displays the general information message.
```bash
docgen --help
```
- `-t template`:
To use custom style and format, place the template file of the name `docgen_template_{custom_name}.docx` in the `YourProjectFolder/Profiles/Common/bin/codegen` folder.
- `input_file`: The input file is the content of the user manual. File scripted in Lua format is now supported. It is allowed not to include the file extension, but it is recommended to include '.lua'.
- `output_file`: The output file is the generated user manual. It will be the output file, so it is recommended not to include the file extension. If this option is not specified, the output file will be automatically generated with the properties of the input Lua file.

For example, if the input file is `test.lua`, the prepared template file is `docgen_template_common.docx`, and the output file name is `manual.docx` or `manual.pdf`, the command is as follows.
```bash
docgen -t common test.lua manual
```

### Before You Write
### Template
To set standardized style and format, 

## References
- [TestDrive Profiling Master Wiki](https://testdrive-profiling-master.github.io/wiki/?top.md)