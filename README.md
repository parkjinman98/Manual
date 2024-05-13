# User Manual Writer's Guide

## Prerequisites
### Environment Setup
- OS: Windows 10 or 11 with the latest updates
- Microsoft Word 
- Git
- TortoiseGit(Recommended)

Set Microsoft Word's equation editor to LaTeX. To do this, go to `Insert` -> `Equation` -> `Insert New Equation`. When you click the created equation, you can set the `Unicode` option to `LaTeX` in the `Equation Tools` tab.
### Installation

#### Method 1: [Direct Download](https://testdrive-profiling-master.github.io/download/TestDrive_Profiling_Master.exe)

#### Method 2: Installable Package by Git 
1. Install from repositories
    ```bash
    cd YourInstallationFolder
    mkdir Project
    cd Project
    git clone https://github.com/testdrive-profiling-master/release TestDrive
    git clone https://github.com/testdrive-profiling-master/profiles Profile
    ```

2. Program environment registration

    (To ensure the Visual Studio redistributable package is installed, execute `VC_redist.x64.exe` in the `Project/TestDrive/preinstallation` folder.)
    ```bash
    cd TestDrive
    TestDrive.exe
    ```

#### Method 3: Mobilint® Software Development Kit (SDK) **(Recommended)**
Available at https://dl.mobilint.com/view.php. Download after signing up for an account.

* **Issue**: During installation, the installer requests the user to log in at https://repo.mobilint.com/user/login. If you need access, please contact <jisung@mobilint.co>.

### Post-Installation
0. Reboot the system before installing the external tools. Therefore, the environment variables are updated.
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
    In addition, execute the following commands.

    ```bash
    cd YourInstallationFolder
    cd ./Project/TestDrive/bin/
    ./ecilpse/eclipse.exe
    ./notepad/notepad++.exe
    ```
### Update (Only Available for Method 3 Installation)
Go to `YourInstallationFolder/Project/Profiles/MobilintCI` and execute `start.bat`. The update process will be automatically executed.


## Writing a User Manual
### Before You Write
#### Template

Prepare a template file in the `YourProjectFolder/Profiles/Common/bin/codegen` folder to set a standardized style and format. The template file should be named `docgen_template_{custom_name}.docx`.

It is recommended to start with the existing template file, and the following process is an example of editing the template file using Microsoft Word.
1. Open the template file in Microsoft Word.
2. Press `Alt+F9` to display the field codes.
3. Find the properties that you want to modify the style and format, and modify them.

For example, if you want to modify the title style, find the field code including `DOCPROPERTY Comprehensive_IP_Title`. Then, apply the custom style to the object corresponding to the field code.

#### Lua Script
1. Properties

    The following properties are required to be included in the Lua script.
    ```lua
    -- Document Properties for title, author, etc.
    property["Document_Name"]           -- Document Name
    property["IP_Version"]              -- Version
    property["Comprehensive_IP_Title"]  -- Comprehensive IP Title. This will be used in the document title page.
    property["IP_Name_First_Page"]      -- IP Name for the first page
    property["Business_Unit"]           -- Business Unit
    property["IP_Name_Header"]          -- IP Name for the header
    property["Ownership"]               -- Owner
    property["Document_Name_Header"]    -- Document Name for the header
    property["Security_Level"]          -- Security Level of the document. This will be displayed in the header of title pages and the footer of contents.
    property["Water_Mark"]              -- Watermark(empty when no need)
    ```
    For example,
    ```lua
    property["Document_Name"]           = "Model Compiler Suite for Aries™"
    property["IP_Version"]              = "v0.8.1"
    property["Comprehensive_IP_Title"]  = "Model Compiler Suite for Aries"
    property["IP_Name_First_Page"]      = "Developers Guide"
    property["Business_Unit"]           = "Software"
    property["IP_Name_Header"]          = "Developers Guide"
    property["Ownership"]               = "Mobilint"
    property["Document_Name_Header"]    = "Developers Guide"
    property["Security_Level"]          = "Mobilint Confidential"
    property["Water_Mark"]              = "Provided to NOTA, 05JUL23"
    ```
2. Revision History

    Revision history can be written in the following format.
    ```lua
    AddRevision("version", YYYY, MM, DD, "description")
    ```
    Moreover, this can be added multiple times. For example,
    ```lua
    AddRevision("0.2",  2021, 12,  01,  "Revised for v0.2")
    AddRevision("0.3",  2022, 02,  05,  "Revised for v0.3")
    AddRevision("0.4",  2022, 02,  23,  "Revised for v0.4")
    ```
3. Definitions

    If you want to prescribe the essential terms, you can use the following format.
    ```lua
    AddDefinition("term", "description")
    ```
    The description can be written in the same way as the paragraph. For example,
    ```lua
    AddTerm("TestDrive", "TestDrive Profiling Master (@<link:https://testdrive-profilingmaster.github.io/>)") 
    AddTerm("Lua", "Lua script language (@<link:https://ko.wikipedia.org/wiki/루아_(프로그래밍_언어);Wiki>, @<link:http://www.lua.org/;Homepage>)") 
    ```
4. Paragraphs
    
    The main content of the manual, which is already written in markdown format, can be included in the document using the following format.
    ```lua
    AddParagraph("[[content.md]]")
    ```
    The way to write the content is explained in the [reference](https://testdrive-profiling-master.github.io/wiki/?top.md). It is generally the same as the markdown format.
    If you want to include multiple content files, you can use the command multiple times. For example,
    ```lua
    AddParagraph("[[content1.md]]")
    AddParagraph("[[content2.md]]")
    ```

### Writing Your Manual
**Note: Writing the manual contents in markdown format is recommended. Technical details are available on [reference](https://testdrive-profiling-master.github.io/wiki/?top.md).**

### After You Write
After writing the Lua script and manual contents in markdown format, you can generate the user manual by executing the DocGen tool.
To execute DocGen to generate the user manual, the following command is used.
```bash
docgen [--help] [-t template] input_file [output_file]
```
#### Command Line Options
- `--help`: Displays the general information message.
    ```bash
    docgen --help
    ```
- `-t template`:
To use custom style and format, place the `docgen_template_{custom_name}.docx` template file in the `YourProjectFolder/Profiles/Common/bin/codegen` folder.
- `input_file`: The input file is the content of the user manual. File scripted in Lua format is now supported. It is allowed not to include the file extension, but it is recommended to include '.lua'.
- `output_file`: The output file is the generated user manual. It will be the output file, so it is recommended not to include the extension. If this option is not specified, the output file will be automatically generated with the properties of the input Lua file.

For example, if the input file is `test.lua`, the prepared template file is `docgen_template_common.docx`, and the output file name is `manual.docx` or `manual.pdf`, the command is as follows.
```bash
docgen -t common test.lua manual
```
## References
- [TestDrive Profiling Master Wiki](https://testdrive-profiling-master.github.io/wiki/?top.md)