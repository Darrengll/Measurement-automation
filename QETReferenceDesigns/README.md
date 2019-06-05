# QET Reference Designs

QET Reference Designs contains a collection of example programs that demonstrate how to run typical quantum experiments using the hardware and software packages in the Quantum Engineering Toolkit. 

### Source
The git repository for QET Reference Designs can be found at:
[https://bitbucket.it.keysight.com/scm/qes/qet-reference-designs.git](https://bitbucket.it.keysight.com/scm/qes/qet-reference-designs.git)

To clone this repository from the command:

    $ cd repos
    $ git clone --recursive https://bitbucket.it.keysight.com/scm/qes/qet-reference-designs.git
    $ cd qet-reference-designs
    $ git checkout develop

#### Project organization

QET Reference Designs adheres to the established development processes outlined in the Keysight Source Code Management standard.  This document can be found on the [KOSI wiki][sourceCodeStandard] 

For the specific repository layout recommendations see [this section][recommendedGitDirectories].

| Top level    | Description                                              |
| -----------  | -------------------------------------------------------- |
| doc          | Code documentation                                       |
| include      | Artifacts required by the component (libraries)          |
| src          | Software source code                                     |
| test         | Test related code and scripts (unit test)                |

### Dependencies

Keysight SD1 Driver
Keysight M3601A HVI Programming Environment

### Notes

This readme file is written with markdown syntax, a lightweight markup language. 
An easy-to-use online tool for editing markdown files can be found at [this site][markdownEditor]

[markdownEditor]: http://dillinger.io/
[sourceCodeStandard]: https://wiki2.collaboration.is.keysight.com/display/KOSi/Keysight+Source+Code+Management+Standard
[recommendedGitDirectories]: https://wiki2.collaboration.is.keysight.com/display/KOSi/Keysight+Source+Code+Management+Standard#KeysightSourceCodeManagementStandard-_Toc453757340