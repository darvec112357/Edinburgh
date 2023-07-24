# IAML 2020 - INFR10069 (Level 10) Assignment 2
This is the repository for Assignment 2 for IAML 2020.

**N.B.: This is the assignment for the Level 10 version of the course. If you are on the Level 11 course (i.e. likely an MSc student), then you should use [this](https://github.com/uoe-iaml/INFR11182-2020-CW2) version instead.**

## Repository Contents

 * `IAML_2020_CW2_INFR10069_Instructions.pdf`: This is the pdf with the question-sheet for the assignment.
 * `Assignment_2.tex`: This is the template you should modify to writeup your assignment.
 * `style.tex`: This is the style file for the assignment template. **Do not modify** this file in any way.
 * `data`: This is the directory (folder) which contains the data you need to complete the 3rd part of assignment (Question 3).
 * `templates`: This is the directory (folder) which contains Python template files you should use.
 * `helpers`: This is the directory (folder) which contains helper functions.

## Conda Environment

For this assignment you **must** use the same versions of the packages that are specified in the [IAML Labs repository](https://github.com/uoe-iaml/iaml-labs). 
Failing to do will result in you loosing points. 

## Building the PDF

This repository provides a latex template (`Assignment_2.tex`) for generating the final PDF that you will submit. 
You **must not** edit the style or formating of this document.
The only thing you need to do is enter your student number (see PDF for instructions) and put your answers in the correct answer boxes. 

You have a few options to create the final PDF:
* From a DICE machine you can run `pdflatex Assignment_2.tex` twice from the command line to compile the PDF. 
* You can do the same from your own computer, provided you have latex and all the necessary packages installed. Unfortunately, we do not support troubleshooting this step for you. 
* You can also use a free browser based latex editor such as [Overleaf](https://www.overleaf.com). **If you use such as tool, make sure to keep your document private and do not share the link to it.** 

## Submitting the Final PDF

Instructions will be provided on the IAML Learn page telling you where to upload your final PDF and code. 

## Change log
   Abbreviation: [I]: Instruction sheet (PDF file), [T]: Assignment_2.tex (Latex file)
   
* 03/Nov/2020 Ver. 1.0.0 -> 1.0.1 [I,T]
  * Fixed a typo in Question 1.8
    * Plot all the test samples (Xtrn_nm)" --> "Plot all the training samples (Xtrn_nm)"

* 02/Nov/2020 Ver. 1.0.0 [I,T]
  * Initial version
