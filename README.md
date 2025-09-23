<!-- start abstract -->
# Loop Extrusion Contact Dynamics

![alt text](image_analysis/image/20241212_1a2_cont_15_15_100_standard_54__0002_movie_3panel_rgb_13.gif)

This is a repository to analyze two color contact dynamics and estimate the localization precision from live cell imaging data


<!-- end abstract -->

## Repository Overview

The code for the image analysis can be found in the folder [image_analysis](image_analysis/). The folder contains:
    * `docs`: Contains all project documentation.
    * `infrastructure`: Contains detailed installation instructions for all requried tools.
    * `ipa`: Contains all image-processing-and-analysis (ipa) scripts which are used for this project to generate final results.
    * `runs`: Contains all config files which were used as inputs to the scripts in `ipa`.
    * `scratchpad`: Contains everything that is nice to keep track of, but which is not used for any final results.

The code for the polymer simulation can be found in the folder [3D_loop_extrusion](3D_loop_extrusion/)

## Setup
Detailed install instructions can be found in [infrastructure/README.md](image_analysis/infrastructure/README.md).

## Citation
Do not forget to cite our [publication]() if you use any of our provided materials.

---
This project was generated with the [faim-ipa-project](https://fmi-faim.github.io/ipa-project-template/) copier template.
