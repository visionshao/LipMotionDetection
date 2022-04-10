# Lip Motion Detection
This project uses two methods (Dlib and SAN) to do lip motion detection.

## Preparation

### Environment
I prepare the `LMD_env..yaml` for you. You can run `conda env create -f LMD_env.yaml` to create the environment.

### Datasets Download
Please open the folder of the each method to view the detailed description.

## Instructions for Use
Run `python GUI.py` to start.
- Step-1 : Choose a method for handling the media, Dlib or SAN. Please notice that the SAN requires GPU.
- Step-2 : Determine the type of input, you can choose images, videos, or real-time camera input.
- Step-3 : Select the media path you want to process. If you select real-time input in the second step, you can skip this step.
- Step-4 : Set the save path of the processed media. If this is not filled in, the processed media will not be saved.
