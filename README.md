# EmoTorch - Paper
This repository contains Facial Emotion Recognition models build in pytorch that can be run out of the box. And their entire training process including dataset creation can be found here. This project was build for a university project at TU Graz from Gregor Autischer with Know Center. The purpuse of this project is to refine a existing Emotion Recogntion model with fairness in mind to make it certifyable in light of the EU AI Act. 

A precursor to this project was the paper: Practical Application and Limitations of AI Certification Catalogues in the Light of the AI Act [arXiv](https://arxiv.org/abs/2502.10398).
The baseline model in this repository is a faithful recreation of the ConvlutionalNN model from the EmoPy Project [EmoPy](https://github.com/thoughtworksarts/EmoPy).
The new model in the repository is a improved version that tries to address the shortcoming of the original model that came to light in the above mentioned paper during our certification attempt.

This repository has a sister repo that just includes the two trained and ready to use facial emotion recogniton models. It can be found here: [EmoTorch](https://github.com/gregor-autischer/EmoTorch).

# General Structure
This entire repository is split into basically two sections. One for the baseline model and one for the new model, both of which have several directories and files. The general datastructure is:
```
EmoTorch_Paper/
├── develop_baseline_model/
│   └── [...]
├── baseline_model/
│   └── [...]
├── build_new_model/
│   └── [...]
├── build_new_dataset/
│   └── [...]
└── new_model/
    └── [...]
README.md
requirements.txt
```

## Baseline Model
The baseline model has the two directories "baseline_model" and "develop_baseline_model". The folder "develop_baseline_model" has all the scripts to retrain the baseline model from the ground up. The description of how to do this can be found in [BASELINE_MODEL.md](BASELINE_MODEL.md).

There is also a pretrained model shipping with this repo that represents the baseline for the associated research paper. It was created with the scripts inside of "develop_baseline_model" and can also be found there. But for convinience it is in the directory "baseline_model". In this folder you can find several scripts to evaluate the pretrained model and run predictions with it. The general structure is this:

```
EmoTorch_Paper/
└── baseline_model/
    ├── helpers
    │   └── generate_dataset_catalog.py
    └── model
        ├── looking_at_data.ipynb
        ├── convolutional_nn_pytorch.py
        ├── eval_model.py
        ├── predict_image_baselinemodel.py
        ├── model.pth
        └── [...]
```

### In dir "helpers"
**`generate_dataset_catalog.py`**
- Generates a comprehensive CSV catalog of all images from both original and augmented datasets
- Includes emotion labels, usage classifications (training/validation), and demographic attributes (race, gender, age)
- Handles path normalization and matches augmented images to their source images for demographic data
- **Requires**: `faces_gender_race.csv` from [EmoTorch_Run_FairFace](https://github.com/gregor-autischer/EmoTorch_Run_FairFace) copied to `baseline_model/model/`
- Usage: `python baseline_model/helpers/generate_dataset_catalog.py`

### In dir "model"
- "looking_at_data.ipynb" looks into all the statistics for fairness on the dataset.
- "convolutional_nn_pytorch.py" is the architecture of the baseline model.
- "eval_model.py" does evaluation of the model with basic metrics that it saves to files.
- "predict_image_baselinemodel.py" performs single image emotion prediction using the trained baseline model. Use it: ```python predict_image_baselinemodel.py <image_path>```
- "model.pth" is the description of the pretrained model these scripts use to do prediction and eval.
- Other files (like CSVs) are created from one of those scripts.

## New Model
The new model has three directories "build_new_model", "build_new_dataset" and "new_model". The directories "build_new_dataset" and "build_new_model" and its files are used to train the new model from the ground up. The description of how to do this can be found in [NEW_MODEL.md](NEW_MODEL.md).

There is also a pretrained model shipping with this repo that represents the new improved model for the associated research paper. It was created with the scripts inside of "build_new_dataset" and "build_new_model" and can also be found there. But for convinience it is in the directory "new_model". In this folder you can find the new model and a script that allows you to easily predict images. The general structure is this:

```
EmoTorch_Paper/
└── new_model/
    ├── new_model
    │   └── [...]
    ├── enhanced_model.py
    └── predict_images.py
```

### In dir "new_model/new_model"
- In this folder are several files that have all the statistical information for the new models performance.

### In dir "new_model"
- "enhanced_model.py" is the architecture of the new model.
- "predict_images.py" performs single image emotion prediction using the trained baseline model. Use it: ```python predict_images.py <image_path>```