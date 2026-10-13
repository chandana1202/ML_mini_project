# Bird Species Identification from an Image

A machine learning mini-project that predicts the species of a bird from an uploaded photograph using **MobileNetV2 transfer learning**, TensorFlow/Keras, and Streamlit.

## Project Overview

Bird species identification from photographs can be challenging because many species have similar colors, shapes, and visual patterns. This project uses a convolutional neural network (CNN) based on MobileNetV2 to classify an image into one of **200 bird species** from the CUB-200-2011 dataset.

The Streamlit web application allows a user to upload a bird image and view the predicted species, the model's predicted probability, and the top five predictions.

## Features

- Upload a bird photograph in JPG, JPEG, or PNG format.
- Predict one of 200 bird species.
- Display the top prediction and its predicted probability.
- Display the top five predictions with scores.
- Simple browser-based interface built with Streamlit.

## Dataset

This project uses the **CUB-200-2011 (Caltech-UCSD Birds-200-2011)** dataset.

- **Images:** 11,788
- **Species/classes:** 200
- **Labels:** Bird species
- **Official split:** Training and test partitions supplied with the dataset

The dataset contains images of birds belonging to 200 categories. The dataset is not included in this repository. Obtain it from the official dataset source and follow its terms of use.

## Model and Methodology

The model was built with TensorFlow/Keras using **MobileNetV2 pretrained on ImageNet**.

1. Load the pretrained MobileNetV2 convolutional base without its original classification head.
2. Freeze the pretrained base during initial training.
3. Apply image augmentation using random horizontal flips and random rotations.
4. Use global average pooling to summarize the extracted image features.
5. Apply dropout with a rate of 0.3 to help reduce overfitting.
6. Add a dense output layer with 200 units and softmax activation.
7. Train the classifier using the Adam optimizer and sparse categorical cross-entropy loss.
8. Use early stopping during training.

The dataset's official training partition was divided into training and validation sets using a 15% validation split with stratification. The official test partition was kept for evaluation.

## Results

The model achieved the following result on the held-out test partition:

| Metric | Result |
|---|---:|
| Test accuracy | **56.82%** |
| Test loss | **1.6459** |

These values are from the model evaluation recorded during the project. The probability displayed for an individual uploaded image is the model's predicted probability, not a measure of overall test accuracy and not a guarantee that the prediction is correct.

## Project Structure

Keep these files together in the project folder:

```text
Bird-Species-Identification/
├── app.py
├── bird_species_model.keras
├── bird_species_classes.json
├── requirements.txt
├── README.md
├── report.pdf
├── presentation.pptx
├── notebooks/
│   └── bird_species_identification.ipynb
└── screenshots/
```

Include the report, presentation, notebook, and screenshots if they are part of your final submission. The notebook and screenshot folders are optional if you do not have those files.

## Requirements

- Python 3.12 is recommended for the local environment used for this project.
- pip
- The trained model file: `bird_species_model.keras`
- The class-name file: `bird_species_classes.json`

## Installation and Setup

### 1. Clone the repository

```bash
git clone <YOUR-PRIVATE-REPOSITORY-URL>
cd Bird-Species-Identification
```

Replace `<YOUR-PRIVATE-REPOSITORY-URL>` with your private GitHub repository URL.

### 2. Create a virtual environment (recommended)

**Windows:**

```powershell
py -3.12 -m venv .venv
```

### 3. Install dependencies

**Windows:**

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 4. Run the application

From the project directory, run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit will print a local URL in the terminal, usually `http://localhost:8501`. Open that URL in your browser.

## How to Use the App

1. Start the application using the command above.
2. Upload a bird image in JPG, JPEG, or PNG format.
3. Click **Identify Bird**.
4. View the predicted species, predicted probability, and top five predictions.

For the best chance of a useful prediction, try a clear photograph where the bird is visible. Results can be less reliable for blurry images, unusual viewpoints, multiple birds, or species outside the dataset's 200 categories.

## Limitations

- The model only predicts among the 200 species represented in the training dataset.
- The reported test accuracy is 56.82%, so predictions can be incorrect.
- Predicted probabilities may be overconfident and should not be interpreted as calibrated certainty.
- Performance can vary with image quality, lighting, background, bird size, and viewpoint.
- This is an educational mini-project and should not be treated as a definitive wildlife identification tool.

## Possible Future Improvements

- Fine-tune selected MobileNetV2 layers.
- Use learning-rate scheduling and more systematic hyperparameter tuning.
- Evaluate per-class precision, recall, and a confusion matrix.
- Improve confidence calibration and test on photographs from outside the training dataset.
- Compare with other pretrained image-classification architectures.

## Technologies Used

- Python
- TensorFlow and Keras
- MobileNetV2
- NumPy
- Pillow (PIL)
- Streamlit
- Google Colab for model training

## Team / Author

**Project:** Bird Species Identification from an Image  
**Type:** Machine Learning Mini-Project

Add your name, team members, course, and institution here if required by your submission instructions.

## Notes for Repository Maintenance

- Do **not** upload the entire CUB-200-2011 dataset to GitHub.
- Ensure `app.py`, `bird_species_model.keras`, and `bird_species_classes.json` are available to anyone who needs to run the application.
- GitHub blocks ordinary Git files larger than 100 MB. If the model file exceeds this limit, use Git LFS or provide it through an approved download location and explain the download step here.
- Keep the repository private if required by the assignment, and grant access to the instructor or teaching assistant.
