# 🎧 Deepfake Audio Detector - KNN

A K-Nearest Neighbors (KNN)-based system to classify synthetic (deepfake) and real audio using Mel-Frequency Cepstral Coefficients (MFCCs).

---

## 🔍 Overview

This project identifies whether an audio sample is real or AI-generated using:
- **K-Nearest Neighbors (KNN)** classifier
- **MFCC feature extraction**
- **Dataset of real and fake voices**
- **Audio preprocessing & segmentation**

---

## 🧠 Key Features

- 📊 **KNN Classification** with scikit-learn  
- 🎵 **MFCC Extraction** using `librosa`  
- 🔈 **10s Audio Segmentation** for dataset balancing  
- 📁 Train/Test split with evaluation metrics  
- 📦 Model saved using `joblib` for reuse  

---

## 📁 Project Structure
deepfake-audio-detector-knn/
│
├── data/ # Processed dataset and demo audio
│ ├── REAL/
│ └── FAKE/
│
├── notebooks/ # Colab-compatible Python code
│ └── deepfake_knn.ipynb
│
├── models/ # Saved model
│ └── knn_model.pkl
│
├── results/ # Evaluation results and visualizations
│ └── confusion_matrix.png
│
└── README.md # This file


---

## 🛠️ Technologies Used

- Python
- Librosa
- Pydub
- Scikit-learn
- Matplotlib & Seaborn

---

## 📊 Evaluation Metrics

| Metric     | Value (Example) |
|------------|-----------------|
| Accuracy   | 92%             |
| Precision  | 90%             |
| Recall     | 93%             |
| F1-Score   | 91%             |

> 📌 Confusion matrix and classification report are included in the `results/` folder.

---

## 🚀 How to Run

1. Clone this repo  
2. Install required packages:
   ```bash
   pip install -r requirements.txt

3. Run the Jupyter notebook or convert it to a Python script:

   ```bash
   jupyter notebook notebooks/deepfake_knn.ipynb

## 📦 Model Deployment

Use the saved  model to make predictions on new audio features:

   ```bash
   import joblib
   
   model = joblib.load('models/knn_model.pkl')
   prediction = model.predict([your_features])
   print("Prediction:", prediction[0])
   ```
---

## 📌 Notes

- The dataset requires balanced classes (equal number of fake and real audio clips).  
- Audio clips are segmented into fixed 10-second intervals.  
- Features are based on MFCCs for effective audio representation.  
- This project focuses on KNN; additional models (CNN, Naive Bayes) are in separate branches/repositories.  
- Make sure dependencies and data paths are properly configured for smooth execution.

---

## 📬 Contact

**Fathima Hana**  
📧 [fathimahanaks@gmail.com](mailto:fathimahanaks@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/fathimahana/)  <!-- Replace with your actual LinkedIn URL -->

Feel free to reach out for collaborations or questions!
