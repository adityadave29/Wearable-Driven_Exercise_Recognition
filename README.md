# Wearable-Driven Exercise Recognition

As we move further into the era of automation, the fitness industry still has significant scope for improvement in workout monitoring and tracking. Most individuals with regular gym experience rely on applications to log their workouts, such as the number of exercises, sets, and repetitions. However, this process is largely manual—users must select the exercise type and enter repetitions themselves, which can be time-consuming and error-prone.

To address this limitation, this project aims to automate workout tracking by designing a machine learning model that can detect and classify exercises using motion data captured from a fitness band worn on the wrist. By analyzing sensor data, the model can identify the performed exercise without requiring manual input from the user.

Currently, the model is trained to recognize barbell exercises, and future extensions may include additional workout types. This document focuses on the technical aspects of the project, including data processing, model design, and evaluation.


## Participants

This data was collected by a master student at a university with the help of 5 people.
| Participant | Gender | Age | Weight (kg) | Height (cm) | Experience (years) |
| --- | --- | --- | --- | --- | --- |
| A | Male | 23 | 95 | 194 | 5+ |
| B | Male | 24 | 76 | 183 | 5+ |
| C | Male | 16 | 65 | 181 | < 1 |
| D | Male | 21 | 85 | 197 | 3 |
| E | Female | 20 | 58 | 165 | 1 |

## Project Workflow & Methodology

This project follows a structured pipeline to transform raw wearable sensor data into meaningful exercise recognition and repetition counting results. The complete process is outlined below.

### 1. Data Collection
Motion data is collected using a wrist-worn fitness band equipped with:
- **Accelerometer** (X, Y, Z axes)
- **Gyroscope** (X, Y, Z axes)

Each recording corresponds to a specific:
- Participant
- Exercise type (bench press, squat, row, overhead press, deadlift)
- Weight category (heavy / medium)
- Set number

---

### 2. Dataset Construction & Merging
The raw dataset is distributed across multiple CSV files (one per participant, exercise, and sensor type).  
To create a unified dataset:
- All accelerometer and gyroscope files are loaded
- Metadata (participant, label, category, set) is extracted from filenames
- Files are merged into consolidated accelerometer and gyroscope DataFrames

Sensor timestamps stored as epoch milliseconds are converted to a datetime index to enable time-series processing.

---

### 3. Sensor Alignment & Resampling
Since accelerometer and gyroscope sensors operate at different sampling rates:
- Both signals are aligned using their datetime index
- Data is resampled into **fixed 200 ms windows**
- Sensor values are averaged within each window
- Metadata is forward-filled

The processed dataset is stored as a `.pkl` file to preserve data types and structure.

---

### 4. Exploratory Data Analysis (EDA)
Initial visualization is performed to understand motion patterns:
- Accelerometer and gyroscope signals are plotted per exercise
- Differences between exercises and weight categories are visually analyzed

This step confirms that sensor data contains discriminative patterns suitable for classification.

---

### 5. Outlier Detection
Sensor data may contain unintended movements or noise. Three outlier detection techniques are evaluated:
- **IQR (Interquartile Range)**
- **Chauvenet’s Criterion**
- **Local Outlier Factor (LOF)**

After comparison, **Chauvenet’s method** is selected as it removes statistically improbable values without over-filtering valid motion.  
Outliers are marked as missing values rather than fully removed.

---

### 6. Signal Smoothing (Low-Pass Filtering)
To remove high-frequency noise while preserving meaningful human motion:
- A **low-pass filter** is applied to all sensor axes
- Frequencies above human movement range are suppressed

This results in smoother signals, which improves feature extraction and repetition detection.

---

### 7. Feature Engineering
Multiple feature types are extracted to capture different aspects of motion:

- **Vector Magnitude Features**  
  - Resultant acceleration (`acc_r`)
  - Resultant gyroscope magnitude (`gyr_r`)

- **Dimensionality Reduction (PCA)**  
  - Principal components capture dominant movement directions

- **Temporal Features**  
  - Sliding-window mean and standard deviation
  - Capture motion intensity and consistency over time

- **Frequency-Domain Features (FFT)**  
  - Dominant repetition frequency
  - Movement rhythm and regularity
  - Power spectral entropy (PSE)

- **Clustering Features (K-Means)**  
  - Unsupervised clustering to identify natural motion groupings

---

### 8. Feature Selection
As the feature space grows large, forward feature selection is applied to:
- Identify the most informative feature subset
- Reduce redundancy and overfitting
- Improve model efficiency

Features are grouped into progressive feature sets to evaluate the impact of each processing stage.

---

### 9. Model Training & Evaluation
Multiple machine learning models are trained and compared, including:
- Neural Networks
- Random Forest
- XGBoost
- KNN
- Decision Tree
- Naive Bayes

Grid search is used to tune hyperparameters and compare performance across feature sets.  
The best-performing models achieve **~99% classification accuracy**.

---

### 10. Repetition Counting
Beyond exercise classification, the project also estimates repetitions:
- Rest periods are removed
- A low-pass filter is applied to smooth motion signals
- **Peak detection** is used to identify repetitions  
  - Each dominant peak corresponds to one completed repetition
- Exercise-specific filter cutoffs are used to account for movement speed differences

---

### 11. Evaluation
Predicted repetitions are compared against ground truth using:
- **Mean Absolute Error (MAE)**

This validates the effectiveness of the repetition-counting approach across different exercises and intensity levels.

---

### Key Insight
The combination of signal processing, feature engineering, and machine learning enables accurate **exercise recognition** and **automatic repetition counting** using only wearable sensor data—without any manual user input.

## 📊 Model Performance Comparison

| Model | Feature Set | Accuracy |
|------|------------|----------|
| NN | Feature Set 4 | 0.992589 |
| XG | Feature Set 4 | 0.992589 |
| RF | Feature Set 3 | 0.989810 |
| RF | Feature Set 4 | 0.988421 |
| RF | Selected Features | 0.978694 |
| NN | Selected Features | 0.977304 |
| KNN | Feature Set 4 | 0.977304 |
| DT | Feature Set 3 | 0.974062 |
| DT | Feature Set 4 | 0.973599 |
| DT | Selected Features | 0.967114 |
| RF | Feature Set 1 | 0.965262 |
| RF | Feature Set 2 | 0.965262 |
| KNN | Feature Set 3 | 0.962019 |
| DT | Feature Set 2 | 0.951366 |
| NB | Feature Set 3 | 0.949050 |
| DT | Feature Set 1 | 0.948124 |
| NN | Feature Set 2 | 0.938861 |
| NN | Feature Set 1 | 0.930987 |
| KNN | Selected Features | 0.919407 |
| NB | Selected Features | 0.914312 |
| NB | Feature Set 4 | 0.911533 |

## Future Scope

This project can be extended by integrating the trained model into a mobile or wearable application, enabling users to automatically track exercises and repetition counts in real time without manual input.

From a modeling perspective, the overall accuracy is high but not perfect. One of the main challenges observed is the similarity in motion patterns between certain exercises—particularly **bench press** and **overhead press (OHP)**. Since both involve comparable upper-body pushing movements, the model occasionally confuses these two classes.

To address this limitation, future work may include:
- Performing a dedicated, fine-grained analysis focused only on similar exercise pairs (e.g., bench vs OHP)
- Designing specialized features or models to better distinguish between subtle motion differences
- Incorporating additional sensor placements or contextual information to improve class separation

These improvements have the potential to further enhance model robustness and push classification accuracy closer to optimal performance.
