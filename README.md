# Heart Sound Classification - Baseline (PhysioNet/CinC Challenge 2016)

This project uses the **PhysioNet/CinC Challenge 2016 heart sound dataset** for training and testing baseline machine learning models.

---

## Dataset Description
- The dataset contains **PCG recordings** (`.wav`) and **header files** (`.hea`) from multiple subjects.
- Each recording is labeled as:
  - `Normal`
  - `Abnormal`
  - (Some recordings marked as `Unsure` in original Challenge, due to poor quality)

Training and test splits are predefined by the Challenge organizers.

---

## Preprocessing Steps (based on Challenge paper)
To prepare the dataset for training, the following preprocessing steps are applied:

1. **Resampling**  
   - All recordings are resampled to **2000 Hz** using an anti-alias filter.

2. **Segmentation**  
   - Heart sounds are segmented into four states:  
     - **S1 (First heart sound)**  
     - **Systole**  
     - **S2 (Second heart sound)**  
     - **Diastole**  
   - Springer’s segmentation algorithm (2015) is recommended for this step.

3. **Feature Extraction**  
   From each segmented beat, 20 features are extracted (as described in the paper):  
   - RR interval statistics (`m_RR`, `sd_RR`)  
   - S1, S2 interval durations and variabilities  
   - Systolic/diastolic durations and ratios  
   - Amplitude ratios between systole/diastole and S1/S2  
   - (See Section 6.2 of the paper for full feature list)

4. **Feature Selection**  
   - Logistic regression with forward likelihood ratio selection is applied.  
   - From 20 features, 7 key predictors were found most useful in the paper.  

---

## Machine Learning Methods to Implement
We will implement and test the following methods step by step in `baseline_demo.ipynb`:

1. **Baseline Logistic Regression (BLR)** – as described in the paper.  
2. **K-Means clustering** (unsupervised segmentation/classification).  
3. **Hidden Markov Models (HMM / HSMM)** for sequence modeling.  
4. **Neural Network models** (MLP, Time-Delay NN).  
5. **Modern classifiers** (Random Forest, SVM, XGBoost, etc. for comparison).

Each model will be tested using:
- **10-fold cross-validation (stratified by patient)**  
- **Leave-one-database-out validation**  

---

## How to Run
1. Clone this repo in GitHub Codespaces.  
2. Place the dataset inside `data/training-b/`.  
3. Open `baseline_demo.ipynb`.  
4. Run cells step by step to:
   - Preprocess data  
   - Extract features  
   - Train models  
   - Evaluate results  

---

## References
- PhysioNet/CinC Challenge 2016: [https://physionet.org/challenge/2016/](https://physionet.org/challenge/2016/)  
- Springer et al. (2015): State-of-the-art segmentation algorithm for heart sounds.  
- Clifford & Moody (2012): Prior PhysioNet Challenges.  
