# UC Berkeley ML/AI Capstone project - Skin Cancer Detection from Dermoscopic Images Using Deep Convolutional Neural Networks (CNN)

*PyTorch & Torchvision (CNNs, EfficientNet-B0), Albumentations-based Image Augmentation, Class-Imbalance Handling (Weighted Sampling, Loss Weighting), Model Evaluation (PR-AUC, ROC-AUC, Precision–Recall Analysis), Threshold Optimization for Screening, and Ensemble Stacking with scikit-learn*

**Project Repository:**  
https://github.com/parthabiswas1/Berkeley_MLAI_Capstone_-Skin-Cancer-Detection-from-Dermoscopic-Images-Using-deep-CNN

---
## **Sample Dermoscopic images**
<img width="247" height="246" alt="image" src="https://github.com/user-attachments/assets/f4634b46-8a1d-4c39-b741-9b271a22de24" />

ISIC_0250839	IP_6234053	male	75	head/neck	melanoma	malignant	1


<img width="248" height="248" alt="image" src="https://github.com/user-attachments/assets/9bc24baf-25bd-4ef4-8885-36ece08f44fd" />

ISIC_0250064	IP_4921034	female	75	lower extremity	nevus	benign	0


<img width="247" height="249" alt="image" src="https://github.com/user-attachments/assets/2e04e6a1-c2a1-491b-b6c2-e8a23c9a86c6" />

ISIC_0247330	IP_3232631	female	65	lower extremity	melanoma	malignant	1


<img width="246" height="244" alt="image" src="https://github.com/user-attachments/assets/08c880d7-57c2-4ffc-8875-89ae93648b7d" />

ISIC_0250455	IP_7748972	male	45	torso	unknown	benign	0





---

## **Executive Summary**

Skin cancer screening systems face a difficult trade-off: they must identify as many cancer cases as possible while working with limited data quality, extreme class imbalance, and practical resource constraints. Missing a malignant case can have serious consequences, yet aggressively flagging every suspicious image can overwhelm clinicians with false positives.

In this project, I explored whether modern deep learning models can still be useful for **screening purposes** when trained on **low-resolution dermoscopic images** and limited patient metadata. Rather than chasing maximum benchmark accuracy, the focus was on building a **realistic, reproducible pipeline** that mirrors real-world constraints such as storage limits, compute budgets, and noisy data.

The work compares traditional metadata-based models, image-only deep learning models, and a combined ensemble approach. By carefully selecting evaluation metrics and decision thresholds, the final models demonstrate how performance can be tuned toward **high sensitivity screening**, with clearly understood trade-offs. <!-- The project concludes with an interactive demo that shows how these models could be used in practice, not just evaluated offline.-->

---

## **Project Objective**

The objective of this capstone is to predict whether a dermoscopic skin image is **malignant (skin cancer)** using low-resolution dermoscopic images and a small set of patient metadata. The emphasis is on building a **leakage-safe, imbalance-aware, and clinically motivated pipeline**, rather than maximizing leaderboard-style performance.

---

## **Dataset and Practical Constraints**

This project is based on the **ISIC 2020 Skin Lesion Analysis: Towards Melanoma Detection** dataset (~33,000 images, ~50GB at full resolution):

https://challenge2020.isic-archive.com/

Due to storage and compute constraints, training on the original high-resolution images was not feasible. Instead, a **pre-resized 224×224 JPG version** of the dataset was used:

https://www.kaggle.com/datasets/nischaydnk/isic-2020-jpg-224x224-resized

**Important trade-off:**  
This is an extremely difficult dataset due to severe class imbalance. Malignant images make up only ~1–2% of the data. Performance is therefore expected to be lower than approaches trained on full-resolution dermoscopy images. This trade-off is intentional due to storage and compute limits as mentioned before.

---

## **Key Challenges and Design Decisions**

*   **Google Colab instability resolution** Training directly from Google Drive createed a performance and reliability problem because reading thousands of small image files incured high I/O latency and intermittently stalled and disconnected during long runs.  The solution was to copy and unzip the image dataset into the local runtime once, then load metadata, construct file paths, and verify image integrity so training used fast, stable local disk access.

*   **Severe Class Imbalance:** Malignant cases are rare relative to benign lesions. Accuracy calculation is therefore misleading. The model can achieve high accuracy by simply predicting the majority class (Benign or not cancer) most of the time, while completely failing to identify the minority class (malignant or cancer). To properly evaluate performance under imbalance, **PR-AUC (Precision-Recall Area Under Curve)** was used as the primary evaluation metric, with **ROC-AUC (Receiver Operating Characteristic Area Under Curve)** reported as a secondary reference

*   **Leakage Prevention:** There are multiple images per patient in the dataset and the images are highly correlated. To avoid leakage, I have split the dataset strictly at the **patient level** using patient_id, to ensure that no patient appears in more than one of train, validation, or test sets.

*  **Image level prevalance** Malignant images are rare in the dataset, so care had to be taken to ensure that the right proportion of maligant images were in train, val and test datasets.

---
## **Exploratory Data Analysis (EDA)**

A very limited set of metadata were provided with the images. 
<img width="1276" height="233" alt="image" src="https://github.com/user-attachments/assets/d1089009-cac4-492a-a233-00ccf61ed414" />

The following was done:

*   Missing-value and metadata sanity checks
*  <img width="445" height="325" alt="image" src="https://github.com/user-attachments/assets/74466486-f474-4404-9765-98ac185a4a01" />

*   **Class imbalance analysis:** The splits had to be carefully done to ensure both benign and malignant images were present in the train, val and test dataset in similar proportions.
*   <img width="436" height="322" alt="image" src="https://github.com/user-attachments/assets/b6f8b1ba-54ae-4b45-952e-3b2359d32454" />

*   **Patient-level distribution checks:** Ensured that the train, val , test splits did not contain the same patient's dermascoping images.
*   <img width="437" height="318" alt="image" src="https://github.com/user-attachments/assets/b1f84680-ec34-4736-a69d-35de440c23b0" />

*   <img width="438" height="324" alt="image" src="https://github.com/user-attachments/assets/c0eea725-d40a-49f0-ba5d-8f4b3c3c60c2" />


---

## **Feature Engineeering**

*  **Age normalization and binning**
Age was converted to a numeric value, bucketed into ordered ranges, and augmented with an explicit missing flag.

*  **Sex and site missingness handling**
Missing sex and anatomical site values were encoded using binary indicators and  "Unknown" category.

*  **Anatomical site consolidation**
Rare anatomical sites were grouped into an "Other" category using training-only split to reduce sparsity and overfitting.

*  **Categorical interaction features**
Sex–site and age–site interaction features were created to capture conditional risk patterns beyond marginal effects.

*  **Leakage-safe default handling**
Unseen patients were assigned neutral defaults, with positive-rate imputation based on the training-set mean only.

*  **Split-safe application**
All feature statistics were learned on the training set and applied identically to train, validation, and test splits.

---

## **Models Implemented**

As the dataset is highly imbalanced with mostly benign images and rare positive images, PR-AUC (Precision–Recall curve) was used to rank the models. 

### **1. Baseline Model with metadata: LogisticRegression**

Used patient-level metadata with guaranteed malignant/ benign data within train and val sets to establish a baseline. this also demonstrates the predictive value of metadata alone.

<img width="594" height="206" alt="image" src="https://github.com/user-attachments/assets/cf337106-951d-4110-ae59-4543a6fda6c8" />


### **2.LogisticRegression with Hyperparameter tuning - Cross Validation and GridSearch**
Class-weighted Logistic Regression (which up-weights the rare positive class to counter the 1–2% prevalence) is fitted and tuned with **5-fold GroupKFold** by patient_id to avoid same-patient leakage across folds. Then **GridSearch** explores **L1 vs L2 regularization** (sparsity/selective feature use vs smooth shrinkage) and **different regularization strengths** (C) to select the best setting by **cross-validated** PR-AUC.

<img width="594" height="206"  alt="image" src="https://github.com/user-attachments/assets/6f7e22c9-9a1a-42c1-a2b1-af46b08537bf" />


### **3. BGradient Boosted Decision Trees (GBDT) - HistGradientBoostingClassifier**

Trained a metadata-only **HistGradientBoostingClassifier** with one-hot encoded categorical features and median-imputed numeric features. I used HistGradientBoostingClassifier because **Gradient Boosted Decision Trees** can capture non linear relationships and feature interactions that logistic regression often misses. 

<img width="594" height="206"  alt="image" src="https://github.com/user-attachments/assets/45612ddb-70c4-4412-bd18-aa48c072df6a" />


### **4. Image-Only Convolutional Neural Network (CNN) Model EfficientNet-B0**

*(EfficientNet-b0 is a convolutional neural network that is trained on more than a million images from the ImageNet database.)*
I used reduced size 224×224 pixel dermoscopic images in all CNN models.

I chose it because it is widely used for image classification, often outperforming much larger models like ResNet with significantly fewer parameters. Pretrained on ImageNet dataset images features that transfer well, so head-only training is a reliable.

Training strategy: Freeze backbone and train only classifier head.

Used (torchvision) for light augmentation (flip and small rotation). 

Imbalance handling: Used pos_weight = neg/pos inside **BCEWithLogitsLoss** to upweight positives (malignant).

Used **AdamW** optimizer during training to update the model’s trainable weights and to reduce the **loss**. 

<img width="594" height="206"  alt="image" src="https://github.com/user-attachments/assets/f507ed81-45b1-46e8-919b-37b6cd8acfb3" />


### **5. Convolutional Neural Network (CNN) Model - EfficientNet-B0 with image aumentaion using Albumentations**

The training pipeline uses **Albumentations** to apply a richer and more diverse set of image augmentations. Each training image is first **resized** to a fixed input size. It is then **randomly flipped horizontally and vertically** to improve invariance to orientation. Next, an affine transformation is applied, which includes **small random translations, scaling, and rotations** of up to ±20 degrees to **increase geometric robustness**.

After the geometric transformations, **color and illumination variations** are introduced through random **brightness and contrast adjustments, hue and saturation shifts, and gamma corrections**. To further improve robustness to image quality variations common in real-world data, the pipeline occasionally applies **mild blurring or Gaussian noise**. Finally, the image is normalized using ImageNet mean and standard deviation and converted into a PyTorch tensor using ToTensorV2

Training strategy is the same as before, freeze backbone and train only classifier head.

<img width="594" height="206" alt="image" src="https://github.com/user-attachments/assets/49acb4bc-2965-470b-9575-70ee0daac16a" />


### **6. Convolutional Neural Network (CNN) Model - EfficientNet-B0 classifier using a two-stage fine-tuning approach with Albumentations-based image augmentation**

* Images are resized, augmented, normalized with ImageNet statistics, and converted to tensors, while class imbalance is handled using a weighted binary cross-entropy loss. 
* **In Stage 1**, the pretrained backbone is frozen and only the classifier head is trained, with validation PR-AUC used for early stopping and checkpoint selection. 
* **In Stage 2**, the best Stage 1 model is reloaded and only the final EfficientNet block plus the classifier head are fine-tuned using a lower learning rate for the backbone. 
* The final model is evaluated on the test set using PR-AUC and ROC-AUC.


<img width="594" height="206" alt="image" src="https://github.com/user-attachments/assets/0a26626d-e839-48d9-9219-6f45c9f13ff3" />


### **7 Convolutional Neural Network (CNN) Model - EfficientNet-B0 classifier using a two-stage fine-tuning approach with Albumentations-based image augmentation and imbalance handling with WeightedRandomSampler**

Two-stage fine-tune: (S1) head-only, (S2) last block + head (lower LR)
Imbalance handling : WeightedRandomSampler
Loss : BCEWithLogitsLoss
Stronger Albumentations aug: Affine and color and blur/noise and coarse dropout

Best reseults.

<img width="594" height="206" alt="image" src="https://github.com/user-attachments/assets/621daf97-8044-43ed-93c3-011f3790fb90" />

### **8 ##STACKING - IMAGE MODEL: Convolutional Neural Network (CNN) Model - EfficientNet-B0 using two stage fine tuning and strong albumentation and weightedRandomSampler with METADATA MODEL: LogisticRegression with Cross Validation and GridSearch**

This code builds a stacked ensemble that combines the EfficientNet-B0 CNN trained with strong Albumentations augmentation and WeightedRandomSampler with the tuned metadata LogisticRegression (GroupKFold/GridSearch). It generates out-of-fold (OOF) probability predictions for both base models on the TRAIN set (K=3), fits a LogisticRegression stacker on those OOF features, and then evaluates the stacker on VAL using model probabilities.

### **9 Evaluate best performing model on unseen TEST set**

Convolutional Neural Network (CNN) Model - EfficientNet-B0 classifier using a two-stage fine-tuning approach with Albumentations-based image augmentation and imbalance handling with WeightedRandomSampler had the best rank using PR-AUC. this model was used on the Test set.

<img width="594" height="206" alt="image" src="https://github.com/user-attachments/assets/d7ab3e3c-501c-426a-95a9-7e039ea899fb" />

## ** Threshold, Recall & Confusion Matrix

Scanned all possible thresholds on the validation set using the precision–recall curve (PR-AUC), and selected the threshold that achieves **Recall** (Of all the positive images, how many did we identify ?)  **≥ 0.85** while maximizing precision among those candidates. That threshold was **0.339062**.

<img width="393" height="382" alt="image" src="https://github.com/user-attachments/assets/b3940a3b-4d9a-4859-b49b-bffd40d85d5e" />


# Capstone Results Summary


|   # | Model                                                                   | Split   |   PR-AUC |   ROC-AUC |   Top-5% Recall |   Top-5% Precision |   Top-10% Recall |   Top-10% Precision |
|----:|:------------------------------------------------------------------------|:--------|---------:|----------:|----------------:|-------------------:|-----------------:|--------------------:|
|   1 | Metadata LR (baseline)                                                  | VAL     |  0.01966 |   0.52767 |          0.0575 |             0.0194 |           0.1494 |              0.0251 |
|   2 | Metadata LR (GroupKFold & GridSearch tuned)                             | VAL     |  0.02533 |   0.58207 |          0.1264 |             0.0426 |           0.2299 |              0.0387 |
|   3 | Metadata HGB (baseline)                                                 | VAL     |  0.01839 |   0.53924 |          0      |             0      |           0.1379 |              0.0232 |
|   4 | CNN EffNet-B0 (head-only, pos_weight)                                   | VAL     |  0.09295 |   0.81424 |          0.3103 |             0.1047 |           0.4483 |              0.0754 |
|   5 | CNN EffNet-B0 (head-only & Albumentations)                              | VAL     |  0.08431 |   0.83707 |          0.2644 |             0.0891 |           0.3563 |              0.06   |
|   6 | CNN EffNet-B0 (2-stage & Albumentations, pos_weight)                    | VAL     |  0.10753 |   0.8502  |          0.3218 |             0.1085 |           0.4943 |              0.0832 |
|   7 | **CNN EffNet-B0 (2-stage & strong Albumentations + WeightedRandomSampler)** | VAL     |  **0.10868** |   0.85861 |          0.3218 |             0.1085 |           0.4713 |              0.0793 |
|   8 | Stacking OOF(K=3): CNN & tuned LR(meta)                                 | VAL     |  0.09434 |   0.84907 |          0.2874 |             0.0969 |           0.4138 |              0.0696 |
|   9 | Best model on TEST (same as #7)                                         | TEST    |  0.13381 |   0.86871 |          0.3761 |             0.1285 |           0.5046 |              0.0863 |


### The Best model performance was by Convolutional Neural Network (CNN) Model - EfficientNet-B0 classifier using a two-stage fine-tuning approach with Albumentations-based image augmentation and imbalance handling with WeightedRandomSampler

---

