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

The work compares traditional metadata-based models, image-only deep learning models, and a combined ensemble approach. By carefully selecting evaluation metrics and decision thresholds, the final models demonstrate how performance can be tuned toward **high sensitivity screening**, with clearly understood trade-offs. The project concludes with an interactive demo that shows how these models could be used in practice, not just evaluated offline.

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

### **1. Baseline Model with metadata: LogisticRegression**

Used patient-level metadata with guaranteed malignant/ benign data within train, val test, data to establish a baseline. this also demonstrates the predictive value of metadata alone.

<img width="641" height="471" alt="image" src="https://github.com/user-attachments/assets/f38f3f9e-0380-4540-ad8c-8cc5b8cd26ef" />


### **2.LogisticRegression with Hyperparameter tuning - Cross Validation and GridSearch**
Class-weighted Logistic Regression (which up-weights the rare positive class to counter the 1–2% prevalence) is fitted and tuned with **5-fold GroupKFold** by patient_id to avoid same-patient leakage across folds. Then **GridSearch** explores **L1 vs L2 regularization** (sparsity/selective feature use vs smooth shrinkage) and **different regularization strengths** (C) to select the best setting by **cross-validated** PR-AUC.

<img width="318" height="558" alt="image" src="https://github.com/user-attachments/assets/13bef9ff-ed05-45af-9c50-aca9458b081f" />


### **3. Baseline Model with metadata: Gradient Boosted Decision Trees (GBDT) - HistGradientBoostingClassifier**

Trained a metadata-only **HistGradientBoostingClassifier** with one-hot encoded categorical features and median-imputed numeric features. I used HistGradientBoostingClassifier because **Gradient Boosted Decision Trees** can capture non linear relationships and feature interactions that logistic regression often misses. 

<img width="659" height="462" alt="image" src="https://github.com/user-attachments/assets/d4472677-b88e-4e9b-961e-4f9094b3c558" />


### **4. Image-Only Convolutional Neural Network (CNN) Model EfficientNet-B0**

*(EfficientNet-b0 is a convolutional neural network that is trained on more than a million images from the ImageNet database.)*
I used reduced size 224×224 pixel dermoscopic images in all CNN models.

I chose it because it is widely used for image classification, often outperforming much larger models like ResNet with significantly fewer parameters. Pretrained on ImageNet dataset images features that transfer well, so head-only training is a reliable.

Training strategy: Freeze backbone and train only classifier head.

Used (torchvision) for light augmentation (flip and small rotation). 

Imbalance handling: Used pos_weight = neg/pos inside **BCEWithLogitsLoss** to upweight positives (malignant).

Used **AdamW** optimizer during training to update the model’s trainable weights and to reduce the **loss**. 

<img width="1543" height="468" alt="image" src="https://github.com/user-attachments/assets/3ae30b9c-b33b-46e5-885f-bcc2685aeaf7" />


### **5. Convolutional Neural Network (CNN) Model - EfficientNet-B0 with image aumentaion using Albumentations**

The training pipeline uses **Albumentations** to apply a richer and more diverse set of image augmentations. Each training image is first **resized** to a fixed input size. It is then **randomly flipped horizontally and vertically** to improve invariance to orientation. Next, an affine transformation is applied, which includes **small random translations, scaling, and rotations** of up to ±20 degrees to **increase geometric robustness**.

After the geometric transformations, **color and illumination variations** are introduced through random **brightness and contrast adjustments, hue and saturation shifts, and gamma corrections**. To further improve robustness to image quality variations common in real-world data, the pipeline occasionally applies **mild blurring or Gaussian noise**. Finally, the image is normalized using ImageNet mean and standard deviation and converted into a PyTorch tensor using ToTensorV2

Training strategy is the same as before, freeze backbone and train only classifier head.


<img width="302" height="310" alt="image" src="https://github.com/user-attachments/assets/490a515f-6817-415d-85c5-a0abf0685ebc" />


### **6. Convolutional Neural Network (CNN) Model - EfficientNet-B0 classifier using a two-stage fine-tuning approach with Albumentations-based image augmentation**

* Images are resized, augmented, normalized with ImageNet statistics, and converted to tensors, while class imbalance is handled using a weighted binary cross-entropy loss. 
* **In Stage 1**, the pretrained backbone is frozen and only the classifier head is trained, with validation PR-AUC used for early stopping and checkpoint selection. 
* **In Stage 2**, the best Stage 1 model is reloaded and only the final EfficientNet block plus the classifier head are fine-tuned using a lower learning rate for the backbone. 
* The final model is evaluated on the test set using PR-AUC and ROC-AUC.

<img width="305" height="220" alt="image" src="https://github.com/user-attachments/assets/9100bbed-626c-427b-a648-8ee188231d89" />



### **7. STACKING - Convolutional Neural Network (CNN) Model - EfficientNet-B0 with Metadata Probability**

* Here CNN image-based probabilities and metadata-based probabilities are combined by training a logistic regression model on the validation set.
* The stacked model is then evaluated on the test set and compared against the standalone CNN and metadata models to measure performance gains.

<img width="309" height="317" alt="image" src="https://github.com/user-attachments/assets/782d566e-02b4-4df7-a3c4-08acde18b47c" />

### **8. Convolutional Neural Network (CNN) Model - EfficientNet-B0 classifier using a two-stage fine-tuning approach with Albumentations-based image augmentation and imbalance handling with WeightedRandomSampler**

Two-stage fine-tune: (S1) head-only, (S2) last block + head (lower LR)
Imbalance handling : WeightedRandomSampler
Loss : BCEWithLogitsLoss
Stronger Albumentations aug: Affine and color and blur/noise and coarse dropout
<img width="1535" height="542" alt="image" src="https://github.com/user-attachments/assets/96c60b3c-fb53-466b-afaa-c36a534be5a7" />

Best reseults.

## Model Results Summary

| Model executed | Val PR-AUC | Val ROC-AUC | Test PR-AUC | Test ROC-AUC | Imbalance handling | Key parameters / hyperparameters |
|---|---:|---:|---:|---:|---|---|
| Metadata baseline: LogisticRegression | 0.01966 | — | 0.01719 | — | none | n_features=12; train/val/test split |
| Metadata tuned: LogisticRegression + GridSearchCV | 0.02533 | — | 0.01756 | — | implicit via tuning | best_cv_pr_auc=0.31942; C=0.01; penalty=l2; solver=liblinear |
| Metadata baseline: HistGradientBoostingClassifier | 0.01839 | — | 0.01783 | — | none | one-hot encoding + median imputation |
| CNN: EfficientNet-B0 (head-only, torchvision aug) | 0.09030 | 0.8289 | — | — | pos_weight=54.63 | IMG=224; batch=32; epochs=10; patience=2; freeze backbone; train head; ckpt=effb0_head_only_best.pt |
| CNN: EfficientNet-B0 (head-only, Albumentations aug) | 0.09340 | 0.8522 | — | — | pos_weight=54.63 | IMG=224; batch=32; epochs=10; patience=2; stronger augmentation; ckpt=effb0_head_only_best_albu.pt |
| CNN: EfficientNet-B0 (2-stage fine-tune, Albumentations) | 0.10043 | 0.8561 | — | — | pos_weight=54.63 | Stage1=head-only; Stage2=last block + head; early stopping; ckpt=effb0_stage2_lastblock_best.pt |
| Stacking: LogisticRegression on (CNN prob + metadata prob) | — | — | 0.10762 | 0.8526 | class_weight=balanced | trained on validation stack features; evaluated on test |
| **Best CNN: EfficientNet-B0 (2-stage, strong Albumentations + sampler)** | **0.11283** | **0.8610** | **0.12724** | **0.8659** | WeightedRandomSampler | Stage1=head-only; Stage2=last block + head; loss=BCE; ckpt=effb0_albu_s2_stage2_lastblock_best.pt |


*(Each run took a long time to complete. Some did not complete and so the cells have been left blank)*

### The Best model performance was by Convolutional Neural Network (CNN) Model - EfficientNet-B0 classifier using a two-stage fine-tuning approach with Albumentations-based image augmentation and imbalance handling with WeightedRandomSampler

---

