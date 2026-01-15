# Berkeley_MLAI_Capstone
Berkeley Capstone project - Using Convolutional Neural Networks (CNN) to predict skin cancer

 UC Berkeley ML/AI Capstone project: What Drives the Price of a Car?
( Data analysis and visualization with python, pandas, dataframe, matplotlib, seaborn, sklearn, StandardScaler, PolynomialFeatures, 
OneHotEncoder, train_test_split, GridSearchCV, LinearRegression, Ridge, Lasso, RidgeCV, LassoCV, ElasticNetCV, mean_absolute_error, mean_squared_error, r2_score, SequentialFeatureSelector )


Project location: https://github.com/parthabiswas1/UCBerkeley-project-whatDrivesPriceOfCar 

## Problem Statement

### Understand why some used cars sell at a higer price, compared to others.
There are many factors like age, manufacturer, model, condition, odometer reading, fuel type, transmission, and drive that influence a used car buyer's decision to buy a car at a particular asking price. We need to find a way to identify the most important factors and their weightages that drive used car pricing so that the dealership can better price the car for quick sell.

### Objective of this project
Identify the most important factors that detrmine used car prices so the dealership can better price their vehicles and understand consumer preferences.

## Source of data

The dataset was provided by UC Berkeley School of Engineering and UC Berkley Haas school of Management for assigment for Professional Certificate in Machine Learning and Artificial Intelligence

## Approach

- Exploration ofthe data and formulation of data cleaning strategy
- Execution of the data cleaning strategy
- Creating a Modeling Plan
- Execution of the Models
- Evaluation of the Models
- Recommendatons to Car Dealers
  
Below are a summary of observations and findings.

## Data exploration results ** vehicles ** dataset

### Sample from original dataset
![Original Dataset](images/original_data.png)

1. Did basic inspection of data - head(), dtypes, shape, describe()

2. Calculated missing counts percentages with isnull().sum() and identified columns that can be dropped because high % of data is missing. **size** qualifies as **70%** of size is missing.

3. VIN does not contribute much to the buying decision and 37% of VIN is missing. Good candidate to be dropped.

4. Decided that 'year' (more like age), 'odometer' readings are critical to buying decision. Only 0.28% of rows with 'year' are null and 1.03% of 'odometer' are null. These are small numbers and dropping these rows will ensure that all remaining rows have these values critical for the model to predict future price.

5. Looked for data outliers.   

   a) Cars older than 1950. Many of these cars have **placeholder price of $1.**.

   b) many have fake odometer readings (1, 10, 100 etc)     
   a)and b) together is **1213 rows only**, so decided to drop them.    

   c) There were **1386 rows** with odometer reading of 500K miles or more. Decided to drop them as they can skew the data.  

   d) Incorect pricing - Less than 100 and more than $200K. These were about 8.5% of the rows. Since these also can distort the model performance, decided to drop them.

## Data cleaning execution
![Data after cleaning](images/data_after_cleaning.png)
(Data after cleaning)

1. **Data imputation** - All nulls in non numeric features are replaced with 'unknown'

2. **Drop Features (columns):**

   A. **id** - as this add no value to the car value.

   B. **size** - though size matters to a car value, **70%** of the data is missing.  

   C. **region and model** - Each contains too many elements, very difficult to convert to 'one hot encoding'.   

   D. **VIN** - adds no value to car value.  

   E. **state** - Though state is important (cars that sell in Texas more many be different from what sells in California) however with 'one hot encoding' it will explode the columns, so dropping it for how.

3. **Drop Observations (rows):**

   A.  **Year, Odometer** - These are critical and influences the price of used car, and the null rows are very few (Year - 0.28% and Odometer 1.03%). Dropping them will ensure that this data is avaliable in all remaining rows and help the performance of the models.  

   B. **Outliers** -
   
      i) Drop any vehicle older than 1950 and younger than 2025   

      ii) Odometer readings in negative or greater than 500K.

      iii) Price less than 100 or more than 200K

4. **Feature Engineering** -

   A. **Ordinal Encoding** -

     i) **Condition** has meaning- Encoded "salvage": 0, "fair": 1, "good": 2, "excellent": 3, "like new": 4, "new": 5, "unknown": -1

     ii) **title_status** has a clear hierarchy : "parts only": 0, "salvage": 1, "rebuilt": 2, "lien": 3, "unknown": 3, "missing": 3, "clean": 4. (gave same weightage to lien, unkown, missing )

   B. **One-hot encoding**
     
     fuel, transmission, drive, paint_color, type, manufacturer.


   C. **Feature creation**
   
     i) Created a column called **car_age**. Everyone wants to know how old is the car and the age is important to the buying decision.  

     Drop **year**, no more needed

   D **Scaling** Normalized numerical Features odometer, car_age.

(Note: I did not do PCA because I will be doing Ridge and LASSO. Ridge handles multicollinearity (features highly corelated to each other) by shrinking correlated coefficients. LASSO adds both shrinkage and feature selection. PCA removes collinearity but basically redundant in this scenario, it is also hard to interpret the data.)

![Price Distribution of vehicles after cleaning](images/vehicle_price_dist.png)

Histogram has a right scewed shape - as price increases count decreases. Main cluster is around 5k - 20k. Long tail repesents expensive/luxury cars. Looks good. Ready to create model 

## **Modeling Plan**

**Objective:**
We want to understand what drives used car prices and build a model that can reasonably predict them.

1. I will start with a baseline **Linear Regression** model and establish a benchmark of MAE (Mean Squared Error, RMSE (Root Mean Squared Error) and R^2 (Coeff of Determination, If 1 then perfect fit, If 0= Model did not understand anything :-) , a -ve value will be worse than baseline)

2. I will do **Polynomial & Interaction** Features to expand the datset and capture non linear effects.

3. I will do Regularization: with **Ridge** (adds penalty to shrink coefficients) & **Lasso** (can reduce some coeffs to 0 there by removing features - feature selection) this is the way Lasso will reduce complexity of the model.

4. Next I wil create a pipeline and combine Ridge and Lasso to see if the model performance improves.  

5. After that I will do Sequential Feature Selector to reduce feature sets further if needed.

## **Model Execution Observations**

### Notes: Resource is a huge issue. Many of the Model executions did not complete. Moved to Google Colab and was able to complete some executions. Had to reduce parameters Cross Validation and Max iterations in LASSO to get the execution to complete. Could not club together ridge and lasso in a pipeline as it never completed.

### Strategy I
   Did **baseline LinearRegression** - Training and Test R^2 where close .555 and .558 which meant the model was stable. This also meant the model was only able to explain half the variations in the used car prices.     Test MAE: $ 6480.52, Test RMSE: $ 10011.37. So the predictions were off by quite a bit (comaprison table presented later ).

   After execution of **Ridge** there was not much change. Alpha=1

   After execution of **LASSO** very minimal change Test R^2: 0.555, Test MAE: $ 6479.7, Test RMSE: $ 10012.63

   ![Residual](images/residual.png)\
   I was expecting residuals (errors) to be scattered randomly around 0. What I got was triangle/funnel shape. The residual spread gets wider as predicted price increases. SO errors aren’t constant across price levels. This could mean the model predicts cheaper cars more consistently, but does not do well with expensive ones.

   ![Actual vs Predicted](images/actual_predicted.png)\
   the plot shows heavy clustering at lower prices (0–40k). Looks like the predictions are fairly reasonable in this range, but still spread out. The cloud falls below the diagonal and clustoring below 0 which many mean that the model predicts low prices for more expensive cars.

   ![Coefficients](images/coeff.png)

### Strategy II
   Log-transform target variable Price. Since there is a wide spread, prices are skewed, most are cheap cars but few are very expensive. Log will compress the scale. This will hopefully stabilize variance and makes regression handle extremes better.

![Log Transform](images/log_transform.png) \
Test R^2: 0.443, Test MAE: $ 6447.59, Test RMSE: $ 10414.08. So there was minimal improvement, Model could explain only 44% of the variations, MAE and RMSE came down a bit.

### Strategy III
Explore non linear relationships for 'car_age', 'odometer', 'condition', 'title_status', PolynomialFeatures with degree=2. Polynomial expansion often creates features on very different scales. So standardization is important. Apply StandardScaler and then do RidgeCV and LassoCV to see if there is any improvements

![Log Transform](images/log_transform.png) \
Test R^2: 0.4912195805665599, Test MAE: $ 5949.25, Test RMSE: $ 9878.06


### Strategy IV

Split the dataset into Luxury and Regular cars and try a two model strategy. Did not see any significant improvement

Linear Regression - Luxury Cars R^2 =0.425, MAE  8983.38, RMSE= 16381.599799
Linear Regression - Regular Cars R^2 =0.44, MAE  6384.412946, RMSE= 10275.45
<br>
<br>
<br>


# Why some used cars sell at a higer price, compared to others

##  Used Car Price drivers: Data-Driven Insights for Dealers

### Executive Summary

Using regression analysis on a large dataset of used car listings, I examined what drives vehicle prices in today’s market. My models achieved predictive performance 0.49–0.56, meaning they explain about half of the variation in prices. These results highlight key factors that consistently impact resale value.

The findings confirm that: car age, mileage, condition, brand, and title status are the most important price drivers.

![Top Drivers of Price](images/top_drivers.png)

### Age of the car is the top driver of price
Cars under five years will sell at a premium.

### Odometer is a prime driver of price
Low milage will sell at a premium. Higher odometer readings reduce price consistently.

### Condition & Title Status
Shows as positive drivers of price. A clean title and a car in good condition will sell at a higher price.

### Fuel type

Hybrid seems to be more preferred and a driver of price.


# **Skin Cancer Detection from Dermoscopic Images Using Deep Convolutional Neural Networks (CNN)**

##**Project Objective**

The goal of this capstone is to predict whether a dermoscopic image (skin image) is malignant (skin cancer) using low-resolution images combined with a small set of patient metadata. The emphasis is on building a realistic, reproducible pipeline under practical constraints rather than maximizing benchmark performance.

##**Dataset and Practical Constraints**

This project is based on The International Skin Imaging Collaboration (ISIC) 2020 'Skin Lesion Analysis: Towards Melanoma Detection' dataset (≈33,000 images, ~50GB in full resolution). https://challenge2020.isic-archive.com/

Working with the original high-resolution dermoscopic images was not feasible for this capstone due to storage limits and compute constraints. To address this, I used a pre-resized 224×224 JPG version of the ISIC 2020 dataset available on Kaggle:

https://www.kaggle.com/datasets/nischaydnk/isic-2020-jpg-224x224-resized

**This decision introduced an important trade-off**: This is an extemely difficult dataset as it has extreme imbalance. The model performance is expected to be lower than approaches trained on full-resolution dermoscopy images. This trade-off is intentional due to storage and compute limits as mentioned before.

##**Key Challenges and Design Decisions**

*   **Google Colab instability resolution** Training directly from Google Drive createed a performance and reliability problem because reading thousands of small image files incured high I/O latency and intermittently stalled and disconnected during long runs.  The solution was to copy and unzip the image dataset into the local runtime once, then load metadata, construct file paths, and verify image integrity so training used fast, stable local disk access.

*   **Severe Class Imbalance:** Malignant cases are rare relative to benign lesions. Accuracy calculation is therefore misleading. The model can achieve high accuracy by simply predicting the majority class (Benign or not cancer) most of the time, while completely failing to identify the minority class (malignant or cancer). To properly evaluate performance under imbalance, **PR-AUC (Precision-Recall Area Under Curve)** was used as the primary evaluation metric, with **ROC-AUC (Receiver Operating Characteristic Area Under Curve)** reported as a secondary reference

*   **Leakage Prevention:** There are multiple images per patient in the dataset and the images are highly correlated. To avoid leakage, I have split the dataset strictly at the **patient level** using patient_id, to ensure that no patient appears in more than one of train, validation, or test sets.

*  **Image level prevalance** Malignant images are rare in the dataset, so care had to be taken to ensure that the right proportion of maligant images were in train, val and test datasets.

##**Exploratory Data Analysis (EDA)**

A very limited set of metadata were provided with the images. The following was done:

*   Missing-value and metadata sanity checks
*   **Class imbalance analysis:** The splits had to be carefully done to ensure both benign and malignant images were present in the train, val and test dataset in similar proportions.
*   **Patient-level distribution checks:** Ensured that the train, val , test splits did not contain the same patient's dermascoping images. 

##**Feature Engineeering**

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

##**Models Implemented**

###**1. Baseline Model with metadata: LogisticRegression**

Used patient-level metadata with guaranteed malignant/ benign data within train, val test, data to establish a baseline. this also demonstrates the predictive value of metadata alone.

###**2.LogisticRegression with Hyperparameter tuning - Cross Validation and GridSearch**
Class-weighted Logistic Regression (which up-weights the rare positive class to counter the 1–2% prevalence) is fitted and tuned with **5-fold GroupKFold** by patient_id to avoid same-patient leakage across folds. Then **GridSearch** explores **L1 vs L2 regularization** (sparsity/selective feature use vs smooth shrinkage) and **different regularization strengths** (C) to select the best setting by **cross-validated** PR-AUC.

###**3. Baseline Model with metadata: Gradient Boosted Decision Trees (GBDT) - HistGradientBoostingClassifier**

Trained a metadata-only **HistGradientBoostingClassifier** with one-hot encoded categorical features and median-imputed numeric features. I used HistGradientBoostingClassifier because **Gradient Boosted Decision Trees** can capture non linear relationships and feature interactions that logistic regression often misses. 

###**4. Image-Only Convolutional Neural Network (CNN) Model EfficientNet-B0**

*(EfficientNet-b0 is a convolutional neural network that is trained on more than a million images from the ImageNet database.)*
I used reduced size 224×224 pixel dermoscopic images in all CNN models.

I chose it because it is widely used for image classification, often outperforming much larger models like ResNet with significantly fewer parameters. Pretrained on ImageNet dataset images features that transfer well, so head-only training is a reliable.

Training strategy: Freeze backbone and train only classifier head.

Used (torchvision) for light augmentation (flip and small rotation). 

Imbalance handling: Used pos_weight = neg/pos inside **BCEWithLogitsLoss** to upweight positives (malignant).

Used **AdamW** optimizer during training to update the model’s trainable weights and to reduce the **loss**. 


###**5. Convolutional Neural Network (CNN) Model - EfficientNet-B0 with image aumentaion using Albumentations**

The training pipeline uses **Albumentations** to apply a richer and more diverse set of image augmentations. Each training image is first **resized** to a fixed input size. It is then **randomly flipped horizontally and vertically** to improve invariance to orientation. Next, an affine transformation is applied, which includes **small random translations, scaling, and rotations** of up to ±20 degrees to **increase geometric robustness**.

After the geometric transformations, **color and illumination variations** are introduced through random **brightness and contrast adjustments, hue and saturation shifts, and gamma corrections**. To further improve robustness to image quality variations common in real-world data, the pipeline occasionally applies **mild blurring or Gaussian noise**. Finally, the image is normalized using ImageNet mean and standard deviation and converted into a PyTorch tensor using ToTensorV2

Training strategy is the same as before, freeze backbone and train only classifier head.


###**6. Convolutional Neural Network (CNN) Model - EfficientNet-B0 classifier using a two-stage fine-tuning approach with Albumentations-based image augmentation**

* Images are resized, augmented, normalized with ImageNet statistics, and converted to tensors, while class imbalance is handled using a weighted binary cross-entropy loss. 
* **In Stage 1**, the pretrained backbone is frozen and only the classifier head is trained, with validation PR-AUC used for early stopping and checkpoint selection. 
* **In Stage 2**, the best Stage 1 model is reloaded and only the final EfficientNet block plus the classifier head are fine-tuned using a lower learning rate for the backbone. 
* The final model is evaluated on the test set using PR-AUC and ROC-AUC.

###**7. STACKING - Convolutional Neural Network (CNN) Model - EfficientNet-B0 with Metadata Probability**

* Here CNN image-based probabilities and metadata-based probabilities are combined by training a logistic regression model on the validation set.
* The stacked model is then evaluated on the test set and compared against the standalone CNN and metadata models to measure performance gains.

###**8. Convolutional Neural Network (CNN) Model - EfficientNet-B0 classifier using a two-stage fine-tuning approach with Albumentations-based image augmentation and imbalance handling with WeightedRandomSampler**

Two-stage fine-tune: (S1) head-only, (S2) last block + head (lower LR)
Imbalance handling : WeightedRandomSampler
Loss : BCEWithLogitsLoss
Stronger Albumentations aug: Affine and color and blur/noise and coarse dropout

Best reseults.

##**Going above and Beyond**

I have used the Model weights and Gradio 6.3.0 to create an UI based application in HuggingFace Spaces
**Skin Lesion Risk Demo**
This App

* accepts a skin lesion photo upload
* runs EfficientNet-B0 inference from a saved .pt checkpoint
* shows probability + decision at a chosen threshold
* shows Grad-CAM overlay and heatmap
* provides Q&A using RAG + OpenAI API
