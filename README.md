# **Airline Passenger Referral Program Development with Classification Techniques**  

## **Overview**  
Airlines operate in a highly competitive market where retaining existing customers and acquiring new ones are critical for growth. This project focuses on solving one such challenge: predicting whether a referred passenger will book a flight. Using machine learning classification techniques, we analyzed historical passenger data to uncover the key drivers behind successful referrals.  

The results from this project empowered airlines to:  
- Improve their referral success rate by targeting high-potential customers.  
- Enhance customer satisfaction by addressing specific service gaps.  
- Drive long-term growth through data-informed decision-making.  


## **Objective**  
The primary objectives were:  
1. To build a robust classification model to predict referred passenger bookings.  
2. To provide actionable insights into what influences customer recommendations and referral success.  
3. To identify service areas for improvement based on data trends and customer behavior.


## **Dataset Summary**  
The dataset contains **131,895 entries** from airline reviews (2016–2019). It includes passenger ratings, travel details, and review text.  

### **Key Features**  
- **Customer Experience**: Seat Comfort, Cabin Service, Food & Beverage, Entertainment, Ground Service.  
- **Demographics**: Traveler Type (Solo, Couple, Family), Travel Class (Economy, Business, etc.).  
- **Flight Details**: Airline, Aircraft, Route, Date Flown.  
- **Target Variable**: Recommendation (whether the passenger recommended the airline to others).  

### **Notable Observations from Data Exploration**  
- **Traveler Type Distribution**:  
  - **37%** were solo travelers (key for referrals).  
  - **28%** traveled as couples, with higher satisfaction ratings.  
- **Class Preference**:  
  - **78%** chose economy, suggesting that economy-class services significantly impact overall satisfaction.  
- **Service Ratings**:  
  - **50%** of passengers rated cabin service 4 or 5 (positively influencing referrals).  
  - **30%** expressed dissatisfaction with entertainment services.  



## **Pipeline Architecture**  

```plaintext

Data Ingestion  
  |-- Load raw dataset (CSV) using Pandas  
  |-- Verify dataset schema and inspect data (data types, size, etc.)  
  |-- Identify missing values and outliers  
  |  
  v  
Data Cleaning  
  |-- Handle missing values:  
  |     - Drop rows with critical missing data (e.g., Overall Rating)  
  |     - Impute other missing values (e.g., median for Cabin Service)  
  |-- Handle duplicates (if any)  
  |-- Standardize date columns (e.g., convert to consistent datetime format)  
  |-- Remove unnecessary columns (e.g., Reviewer Name)  
  |  
  v  
Data Exploration  
  |-- Perform exploratory data analysis (EDA):  
  |     - Visualize distributions (e.g., histograms for Seat Comfort)  
  |     - Examine correlations (heatmap of numeric features)  
  |     - Analyze class imbalance in target variable (Recommendation)  
  |-- Generate key insights for business understanding:  
  |     - High dissatisfaction with entertainment service  
  |     - Solo travelers contribute the most to referrals  
  |  
  v  
Feature Engineering  
  |-- Encode categorical variables (e.g., Traveler Type, Cabin Class):  
  |     - Use One-Hot Encoding for nominal variables  
  |-- Extract features from text (Customer Review):  
  |     - Apply TF-IDF vectorization for sentiment analysis  
  |-- Normalize/scale numerical features (e.g., ratings, overall score)  
  |-- Drop features with low correlation to target variable  
  |  
  v  
Data Splitting  
  |-- Split dataset into training, validation, and test sets:  
  |     - 70% training, 15% validation, 15% testing  
  |-- Stratified sampling to preserve class distribution in all splits  
  |  
  v  
Model Selection  
  |-- Train multiple classification models:  
  |     - Logistic Regression  
  |     - Decision Tree  
  |     - Random Forest  
  |     - Support Vector Machine (SVM)  
  |     - K-Nearest Neighbor (KNN)  
  |-- Perform initial evaluation using:  
  |     - Accuracy  
  |     - Precision, Recall, F1-Score  
  |     - ROC-AUC Score  
  |  
  v  
Hyperparameter Tuning  
  |-- Use Grid Search with Cross-Validation for the top-performing models:  
  |     - Logistic Regression: Regularization strength (C)  
  |     - Random Forest: Number of trees, max depth  
  |     - SVM: Kernel type, regularization  
  |-- Optimize for highest validation F1-Score and accuracy  
  |  
  v  
Model Evaluation  
  |-- Evaluate final models on the test set using key metrics:  
  |     - Accuracy  
  |     - Precision, Recall, F1-Score  
  |     - Confusion Matrix analysis (e.g., false positives/negatives)  
  |-- Compare models based on:  
  |     - Training speed and inference time  
  |     - Interpretability (Logistic Regression preferred)  
  |-- Select Logistic Regression as the final model for deployment  
  |  
  v  
Feature Importance Analysis  
  |-- Analyze feature importance from Logistic Regression coefficients  
  |-- Identify most impactful features:  
  |     - Overall Rating, Value for Money, Seat Comfort  
  |  
  v  
Deployment Preparation  
  |-- Save trained model as a serialized file (e.g., Pickle or Joblib)  
  |-- Prepare input-output scripts:  
  |     - Input: New passenger referral data  
  |     - Output: Prediction (Will the referred passenger book?)  
  |  
  v  
Post-Deployment Monitoring  
  |-- Deploy model (e.g., Flask or FastAPI endpoint) for real-time predictions  
  |-- Monitor performance using live data:  
  |     - Accuracy drift  
  |     - Changes in customer behavior  
  |  
  v  
Business Impact Analysis  
  |-- Analyze model outcomes on referral success rates  
  |-- Measure ROI for targeted marketing campaigns  
  |-- Identify areas for service improvement (e.g., entertainment dissatisfaction)  
  |  
  v  
Future Enhancements  
  |-- Incorporate new features (e.g., passenger demographics)  
  |-- Use ensemble methods for further improvement (e.g., XGBoost)  
  |-- Retrain periodically with updated data

```

### **1. Data Ingestion and Cleaning**  
- **Input**: CSV dataset loaded using Pandas.  
- **Cleaning Steps**:  
  - Removed rows with missing values in critical features (e.g., Overall Rating).  
  - Imputed values where feasible (e.g., median rating for cabin service).  
  - Extracted useful features from text-based reviews (e.g., sentiment score).  

**Why Remove Missing Values?**  
Critical features like "Overall Rating" and "Value for Money" are strong predictors of customer satisfaction. Retaining incomplete rows could introduce noise and bias into the model, reducing prediction accuracy.  



### **2. Feature Engineering and Selection**  
- **One-Hot Encoding**: Encoded categorical variables like "Traveler Type" and "Cabin Class."  
- **Sentiment Analysis**: Processed free-text reviews to extract sentiments using TF-IDF vectorization.  
- **Feature Selection**: Retained features with high correlation to the target variable.  

**Key Features Identified**:  
1. Overall Rating  
2. Value for Money  
3. Seat Comfort  
4. Cabin Service  
5. Food & Beverage  

These features emerged as the most influential in determining referrals based on their weights in the Logistic Regression model and feature importance in ensemble models like Random Forest.


### **3. Model Development**  
We explored multiple classification algorithms to identify the best fit:  

#### **Models Evaluated**  
1. **Logistic Regression**  
   - Pros: Highly interpretable, fast, and effective for linearly separable data.  
   - Result: Delivered **~90% accuracy** with proper regularization to avoid overfitting.  
2. **Random Forest**  
   - Pros: Robust against overfitting; provides feature importance.  
   - Result: Marginally higher accuracy but computationally expensive.  
3. **Support Vector Machine (SVM)**  
   - Pros: Effective for high-dimensional data.  
   - Result: Slightly better accuracy than Logistic Regression but slower inference.  
4. **Decision Tree**  
   - Pros: Simple and interpretable.  
   - Result: Prone to overfitting; less effective on this dataset.  
5. **K-Nearest Neighbors (KNN)**  
   - Result: Performed poorly due to high dimensionality.  


**Why Logistic Regression?**  
- Logistic Regression outperformed other models in terms of interpretability and training efficiency.  
- Regularization (L2 penalty) helped handle multicollinearity and improved generalization.  
- Despite SVM's marginally better accuracy, its higher computational cost made it less practical for production deployment.  



### **4. Model Evaluation and Hyperparameter Tuning**  
- **Cross-Validation**: Used Stratified K-Fold to ensure balanced class representation.  
- **Hyperparameter Optimization**: Grid Search tuned hyperparameters like regularization strength for Logistic Regression and tree depth for Random Forest.  

#### **Final Model Performance**  
- Logistic Regression achieved:  
  - Accuracy: **90%**  
  - F1-Score: **0.89**  
- Insights from coefficients confirmed the importance of "Overall Rating" and "Value for Money."  


## **Setup Instructions**  

1. **Clone the Repository**  
   ```bash
   git clone <repository-link>
   cd Airline-Passenger-Referral-Program
   ```

2. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Preprocessing Scripts**  
   ```bash
   python preprocess_data.py
   ```

4. **Train the Model**  
   ```bash
   python train_model.py
   ```

5. **Run Predictions**  
   ```bash
   python predict.py --input <data_file>
   ```


## **Future Work**  
1. **Real-Time Integration**:  
   - Deploy the model as an API for real-time referral likelihood scoring.  

2. **Advanced Techniques**:  
   - Experiment with ensemble methods like XGBoost or Stacking to improve predictions.  

3. **Data Expansion**:  
   - Incorporate demographic and geographic data for a more personalized approach.  

4. **Continuous Learning**:  
   - Periodically retrain the model to adapt to changing customer preferences.  

