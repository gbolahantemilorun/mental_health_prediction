# Mental Health Prediction: Data Analysis and Model Development

### Table of Contents

1. [Installation](#installation)
2. [Project Overview](#overview)
3. [File Descriptions](#files)
4. [Instructions](#instructions)
5. [Analysis](#Analysis)
6. [Answers to Assessment Questions](#answers)
7. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

To run the Python scripts and web app, make sure to install the libraries provided in the requirements.txt file.

## Project Overview<a name="overview"></a>

This project explores factors associated with mental illness, specifically depression, through data analysis and machine learning. By examining variables such as lifestyle habits, employment status, family history, and health conditions, we gain insights into correlations with depression. Using a synthetic dataset, we developed predictive models, including Random Forest and XGBoost classifiers, to assess an individual’s likelihood of experiencing mental illness. Techniques like SMOTE and class weighting were applied to manage class imbalance. Although the dataset's synthetic nature limits generalizability, this project lays a foundation for future research using real-world data to improve predictive accuracy and support mental health interventions.

## File Descriptions<a name="files"></a>

### app folder

- **app.py**: This file contains the main script to run the end point.
- **Random_Forest_model.joblib**: Pickle file containing the trained classifier model.
- **XGBoost_model.joblib**: Pickle file containing the trained classifier model.
- **preprocessor_pipeline.joblib**: Pickle file containing the data preprocessing pipeline.

### data folder

- **depression_data.csv**: CSV file containing the depression data.
- **features.csv**: CSV file containing the features.
- **target.csv**: CSV file containing the target data.
- **process_data.py**: Python script for processing and cleaning data.

### models folder

- **train.py**: Python script for training the classifier model.

### models folder

- **depression.ipynb**: Notebook containing the experiments.
- **depression.ipynb-Colab.pdf**: Notebook in PDF format.

### mental-health folder (python virtual environment)

## Instructions - Running the Python Scripts and Web App<a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/depression_data.csv data/features.csv data/target.csv`
    - To run ML pipeline that trains classifier and saves
        `python models/train.py data/features.csv data/target.csv`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://127.0.0.1:5000/predict

4. Send a request to the server in this format via Postman:
{
    "Name": ["Christine Barker"],
    "Age": [31],
    "Marital Status": ["Married"],
    "Education Level": ["Bachelor's Degree"],
    "Number of Children": [4],
    "Smoking Status": ["Non-smoker"],
    "Physical Activity Level": ["Active"],
    "Employment Status": [0],
    "Income": [26265.67],
    "Alcohol Consumption": ["Moderate"],
    "Dietary Habits": ["Moderate"],
    "Sleep Patterns": ["Fair"],
    "History of Substance Abuse": [0],
    "Family History of Depression": [0],
    "Chronic Medical Conditions": [1]
}

## Analysis <a name="analysis"></a>

1. **Data Cleaning and Processing**:
   - The dataset was loaded and inspected for missing values and duplicates.
   - Categorical data was encoded.
   - Low-value features were dropped.

2. **Model Training**:
   - In the depression notebook, a machine learning pipeline was created using  `RandomForestClassifier`, and XGBoost. However, the random forest's performance was better.
   - The model was trained on the preprocessed dataset.
   - Cross-validation was used to evaluate the model's performance, resulting in the following metrics:
     - **Accuracy**: 62%
     - **Precision**: 59% (macro avg)
     - **Recall**: 62% (macro avg)
     - **F1-Score**: 60% (macro avg)

3. **Justification for Chosen Metrics**:

    - Given the highly imbalanced nature of the data in this project, it's crucial to choose metrics that adequately address the challenges posed by such an imbalance. Below, I have provided a justification for the selected metrics, specifically tailored to handle the imbalanced dataset in this project.

        - **Precision (per class)**: Precision is crucial to reduce false positives, especially in minority classes like neutral and positive sentiments.

        - **Recall (per class)**: Recall ensures that the model captures most of the relevant instances, reducing false negatives, which is important for comprehensive sentiment detection.

        - **F1-Score (per class and macro average)**: The F1-score, particularly the macro average, balances the performance across all classes, ensuring that the evaluation does not overly favour the majority class (negative). This is essential for an imbalanced dataset to provide a holistic view of the model performance.

        - **Accuracy**: While accuracy gives an overall measure of correctness, it can be misleading in imbalanced datasets. However, it is still useful as a basic metric to gauge the model’s performance. Accuracy indicates a general performance but needs to be supplemented with other metrics for a comprehensive evaluation.

4. **Endpoint Deployment**:
   - Due to the Random Forest model's performance, efficiency, scalability and it's robustness to overfitting, it was chosen as the model to be deployed for this project.
   - A Flask app was developed to allow users to send requests to the server and receive mental illness predictions.

## Answers to Assessment Questions<a name="answers"></a>

## Question 1: Insights from the Data Analysis

### 1. Children and Depression
The analysis shows that individuals without children are more prone to depression than those with children. This was observed through the visual comparison of depression rates between these two groups, indicating a clear difference. This suggests that parental responsibilities or the presence of children may provide a form of emotional support or distraction from depressive symptoms.

### 2. Smoking and Depression
The plots reveal that non-smokers are more prone to depression than both former and current smokers. This counterintuitive finding may be related to the idea that smoking can be a coping mechanism for some individuals, while those who do not smoke might experience higher rates of depression due to other factors not captured by the dataset.

### 3. Physical Activity and Depression
A significant trend observed is that those with sedentary or moderate physical activity levels have higher depression rates than individuals with high levels of physical activity. This emphasizes the protective effect of regular physical activity on mental health, suggesting that an active lifestyle may help mitigate depression.

### 4. Employment Status and Depression
The analysis shows that employed individuals are more prone to depression than those who are unemployed. This could be reflective of the stress, workload, and pressure associated with employment, while unemployed individuals may experience different forms of stress or coping mechanisms that are not captured in the dataset.

### 5. Alcohol Consumption and Depression
The findings suggest that individuals with moderate or low alcohol consumption have higher rates of depression compared to those who drink heavily. This is an unexpected result, and may point to a more complex relationship between alcohol consumption and mental health, where moderate drinking might correlate with other factors that contribute to depression.

### 6. Dietary Habits and Depression
The data shows that individuals with unhealthy and moderate dietary habits are more likely to experience depression than those with healthy dietary habits. This highlights the important role of nutrition in mental health, where poor dietary habits may increase the risk of developing or worsening depression.

### 7. Sleep Patterns and Depression
A key finding is that individuals with poor or fair sleep patterns are more prone to depression than those with good sleep quality. This insight aligns with existing research on the link between sleep disturbances and mental health issues, suggesting that improving sleep quality could be an important factor in preventing or treating depression.

### 8. Substance Abuse History and Depression
The analysis indicates that individuals with no history of substance abuse are more likely to experience depression than those with a history of substance abuse. This could imply that those with a history of substance abuse may have developed coping strategies or interventions that mitigate the effects of depression, whereas those without such a history might face depression for other reasons.

### 9. Family History of Depression and Depression
Interestingly, the plots suggest that individuals with no family history of depression are more prone to depression than those with a family history. This is a surprising finding and may point to the complex nature of depression, where factors like genetic predisposition and environmental stressors play a role in the development of the condition.

### 10. Chronic Medical Conditions and Depression
Finally, the analysis indicates that those without chronic medical conditions have higher depression rates compared to those with chronic medical conditions. This might reflect the impact of living with chronic health issues, where the mental burden of managing a chronic condition could serve as a protective factor against depression, whereas those without such conditions may be more likely to experience depression from other sources of stress.

### Conclusion
In conclusion, the data analysis reveals several noteworthy insights about the factors contributing to depression. These insights underscore the importance of lifestyle, health conditions, and personal history in influencing mental health outcomes. However, given the synthetic nature of the dataset, the findings should be interpreted with caution, as they may not necessarily reflect real-world patterns or causality. Further analysis with real-world data would be necessary to validate these results and refine our understanding of the relationships between these factors and depression.

---

## Question 2: Predictive Model Selection and Justification

### Predictive Model Selection
To predict an individual’s likelihood of suffering from mental illness, we experimented with both a Random Forest Classifier and XGBoost Classifier. The Random Forest model was ultimately chosen as the primary model based on its generally strong performance in classification tasks, its interpretability in terms of feature importance, and its robustness against overfitting, especially when used with class weights to handle class imbalance.

The inclusion of class weighting in the Random Forest allowed for better handling of the expected imbalance in cases of mental illness. XGBoost was used as a comparative model due to its ability to handle imbalanced data and its success in high-dimensional datasets with nonlinear relationships.

### SMOTE and Cross-validation
To improve predictive accuracy, SMOTE (Synthetic Minority Over-sampling Technique) was applied to address class imbalance, creating synthetic examples in the minority class (individuals with a history of mental illness). StratifiedKFold cross-validation was used to further validate results and reduce model variance.

### Variable Selection and Encoding Strategy

#### Included Variables:
- **Age, Income**: Continuous features directly influencing mental health risks.
- **Marital Status, Education Level, Physical Activity Level, Alcohol Consumption, Dietary Habits, Sleep Patterns**: Important lifestyle and social factors known to correlate with mental health.
- **Employment Status, Family History of Depression, Chronic Medical Conditions**: Strong predictors due to socioeconomic and hereditary aspects.

#### Excluded Variables:
- **Name**: Removed as it does not contribute to predictive power and can be treated as a unique identifier rather than an attribute.
- **History of Substance Abuse**: Removed as it doesn't show a significant association with the target variable, removing it can reduce model complexity and potentially improve performance by eliminating noise. All other features with p-values less than 0.05 will be retained as they contribute meaningful information for predicting mental illness history.
- **Income**: Removed as it has a negative relationship with the target.

#### Encoding Strategy:
- **Ordinal encoding** for features with a natural ranking (e.g., education, physical activity).
- **One-hot encoding** for unordered categorical variables (e.g., marital status).
- **Binary encoding** for binary variables such as employment status, history of substance abuse, etc.

These transformations were essential to ensure that the data was in a suitable format for machine learning models while minimizing multicollinearity.

### Model Performance Assessment
The Random Forest and XGBoost models were evaluated based on accuracy, F1 score, and classification report metrics. Hyperparameter tuning for the Random Forest with weighted classes showed improved performance metrics. Final model performance on the test set was:

- **Accuracy**: Approximately 0.61 for the Random Forest model after tuning.
- **F1 Score**: Provided a balanced evaluation of precision and recall, critical for the minority class (mental illness cases).

### Potential Model Biases
- **Class Imbalance**: Despite using SMOTE and class weighting, there remains potential bias due to fewer instances of individuals with mental illness.
- **Socioeconomic Bias**: Variables like income and employment status could introduce socioeconomic bias, disproportionately affecting predictions for lower-income individuals.
- **Selection Bias**: The dataset’s origin may not fully represent the diversity of broader populations, limiting model generalizability across different demographic groups.

---

## Question 3: Limitations and Suggested Improvements

### Limitations of the Chosen Approach:
- **Data Imbalance**: Even with SMOTE and class weighting, the model may still favor the majority class, reducing sensitivity to mental illness prediction.
- **Feature Engineering Constraints**: Current features did not capture all psychological, environmental, or genetic factors relevant to mental illness, which limits the model's effectiveness.
- **Cross-validation Limitations**: Although cross-validation was used, it cannot fully prevent variance if there are unseen patterns in the test set.
- **Interpretability and Transparency**: Random Forests and XGBoost, while effective, are relatively black-box models. This reduces interpretability, especially in healthcare contexts where decisions should ideally be transparent.

### Recommendations for Improvement:
- **Additional Features**: Including psychological assessment scores, stress levels, social network factors, or genetic information could improve predictive power.
- **Alternative Models**: Testing models known for handling class imbalance more naturally (e.g., Gradient Boosting Machines with custom loss functions or Balanced Random Forest) might further improve minority class predictions.
- **Deep Learning with Class Weights**: Neural networks using class weights or specialized architectures like Autoencoders for anomaly detection may better capture complex relationships.
- **Explainable AI**: Implement SHAP or LIME for feature importance analysis in complex models, enhancing transparency and potentially revealing new insights into variable importance.
- **Data Augmentation**: In addition to SMOTE, using a broader range of data sources could help generalize the model across different populations, reducing bias.

These improvements aim to enhance the model’s accuracy, reduce biases, and support better decision-making by increasing interpretability and transparency in mental illness prediction.

## Conclusion

This project explores the relationship between various factors and mental illness, using machine learning techniques to predict the likelihood of depression. While the synthetic dataset has its limitations, it provides useful insights into the features affecting mental health. Further work is needed to improve the model's accuracy and ensure that it generalizes well to real-world data.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to AXA Health for the project idea.