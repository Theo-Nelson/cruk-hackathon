# AI x Cancer Bio Hackathon: Project Repository

## Acknowledgments
This repository contains the work produced during the first-ever **AI x Cancer Bio Hackathon**, held at **Newnham College**. We would like to thank the **sponsors**, **mentors**, and **organizers** who made this event possible:

- **Tanmay Gupta** from **GetSeen Ventures**
- Keynote speaker **Prof Raj Jena**, Dept of Oncology
- Mentors, including **Jonathan Hsu** from **Valence Lab** and **Dr Maxime Allard** from **Helical**
- **Dr Namshik Han**, Head of Computational Research & AI at the Milner Therapeutics Institute

We are grateful for the support of **Newnham College**, which hosted this exciting event.

## Authors
This work was collaboratively produced by:
- **Aiswarya Menon**
- **Alagu Subramanian**
- **Ethan Wong**
- **Jessica Tang**
- **Theodore Nelson**

## Repository Overview
This repository contains the outputs of various analyses, scripts, and visualizations generated during the hackathon. Below is a description of each file and folder for easier navigation.

### Folders and Files

### **IC50_pred_CNV_EXPR_NO_PROTEOMICS**
- **`IC50_prediction_CNV_EXPR_NO_PROTEOMICS_Deep_Models.py`**: Python script implementing deep learning models for IC50 predictions without proteomics data.
- **`IC50_prediction_CNV_EXPR_NO_PROTEOMICS_Standard_Models.py`**: Python script applying standard machine learning models for IC50 predictions without proteomics data.

### **Scatterplots**
This folder contains scatterplot visualizations for model performance:
- **`CNN_scatterplot.png`**: Scatterplot of predictions from a Convolutional Neural Network (CNN).
- **`ElasticNet_scatterplot.png`**: Scatterplot showing ElasticNet regression performance.
- **`Lasso_scatterplot.png`**: Scatterplot displaying Lasso regression predictions.
- **`RandomForest_scatterplot.png`**: Scatterplot of predictions using a Random Forest model.
- **`Ridge_scatterplot.png`**: Scatterplot showing Ridge regression predictions.
- **`RNN_scatterplot.png`**: Scatterplot for Recurrent Neural Network (RNN) predictions.
- **`SimpleNN_scatterplot.png`**: Scatterplot displaying performance of a simple neural network.
- **`SupportVectorRegressor_scatterplot.png`**: Scatterplot for Support Vector Regressor predictions.
- **`XGBoost_scatterplot.png`**: Scatterplot showing XGBoost regression predictions.

### **IC50_pred_CNV_EXPR_WITH_PROTEOMICS**
- **`Deep_Model_NO_Proteomic_Control.png`**: Visualization of predictions from a deep model excluding proteomic control.
- **`Deep_Model_Proteomic_Group.png`**: Visualization of predictions from a deep model including proteomic group.
- **`IC50_prediction_CNV_EXPR_PROTEOMICS_Deep_Models.py`**: Python script implementing deep learning models for IC50 predictions with proteomics data.
- **`merge_IC50_proteome_data.py`**: Script to preprocess and merge IC50 and proteomics datasets.

### **Response_Plasma_Proteomic_pred**
- Folder containing results of predictions based on plasma proteomic response. Specific contents can be explored for detailed analyses.

### **NSCLC_data**
- **`NSCLC_cleaner_data.txt`**: Cleaned dataset containing information about NSCLC patients.
- **`NSCLC_cleaner_metadata.txt`**: Metadata corresponding to the NSCLC dataset.

### **Root-Level Files**
- **`model_comparison_results.csv`**: A CSV file summarizing the performance metrics (accuracy, ROC AUC, etc.) of various machine learning models.
- **`Patient_Response_Based_on_NSCLC_Plasma_Protein_Profiles.py`**: Python script analyzing NSCLC patient response based on plasma protein profiles.
