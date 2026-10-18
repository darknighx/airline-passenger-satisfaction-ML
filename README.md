# Airline Passenger Satisfaction Prediction

Predicting airline passenger satisfaction using machine learning, with a focus on feature engineering, cross-validation, and model optimization.

## Project Overview

This project was developed for the [Kaggle Playground Series S6E10 competition](https://www.kaggle.com/competitions/playground-series-s6e10).

The objective is to predict whether an airline passenger is satisfied with their experience based on information such as flight distance, travel class, customer type, and service ratings.

I used **LightGBM** to build a classification model and experimented with feature engineering, hyperparameter tuning, and cross-validation to improve its performance.

## Dataset

The competition provides:

- **Training data:** 699,635 passengers with satisfaction labels.
- **Test data:** 299,844 passengers without satisfaction labels.
- **Target variable:** `satisfaction`
- **Evaluation metric:** ROC-AUC

The dataset includes numerical and categorical features related to passenger demographics, travel details, and airline services.

## Methodology

### 1. Data Preprocessing

- Separated the target variable from the input features.
- Removed the `id` column from model training.
- Converted categorical columns to pandas `category` dtype for LightGBM.
- Used LightGBM's native handling of missing numerical values.
- Applied consistent preprocessing to the training and test datasets.

### 2. Feature Engineering

Created a new feature called `Travel_class`, combining:

- `Type of Travel`
- `Class`

The goal was to capture possible relationships between a passenger's reason for travelling and their cabin class.

I also experimented with flight distance groupings and interactions between distance and travel type. These experiments did not improve performance consistently, so they were not retained in the final feature set.

### 3. Model Training and Validation

I trained a **LightGBM classifier** and evaluated it using **3-fold Stratified Cross-Validation**.

Stratification helped maintain similar target-class proportions across the folds.

I used out-of-fold (OOF) predictions to calculate an overall ROC-AUC score, providing a more reliable estimate of model performance than evaluating predictions on the training data.

### 4. Hyperparameter Tuning

I manually experimented with several LightGBM hyperparameters, including:

- `num_leaves`
- `learning_rate`
- `min_child_samples`
- `n_estimators`

I also implemented early stopping to stop training when validation performance was no longer improving.

Some configurations improved the model's performance, while others had little or no benefit.

### 5. Final Predictions

For the Kaggle submission, I trained three models using stratified cross-validation.

Each model generated probabilities for the competition test dataset. I averaged the predictions from all three models to produce the final submission.

## Results

| Experiment | ROC-AUC |
|---|---:|
| Initial LightGBM model (3-fold OOF) | 0.958215 |
| Improved LightGBM model (3-fold OOF) | **0.958604** |
| Kaggle public leaderboard | **0.95800** |

The final model achieved a public leaderboard ROC-AUC of **0.95800**.

The difference between the OOF score and public leaderboard score also highlighted the importance of evaluating models on unseen data.

## What I Learned

Through this project, I gained practical experience with:

- Building an end-to-end machine learning classification pipeline.
- Handling numerical and categorical features.
- Using stratified cross-validation and out-of-fold predictions.
- Understanding and evaluating ROC-AUC.
- Experimenting with feature engineering.
- Tuning hyperparameters and implementing early stopping.
- Averaging predictions from multiple models.
- Preparing and submitting predictions to Kaggle.

One important takeaway was that **adding more features or increasing model complexity does not always improve performance**. Testing changes systematically and comparing validation results helped me understand which approaches were useful.

## How to Run

1. Download the notebook from this repository.
2. Open it in Kaggle or another compatible Jupyter environment.
3. Obtain the dataset from the [competition page](https://www.kaggle.com/competitions/playground-series-s6e10).
4. Ensure the dataset paths match those used in the notebook.
5. Install the required Python packages if necessary: `pandas`, `numpy`, `scikit-learn`, and `lightgbm`.
6. Run the notebook cells in order.

The notebook generates a `submission.csv` file containing predicted satisfaction probabilities for Kaggle's test dataset.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- LightGBM
- Jupyter Notebook / Kaggle Notebooks

## Competition

[Kaggle — Playground Series S6E10](https://www.kaggle.com/competitions/playground-series-s6e10)
