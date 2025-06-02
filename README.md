# Black Carbon (BC) Concentration Prediction

This project focuses on predicting Black Carbon (BC) concentrations using various environmental and air quality parameters. The analysis involves data preprocessing, feature selection, and the implementation of multiple machine learning models to achieve optimal prediction accuracy.

## Dataset

The dataset (`BC-Data-Set.csv`) contains various environmental measurements including:
- BC (Black Carbon)
- N_CPC (Particle Count)
- PM-10, PM-2.5, PM-1.0 (Particulate Matter)
- NO2, O3, SO2, CO, NO, NOX (Air Pollutants)
- TEMP (Temperature)
- HUM (Humidity)

## Features

- Comprehensive data preprocessing and analysis
- Feature selection using Forward Sequential Selection (FSS) and Lasso
- Implementation of multiple machine learning models:
  - Support Vector Regression (SVR)
  - Random Forest
  - Gradient Boosting
  - Feed-Forward Neural Network (FFNN)
- Hyperparameter tuning using GridSearchCV
- Model performance comparison
- Temporal analysis with both shuffled and non-shuffled data splits

## Requirements

```bash
pip install pandas matplotlib seaborn scikit-learn numpy
```

## Project Structure

- `main.ipynb`: Main Jupyter notebook containing the complete analysis
- `BC-Data-Set.csv`: Input dataset

## Key Findings

1. Feature Selection:
   - Forward Sequential Selection (FSS) was used to identify the most important features
   - Different feature combinations were tested to optimize model performance

2. Model Performance:
   - Multiple models were implemented and compared
   - Hyperparameter tuning was performed for each model
   - Performance metrics (R² and RMSE) were used for evaluation

3. Temporal Analysis:
   - Both shuffled and non-shuffled data splits were analyzed
   - Temporal patterns in BC concentration predictions were examined

## Usage

Run the analysis:
   - Open `main.ipynb` in Jupyter Notebook

## Results

The project includes comprehensive visualizations and analysis of:
- Correlation matrices
- Feature importance
- Model performance comparisons
- Learning curves
- Temporal prediction patterns
