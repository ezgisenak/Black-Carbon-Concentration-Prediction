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

## Machine Learning Models

We implemented and compared multiple regression algorithms using both **raw** and **log-transformed** features:

### 1. **Support Vector Regression (SVR)**
- Kernel: RBF
- Tuned over `C`, `epsilon`, `gamma`, and `kernel`
- Best R²: **0.789**, RMSE: **0.477** (with tuned parameters and FSS features)

### 2. **Random Forest Regression**
- Tuned over `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`
- Best R²: **0.690**, RMSE: **0.579**

### 3. **Gradient Boosting Regression**
- Tuned over `n_estimators`, `learning_rate`, `max_depth`, `min_samples_split`, `min_samples_leaf`
- Best R²: **0.730**, RMSE: **0.540**

### 4. **Feedforward Neural Network (FFNN)**
- Initial architecture: `(64, 32)` with ReLU and Adam optimizer
- Best performance: **R²: 0.796**, RMSE: **0.469**
- Tuned variants explored architectures and learning rates; the original configuration surprisingly remained best

---

## Temporal Generalization

- **Shuffled vs. Non-Shuffled Splits:** 
  - Shuffled splits simulate i.i.d. training conditions
  - Non-shuffled (time-aware) splits simulate forecasting future values
- **Findings:** Performance drops in non-shuffled setup (e.g., FFNN drops from R² 0.796 to 0.739), highlighting the challenge of temporal generalization.

---

## Final Model Comparison

| Model                       | Feature Selection | Tuned | R²     | RMSE   |
|----------------------------|-------------------|-------|--------|--------|
| SVR                        | FSS               | ✔️     | 0.789  | 0.477  |
| Random Forest              | FSS               | ✔️     | 0.690  | 0.579  |
| Gradient Boosting          | FSS               | ✔️     | 0.730  | 0.540  |
| FFNN                       | FSS               | ❌     | 0.796  | 0.469  |
| FFNN                       | FSS               | ✔️     | 0.783  | 0.484  |
| FFNN                       | FSS               | Non-Shuffled | 0.739 | 0.571 |

---

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
