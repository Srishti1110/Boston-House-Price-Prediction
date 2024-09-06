# Boston-House-Price-Prediction
This repository contains two machine learning models focused on predicting Boston house prices. The models utilize different techniques: **Model 1** applies traditional Linear Regression, while **Model 2** incorporates Principal Component Analysis (PCA) to address multicollinearity and improve model stability.

---

#### Dataset Overview

The dataset used in both models is the **Boston Housing Dataset**, originally compiled by the U.S. Census Service and available from the StatLib archive. It consists of 506 records, each representing a housing district in the Boston area, with 14 attributes capturing various socio-economic, geographic, and infrastructural factors that influence house prices.

**Features:**
1. **CRIM**: Per capita crime rate by town.
2. **ZN**: Proportion of residential land zoned for lots larger than 25,000 sq. ft.
3. **INDUS**: Proportion of non-retail business acres per town.
4. **CHAS**: Charles River dummy variable (1 if the tract bounds the river, 0 otherwise).
5. **NOX**: Nitric oxides concentration (parts per 10 million).
6. **RM**: Average number of rooms per dwelling.
7. **AGE**: Proportion of owner-occupied units built before 1940.
8. **DIS**: Weighted distances to five Boston employment centers.
9. **RAD**: Index of accessibility to radial highways.
10. **TAX**: Full-value property tax rate per $10,000.
11. **PTRATIO**: Pupil-teacher ratio by town.
12. **B**: 1000(Bk - 0.63)^2 where Bk is the proportion of Black residents by town.
13. **LSTAT**: Percentage of the population considered lower status.
14. **MEDV**: Median value of owner-occupied homes in $1,000s (target variable).

The target variable (**MEDV**) represents the median home price in each district, while the other 13 variables serve as predictors. The dataset offers a diverse range of features, covering demographic (e.g., CRIM, B, LSTAT), infrastructural (e.g., TAX, RAD), and environmental (e.g., NOX, CHAS) factors, making it well-suited for real estate price prediction tasks.

---

### Model 1: Linear Regression Model

**Objective:**  
This model aims to predict house prices using traditional **Linear Regression**, providing a direct mapping from input features to the target variable.

**Steps:**

1. **Data Preprocessing:**
   - The data is loaded, and columns are renamed for clarity.
   - Missing values are checked, and summary statistics are computed to understand the distribution of the features.
   - An exploratory data analysis (EDA) is performed using visualizations like pair plots and correlation matrices to identify key relationships between features and the target variable (e.g., strong correlations between **RM** and **MEDV**, and **LSTAT** and **MEDV**).

2. **Data Splitting & Normalization:**
   - The dataset is split into training and testing sets using an 80-20 split ratio.
   - **StandardScaler** is applied to standardize the features, ensuring that variables with different scales (e.g., TAX, RM) do not disproportionately influence the model.

3. **Model Training:**
   - A **Linear Regression** model is trained on the standardized training data.
   - The model's coefficients are extracted, providing insights into how each feature influences house prices (e.g., the positive impact of **RM** and the negative impact of **LSTAT**).

4. **Model Evaluation:**
   - Predictions are made on the test set.
   - The model's performance is evaluated using metrics like **Mean Squared Error (MSE)**, **Root Mean Squared Error (RMSE)**, and **R² score**. These metrics provide a quantitative assessment of the model’s predictive accuracy and goodness-of-fit.

5. **Residual Analysis:**
   - A residuals plot is generated to evaluate the distribution of prediction errors, helping identify potential outliers or model deficiencies.
   - A histogram of residuals is plotted to check if the errors are normally distributed, as assumed by linear regression models.

**Advantages:**
- **Interpretability**: The model provides easily interpretable coefficients, allowing us to directly quantify the impact of each feature on house prices.
- **Simplicity**: Linear Regression is computationally efficient and straightforward to implement.

**Limitations:**
- **Multicollinearity**: The model assumes that the predictor variables are not highly correlated, which is often not the case, leading to instability and overfitting.
- **Assumes Linearity**: Linear Regression assumes a linear relationship between the features and the target variable, which may not hold true for all features in this dataset.

---

### Model 2: PCA + Linear Regression Model

**Objective:**  
This model addresses the issue of **multicollinearity** among the predictor variables by applying **Principal Component Analysis (PCA)** before using Linear Regression, aiming for a more robust and stable model.

**Steps:**

1. **Multicollinearity Analysis:**
   - A correlation matrix reveals high correlations between several predictor variables (e.g., **DIS**, **INDUS**, and **AGE**), which can negatively affect model performance.
   - **Variance Inflation Factor (VIF)** is calculated for each feature to quantify the degree of multicollinearity. High VIF values indicate strong correlations that need to be addressed.

2. **Dimensionality Reduction with PCA:**
   - **PCA** is applied to reduce the dataset's dimensionality, transforming the correlated features into a set of **uncorrelated principal components**.
   - This not only mitigates the multicollinearity problem but also reduces the model’s complexity by focusing on the components that explain the most variance in the data.
   - The first few principal components capture most of the variation in the original dataset, ensuring that important information is retained.

3. **Model Training:**
   - After applying PCA, a **Linear Regression** model is trained on the principal components rather than the original features.
   - The model now works with uncorrelated predictors, improving stability and reducing overfitting risks.

4. **Model Evaluation:**
   - Similar evaluation metrics (MSE, RMSE, R² score) are used to assess the model’s performance.
   - The absence of multicollinearity improves the model’s generalization ability, leading to potentially better performance on the test set.

5. **Visualizing Principal Components:**
   - A heatmap is used to visualize the correlation matrix of the principal components, confirming that the transformed features are indeed uncorrelated.

**Advantages:**
- **Handles Multicollinearity**: PCA transforms correlated features into uncorrelated principal components, making the model more stable and reliable.
- **Dimensionality Reduction**: By focusing on the most important components, PCA reduces the complexity of the dataset, leading to faster training times and potentially improved model performance.

**Limitations:**
- **Loss of Interpretability**: The principal components are linear combinations of the original features, making it harder to interpret how specific features influence the target variable.
- **Potential Information Loss**: While PCA retains most of the variance, some information may be lost when transforming the dataset.

---

### Comparative Analysis

| **Aspect**                  | **Model 1: Linear Regression**             | **Model 2: PCA + Linear Regression**     |
|-----------------------------|--------------------------------------------|------------------------------------------|
| **Data Preprocessing**       | StandardScaler normalization applied       | StandardScaler + PCA for dimensionality reduction |
| **Handling Multicollinearity**| Multicollinearity remains an issue         | Effectively eliminated through PCA       |
| **Feature Set**              | Original features used                    | Uncorrelated principal components used   |
| **Interpretability**         | High: Clear coefficients for each feature | Lower: Components are linear combinations of features |
| **Model Stability**          | May suffer due to multicollinearity        | Improved stability and generalization    |
| **Performance**              | Adequate for simple linear relationships   | Better suited for high-dimensional, multicollinear data |
| **Use Case**                 | Suitable for quick insights and easy interpretation | Ideal when multicollinearity impacts model performance |

**Conclusion:**  
Model 1 is ideal for scenarios where interpretability and quick insights are crucial. However, it suffers from multicollinearity, which impacts its performance. Model 2, by incorporating PCA, eliminates multicollinearity, making it better suited for more complex datasets. Although PCA reduces interpretability, it improves model stability and generalization, particularly when dealing with correlated features.
