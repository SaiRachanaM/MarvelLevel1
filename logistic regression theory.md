# Logistic Regression for Heart Disease Prediction

## 1. Problem Statement

The goal of this task is to predict the presence of heart disease using Logistic Regression. Since the output is binary (disease or no disease), this is a **binary classification problem**.

-   `1` → Heart disease present
    
-   `0` → No heart disease
    

----------

## 2. Why Logistic Regression (and not Linear Regression)

Linear Regression can produce values outside the valid probability range:

`y = 2.3 , y = -1.5` 

However, for classification we require:

`0 ≤ y ≤ 1` 

Logistic Regression solves this by:

1.  Computing a **linear score**
    
2.  Passing it through a **sigmoid function** to convert it into a probability
    

----------

## 3. Logistic Regression Model

### 3.1 Linear Score

The model computes a weighted sum of input features:

`z = w_age × age + w_bp × bp + w_chol × cholesterol + b` 

Each feature has its own weight:

-   Large weight → feature strongly influences prediction
    
-   Small weight → feature has less impact
    

The bias term `b` represents the baseline risk.

----------

### 3.2 Sigmoid Function

The sigmoid function converts the score into a probability:

-   Probability ≥ 0.5 → predict **heart disease**
    
-   Probability < 0.5 → predict **no heart disease**
    

This allows the model to output meaningful probabilities instead of hard decisions.
![https://media.licdn.com/dms/image/v2/D4D12AQGIXdSG7IJCNw/article-cover_image-shrink_600_2000/article-cover_image-shrink_600_2000/0/1694183259537?e=2147483647&t=lJ_qEzot0iGYhNpez9XGRNHjS-CDKHn3Wj-6iCQxRO0&v=beta](https://media.licdn.com/dms/image/v2/D4D12AQGIXdSG7IJCNw/article-cover_image-shrink_600_2000/article-cover_image-shrink_600_2000/0/1694183259537?e=2147483647&t=lJ_qEzot0iGYhNpez9XGRNHjS-CDKHn3Wj-6iCQxRO0&v=beta)

----------

## 4. Learning Process

### 4.1 Loss Function (Binary Cross-Entropy)

To learn, the model needs feedback on how wrong it is.

Binary Cross-Entropy Loss:

-   Low loss → confident and correct prediction
    
-   Very high loss → confident but wrong prediction
![](https://i.imgur.com/O2ZSy6d.png)    

This is especially important in medical problems, where incorrect confident predictions can be dangerous.

----------

### 4.2 Gradient Descent

Each weight is updated separately based on its contribution to the error:

`w_age ← w_age − α × gradient_age
w_bp          ← w_bp − α × gradient_bp
w_cholesterol ← w_cholesterol − α × gradient_cholesterol` 

This allows the model to automatically learn which features matter more.
![enter image description here](https://i.imgur.com/UG3ZhxW.png)

----------

## 5. Machine Learning Workflow 

The correct pipeline followed is:

1.  Load dataset
    
2.  Split into training and test sets
    
3.  Handle missing values
    
4.  Apply feature scaling
    
5.  Train the model
    
6.  Evaluate performance
    

This order is critical to avoid **data leakage**.

----------

## 6. Feature Scaling

Without scaling:

-   Age ranges from 20–80
    
-   Cholesterol ranges from 150–300
    

Larger values dominate gradients and slow down training.

With **standardization**:

`x' = (x − μ) / σ` 

All features are brought to a comparable scale, ensuring stable gradient descent.

----------

## 7. Handling Missing Values

Using `dropna()` removes rows with missing values, which leads to:

-   Data loss
    
-   Smaller training set
    
-   Reduced learning capacity
    

A better approach is:

-   Mean or median imputation
    
-   Applied **after train–test split**
    

----------

## 8. Data Leakage

Data leakage occurs when the model accidentally uses information from the test set during training.

Common causes:

-   Scaling before train–test split
    
-   Imputation using full dataset
    
-   Fitting preprocessing on test data
    

Correct approach:

-   Fit preprocessing only on training data
    
-   Apply the same transformation to test data
    

----------

## 9. `fit_transform()` vs `transform()`
|dataset| what to use |
|--|--|
|training data  | fit_transform() |
|test data|transform()|


Reason:

-   `fit()` learns mean and standard deviation
    
-   Test data must not influence these statistics
    

----------

## 10. Feature Leakage vs Label Leakage

**Feature Leakage**

-   Input feature directly reveals the answer
    
-   Example: diagnosis result used as input
    

**Label Leakage**

-   Target information leaks indirectly
    
-   Example: preprocessing before splitting
    

Both result in unrealistically high performance.

----------

## 11. Evaluation Metrics (Medical Perspective)

Accuracy alone is misleading for medical datasets.

Metrics used:

-   **Accuracy** – overall correctness
    
-   **Precision** – how many predicted positives are correct
    
-   **Recall** – how many actual patients are detected
    
-   **F1-score** – balance between precision and recall
    

Recall is especially important because missing a patient is more harmful than a false alarm.
![enter image description here](https://i.imgur.com/qFqoWxD.png)

![](https://i.imgur.com/gJmp9Ij.png)
![](https://i.imgur.com/2obHxaW.png)
![](https://i.imgur.com/cleLbyc.png)
![](https://i.imgur.com/uPZ8IJC.png)
----------

## 12. Model Results Summary

-   The from-scratch model achieved higher accuracy but lower recall.
    
-   The scikit-learn model achieved significantly higher recall and F1-score.
    
-   This indicates better detection of heart disease cases by the scikit-learn model.
    

In medical screening, **higher recall is preferred**, even at the cost of lower accuracy.

----------

## 13. Why F1-Score Was Initially Low

The low F1-score was mainly due to:

-   Class imbalance
    
-   Conservative decision threshold (0.5)
    
-   Fewer positive samples influencing learning
    

This caused the model to predict “no disease” too often.

----------

## 14. Ways to Improve F1-Score 

### 1. Stratified Train–Test Split

Ensures both training and test sets contain similar proportions of heart disease cases, allowing the model to learn disease patterns properly.

----------

### 2. Threshold Tuning

Instead of using a fixed threshold of 0.5, lowering the threshold (e.g., 0.3–0.4) increases recall and improves F1-score.

----------

### 3. Class Weight Balancing

Assigning higher weight to the minority class penalizes false negatives more heavily and improves recall and F1-score.
