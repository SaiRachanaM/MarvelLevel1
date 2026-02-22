# Decision Tree for Fair Recruitment Analysis (ID3 Algorithm)

----------

![desicion tree](https://i.imgur.com/TWq4trD.png)

## 1. Problem Statement

The goal of this task is to build a **Decision Tree classifier using the ID3 algorithm** to predict whether a candidate should be hired.

Since the output is binary:

-   `1` → Hire
    
-   `0` → Do not hire
    

This is a **binary classification problem**.

In addition to prediction, we must analyze whether the model makes **fair decisions across demographic groups (Gender and Age)**.

----------

## 2. Why Decision Tree?

Decision Trees are:

-   Interpretable
    
-   Rule-based
    
-   Easy to visualize
    
-   Suitable for categorical data
    

Unlike Logistic Regression (which learns linear boundaries), Decision Trees:

-   Split data step-by-step
    
-   Create rule-based decisions
    

Example rule:

If test_result >= 2.0 → Hire  
Else → Do not hire

----------

## 3. Core Idea of ID3

ID3 builds a tree by:

1.  Measuring impurity (Entropy)
    
2.  Trying all possible features
    
3.  Selecting the feature that reduces impurity the most
    
4.  Recursively repeating the process
    

----------

# 4. Entropy (Measuring Disorder)

Entropy measures how mixed the data is.

### Formula:

![](https://i.imgur.com/7w3Ot0y.png)

Where:

-   pi = probability of class i
    
-   c = number of classes
    

----------

## 4.1 Example Calculation

Suppose:

100 candidates:

-   60 hired
    
-   40 not hired
    

![](https://i.imgur.com/Xm7GZfK.png)

Entropy close to 1 → highly mixed.

Goal of tree: Reduce entropy

----------

# 5. Information Gain

Information Gain measures how much entropy decreases after a split.

### Formula:

![](https://i.imgur.com/isHu1FB.png)

Where:

-   A = feature
    
-   Sv= subset after split
    

----------

## 5.1 Numerical Example

Split by test_result:

### Before Split:

Entropy = 0.97

### After Split:

Group 1:

-   50 hired
    
-   10 not hired
    

Entropy ≈ 0.65

Group 2:

-   10 hired
    
-   30 not hired
    

Entropy ≈ 0.81

![](https://i.imgur.com/Dv3GYHI.png)

Higher IG → better feature.

----------

# 6. Recursive Tree Construction

ID3 repeats:

1.  Compute entropy
    
2.  Compute IG for each feature
    
3.  Select best feature
    
4.  Split data
    
5.  Repeat on subsets
    

Stopping conditions:

-   All samples same class
    
-   Maximum depth reached
    
-   Minimum samples per split reached
    

----------

# 7. Model Evaluation

1. Accuracy
2. Precision
3. Recall
4. F1 - Score

----------

# 8. Feature Importance

Feature importance = total information gain contributed.

If:

-   test_result → 0.30
    
-   gender → 0.25
    
-   age → 0.15
    

Then test_result is most influential.

If gender has high importance → possible bias indicator.

----------


# 9.1 Demographic Parity

![](https://i.imgur.com/FkG0ADa.png)


----------

### Example:

Male:

-   60 total
    
-   42 predicted hired
    

=42/60=0.70= 42/60 = 0.70=42/60=0.70

Female:

-   40 total
    
-   12 predicted hired
    

=12/40=0.30= 12/40 = 0.30=12/40=0.30

Since:

0.70≠0.30

Demographic parity violated.

----------

# 9.2 Equal Opportunity

![](https://i.imgur.com/iWqR9Mn.png)
----------

### Example:

Qualified males = 50  
True positives = 40

=40/50=0.80= 40/50 = 0.80=40/50=0.80

Qualified females = 20  
True positives = 10

=10/20=0.50= 10/20 = 0.50=10/20=0.50

Since:

0.80≠0.50

Equal opportunity violated.

----------

# 10. Why Bias Happens

Decision Tree optimizes: Maximize Information Gain

It does NOT optimize: Fairness

If historically:

-   More males were hired
    
-   Older candidates rejected
    

Then splitting by gender or age may reduce entropy.

Thus, tree learns historical bias.

----------


# 11. Responsible Considerations

To reduce unfair discrimination:

-   Limit tree depth
    
-   Remove sensitive features
    
-   Monitor fairness metrics
    
-   Balance dataset
    
-   Apply fairness constraints
    

Accuracy alone is insufficient.

----------

# 12. Final Insight

Decision Tree minimizes: Entropy

Fairness requires controlling:

P(Y^∣group)

These are separate objectives.

A model can be:

-   Accurate but unfair
    
-   Fair but less accurate
    

The goal is balance.

----------


    


