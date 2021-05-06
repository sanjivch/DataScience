
# 1. Probablility and Statistics
## General
1. - **Central Limit Theorem/Tendency**
The Central Limit Theorem states that the sampling distribution of the sample means approaches a normal distribution as the sample size gets larger — no matter what the shape of the population distribution. This fact holds especially true for sample sizes over 30. 

   - **Law of Large Numbers**
The law of large numbers states that as a sample size grows, its *mean* gets closer to the *average of the whole population*.


   - **Under what circumstances**
## Statistical tests
1. **ANOVA - Analysis of Variance** 
ANOVA is method where we check the means of one or more groups are statistically different.

Terms:
Example: 

5. **What is Mahalanobis distance how is it related to Gaussian distribution**

7. **Please shared your views on Bayes Theorem and it’s link to Likelihood estimation**

9. **Share an example Hypothesis testing you performed recently, How is it fundamentally similar or different from a. Classification problem?**


# 2. Linear Regression
1. **How is correlation similar to or different from covariance**
Covariance tells us the relationship between two variables. Correlation tells us the strength of the relationship. p-value tells us the statistical significance of the correlation. 
Covariance is hard to interpret as the value changes with scale, however, correlation does not change with scale.

6. **What is multi-collinearity? What is its significance?**
Correlation between one or more independent variables. Usually identified by the VIF Variance Inflation Factor needs to be closer to 1

10. **Define Goodness of fit test, discuss cases when it is used**
R^2 test

11. **In a Regression setting, when do you user R-squared vs RMSE**
The RMSE is the square root of the variance of the residuals. It indicates the *absolute fit of the model* to the data–how close the observed data points are to the model’s predicted values. 
Whereas R-squared is a *relative measure of fit*, RMSE is an absolute measure of fit. As the square root of a variance, RMSE can be interpreted as the standard deviation of the unexplained variance, and has the useful property of being in the same units as the response variable. Lower values of RMSE indicate better fit. RMSE is a good measure of how accurately the model predicts the response, and it is the most important criterion for fit if the main purpose of the model is prediction.
7. What is regularisation, why is it important. Share regularisation techniques you have used

# 3. Logistic Regression

Precision

Recall

Confusion Matrix


6 What is F measure, under what circumstances can we use it in its base form (Will you use HM every time, Precision recall related)
F = HM of Precision and Recall. Presicion and Recall have an inverse relation and F score is true indicatior of a classifier.
# 4. Decision trees and Random Forest
8. What is entropy, explain the mathematical formulation
Entropy is measure of disorder and in the data sense measure of how unorganised/ messy the data is.

4. How is Random Forest Regressor different from other Regressors. Please explain using its base formulation (Entropy Info gain)
builds on randomly generated decision trees and the model is average of the decision trees, inherently reducing the overfit. Also, does not need the features to be scaled/transformed


# 5. Principal Component Analysis
Scenario 1 - Training Data for Classification (5000 cols and 10 mn rows)


A) How do you you go about dimensionality reduction

B) Comment on above if the system has limited memory resources

# 6. Time series

ARIMA

ARMA

# 7. Topic Modeling

Part B


# 8. Machine Learning - General

2 How is classification fundamentally different from Clustering
Supervised vs Unsupervised learning

3 Can a supervised learning setting be thought of as a conditional expectation? If yes, can we say that a supervised learning is inherently biased? Open ended

5 Explain Bias Variance tradeoff
ML models overfit - high variance, low bias
underfit - high bias, low variance 
reasonable fit is where achieve the balance in bias and variance

# 9. Deep Learning - General

8 What is convolution, please share an example which does not involve images

9 What are LSTMs, how are they different from CNN

10 How would you go about finding a batch size in a deep learning setting?


Part C





Scenario 2 - Consider a classical scheduled system (ON TIME PERFORMANCE is IMPORTANT)


How do you go about solving such a business problem and which Machine Learning Techniques would you use for the same


Skills for which Candidates have been evaluated


    Probability Statiscs, NLP, Signal Processing
    Model PP (Autonomous systems)
    LSTMs, GRUs
    CNNs
    SVD
    Fourier transforms and Power transforms
    Word Embeddings, GloVe, NER 
    Seq2Seq models
    Applied Mathematical modeling 
