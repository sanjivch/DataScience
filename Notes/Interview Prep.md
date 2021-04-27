
# 1. Probablility and Statistics
## 1.1. General
1. - **Central Limit Theorem/Tendency**
The Central Limit Theorem states, given a population with mean and standard deviation, the means of large random samples taken from the population with replacement will be approximately normally distributed.
Alternately, the sampling distribution of the sample means approaches a normal distribution as the sample size gets larger — no matter what the shape of the population distribution. This fact holds especially true for sample sizes over 30. 

   - **Law of Large Numbers**
The law of large numbers states when random experiments when repeated for many times, the average of the outcome converges towards the expected value. that as a sample size grows, its *mean* gets closer to the *average of the whole population*.


   - **Under what circumstances**
  
## 1.2. Statistical tests

1. **ANOVA - Analysis of Variance** 
ANOVA is method where we check the means of one or more groups are statistically different.

Terms:
Example: 

5. **What is Mahalanobis distance how is it related to Gaussian distribution**
   Mahalanobis distance is teh distance between the data vectoor and the distribution. It takes into account 

6. **Please shared your views on Bayes Theorem and it’s link to Likelihood estimation**
   Bayes Theorem states that the conditional probability of an event can be estimated from the marginal probablility. Likelihood estimation is P(B|A)

7. **Share an example Hypothesis testing you performed recently, How is it fundamentally similar or different from a. Classification problem?**
   Hypothesis testing is inference related while classification is prediction related. Hupo infere an unkwown truth from an observed data. Classi is predicting given your input data


# 2. Linear Regression
1. **How is correlation similar to or different from covariance**
Covariance tells us the relationship between two variables. Correlation tells us the strength of the relationship. p-value tells us the statistical significance of the correlation. 
Covariance is hard to interpret as the value changes with scale, however, correlation does not change with scale.

6. **What is multi-collinearity? What is its significance?**
Correlation between one or more independent variables. Usually identified by the VIF Variance Inflation Factor needs to be closer to 1

10. **Define Goodness of fit test, discuss cases when it is used**
The goodness-of-fit test is a statistical hypothesis test to see how well sample data fit a distribution from a population with a normal distribution. Put differently, this test shows if your sample data represents the data you would expect to find in the actual population or if it is somehow skewed. Goodness-of-fit establishes the discrepancy between the observed values and those that would be expected of the model in a normal distribution case. Goodness-of-fit tests are statistical methods often used to make inferences about observed values. These tests determine how related actual values are to the predicted values in a model, and when used in decision-making, goodness-of-fit tests can help predict future trends and patterns. 

11. **In a Regression setting, when do you user R-squared vs RMSE**
The RMSE is the square root of the variance of the residuals. It indicates the *absolute fit of the model* to the data–how close the observed data points are to the model’s predicted values. 
Whereas the R-squared is a *relative measure of fit*. As the square root of a variance, RMSE can be interpreted as the standard deviation of the unexplained variance, and has the useful property of being in the same units as the response variable. Lower values of RMSE indicate better fit. RMSE is a good measure of how accurately the model predicts the response, and it is the most important criterion for fit if the main purpose of the model is prediction.
7. **What is regularisation, why is it important. Share regularisation techniques you have used**
Regularization is a technique used to reduce the error by fitting a function appropriatelyon given training set and avoid overfitting. The features which are not adding value to the model are penalized. 
# 3. Logistic Regression

Precision

Recall

Confusion Matrix


1. What is F measure, under what circumstances can we use it in its base form (Will you use HM every time, Precision recall related)
F = HM of Precision and Recall. Precision and Recall have an inverse relation and F score is true indicator of a classifier.

# 4. Decision trees and Random Forest

1. **What is entropy, explain the mathematical formulation**
Entropy is measure of disorder and in the data sense measure of how unorganised/ messy the data is.

4. **How is Random Forest Regressor different from other Regressors. Please explain using its base formulation (Entropy Info gain)**
Random forest is an ensemble of multiple decision trees and the training is performed by random sampling of the decision tree regressors with replacement (known as bootstrap aggregation or bagging). The predictor is an average of the such bagged decision trees, inherently reducing the overfit. Ensemble models usually produce high bias low variance models. Also, does not need the features to be scaled/transformed


# 5. Principal Component Analysis

1. **How do you you go about dimensionality reduction?** PCA is a Dimensionality reduction technique where the data is projected to a new feature space reducing the dimensions.

2. **What if the memory is a constraint?**
Incremental principal component analysis (IPCA) is typically used as a replacement for principal component analysis (PCA) when the dataset to be decomposed is too large to fit in memory. IPCA builds *a low-rank approximation* for the input data using an amount of memory which is independent of the number of input data samples. It is still dependent on the input data features, but changing the batch size allows for control of memory usage.

# 6. Time series

ARIMA

ARMA

# 7. Topic Modeling

Part B


# 8. Machine Learning - General

1. **How is classification fundamentally different from Clustering**
Supervised vs Unsupervised learning

2. **Can a supervised learning setting be thought of as a conditional expectation? If yes, can we say that a supervised learning is inherently biased? Open ended**

3. **Explain Bias Variance tradeoff**
ML models overfit - high variance, low bias
underfit - high bias, low variance 
reasonable fit is where achieve the balance in bias and variance

# 9. Deep Learning - General

1. **What is convolution, please share an example which does not involve images**
   Convolution is a grouping function. In CNNs, convolution happens between two matrices (rectangular arrays of numbers arranged in columns and rows - they are convoluted or multipled) to form a third matrix as an output.
   A CNN uses these convolutions in the convolutional layers to filter input data and find information. Learns about local patterns in the input feature space . CNNs are:
   - **Shift/ space invariant** - Once a certain pattern is learnt, it can be recognized anywhere even if it appears in a new location. Because of this they need fewer examples to train. 
   - In addition to this, they can also learn spatial hierarchies - hence, they can be trained to efficiently lear increasingly complex and abstract visual concepts.

1. **What are LSTMs, how are they different from CNN**
LSTMs (Long Short-Term Memory networks are specialised form of RNNsAn RNN is a neural network with an active data memory, known as the LSTM, that can be applied to a sequence of data to help guess what comes next.With LSTMs, the outputs of some layers are fed back into the inputs of a previous layer, creating a feedback loop.

10. **How would you go about finding a batch size in a deep learning setting?**
    Mini-batch sizes, commonly called “batch sizes” for brevity, are often tuned to an aspect of the computational architecture on which the implementation is being executed. Such as a power of two that fits the memory requirements of the GPU or CPU hardware like 32, 64, 128, 256, and so on.

   Batch size is a slider on the learning process.

   - Small values give a learning process that converges quickly at the cost of noise in the training process.
   - Large values give a learning process that converges slowly with accurate estimates of the error gradient.

11.  **Gradient Descent and types**
    Gradient descent can vary in terms of the number of training patterns used to calculate error; that is in turn used to update the model.

   The number of patterns used to calculate the error includes how stable the gradient is that is used to update the model. We will see that there is a tension in gradient descent configurations of computational efficiency and the fidelity of the error gradient.
    - *Stochastic gradient descent SGD* - is a variation of the gradient descent algorithm that calculates the error and updates the model for each example in the training dataset.
    - *Batch gradient descent* - is a variation of the gradient descent algorithm that calculates the error for each example in the training dataset, but only updates the model after all training examples have been evaluated.
    - *Mini-batch gradient descent* - is a variation of the gradient descent algorithm that splits the training dataset into small batches that are used to calculate model error and update model coefficients.

One cycle through the entire training dataset is called a training epoch. Therefore, it is often said that batch gradient descent performs model updates at the end of each training epoch.


Scenario 2 - Consider a classical scheduled system (ON TIME PERFORMANCE is IMPORTANT)


How do you go about solving such a business problem and which Machine Learning Techniques would you use for the same

# 10. General

1. **Machine Learning vs Deep Learning**
   - Feature Engineering which involves feature extraction and feature selection is absolutely necessary in build robust machine learning models. However, they need human intervention coupled with domain knowledge to extract and select the features. In deep learning, this process is automatic and the neural network automatically learns from the pattern 
  
