# 1. Generalized Linear Models

## 1.1. Linear Regression

- Predict the value of one or more continuous target variables $t$ given the value of a $D$\-dimensional vector $x$ of input variables.
- If there is one input variable/feature/predictor then it called  *Simple Linear Regression*
$$\hat{y} = β_{0} + β_{1}X$$
- More than one features - *Multiple Linear Regression*
$$\hat{y} = β_{0} + β_{1}X_{1} + β_{2}X_{2} +....+ β_{n}X_{n}$$

### 1.1.1. Assumptions
1.   **Linearity**: The relationship between $X$ and the mean of $Y$ is linear. It is also important to check for outliers since linear regression is sensitive to outlier effects.
	<u>How to check</u>: *Scatter plots*
	
2.  **Homoscedasticity**: The variance of residual is the same for any value of $X$.
	(*homo + scedastic* -> meaning the same variance of residuals across the regression line)
	<u>How to check</u>: *Scatter plots*
3.   **Independence**: Observations are independent of each other a.k.a not auto-correlated.
 	<u>How to check</u>: *Scatter plots*
4.   **Normality**: For any fixed value of $X$, $Y$ is [[Distributions#Gaussian Distribution|normally distributed]]
	<u>How to check</u>: *Histogram or a Q-Q-Plot*

Reference : [Duke Univ.](https://people.duke.edu/~rnau/testing.htm)

### 1.1.2. Evaluation Metrics
1. **$R^2$ Statistic** 
	- Defined as $$R^{2} = \frac{TSS - RSS}{TSS} = 1 - \frac{RSS}{TSS} = 1 -\frac{(y_{i}-\hat{y})^{2}}{(y_{i}-\bar{y})^{2}}$$
	-  $R^2$ *measures the proportion of variability* in $Y$ that *can be explained* using $X$.
	- $TSS$ measures the total variance in the response Y , and can be thought of as the <u>amount of variability inherent in the response before the regression is performed</u>
	- $RSS$ measures the amount of variability that <u>is left unexplained after performing the regression</u>
	- Its a relative measure and varies between $0$ and $1$
	
	Related : [[Correlation]]. [[Covariance]]
	
2. **Adjusted $R^2$ Statistic**
	- Defined as $$Adjusted \space R^2 = 1− \frac{RSS(n − 1)}
{TSS(n − d -1)} = 1 -  \frac{(1-R^2)(n-1)}{(n-d-1)}$$
	- the model with the largest adjusted $R^2$ will have only correct variables and no noise variables. Unlike the $R^2$ statistic, the adjusted $R^2$ statistic pays a price for the inclusion of unnecessary variables in the model
3. **RMSE - Root Mean Squared Error**


### 1.1.3. Multicollinearity
- Independent variables are too highly correlated with each other
<u>How to check :</u>
1. Correlation matrix - pandas' `df.corr()`

2. Tolerance – the tolerance *measures the influence of one independent variable on all other independent variables* and  is defined as,
 $$T = 1 – R²$$
$T < 0.1$ -> possible multicollinearity in the data 
$T < 0.01$ -> definite multicollinearity.

3. Variance Inflation Factor (VIF) – the variance inflation factor of the linear regression is defined as 
$$VIF = \frac{1}{1-R²}$$. 
$VIF > 5$ -> possible multicollinearity  
$VIF > 10$ -> definite multicollinearity 

**Ways to fix multicollinearity: **
- centering the data (deduct the mean of the variable from each score)   
- remove independent variables with high VIF values.

### 1.1.4. Regularization
Regularization is a way to reduce [[Bias vs Variance#Bias variance Trade-off|overfitting]]. 
- We reduce the overfitting by constraining the degrees of freedom.  
- Achieved by adding a regularization term to the [[cost function]]
1. Ridge Regression- $\ell_2$ Norm 
	- *Shrinks* the weight of least important features
	- Cost function is defined as $$J(\theta) = MSE(\theta) + \alpha\frac{1}{2}\Sigma^{n}_{i=1}\theta_{i}^{2}$$
1. LASSO Regression-$\ell_1$ Norm
	- *Eliminates* the weight of least important features; outputs a sparse model - useful for [[feature selection]]
	- Cost function is defined as
	$$J(\theta) = MSE(\theta) + \alpha\Sigma^{n}_{i=1}|\theta_{i}|$$
1. ElasticNet Regression
	- Middle ground
	- Cost function defined as 
	$$J(\theta) = MSE(\theta) + r\alpha\Sigma^{n}_{i=1}|\theta_{i} + \alpha\frac{1-r}{2}\Sigma^{n}_{i=1}\theta_{i}^{2}|$$
	- $r$ is called the mix ratio.
	-  When r-> 0, Elastic Net -> Ridge; r->1, Elastic Net -> LASSO
	
	Related : [[Bias vs Variance#Bias variance Trade-off|Bias variance Trade-off]], [[Learning Rate]]
## 1.2. Logistic Regression

Linear regression applied to a [[Activation Functions#sigmoid|sigmoid]]/logit function

