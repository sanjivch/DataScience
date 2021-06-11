# Bias vs Variance

## Bias Variance Decomposition

Mean Squared Error (MSE) for any test set can decomposed into three fundamental quantities:
1. Variance, $Var(\hat{y_i})$
	- This is the error due to <u>model's excessive sensitivity to small variations in training data</u>
	- Models with high degree of freedom (more features) is likely to have **high variance and overfit the model**
1. Bias, $[Bias(\hat{y_i})]^2$
	- This is the error due to wrong assumptions
	- High bias models **underfit the model**
1. Irreducible Error, $Var(\epsilon)$
	- The error is because of the noise in the data
	- cleaning up data might reduce this error

$$E(y_i - \hat{y_i})^2 = Var(\hat{y_i}) + [Bias(\hat{y_i})]^2 + Var(\epsilon)$$

## Bias variance Trade-off
![[Pasted image 20210521151724.png]]

- Increasing model's complexity -> typically increases the variance $\uparrow$ and reduce the bias $\downarrow$ 
- Decreasing model's complexity -> increases the bias $\uparrow$ and decreases the variance $\downarrow$
- Balancing the model's complexity to balance the bias and variance. Hence, it is called a trade-off. 