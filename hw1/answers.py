r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**

**1. False. Consider the following split: $|Test| = |D|-1, |Train| = 1$.  We will train the model on a single datum (quite useless) and test on the entire dataset (almost). this will result in a poorly trained model with poor test results.**

**2. False. The test set should only be used to test the models accuracy. The purpose of the test set is to measure how the model performs on unseen data.**

**3. True.**

**4. True. Noise injection could be used to decrease the generalization error of the model.**
"""

part1_q2 = r"""
**Your answer:**

The question describes a data leakage. the test hyper parameters should be tuned on the validation set, not the test set.

"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**
If we allow $\Delta < 0$, we will fail to train a gap next to the decision boundary. The loss function will return 0 for incorrectly classified samples.

$L_i(W) = \sum_{j\neq y_i}max(0,\Delta + w^{\top}_jx_i - w^{\top}_{y_i}x_i)$

If for sample $i$: $|\Delta| > w^{\top}_jx_i - w^{\top}_{y_i}x_i$

Then: $max(0,\Delta + w^{\top}_jx_i - w^{\top}_{y_i}x_i)$

This means our loss function is alot more lenient towards misclassified samples.

"""

part2_q2 = r"""
**Your answer:**

We can see that our model learns the elements of the classes based on their shapes and produces a heatmap constructed from the examples it is given.
In one of the mistakes, it interpreted the digit 5 as a 6, and when comparing this sample to the heatmaps generated for each digit, this example indeed matched the heatmap of 6 more than that of 5, which explains the misclassification.

"""

part2_q3 = r"""
**Your answer:**

We can see that in the training set loss graph, there is a smooth slope that ends in a low value (but not zero!). this means that the learning rate we chose is fitting to the model and data. the model is not missing the minimum because the learning rate is too high (spikes in the training loss graph would indicate this.) and the graph is converging quickly to 0, which means the learning rate isn't too low.

According to the accuracy graph, our model is slightly overfitted because: the validation accuracy and train accuracy are close to each other in value (although the validation accuracy is a little lower). the validation loss is consistently above the train loss which indicates that the model is slightly specializing on its training data, but the validation loss graph isn't increasing which means its not a severe case of overfitting. Overfitting can also be detected by very high train accuracy and very low test accuracy. Our model doesn't show that type of behaviour.
On the other hand, we can tell that the model isn't underfitted because both validation, train and test accuracies are high (above 89%). An underfitted model would produce much lower validation, train and test accuracies. An underfitted model would show much higher losses and the convergence to zero would stop quite early (and not end very close to zero)

"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**

The ideal residual plot shows residuals randomly scattered around zero with no clear pattern. This indicates that the model captures the underlying relationships adequately and that the assumptions of linear regression are satisfied.
The residuals for the model using only the top-5 features exhibit a curving pattern, with non-constant variance, suggesting a failure to capture a non-linear distribution. In contrast, the residuals for the final model after cross-validation, which includes polynomial features and optimized hyperparameters, are more evenly scattered around zero (and closer to it) and show less structure, indicating a improved fit and reduced bias
"""

part3_q2 = r"""
**Your answer:**
adding non linear features allows a model to capture more complex relationships between the input features and the target variable. In linear regression, this means that the model can now fit curves, interactions, and other non-linear patterns that a plain linear model could not. As a result, the model’s predictions often improve, bias is reduced, and the fit becomes more flexible.

1. yes, The linearity refers to the parameters $w_i$ , not the original input features $x_i$ . Therefore, even though the function of the original features may be non-linear, the regression is still linear in the parameters.

2. yes (only continuous ones), according to the polynomial approximation theorem, any continuous function can be approximated arbitrarily well by a polynomial on a bounded domain.

3. The hyperplane in the transformed space corresponds to a non-linear boundary when mapped back to the original feature space. Therefore, the boundary is no longer a simple hyperplane in terms of the original features, but the model is still linear in the transformed feature space.

"""

part3_q3 = r"""
**Your answer:**
# 1.
Joint PDF is: $f(x,y) = 1$ (Uniform variables)

$\mathbb{E}_{x,y}[y-x] = \int_{0}^{1} \int_{0}^{1} |y-x| dx dy$

Because of symmetry:

$\mathbb{E} = 2 \int_{0}^{1} \left[ \int_{0}^{y} (y-x) dx \right] dy$

$\int_{0}^{y} (y-x) \, dx = \left[ yx - \frac{x^2}{2} \right]_{0}^{y} = \frac{y^2}{2}$

$\mathbb{E} = 2 \int_{0}^{1} \frac{y^2}{2} \, dy = 2 \cdot \frac{1}{2} \left[ \frac{y^3}{3} \right]_{0}^{1} = 1 \cdot \frac{1}{3}$

# 2.

We treat $\hat{x}$ as a constant and split the integral in two parts:

$\mathbb{E}_{x}[|\hat{x}-x|] = \int_{0}^{\hat{x}} (\hat{x}-x) dx + \int_{\hat{x}}^{1} (x-\hat{x}) dx$

$\int_{0}^{\hat{x}} (\hat{x}-x) dx = \hat{x}^2 - \int_{0}^{\hat{x}}x dx  = \hat{x}^2 - \frac{\hat{x}^2}{2} = \frac{\hat{x}^2}{2}$

$\int_{\hat{x}}^{1} (x-\hat{x}) dx = \int_{\hat{x}}^{1} x dx - \int_{\hat{x}}^{1}\hat{x}dx = \left[ \frac{x^2}{2} - \hat{x}x \right]_{\hat{x}}^{1} = \frac{1}{2}  - \hat{x} - \frac{\hat{x}^2}{2} + \hat{x}^2 =  \frac{1}{2}  - \hat{x} + \frac{\hat{x}^2}{2}$

$\mathbb{E}_{x}[|\hat{x}-x|] =  \frac{1}{2}  - \hat{x} + \frac{\hat{x}^2}{2} + \frac{\hat{x}^2}{2}$

$\mathbb{E}_{x}[|\hat{x}-x|] = \frac{1}{2} - \hat{x} + \hat{x}^2$

# 3.

The model is linear in the parameters, not necessarily in the original inputs, Multiplying a feature by a constant just rescales the parameter so it does not add new information. Therefore, you can “drop” the scalar when generating polynomial features
"""

# ==============

# ==============
