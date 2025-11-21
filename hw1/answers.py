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


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
