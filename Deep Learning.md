# Deep Learning

## Foundations

### Perceptron

The **perceptron** set the foundations for Neural Network models. It can be used to predict two-class classification problems, using a nonlinear activation **sign** function, which outputs $-1$ for negative values, $+1$ for non-negative values. The general form is therefore:

$$
y(x) = sign(w \cdot x + b)
$$

According to this, if a record is classified correctly, we get $y(w\cdot x + b) > 0$, so we'll want to maximize the sum of these quantities.
We obtain a cost function (just the inverse of this) and find the gradient wrt the weights and the bias:

$$
\begin{gathered}
\nabla_{w} L(w, b)=-\sum_{x_{i} \in M} y_{i} \cdot x_{i} \\
\nabla_{b} L(w, b)=-\sum_{x_{i} \in M} y_{i}
\end{gathered}
$$

and the update rule:

$$
\begin{gathered}
w_{i+1}=w_{i}+a \cdot y_{i} \cdot x_{i} \\
b_{i+1}=b_{i}+a \cdot y_{i}
\end{gathered}
$$

where $a$ is the learning rate.
Note that this algorithm is not guaranteed to converge: when the data are not linearly separable, the algorithm will not converge.

### Gradient Descent

We know gradient to be a vector pointing at the greatest increase of a function: combining this concept with a **cost function**, we can find the direction in which this cost is minimized. Parametrising our cost function with respect to the model (whatever it is, a neural network, a perceptron, etc.) we can find the direction in which the cost is minimized, iteratively:

$$
x_{n}^{(i+1)}=x_{n}^{(i)}-a \cdot \frac{\partial f}{\partial x_{n}}\left(x^{(i)}\right)
$$

Remember that normalisation plays a big role in gradient descent: if features are very different in magnitude, the gradient will be dominated by the largest.

### Shallow Neural Networks

A shallow neural network is just a NN with one hidden layer:

```
X1  \
X2   =>  z1 = XW1 + B1 => a1 = Sigmoid(z1) => z2 = a1W2 + B2 => a2 = Sigmoid(z2) => l(a2,Y)
X3  /
```

### Activation functions

The sigmoid was the first activation function, very commonly used at the beginning. It though turns out that it can lead to small updates in the weights. A second fuction is the $tanh$, having a range $[-1,1]$, which, having mean $0$, turns out to be more stable. This, though, still suffers from the same problem the Sigmoid has: when the input is too large or too small, the gradient will be too small as the slope will be around zero. The **ReLU** function is a nonlinear function that has a range $[0,\infty)$ and a mean of $0$, and it solves this problem. It's defined as $ReLU(x) = \max(0,x)$. Generally, when dealing with classification problems, we use the **ReLU** function for the hidden layers and a sigmoid or a softmax for the output layer. Note that we're adding non-linearities to _squash_ the space in order to better find patterns. If we didn't add non-linearities, whatever number of layers we had, we could always simplify to a logistic regressor.

### Random initialization

While in logistic regressors we could initialize the parameters to 0, if we do it now, all the layers will compute the same function, updates will be the same, and the model will be the same. We start with small values, though, as otherwise our activation functions (sigmoid and tanh) would saturate fast.

### Key points

#### Why do we use NumPy?

By using NumPy, we are able to perform operations on vectors, avoiding costly for loops. The NumPy dot product uses vectorization by default, being performing. Reshapes are not costly.

#### What's gradient checking?

Gradient checking allows us to compute the gradient numerically, and compare it to the gradient computed by the backpropagation algorithm. This is useful for debug matters.

## Improving Deep Neural Networks

### Train/Test/Dev

We usually split the dataset into 3 different sets:

- The **training** dataset, the biggest one, which we use to train the model
- The **validation** dataset, which we use to validate the model in order to tune the hyperparameters
- The **test** dataset, which we use to test the model when we reached a _final_ configuration

Usually, these 3 sets are split as $60/20/20$ for $N\le 10^6$, while the training percentage becomes bigger and bigger for larger datasets.

### Bias and variance

The L2 loss allows us to naturally divide our error into two parts: the **bias** and the **variance** (overfitting). The first one is also known as internal variance, meaning an error that is introduced by the learning algorithm, while the second is known as parametric variance, representing the error that is due to the limitedness of available data. We're always dealing with a tradeoff between these two. Generally, when we have high bias, we'll want to make our model more complex (for example, a bigger NN), try a different model, to run it longer. When we have high variance, we'll want more data, some kind of regularization, or a different model.

### Regularization

Regularization is a technique that allows us to reduce the variance. We can cite two different techniques, the **L1** and the **L2**. The L1 regularization is the sum of the absolute values of the weights, while the L2 regularization is the sum of the squares of the weights. The L1 regularization is a good choice when we have a lot of weights, as it forces the model to be more sparse: it tends to take weights to exactly 0, while L2 tends to make them small. Regularization introduces a $\lambda$ hyperparameter.

### Dropout

Dropout is another technique that allows us to reduce the variance. It's a technique that randomly sets some weights to zero, while keeping the rest of the model intact. **Inverted Dropout** is a famous technique for this, which just initializes a vector of 0/1 randomly picked with probability $p$, and multiplies it by the weights. Note that **dropout is not used in the final model**, but rather in the training process: this would introduce useless noise. Notice how dropout avoids neurons to compensate each other. You can even set different $keep_{rate}$ for different layers, if you think a layer is overfitting more than the others. Dropout is used both in forward and backward passes.

### Other techniques to reduce overfitting

Other techniques we can cite are **data augmentation**, **early stopping** and **ensemble methods**.

### Normalizing inputs

Normalizing inputs speeds up the training. To do so, we subtract the mean from every feature, than divide $X$ by the variance. If we don't normalize, our cost function will be deep and dominated by the largest values, forcing us to keep the learning rate low.

### Vanishing/Exploding Gradients

This happens when our gradient becomes very small or very large. Due to the chain rule, we notice that gradients in the first layers of the network can become very small when $W<I$ and very large when $W>I$ (with $I$ identity matrix). A solution to this is known as _He initialization_, a weight initialization technique that initializes the weights to a small value, but with a variance of 1.

## Optimization algorithms

### Mini-batch gradient descent

Instead of computing the updates on the whole dataset, we can just split the dataset into subsets, and compute the updates on each subset. This is called a **mini-batch**. When we've run all the subsets one time, **an epoch** has passed. The cost won't go down at every step perfectly, as the gradient is computed on a subset of the dataset, but in the long run, it will behave as if we had computed the gradient on the whole dataset. When the mini batch size is 1, we're dealing with **Stochastic Gradient Descent**. This is somewhat noisy and the convergence is hard, but it's fast. Mini-batches having a size in the form $2^n$ are computationally optimal.

### Momentum

### Avoiding overfitting

### Dropouts

### Regularization

### Common structures

#### CNN

#### RNN

#### LSTM
