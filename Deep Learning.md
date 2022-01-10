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

Using **exponentially weighted averages**, we're introducing weighting factors that decrease exponentially. This means that the weighting for each older datum decreases exponentially, though never to zero. This is used for the concept of **momentum**: imagine a ball rolling down the gradient slope, it will gain momentum as it goes down.
We are basically averaging the past updates to the parameters, using them to have some kind of momentum.

$$
\begin{aligned}
\text { update }_{t} &=\gamma \cdot \text { update }_{t-1}+\eta \nabla w_{t} \\
w_{t+1} &=w_{t}-\text { update }_{t}
\end{aligned}
$$

This means that in regions having gentle slopes, we're still able to move fast. We can then introduce the **bias correction**, which is a technique that helps us to converge faster and more accurately. This happens because we normally set $V(0)=0$, meaning that the accuracy suffers at the start. Because of this, we instead compute $v(t)$ as:

$$
v(t) = \frac{\beta v(t-1) + (1-\beta)\theta(t)}{1-\beta^t}
$$

**Nesterov Momentum** is a technique that computes the gradient using the previous update, and then uses the current update to compute the next one. This is done to avoid local minima.

**AdaGrad** (Adaptive Gradient) solves a problem that lies in the data: it can happen that features may be dense/sparse, making their updates faster or slower. AdaGrad introduces a different learning rate for each different feature, at each different iteration. It adapts the learning rate to the parameters, performing smaller updates for parameters associated with frequent features, and larger updates for parameters associated with rare features. Basically, the learning rate decays with respect to the update history, accumulating the squared gradients:

$$
\begin{gathered}
G_{i, t}=\sum_{n=0}^{t} g_{i, n}^{2} \\
\theta_{i, t+1}=\theta_{i, t}-\frac{\eta}{\sqrt{G_{i, t}+\epsilon}} g_{i, t}
\end{gathered}
$$

**RMSProp** is used to perform larger updates on the weights, as it uses the square root of the sum of the squared gradients. We know $dW^2$ to be large, and $db^2$ to be small, so we can use this to tweak the updates:

$$
\begin{gathered}
	sdW = (\beta * sdW) + (1 - \beta) * dW^2 \\
	sdb = (\beta * sdb) + (1 - \beta) * db^2\\
    W = W - learning_{rate} * dW / sqrt(sdW)\\
	b = B - learning_{rate} * db / sqrt(sdb)
\end{gathered}
$$

**Adam**, standing for adaptive moment estimation, is a technique that mixes RMSprop and momentum. It computes $vdW$ and $vdb$ as in the momentum technique, $sdW$ and $sdb$ as in the RMSprop technique, fixes the biases and then computes the updates.
It has 3 parameters, being $\epsilon$ the small constant, $\beta_1$ and $\beta_2$ the exponential decay rates for momentum and RMSprop, respectively.
The pseudocode is the following:

```
vdW = 0, vdW = 0
sdW = 0, sdb = 0
on iteration t:
	# can be mini-batch or batch gradient descent
	compute dw, db on current mini-batch

	vdW = (beta1 * vdW) + (1 - beta1) * dW     # momentum
	vdb = (beta1 * vdb) + (1 - beta1) * db     # momentum

	sdW = (beta2 * sdW) + (1 - beta2) * dW^2   # RMSprop
	sdb = (beta2 * sdb) + (1 - beta2) * db^2   # RMSprop

	vdW = vdW / (1 - beta1^t)      # fixing bias
	vdb = vdb / (1 - beta1^t)      # fixing bias

	sdW = sdW / (1 - beta2^t)      # fixing bias
	sdb = sdb / (1 - beta2^t)      # fixing bias

	W = W - learning_rate * vdW / (sqrt(sdW) + epsilon)
	b = B - learning_rate * vdb / (sqrt(sdb) + epsilon)
```

Finaly, we can use **learning rate decay** to reduce the learning rate as we get closer to our optimum. This is done by multiplying the learning rate by a factor $\eta_0$ that decays exponentially. When dealing with neural networks, it's rare to end up in local optima: it's much more possible, though, to end up in plateaus, i.e. regions where the derivative is close to zero for a long time. Momentum algorithms help in these situations.

### Hyperparameter tuning and normalization

Tuning the hyperparameters is a crucial step for any neural network. The order of importance is something along the lines of:

- Learning rate.
- Momentum beta.
- Mini-batch size.
- No. of hidden units.
- No. of layers.
- Learning rate decay.
- Regularization lambda.
- Activation functions.
- Adam beta1, beta2 & epsilon.

Don't tune with a grid, it's better to use a random search and narrow down when we found decent solutions. Furthermore, it's better to search using a logarithmic scale rather than a linear one. We have two approaches for hyperparameter tuning: the **panda** approach, in which we nudge the parameters a little during training, with one training at a time, or the **caviar** technique, running multiple model in parallel and checking the results in the end.

**Batch normalization** is a technique that speeds up learning by normalizing the outputs from neural layers. This is usually done before the activation function, but it can also be done after it. It's usually applied with mini-batches. Note that if we're using this, the bias in the network gets removed when we subtract the mean. The technique works because it reduces the problem to input values changing, regularizing the network and adding some noise similarly to dropout. Bigger batch sizes will reduce this effect. Don't rely on this technique as a regularization, you should still use L2 or Dropout.
When then using the network to predict one single example, we'll need to estimate mean and variance better than computing it on a single element. This is usually managed by the libraries.

## Convolutional Neural Networks

As computer vision is one of the applications that are growing faster, we're interested in building layers that are optimal for 2D images.
Convolutions are the basic operations that we'll use to build CNNs: we're basically shifting the kernel over the image, and then multiplying it by the image.
Notice that with the kernel being shifted, we also need some kind of padding in order to avoid reducing the size of the image over and over. Generally speaking, if a matrix $n\times n$ is convoluted by a $f\times f$ kernel, the result is $n-f+1 \times n-f+1$. Often, the padding value is just composed by zeros. The **same convolution padding** is a padding such that $p = (f-1)/2$. $f$ is usually odd, in order to have a central value. **Strided convolutions** shift the kernel by $s$ pixels. Notice that in order to process RGB pictures, we'll need to be able to process 3D input. This is done by introducing a **stacked convolution**, where we stack the convolutions on top of each other.

In addition to that, we usually use **Pooling layers**, that reduce the size of the data. MaxPooling is the most common one, and it's done by taking the maximum value of the submatrix. AveragePooling takes the average.

### ResNets

ResNets are a family of deep neural networks that are designed to be more efficient than traditional CNNs. They are usually composed of several layers, and the first one is a convolutional layer. The second one is a residual layer, which adds the output of the first layer to the output of the second layer. This is done several times, and the output of the last layer is the output of the network. This is an example of **Skip-connections**: the output of the first layer is added to the output of the second layer, and the output of the second layer is added to the output of the third layer, and so on. These networks can go deeper and deeper without incurring in vanishing gradients.

### Other networks

**Inception** was proposed by introducing multiple convolution kernels in a single step, and letting the algorithm learn the best.

## Sequential models

Many types of data are indeed _sequential_: we need the data that was analysed in the past to understand the current one.

### Notation

In the past section, we'll index the first element of $x$ as $x^{<1>}$, the second as $x^{<2>}$, and so on. $T_x$ is the number of elements in $x$. $x^{(i)<t>}$ is the element $t$ of input vector $i$. Clearly, $T_x^{(i)}$ is the input sequence length of training example $i$.

### Vectorising words

We can build a vocabulary containing all the words in our training set. Often, we just need to represent the most occurring ones: we can add an `<UNK>` token to the vocabulary, and we can use it to represent all the words that are not in the vocabulary.

### Recurrent Neural Networks

Why can't we just use a normal neural network? There are two problems: the inputs and outputs have no standard length, and features are not shared across different positions of the text sequence. The latter means that if I have a word that's repeating ten times, the ten repetitions will be different from each other.
In a RNN, every time we have an output, we can use it as an input for the next time step. This is called a **recurrent layer**. There are 3 weight matrices now:

- The first one is the **input-to-hidden** matrix, $W_{ax}: (N_{hidden\_neurons}, n_x)$
- The second one is the **hidden-to-hidden** matrix, $W_{aa}: (N_{hidden\_neurons}, N_{hidden\_neurons})$
- The third one is the **hidden-to-output** matrix, $W_{ya}: (n_y, N_{hidden\_neurons})$
  Now, the forward pass is computed as follows:
  $$
  a^{<1>} = g_1(W_{aa} a^{<0>} + W_{ax} x^{<1>} + b_a)\\
  \hat{y}^{<1>}= g_2(W_{ya} a^{<1>} + b_y)\\
  a^{<t>} = g(W_{aa} a^{<t-1>} + W_{ax} x^{<t>} + b_a)\\
  \hat{y}^{<t>}= g(W_{ya} a^{<t>} + b_y)
  $$
  Generally, $g_1$ is a $tanh/ReLU$ activation function, and $g_2$ is a $sigmoid$ or $softmax$ activation function.
  Usually, to perform backpropagation, we use the cross-entropy loss function:
  $$
  \begin{aligned}
  &\mathcal{L}\left(\hat{y}^{\langle t\rangle}, y^{\langle t\rangle}\right)=-\sum_{i} y_{i}^{\langle t\rangle} \log \hat{y}_{i}^{\langle t\rangle} \\
  &\left.\mathcal{L}=\sum \mathcal{L}^{\langle t\rangle} \hat{y}^{\langle t\rangle}, y^{\langle t\rangle}\right)
  \end{aligned}
  $$

### Language models

A **language model** is a model that predicts the next word in a sequence. It's usually trained with a sequence of words, and the model predicts the probabilities for the next word. We just get a training set of target language text, tokenize this by getting the vocabulary and one-hot each word, add `<EOS>` and `<UNK>` tokens.
To predict a whole sentence's probability, we feed one word at a time and multiply the probabilities.

To **sample novel sequences**, we can just pick a random first word from the distribution obtained by $y^{<1>}$, then pick the next word according to the distribution obtained by $y^{<t>}$.

**Character-level language models** are a special case of language models, where the input is a sequence of characters: these tend to create longer sequences and are not as good at capturing long range dependencies.

### Vanishing gradients

As every deep neural network, RNN are subject to **vanishing gradients**. This means that RNNs are not good in long-term dependencies. **Gradient clipping** (i.e. deciding a maximum for gradients) can solve the exploding gradient problem, while a weight initialization (e.g. He) and echo state networks (i.e. RNNs with recurrent dropout) can help to avoid vanishing gradients. The most popular solution, though, is using **GRU/LSTM networks**.

### Gated Recurrent Unit (GRU)

GRUs introduce a **memory cell** that is updated at every time step. This cell is used to remember the output of the previous time step: $C^{<t>} = a^{<t>}$. The _update gate_ decides whether to update the memory cell or not. If the update gate is $1$, the memory cell is updated with the output of the current time step, if $0$ it will just be the previous. The update then works as follows:

$$
\tilde{C}^{<t>} = tanh(W_a[c^{<t-1>}]+b_a)\\
\Gamma_{u} = \sigma(W_u[c^{<t-1>}, x^{<t>}]+b_u)\\
C^{<t>} = \Gamma_{u} \cdot \tilde{C}^{<t>} + (1-\Gamma_{u}) \cdot C^{<t-1>}
$$

With the update usually being a small number (in the order of $10^{-5}$), GRUs don't suffer from vanishing gradients. This makes the equation $C^{<t>}=C^{<t-1>}$ often times.
So, the shapes are as follows:

- $a^{<t>}$: (N\_{hidden_neurons}, 1)
- $c^{<t>}$: (N\_{hidden_neurons}, 1)
- $\tilde{c}^{<t-1>}$: (N\_{hidden_neurons}, 1)
- $u^{<t>}$: (N\_{hidden_neurons}, 1)
  This was true for the **simplified GRU**, but the **full GRU** introduces a new gate, telling us _how relevant the previous memory cell is_.
  We'll call this the **relevance gate**:
  $$
  \Gamma_{r}=\sigma\left(W_{r}\left[c^{\langle t-1\rangle}, x^{(t)}\right]+b_{r}\right)
  $$

### Long Short Term Memory (LSTM)

LSTMs have 3 different gates: an **update** gate, a **forget** gate and an **output** gate.
![LSTM structure](./res/LSTM.png)

### Bidirectional RNN

Some sentences need information from both the past and the future. For example, the sentence "I love you" needs information from both the past and the future. BiRNNs solve this issue by having activations that come from both left and right. BiRNN with LSTM appear to be commonly used, but you obviously need the whole sequence before you can process it: this is not optimal in, for example, live speech recognition.

### Deep RNNs

Sometimes, stacking multiple RNN layers is powerful. In feed-forward deep nects, there could be even 200 layers, while in deep RNNs having 3 is already deep and expensive.

## Word Embeddings

Word embeddings are a way to represent words in a vector space. This is useful for tasks like text classification, where words are represented by a vector space. Up until now, we used a vocabulary, but this is not optimal: we would like to encode the relationship between words, for example between king and queen.
Algorithms used to generate word embeddings examine unlabeled text and learn the representation. Word embeddings tend to make an extreme difference with smaller datasets, and they reduce the size of the input from a one-hot vector to a vector of features. Word embedding technology has even been used for face recognition, being able to analyze similarity.
Word embeddings can be used to analyze analogies: by computing the vector difference between 2 words, you can check whether the difference between them is similar to the one between 2 other words by computing their Cosine Similarity.
