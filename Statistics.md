# Regression

## Regression using SciKit Learn

SciKit Learn provides a linear regression class that follows the standard scikit-learn API, so we just need to call `lr = LinearRegression()` and then `lr.fit(X, y)` to fit the model.

## Linear regression from scratch

An exact way of performing linear regression would be minimising the **squared error**, which is the sum of the squared distances between the model and the data.
To get an approximate result, we can compute the **line of best fit**, being the ratio between covariance and variance of the data:

$$
m=\frac{\sum_{i=1}^{n}\left(x_{i}-\bar{X}\right)\left(y_{i}-\bar{Y}\right)}{\sum_{i=1}^{n}\left(x_{i}-\bar{X}\right)^{2}}
$$

then just compute the intercept as $b=\bar{Y}-m \bar{X}$.

The **Pearson correlation coefficient** is defined as

$$
r=\frac{\sigma_{X Y}}{\sigma_{X} \sigma_{Y}}=\frac{\sum_{i=1}^{n}\left(x_{i}-\bar{X}\right)\left(y_{i}-\bar{Y}\right)}{\sqrt{\sum_{i=1}^{n}\left(x_{i}-\bar{X}\right)^{2}}\sqrt{\sum_{i=1}^{n}\left(y_{i}-\bar{Y}\right)^{2}}}
$$

or in other words, the correlation coefficient is the ratio between the covariance and the standard deviation of the data.

## Confidence interval

The confidence interval can be computed as

$$
\left(m-t \frac{s}{\sqrt{N}}, m+t \frac{s}{\sqrt{N}}\right)
$$

where $t$ is the value that we can get from the t-Student distribution with $n-1$ degrees of freedom.
Notice that we usually use the **critical value** when dealing with approximately normal distributions. This is, for 90% confidence $1.64$, for 95% confidence $1.96$, for 99% confidence $2.58$ and for 99.9% confidence $3.29$.
