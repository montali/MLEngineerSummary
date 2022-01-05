# Probability theory

## Combinatorial analysis

The **basic principle of counting** states that if one experiment can result in any of $m$ possible outcomes, and another experiment can result in any of $n$ outcomes, then there are $mn$ possible outcomes in total.

### Permutations

The **permutation** of a set $S$ is the set of all possible arrangements of the elements in the set.

$$
n(n-1)(n-2)\dots = n!
$$

If we have elements that are alike, we'll want to remove the permutations that are actually the same:

$$
\frac{n !}{n_{1} ! n_{2} ! \cdots n_{r} !}
$$

where $n_1$ are alike, $n_2$ are alike, etc.

### Combinations

We are often interested in determining the number of different groups of $r$ objects that could be formed from a total of $n$ objects. We therefore define $n \choose r$ as the number of ways to choose $r$ objects from a set of $n$ objects.

$$
{n \choose r }= \frac{n !}{(n-r) ! r !}
$$

## Axioms of probability

### Foundations

- The **sample space** is the set of all possible outcomes of an experiment.
- Any subset $E$ of the sample space is known as an **event**.
- The DeMorgan's laws give us useful relationships between the three basic operations of forming unions, intersections and complements:
  $$
  \begin{aligned}
  &\left(\bigcup_{i=1}^{n} E_{i}\right)^{c}=\bigcap_{i=1}^{n} E_{i}^{c} \\
  &\left(\bigcap_{i=1}^{n} E_{i}\right)^{c}=\bigcup_{i=1}^{n} E_{i}^{c}
  \end{aligned}
  $$

### Axioms

- The probability of an event $E$ is always between 0 and 1.
- The probability of an event $E$ is equal to the number of times that event occurs in the sample space divided by the total number of events.
- The probability of the sample space is $1$
- For any sequence of **mutually exclusive** events $E_1,E_2,...$, it's true that
  $$
  P\left(\bigcup_{i=1}^{\infty} E_{i}\right)=\sum_{i=1}^{\infty} P\left(E_{i}\right)
  $$

### Some propositions

- The probability of an event not occurring is 1 minus the probability that it occurs $P\left(E^{c}\right)=1-P(E)$
- If $E \subset F$, then $P(E) \leq P(F)$
- $P(E \cup F)=P(E)+P(F)-P(E F)$, i.e. we get rid of the cases in which both are happening
- For any event E, it holds that $P(E)=\frac{\text { number of outcomes in } E}{\text { number of outcomes in } S}$: if we were asked to find the probability of having two reds and one black in a 6 white+5 red bag, it would be $\frac{\left(\begin{array}{l}
6 \\
1
\end{array}\right)\left(\begin{array}{l}
5 \\
2
\end{array}\right)}{\left(\begin{array}{c}
11 \\
3
\end{array}\right)}$

## Conditional probabilities

If $P(F)>0$, then

$$
P(E \mid F)=\frac{P(E F)}{P(F)}
$$

The **multiplication rule** states that the probability of all the events $E_1..E_n$ occurring is given by:

$$
P\left(E_{1} E_{2} E_{3} \cdots E_{n}\right)=P\left(E_{1}\right) P\left(E_{2} \mid E_{1}\right) P\left(E_{3} \mid E_{1} E_{2}\right) \cdots P\left(E_{n} \mid E_{1} \cdots E_{n-1}\right)
$$

We intuitively know that $P(E)=P(E \mid F) P(F)+P\left(E \mid F^{c}\right)[1-P(F)]$, which allows us to derive Bayes's formula:

$$
P\left(E \mid F\right)=\frac{P(F \mid E)P(E)}{P(F)}
$$

The **odds** of an event $A$ are defined as $\frac{P(A)}{P\left(A^{c}\right)}=\frac{P(A)}{1-P(A)}$.

## Independent events

Two events $E$ and $F$ are said to be independent if the probability of both events occurring is the same as the probability of either event occurring: $P(EF)=P(E)P(F)$.
Furthermore, three events are said to be independent if

$$
\begin{aligned}
P(E F G) &=P(E) P(F) P(G) \\
P(E F) &=P(E) P(F) \\
P(E G) &=P(E) P(G) \\
P(F G) &=P(F) P(G)
\end{aligned}
$$

## Random variables

We define real-valued functions on the sample space as **random variables**. These have a value determined by the outcome of the experiment, and we may assign probabilities to the possible values of the random variable. A random variable that can take at most a countable number of possible values is said to be **discrete**.
The **probability mass function** of a RV $X$ is defined as $p(a) = P(X=a)$.
The **expected value** is the weighted average of the possile values that $X$ can take on, each value being weighted by the probability of that value:

$$
E[X]=\sum_{x: p(x)>0} x p(x)
$$

The expected value of the **indicator variable** $I$ is just $P(A)$.
The expectation of a **function** $f$ is the weighted average of the values of $f$ on the sample space, each value being weighted by the probability of that value:

$$
E[g(X)]=\sum_{i} g\left(x_{i}\right) p\left(x_{i}\right)
$$

Given a random variable along with its distribution function $F$, it would be extremely useful to summarize the essential properties of $F$. If $X$ is a random variable with mean $\mu$, we define its variance $Var(X)=E[(X-\mu)^2]$.
An alternative formula to compute it is represented by

$$
Var(X)=E[X^2]-E[X]^2
$$

### Bernoulli and binomial

If we consider an experiment having possible outcomes success or failure, having probability $p$ of success, we can introduce the random variable $X$ as a **Bernoulli random variable** with probability $p$.
If we now supposed $n$ individual trials, we can introduce the **Binomial random variable** with parameters $(n,p)$. Clearly, a Bernoulli is just a Binomial with $n=1$.
The probability mass function of a Binomial is given by

$$
p(k)=\binom{n}{k}p^k(1-p)^{n-k}
$$

There basically are $n \choose i$ different sequences of the $n$ outcomes leading to $i$ successes and $n-i$ failures.
A Poisson random variable can be used as an approximation for a binomial RV with parameters $n,p$, when $n$ is large and $p$ is small, having $\lambda =np$.
The probability mass function of a Poisson random variable is given by

$$
p(i)=\frac{e^{-\lambda}\lambda^i}{i!}
$$

## Continuous random variables

We can now introduce RVs whose set of possible values is not discrete. We say that $X$ is a continuous RV if there exists a nonnegative function $f$, defined for all reals, having $P\{X \in B\}=\int_{B} f(x) d x$, which we denote as **probability density function**.

$$
P\{a \leq X \leq b\}=\int_{a}^{b} f(x) d x
$$

We define the **expectation** of a continuous RV as

$$
E[X]=\int_{-\infty}^{\infty} x f(x) d x
$$

If $a$ and $b$ are constants, it holds that $E[a X+b]=a E[X]+b$.
A RV is said to be **uniformly distributed** over the interval $(0,1)$ if its probability density function is

$$
f(x)= \begin{cases}1 & 0<x<1 \\ 0 & \text { otherwise }\end{cases}
$$

It follows that $P\{a \leq X \leq b\}=\int_{a}^{b} f(x) d x=b-a$.
We say that $X$ is a normal random variable if the density of $X$ is given by

$$
f(x)=\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}}
$$

The standard normal variable has expectation $0$ and variance $1$.
Its probability density function is given by

$$
\Phi(x)=\frac{1}{\sqrt{2 \pi}} \int_{-\infty}^{x} e^{-y^{2} / 2} d y
$$
