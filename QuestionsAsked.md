# Questions I've been asked

## OpenAI

- Why OpenAI?
- Why this role?
- Python libraries I know
- Pipelines in SKLearn
- Kullback-Leibler divergence: if we want to compare a model to the real distribution, how do we do that?
- What are other divergence functions we can use?
- Supervised/Unsupervised learning
- Exploding gradients
- RegEx, how to find a word, how to make them faster in Python (compile)
- What's a Transformer?
- Why are vanilla RNN not used for NLP? How to solve this? (LSTM)
- SQL accumulators, HAVING

## Parashift

- Python:
  - Sets
  - Tuples vs lists
  - List comprehensions
  - Decorators
  - Pros and cons
- Batch Normalization
- Transfer Learning, Few shots learning
- Docker
  - What is it?
  - Layers
  - Copy on Write
- Git
  - Main commands
  - Rebasing
  - Git flow vs Gitlab flow
- ML
  - Algorithms used
  - Filling NaNs
  - Outliers definition
  - Boxplot
  - Correlation
  - Skewed histograms (normalising, uniformizing)
- PyTorch vs TensorFlow

## Unit8 - Coding

```
1. Given array A of n integers and given a number X, find out if there is a pair of elements (a,b), whose summation is equal to X. If yes, the program should return True otherwise False.

A = [-8, 1, 4, 6, 10, 45]
X = 16, Output= True

def check(l, x):
    s = set()
    for e in l:
        if x-e in s:
            return True
        s.add(e)
    return False

2. There is given non-empty array A of N arrays of two numbers. First number in inner array is always smaller than the second. The task is to write a program to compute the output array with all inner overlapping lists merged.

Example inputs :

 [[1,2], [7,8], [3,4], [2,6], [100,300], [200, 305], [500, 515], [490,550]]
 [[1,2], [2,6], [3,4],[7,8],[100,300], [200, 305], [500, 515]]
  (output: [(1,6), (7,8), (100, 305), (490, 550)])

def merger(couples):
    couples.sort()
    i = 0
    while i<len(couples)-1:
        if couples[i][1]>=couples[i+1][0]:
            couples[i][1] = max(couples[i+1][1], couples[i][1])
            couples.pop(i+1)
        else:
            i += 1
    return couples

3.You have two SQL tables: producers and perfumes.
The producers dataset :
producer_name perfume_name
producer_1    perfume_1
producer_1    perfume_2
producer_2    perfume_3
producer_2    perfume_4
producer_2    perfume_5
producer_3    perfume_6

The perfumes dataset:
perfume_name   sold_bottles
perfume_1      1000
perfume_2      1500
perfume_3      34000
perfume_4      29000
perfume_5      40000
perfume_6      4400

Create an SQL query that shows the TOP 3 producers who sold the most perfumes in total.

SELECT producer_name, SUM(perfumes.sold_bottles) AS sum_perfumes FROM producers, perfumes WHERE producers.perfume_name=perfumes.perfume_name GROUP BY producer_name SORT ASCENDING LIMIT 3

4.We will see N=100 job candidates whose levels are random between 0 and 10. We see the candidates one-by-one, and after seeing each candidate, we have to make a decision on the spot whether to hire this candidate or not. This decision is irrevocable and we cannot change our mind later! Our policy for selecting a candidate is the following:

* Reject the first R candidates, for some value R.
* Then hire the first one we see that is better than the best among the first R.

Write a program that finds the value of R that is _maximizing the probability to hire the best out of the N candidates_ using simulations.
import random

def simulate(candidates, r):
    best_among_first_r = max(candidates[:r])
    for i, val in enumerate(candidates[r:]):
        if val > best_among_first_r:
            if val==max(candidates[i:]):
                return 1
            else:
                return 0
    return 0

def run_n_simulations_for_r(n, r):
    sum = 0
    for i in range(n):
        candidates = [random.randint(0,10) for i in range(100)]
        sum += simulate(candidates, r)
    return sum/n

def try_rs():
    scores = [run_n_simulations_for_r(1000,r) for r in range(0,100)]
    print(scores)
```
