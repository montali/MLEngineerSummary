# Reinforcement Learning

We have an agent in an unknown environment, which can obtain by rewards by interacting with it. The goal is now learning a good policy for the agent from experimental trials.

## Formalization

The agent acts in an **environment**, which is defined by a model that we may or may not know. The agent can be in any of many **states**, pick one of many **actions** to switch between states. When these switches happen, the agent can collect reward. Which state the agent transitions too might be decided stochastically. A **policy** is a function that maps states to actions, and the optimal policy maximizes the total rewards. Transition probabilities are therefore in the form $P(s'|s,a)$. We distinguish between model-based and model-free learning: in the latter, we have no model of the environment and we can only learn from past experiences. In the former, we have a model of the environment and we can learn from it. We can also distinguish between on-policy and off-policy learning. When learning on-policy, the policy is used to sample the next steps after a reward collection. Policies can be deterministic or stochastic, with the latter having form $\pi(a|s)$.

### Markov Decision Processes,

MDPs are a formalization that is often used to describe RL problems: they are sets $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$ in discrete time. The Markov property ensures that the future state only depends on the current state and the action we take: this simplifies the maths by a lot.

## State-value and action-value functions

The long-term cumulative reward over a trajectory $\tau$ is indicated by $G_t (\tau) = \sum_{k=1}^\infty \gamma^{k-1}R_{t+k}$. The state-value function $V_\pi(s)$ is the expected reward for a given state $s$, using the policy $\pi$. The action-value function $Q_\pi(s,a)$ is the expected reward for a given state $s$ and action $a$, using the policy $\pi$: the action $a$ only decides the first step, so we can easily say that $Q_\pi(s,a) = \sum_{s'} P(s'|s,a) [r(s,a,s') + \gamma V_\pi (s')]$.

### Bellman equations

Bellman equations are able to decompose the value function into the immediate reward plus the discounted future values.

$$
\begin{aligned}
V_{\pi}(s) &=\sum_{a \in \mathcal{A}} \pi(a \mid s) Q_{\pi}(s, a) \\
Q_{\pi}(s, a) &=R(s, a)+\gamma \sum_{s^{\prime} \in S} P_{s s^{\prime}}^{a} V_{\pi}\left(s^{\prime}\right)
\end{aligned}
$$

We can make these recursive as follows:

$$
V_\pi(s) = \sum_a \pi(a \mid s) \sum_{s^{\prime} \in S} P(s'|s,a) [r(s,a,s') + \gamma V_{\pi}\left(s^{\prime}\right)]\\
Q_\pi(s,a) =\sum_{s^{\prime} \in S} P(s'|s,a) [r(s,a,s') + \gamma \sum_{a'} q(s', a')
$$

The Bellman equations can also be expressed in matrix form: $\mathbf{v}_{\pi}=\gamma P_{\pi} \mathbf{v}_{\pi}+\mathbf{r}_{\pi}$

#### Bellman optimality equations

As the optimal policy is **greedy**, we can introduce the Bellman optimality equations. These are the same as the Bellman equations, but with the optimal policy substituted:

$$
V_* (s) = \max_{a \in \mathcal{A}} Q_{\pi}(s, a) \\
Q_* (s,a) = \sum_{s'} P(s'|s,a) [r(s,a,s') + \gamma  \max_{a' \in \mathcal{A}} Q_{\pi}(s', a')]
$$

## Methods

### Value and Policy Iteration

Value and policy iterations, techniques in the field of **Dynamic Programming**, allow us to iteratively solve the value and policy functions.
To perform policy iteration, we iteratively evaluate a policy and greedify: we start from $V(s)=0$ for all $s$, then update $V(s)$ for all $s$ by evaluating the policy $V_\pi(s) = \sum_a \pi(a \mid s) \sum_{s^{\prime} \in S} P(s'|s,a) [r(s,a,s') + \gamma V_{\pi}\left(s^{\prime}\right)]$. We obviously need the model for this. Value iteration, instead, first computes the optimal $V_*$, and only greedifies at the end:

$$
v_{k+1}(s) \leftarrow \max _{a} \sum_{s^{\prime}} p\left(s^{\prime} \mid s, a\right)\left[r\left(s, a, s^{\prime}\right)+\gamma v_{k}\left(s^{\prime}\right)\right]
$$

### Monte Carlo Tree Search

We can now dive deep into the field of **Model-free Reinforcement Learning**. In this area, we'll focus on the state-action function more, as it needs to be estimated from collected experiences instead of computed with **one-step look-ahead**. Monte Carlo methods use a simple idea: learning from episodes of raw experience without modeling the environment. The steps are simple: we generate episodes using a policy, we estimate $Q$ using the episode, and we update the policy.
We can distinguish between first-visit and every-visit methods, in which the latter is more data-efficient as it samples trajectories that pass through a state multiply, multiple times. MC doesn't use the Bellman equations at all. For it to converge, it has to be _Greedy in the Limit with Infinite Exploration_.

### Temporal Differencing

In temporal differencing, we're exploiting the knowledge that Bellman equations give us to link values in neighbouring states. This is the basis for many RL algorithms, like SARSA and Q-learning.
While in Monte Carlo we updated our value function with $G_t(S_t)$, we now use a single reward:

$$
v_{\pi}^{\text {new }}\left(S_{t}\right) \leftarrow v_{\pi}^{\text {old }}\left(S_{t}\right)+\alpha\left[R_{t+1}+\gamma v_{\pi}\left(S_{t+1}\right)-v_{\pi}^{\text {old }}\left(S_{t}\right)\right]
$$

TD is said to be a bootstrapping method, as it uses the current value function to estimate the future value function.

#### SARSA

SARSA and Q-learning are two variants of TD: both aim to find the $q(s,a)$ function, with the difference that the first is on-policy, and the second is off-policy.
SARSA updates the action-value function $Q_\pi(s,a)$ with the following equation:

$$
q_{\pi}(s, a) \leftarrow q_{\pi}(s, a)+\alpha(\underbrace{r\left(s, a, s^{\prime}\right)+q_{\pi}\left(s^{\prime}, a^{\prime}\right)}_{\text {new info }}-q_{\pi}(s, a))
$$

#### Q-learning

Q-learning is similar to SARSA, but we have a difference: the update rule now uses the greedy action $a^{\prime}$ instead of the policy-based one:

$$
q^{*}(s, a) \leftarrow(1-\alpha) q^{*}(s, a)+\alpha\left(r\left(s, a, s^{\prime}\right)+\max _{a^{\prime}} q^{*}\left(s^{\prime}, a^{\prime}\right)\right)
$$

Off-policy methods improve data efficiency: they can re-use all sample for training, while on-policy has to avoid considering samples that were obtained with different policies.

### Policy gradient methods

These methods optimise the policy directly, not via a value function. This is useful, for example, in very large state-spaces. We have a parametrised policy $\pi_\theta(a|s)$, and we want to find the optimal parameters $\theta$ by maximising an objective function. The objective function is the expected total reward $J(\theta):=\mathbb{E}_{\tau \sim \pi_{\theta}}[R(\tau)]$.

In more abstract terms, we're trying to optimize an objective function $g(\theta):=\mathbb{E}_{X \sim p_{\theta}}[\phi(X)]=\int \phi(x) p(x \mid \theta) d x$, so we compute the derivative.

$$
\begin{aligned}
\frac{d}{d \theta} g(\theta) &=\int \phi(x) \frac{d}{d \theta}(p(x \mid \theta)) d x \\
&=\int \phi(x)\left[\frac{\frac{d}{d \theta}(p(x \mid \theta))}{p(x \mid \theta)}\right] p(x \mid \theta) d x \\
&=\int \phi(x)\left[\frac{\frac{d}{d \theta}(p(x \mid \theta))}{p(x \mid \theta)}\right] p(x \mid \theta) d x \\
&=\int \phi(x)\left[\frac{d}{d \theta}(\log p(x \mid \theta))\right] p(x \mid \theta) d x \\
&=\mathbb{E}_{X \sim p_{\theta}}\left[\phi(X) \frac{d}{d \theta}(\log p(x \mid \theta))\right]
\end{aligned}
$$

We therefore obtain the Policy Gradient theorem:

$$
\nabla_{\theta} J(\theta)=\nabla_{\theta} \mathbb{E}_{\tau \sim \pi_{\theta}}[R(\tau)]=\mathbb{E}_{\tau \sim \pi_{\theta}}\left[R(\tau) \nabla_{\theta} \log p(\tau \mid \theta)\right]
$$

This is easier than it looks: the $\nabla_{\theta} \log p(\tau \mid \theta)$ tells us how changing the parameters of the policy $\pi_{\theta}$ affects a trajectory, and $R(\tau)$ tells us where the rewards are coming from.

#### Actor-Critic

If we learn the value function in addition to the policy, we can talk about actor-critic: the critic updates value function parameters $w$ (might be $q$ or $v$), and the actor updates policy parameters $\theta$, in the direction suggested by the critic. For example, in a vanilla Actor-Critic method, we sample the reward, and next state, we update the policy parameters $\theta$ with the policy gradient $\theta \leftarrow \theta+\alpha_{\theta} Q(s, a ; w) \nabla_{\theta} \ln \pi(a \mid s ; \theta)$, then update the value function parameters $w$ with the value function gradient $w \leftarrow w+\alpha_{w} G_{t: t+1} \nabla_{w} Q(s, a ; w)$.

In A2C, we're solving a problem that vanilla AC has: the q-value is not very informative, and we'd like to know the **advantage** more. The update becomes:

$$
\nabla_{\theta} J(\theta) \propto \mathbb{E}_{\tau}\left[\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}\left(a_{t} \mid s_{t}\right)\left(q_{\pi_{\theta}}\left(s_{t}, a_{t}\right)-v_{\pi_{\theta}}\left(s_{t}\right)\right)\right]
$$

We obviously don't have two separate value functions, but rather use the Bellman equations to compute one-step look-ahead for $q(s,a)$ from $v(s)$.

### DQN

For all this time, we have been trying to approximate a function, the value or the action-value function. We though know a really powerful tools for function approximation: Neural Networks. Deep Q-Networks try to approximate the $q$ function by learning from experience. This is helpful for another reason too: we're able to generalize to states we've never seen.
We update a $q$ network by computing a target $y_i=r_{i}+\gamma \max _{a_{i}^{\prime}} q_{\theta}\left(s_{i}^{\prime}, a_{i}^{\prime}\right)$, obtaining a loss function $L(\theta)=\frac{1}{N} \sum_{i=1}^{N}\left(y_{i}-q_{\theta}\left(s_{i}, a_{i}\right)\right)^{2}$ and just performing gradient descent on the network parameters:

$$

\theta=\theta-\alpha \nabla_{\theta} L(\theta)


$$

#### Using target networks

We have a moving target: we're computing our target with the function we're trying to approximate. By using a target network, we're using a lagged network to compute the target, to fix it.

#### Experience replay

Experiences along a trajectory are highly correlated, and information rich experiences should be used multiple times. Replay memories allow us to simply feed the network random experiences from the past, saving the most recent in to a buffer.

#### Prioritized Experience Replay

Some experiences are more informative than others: we can use them more often. In PER, we weigh experiences in the buffer by the loss we had when using them to update the network.

#### Double DQN

The problem we solved with the target network can be solved in a smarter way: instead of using a target network, we create two different networks, and use one to update the other. We therefore avoid maximisation bias by disentagling the updates from the biased estimates.

#### Dueling DQN

Dueling DQN splits the Q-values into two different parts, the value function $V(s)$ and the advantage function $A(s,a)$. The same neural network splits its last layer in two parts. This is useful as sometimes it is unnecessary to know the exact value of each action, so just learning the state-value can be enough. To solve the _unidentifiability_ problem we have (as we don't know the value of the advantage), we subtract the maximum value of the advantage function from the advantage function, obtaining 0 for the best, and negative values for the others:

$$

Q(s, a)=V(s)+\left(A(s, a)-\max \_{a^{\prime} \in|\mathcal{A}|} A(s, a)\right)


$$

$$
$$
