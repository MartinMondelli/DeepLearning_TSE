# Report on Reinforcement Learning: Solving MinAtar games using Deep Q-Learning

In this project we implemented a Deep Q-learning algorithms in reinforcement learning context. We developped two DQN agents, one able to play the CartPole game and the other one Breakout from Minatar. In both cases we developped the agents from scratch. The first one is based on a script from the Reinforcement Learning (DQN) Tutorial but adapted in order to make it understandable for users that are not well familiarized with DQNs. The second agent, the one for Minatar, uses a similar way of scripting as the tutorial but instead of using a Multi-Layer Perceptron, we use a Convolutional Neural Network to make the script more closer to the support paper of Mnih et al. (2013).

## About Deep Q-learning

First, we will begin by explaining the mathematical foundations and working of Deep Q-learning, in other words, the theoretical explanation.

### Main idea

Q-learning is included in Reinforcement Learning (RL). The idea of RL is to make an Agent (the virtual robot) and an Environment (here a game), the goal of the Agent is to perform well in the game, this is done through a system of rewards, where the Agent receives a certain reward for good actions. The goal of the Agent is to maximize this reward $$R_T = \sum_{t'=t}^T \gamma^{t'-t} r_{t'}$$ where $$\gamma$$ is the discount factor and r() is the reward a each perior. A key factor is that the Agent is not given any prior training or strategy in the game, it is up to it (thorugh interactions with the environment) to learn the best strategy it can.

In the context of Q-learning, what the Agent observes is an adaptation of the State s, this is similar to the picture of what is happening in the Environment at t. A key remark is that one of the innovations of the paper of Mnih et al. (2013) is the Memory Replay, which gives the agent more information of the previous states, we will develop this further in the next sections.

The center of this method is no other than the $$Q_{\pi}(s,a)$$ function, which receives input "s" and a possible action "a" that belongs to a set of possible actions A. This is a function that, in a current state s, gives the Agent the total reward following action a, given that it will follow a certain policy (strategy) $$\pi$$ afterwards.  

Thus, we can separate this Q function in two part, the present (inmediate) reward in i and the following reward in i+t. This relation is given in the Bellman's equation. Note that there are two reasons why a certain action $$a_1$$ might give a higher value for Q than antoher action $$a_2$$: either it gives an inmediate higher reward or it can also be that because after undertaking either of those actions we end up in a different state where even if we followed the same policy we will not be able to reach the amount of reward.

Now, if we had the function Q, then it would be obvious that the only optimal policy in si would be the one that maximizes Q this is: $$\pi_Q(s) = max_a Q(s,a)$$ at every state: the solution would be trivial. The problem is that we do not know the function Q. **Thus, the main idea would be to estimate Q.**

We can do so in two main ways: either in a parametric way, meaning that we suppose the shape of Q and we estimate it's parameter $$\Theta$$ or we can use Deep Neural Networks.

## About the Bellman equation

Since we use stochastic gradient descent (SGD) and mini-batches, we start from the Bellman optimality equation for the action-value function: $$Q^{*}(s,a) = E\left[r + \gamma \max_{a'} Q^{*}(s', a')\right]$$ where the expectation is over the next state $s'$ given the current state-action pair $(s,a)$.
In practice, this expectation is approximated using a single sampled transition. Therefore, we use the following sample-based update: $$Q^{*}(s,a) = r + \gamma \max_{a'} Q^{*}(s', a')$$

## What was done in the past

In previous works what was done is to do iterations or assume a parametric form of Q, in the paper of Mnih et al. (2013) they propose to estimate it using Neural Networks.

## Important remarks

The model does not know the real behavior of E (like the physics in  CartPole) it only knows a set: ($$s_{t-1}$$, $$a_t$$, r, $$s_{t}$$)

The model tries to explore new actions even if they are not meant to be done in the policy, this is call off-policy and what it is trying to do is to learn new behaviors that might give higher rewards.

The Replay Memory is a dataset containing all previous states. 

## About the DQN

The idea of DQN is that the Neural Network approximates the optimal action-value function by minimizing the loss function

$$
L_i(\theta_i) = \mathbb{E}\left[\left( y_i - Q(s,a;\theta_i) \right)^2\right]
$$

where

$$
y_i = r + \gamma \max_{a'} Q(s',a';\theta_{i-1})
$$

This target value is derived from the Bellman optimality equation and represents the one-step bootstrapped estimate of the optimal Q-value.  

When minimizing the loss with respect to the weights, we arrive at the following gradient:

$$
\nabla_{\theta_i} L_i(\theta_i) =
\mathbb{E}\left[\left(r + \gamma \max_{a'} Q(s',a';\theta_{i-1}) - Q(s,a;\theta_i)\right)
\nabla_{\theta_i} Q(s,a;\theta_i)\right]
$$

When using Stochastic Gradient Descent (SGD), the expectation term can be approximated by computing the gradient over a finite minibatch of samples.

The model is updated at each time step, where a time step corresponds to a single interaction between the agent and the environment: the agent observes the current state, selects an action, receives a reward, and transitions to a new state. Each observed transition $(s_t, a_t, r_t, s_{t+1})$ is stored in a replay memory buffer.

Instead of training the network using only the most recent transition, a random minibatch of transitions is sampled from the replay memory at every time step to perform a gradient descent update. By using this procedure, the model avoids strong correlations between consecutive and similar states. Moreover, it allows previously observed transitions to be reused multiple times, improving data efficiency and making the training process more stable.

Each time the model takes an action, it follows an $\varepsilon$-greedy policy: with probability $\varepsilon$ a random action is selected to encourage exploration, and with probability $1-\varepsilon$ the action with the highest estimated Q-value is chosen according to the current network parameters.

Since the transitions sampled from the replay memory may have been generated by older policies, the learning process is off-policy, which justifies the use of Q-learning in the DQN framework.

## Setting

The idea of the procedure is quite easy: the Agent takes an action, the Environment gives it the Picture $$x_t$$ and a reward r.

The optimal action-value function Q we want is: $$Q^*(s,a) = \max_{\pi} \; \mathbb{E} \left[ \sum_{t=0}^{\infty} \gamma^t r_t \;\middle|\; s_0 = s,\; a_0 = a,\; \pi \right]$$.

What the agent really sees is a State $$s_t = ((x_1,a_1),(x_2,a_2),.....,(x_{t-1},a_{t-1}), x_t)$$

## Metrics

1. Episode Duration (Total Reward):
Purpose: Measures the total reward collected by the agent in an episode

2. Average Maximum Q-value for Fixed States:
Purpose: Provides a more stable and less noisy measure of the policy network's learning progress and confidence by tracking the average of the highest predicted Q-values for a fixed set of states.

3. Maximum Q-value Evolution During an Evaluation Episode:
Purpose: Visualizes how the agent's estimated value for the current state (its maximum predicted Q-value) changes over time within a single evaluation episode. This helps in understanding the agent's perception of value fluctuations.

4. Directional Fall Evaluation:
Purpose: Specifically for CartPole and adapted for Minatar Breakout, this metric assesses if the trained policy has a bias towards the pole falling to the left or right, indicating potential imbalances in the learned control strategy.

## CartPole

In this section we will explain in details what we do in CartPole, ensuring 

## Breakout from Minatar
