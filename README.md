# Classic-CartPole

<h2>Research Papers for DQN and DDQN</h2>
<h3> DQN - https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf</h3>
<h3> DDQN - https://arxiv.org/abs/1509.06461</h3>

<h2>CartPole Introduction</h2>
<p>Cartpole is a classic OpenAI gym environment game. There are 2 possible actions the AI can take: Go Left or Go Right. For each "timestep" alive the agent gets a reward of +1 and the purpose of the game is just to balance the pole and the score of 195 solves the problem.</p>

<h2>Neural Network</h2>
<p> The neural network consists of 2 layers, input layer, hidden layer, and output layer. Layer sizes: 4 x 64, 64 x 128, 128 x 2</p>

<h2>Hyperparameters</h2>
<p> Better scores can be achieved with fine tuning hyperparameters, but in this case I stuck with achieving a score of 200</p>
<p>GAMMA - 0.99<br>
Greedy Strategy - 1, 0.999 for decay - 0.01 for min<br>
Batch Size - 32<br>
Every 500 timesteps I updated my target network </p>


<h1>CartPole AI</h1>
<p> The cartpole starts off with random actions at first since the greedy policy is very high, therefore resulting in a lot of resets</p>
![eResult](https://user-images.githubusercontent.com/41172710/181115048-54dfdaba-bb65-4af1-be6a-781888068a07.gif)<br>

<p>However, after about 50 episodes the AI improves significantly, allowing it to balance the pole for longer</p>

![sResult](https://user-images.githubusercontent.com/41172710/181115206-9a83e514-2367-489c-9e9b-86da511d20a8.gif)<br>

<h2> Deep Reinforcement Learning vs Double Deep Reinforcement Learning</h2>
<p> Vanilla DQN produces acceptable results but extremely unstable</p>

![graph](https://user-images.githubusercontent.com/41172710/181115465-b348ef8f-e384-417e-b4c9-0f408d375c2a.png)

<p>As you can see the Agent learns overtime, however after a period of time results start to drop, a possible cause of catastrophic forgetting and the instablity of the algorithm as a whole.</p>

<p> DDQN reduces the instability of the algorithm by having the target network to evaluate the actions the online network takes. After evaluation, the loss is calculated and the algorithm attempts to reduce the mean squared error in order to achieve high accuracy</p>

![DDQN](https://user-images.githubusercontent.com/41172710/181116064-93467d52-f5a3-40f9-a356-ca7aae0c7899.png)

<p>This graph outcome of DDQN. The DDQN was trained for longer and produced way better and consistent results than the normal DQN. This clearly shows the massive difference between DQN and DDQN. Furthermore, another key concept was implemented in DDQN: LR Scheduling. Overtime I dropped the LR to avoid overfitting and it produced better results.</p>

