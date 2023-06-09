from torch.distributions import Categorical
import gym 
import numpy as np 
import torch 
import torch.nn as nn
import torch.optim as optim

gamma : float = 0.99
learning_rate : float = 0.01

class REINFORCE(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(REINFORCE, self).__init__()
        self.layer1 : function = nn.Linear(in_dim, 64)
        self.relu_activation : function = nn.ReLU()
        self.layer2 : function = nn.Linear(64, out_dim)
        self.onpolicy_reset()
        self.train()

    def onpolicy_reset(self):
        self.log_probs = []
        self.rewards = []
    
    def forward(self, x : torch.Tensor):
        z1 : torch.Tensor = self.layer1(x)
        a1 : torch.Tensor = self.relu_activation(z1)
        z2 : torch.Tensor = self.layer2(a1)
        return z2

    def act(self, state : torch.Tensor):
        state : np.ndarray = np.array(state.astype(np.float32))
        x : torch.Tensor = torch.from_numpy(state)
        logits : torch.Tensor = self.forward(x)
        probs : torch.Tensor = torch.softmax(logits, dim=0)
        pd : Categorical = Categorical(logits=logits)
        action : int = pd.sample()
        log_prob = pd.log_prob(action)
        self.log_probs.append(log_prob)
        return action.item()

def train(model : REINFORCE, optimizer : optim.Adam):
    T = len(model.rewards)
    returns = np.empty(T, dtype=np.float32)
    future_return = 0.0
    for t in reversed(range(T)):
        future_return = model.rewards[t] + gamma * future_return
        returns[t] = future_return
    returns = torch.tensor(returns)
    log_probs = torch.stack(model.log_probs)
    loss = - log_probs * returns
    loss = torch.sum(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def main(enable_render_mode: bool = True) -> None:
    if enable_render_mode: env = gym.make("CartPole-v0", render_mode="human")
    else: env = gym.make("CartPole-v0")
    in_dim : int = env.observation_space.shape[0]
    out_dim : int = env.action_space.n

    model : REINFORCE = REINFORCE(in_dim=in_dim, out_dim=out_dim)
    optimizer : optim = optim.Adam(model.parameters(), lr=learning_rate)

    for episode in range(30000):
        state : tuple = env.reset()
        state : torch.Tensor = state[0]
        for _ in range(200):
            action = model.act(state)
            state, reward, done, _, info = env.step(action)
            model.rewards.append(reward)
            env.render()
            if done: 
                break
        loss : float = train(model, optimizer)
        total_reward : float = sum(model.rewards)
        solved : bool = total_reward > 195.0
        model.onpolicy_reset()
        print(f"Episode: {episode}, loss: {loss}, \
            total reward: {total_reward}, solved: {solved}")

if __name__ == "__main__":
    main()