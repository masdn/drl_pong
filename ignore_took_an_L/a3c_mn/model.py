import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn


class ActorCriticModel(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        c, h, w = input_shape

        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        # Compute the flattened size after convs so LSTM/input layers match.
        conv_out_size = self._get_conv_out(input_shape)

        self.lstm = nn.LSTMCell(conv_out_size, 256)
        self.fc = nn.Linear(conv_out_size, 256)
        self.actor = nn.Linear(256, num_actions)
        self.critic = nn.Linear(256, 1)

    def _get_conv_out(self, shape):
        """Run a dummy tensor through conv layers to infer the flattened size."""
        o = torch.zeros(1, *shape)
        o = self.conv(o)
        return int(np.prod(o.size()))

    def forward(self, x, hx, cx):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        hx, cx = self.lstm(x, (hx, cx))  # TODO mention this in paper
        logits = self.actor(hx)
        value = self.critic(hx)
        return logits, value, (hx, cx)

class A2CAgent:
    def __init__(self, env_name):
        self.env = env_name
        #self.env = gym.make(env_name)
        self.state_shape = self.env.observation_space.shape
        self.num_actions = self.env.action_space.n
        self.lr = 0.0001
        self.n_step = 5

        self.actor, self.critic = ActorCriticModel(self.state_shape, self.num_actions, lr=self.lr, n_step=self.n_step)
        self.actor_optimizer = RMSprop(learning_rate=0.0001)
        self.critic_optimizer = RMSprop(learning_rate=0.0001)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.memory = []
        self.batch_size = 32
        self.train_start = 1000
        self.train_interval = 5
        self.train_steps = 1000

    #def remember():
    #def act():
    #def discount_rewards():
    #def replay():
    #def load():
    #def save():
