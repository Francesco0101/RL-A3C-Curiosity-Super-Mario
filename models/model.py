import torch
import torch.nn as nn

def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out

# Define the Actor-Critic Model
class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.common = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU()
        )
        self.actor = nn.Linear(512, action_dim)
        self.critic = nn.Linear(512, 1)
        self.lstm = nn.LSTMCell(32*6*6, 512)
        self._initialize_weights()
        self.actor.weight.data = normalized_columns_initializer(
            self.actor.weight.data, 0.01)
        self.actor.bias.data.fill_(0)
        self.critic.weight.data = normalized_columns_initializer(
            self.critic.weight.data, 1.0)
        self.critic.bias.data.fill_(0)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTMCell):
                nn.init.constant_(module.bias_ih, 0)
                nn.init.constant_(module.bias_hh, 0)


    def forward(self, x, hx, cx):
        x = self.common(x)
        x = x.view(x.size(0), -1)
        hx , cx = self.lstm(x, (hx, cx))
        action_probs = self.actor(hx)
        value = self.critic(hx)
        return action_probs, value, hx, cx