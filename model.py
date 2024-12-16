
import torch.nn as nn

# Define the Actor-Critic Model
class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.common = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.actor = nn.Linear(512, action_dim)
        self.critic = nn.Linear(512, 1)
        self.lstm = nn.LSTMCell(32*6*6, 512)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTMCell):
                nn.init.constant_(module.bias_ih, 0)
                nn.init.constant_(module.bias_hh, 0)


    def forward(self, x, hx, cx):
        x = x.squeeze(0)
        x = self.common(x)
        x = x.view(x.size(0), -1)
        hx , cx = self.lstm(x, (hx, cx))
        action_probs = self.actor(hx)
        value = self.critic(hx)
        return action_probs, value, hx, cx