import torch
import torch.nn as nn

def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out

class ICM(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ICM, self).__init__()
        self.common = nn.Sequential(
            nn.AvgPool2d(2,2),
            nn.Conv2d(input_dim, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU()
        )
        self.inverse_model = nn.Sequential(
            nn.Linear(288*2, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        self.forward_model = nn.Sequential(
            nn.Linear(action_dim + 288, 256),
            nn.ReLU(),
            nn.Linear(256, 288)
        )
        self._initialize_weights()
        self.inverse_model[0].weight.data = normalized_columns_initializer(
            self.inverse_model[0].weight.data, 0.01)
        self.inverse_model[0].bias.data.fill_(0)
        self.inverse_model[2].weight.data = normalized_columns_initializer(
            self.inverse_model[2].weight.data, 0.01)
        self.inverse_model[2].bias.data.fill_(0)
        self.forward_model[0].weight.data = normalized_columns_initializer(
            self.forward_model[0].weight.data, 0.01)
        self.forward_model[0].bias.data.fill_(0)
        self.forward_model[2].weight.data = normalized_columns_initializer(
            self.forward_model[2].weight.data, 0.01)
        self.forward_model[2].bias.data.fill_(0)


    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTMCell):
                nn.init.constant_(module.bias_ih, 0)
                nn.init.constant_(module.bias_hh, 0)

    def forward(self, s_t0, s_t1, a_t0):
        s_t0 = self.common(s_t0)
        s_t1 = self.common(s_t1)
        s_t0 = s_t0.view(s_t0.size(0), -1)
        s_t1 = s_t1.view(s_t1.size(0), -1)
        inverse_input = torch.cat((s_t0, s_t1), 1)
        forward_input = torch.cat((s_t0, a_t0), 1)
        s_t1_pred = self.inverse_model(inverse_input)
        a_t0_pred = self.forward_model(forward_input)
        return s_t1_pred, a_t0_pred, s_t1


        