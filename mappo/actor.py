import torch
import torch.nn as nn
from torch.distributions import Normal


class LSTMActor(nn.Module):

    def __init__(
        self,
        obs_dim=6,
        hidden_dim=128,
        fc_dim=64,
        action_dim=1,
        action_limit=3.0
    ):
        super().__init__()

        self.action_limit = action_limit

        # --------------------------------------------------
        # LSTM
        # --------------------------------------------------
        self.lstm = nn.LSTM(
            input_size=obs_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # --------------------------------------------------
        # Fully connected layer
        # --------------------------------------------------
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, fc_dim),
            nn.Tanh()
        )

        # --------------------------------------------------
        # Mean of Gaussian policy
        # --------------------------------------------------
        self.mean = nn.Linear(
            fc_dim,
            action_dim
        )

        # --------------------------------------------------
        # Log standard deviation
        # --------------------------------------------------
        self.log_std = nn.Parameter(
            torch.zeros(action_dim)
        )

    def forward(self, observation):

        # observation shape:
        # [batch_size, sequence_length, 6]
        #
        # 7 features:
        #
        # 0 = own distance to conflict
        # 1 = own speed
        # 2 = own TTC
        # 3 = other vehicle distance
        # 4 = other vehicle speed
        # 5 = other vehicle TTC
        # 6 = other vehicle passed status

        lstm_output, _ = self.lstm(
            observation
        )

        # Take the output from the final timestep
        last_output = (
            lstm_output[:, -1, :]
        )

        # Fully connected representation
        features = self.fc(
            last_output
        )

        # Mean of Gaussian policy
        mean = self.mean(
            features
        )

        # Keep the Gaussian standard deviation positive
        std = torch.exp(
            self.log_std
        )

        return mean, std

    def get_action(
        self,
        observation