import torch
import torch.nn as nn
from game_of_life import GameOfLife
import numpy as np
from torch.utils.data import Dataset, DataLoader

class GameOfLifeNet(nn.Module):
    def __init__(self):
        super(GameOfLifeNet, self).__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 100)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.sigmoid(self.fc2(x))
        return x.view(-1, 10, 10)

class GameOfLifeDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, idx):
        prev_state = self.data[idx]
        next_state = self.data[idx + 1]
        return prev_state, next_state

if __name__ == "__main__":
    game = GameOfLife()
    data = []
    for i in range(1000):
        # record the current state of the grid
        data.append(game.grid.copy())
        # advance the game by one timestep
        game.step()

    # convert data to numpy array
    data = np.array(data)

    # create dataset and dataloader
    dataset = GameOfLifeDataset(data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # train the network
    net = GameOfLifeNet()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(10):
        for prev_state, next_state in dataloader:
            optimizer.zero_grad()
            pred_next_state = net(prev_state)
            loss = criterion(pred_next_state, next_state)
            loss.backward()
            optimizer.step()
        print('Epoch %d loss: %.4f' % (epoch+1, loss.item()))