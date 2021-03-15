import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# import numpy as np

class Multiply(nn.Module):
  def __init__(self):
    super(Multiply, self).__init__()

  def forward(self, tensors):
    result = torch.ones(tensors[0].size())
    for t in tensors:
      result *= t
    return result

class SimpleAttention(nn.Module):
  def __init__(self, input_dims):
      super(SimpleAttention, self).__init__()
      self.attention_probs = nn.Linear(input_dims, input_dims)
      self.soft = nn.Softmax(dim=1)
      self.multiply = Multiply()
      self.attention_mul_l1 = nn.Linear(input_dims, 64)
      self.out = nn.Linear(64,1)
      self.sigmoid = nn.Sigmoid()

      self.optimizer = optim.Adam(self.parameters(), lr=0.001)
      # self.loss = nn.BCELoss()
      self.loss = nn.MSELoss()

  def forward(self, x):
      attention_probs = self.attention_probs(x)
      attention_probs_soft = self.soft(attention_probs)
      attention_mul = self.multiply([x, attention_probs_soft])
      output = self.attention_mul_l1(attention_mul)
      out = self.out(output)
      out_ = self.sigmoid(out)

      return attention_probs_soft, out_

  # def fit(self, train_dl, criterion, optimizer, n_epochs, device):
  #     for epoch in range(n_epochs):
  #         print(f"Epoch {epoch+1}/{n_epochs} ...")

  #         # Train
  #         self.train()
  #         for X, y in train_dl:
  #             y = y.type(torch.FloatTensor)
  #             X, y = X.to(device), y.to(device)

  #             self.optimizer.zero_grad()
  #             _, y_ = self(X)

  #             loss = self.loss(y_, y)

  #             loss.backward()
  #             self.optimizer.step()

  def fit(self, X, y):
      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      self.train()
      X = torch.tensor([X]).to(device)
      y = torch.tensor([y]).to(device)
      y = y.type(torch.FloatTensor)
      X = X.type(torch.FloatTensor)
      # X, y = X.to(device), y.to(device)

      self.optimizer.zero_grad()
      attention_probs, y_ = self(X)
      loss = self.loss(y_.squeeze(1), y)
      loss.backward()
      self.optimizer.step()

      return attention_probs