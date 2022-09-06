# Created by Serré Gaëtan
# gaetan.serre93@gmail.com

import torch
import numpy as np

class TorchWrapper():
  def __init__(self, nn, device, optimizer, lossF, metrics=None):
    self.nn = nn
    self.device = device
    self.optimizer = optimizer
    self.lossF = lossF
    self.metrics = metrics

  @staticmethod
  def print_cuda_memory_state():
    t = torch.cuda.get_device_properties(0).total_memory / 1024**3
    r = torch.cuda.memory_reserved(0) / 1024**3
    a = torch.cuda.memory_allocated(0) / 1024**3
    f = t - (r+a)
    print(f"Total mem: {t:.2f} GiB, Reserved mem: {r:.2f} GiB, Allocated mem: {a:.2f} GiB, Free mem: {f:.2f} GiB")
  
  @staticmethod
  def data_to_loader(data, batch_size, num_workers, shuffle):
    loader = torch.utils.data.DataLoader(
              data,
              batch_size=batch_size,
              shuffle=shuffle,
              num_workers=num_workers)
    return loader
  
  def predict(self, X, batch_size=20, num_workers=1):
    self.nn.eval()
    data = torch.from_numpy(X).float()
    loader = self.data_to_loader(data, batch_size, num_workers, shuffle=False)

    preds = None
    for data in loader:
      pred = self.nn(data.to(self.device))
      pred = pred.cpu().detach().numpy()
      del data
      if preds is not None:
        preds = np.concatenate((preds, pred))
      else:
        preds = pred
    return preds
  
  def fit(self, X, Y,
          valid_data=None,
          epochs=1,
          batch_size=20,
          num_workers=1,
          verbose=True,
          shuffle=True):
          
    self.nn.train()
    train_set = torch.utils.data.TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(Y).float())
    loader = self.data_to_loader(train_set, batch_size, num_workers, shuffle=shuffle)

    history = {"loss": [], "val_loss": [], "metrics": [], "val_metrics": []}

    for epoch in range(epochs):  # loop over the dataset multiple times
      running_loss = 0.0
      metrics = 0
      count = 0
      for data in loader:
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(self.device), data[1].to(self.device)

        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
        outputs = self.nn(inputs)
        loss = self.lossF(outputs, labels)
        loss.backward()
        self.optimizer.step()

        if self.metrics:
          metrics += self.metrics(labels, outputs)

        # print statistics
        running_loss += loss.item()
        count += 1

      loss = running_loss / count
      history["loss"].append(loss)

      if self.metrics:
        train_metrics = metrics / count
        history["metrics"].append(train_metrics.cpu().numpy())

      if verbose:
        print(f"Epoch: {epoch+1} Loss: {loss:.2f}", end="")

        if self.metrics:
          print(f" Metric: {train_metrics:.2f}", end="")
        
        if valid_data:
          X_valid, Y_valid = valid_data
          Y_valid = torch.from_numpy(Y_valid)
          preds = torch.from_numpy(self.predict(X_valid, batch_size=batch_size))

          if self.metrics:
            valid_metrics = self.metrics(Y_valid, preds)
            history["val_metrics"].append(valid_metrics.cpu().numpy())

          valid_loss = self.lossF(preds, Y_valid)
          history["val_loss"].append(valid_loss)

          print(f" Validation loss: {valid_loss:.2f}", end="")
          if self.metrics:
            print(f" Validation metric: {valid_metrics:.2f}")
          else: print("")

          self.nn.train()
        else: print("")
      
    return history

  def save(self, filename):
    torch.save(self.nn.state_dict(), filename)
  
  def load(self, filename):
    self.nn.load_state_dict(torch.load(filename))
  
  def get_parameters(self, trainable=False):
    return sum(p.numel() for p in self.nn.parameters() if not trainable or p.requires_grad)