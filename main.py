import os
import copy
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data import MultiViewDataset
from models import TMC


class Experiment:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load dataset
        data_train = MultiViewDataset(data_path='dataset/handwritten_6views.mat', train=True)
        data_valid = MultiViewDataset(data_path='dataset/handwritten_6views.mat', train=False)
        num_classes = len(set(data_train.y))
        self.train_loader = DataLoader(data_train, batch_size=256, shuffle=True)
        self.valid_loader = DataLoader(data_valid, batch_size=1024, shuffle=False)

        # Define model
        self.model = TMC(sample_shapes=[s.shape for s in data_train[0]['x'].values()], num_classes=num_classes)
        self.model = self.model.to(self.device)

        # Define optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=40, gamma=0.1)
        self.epochs = 50

    def train(self, saving_path=None):
        model = self.model
        best_valid_acc = 0.
        best_model_wts = model.state_dict()
        for epoch in range(self.epochs):
            model.train()
            train_loss, correct, num_samples = 0, 0, 0
            for batch in self.train_loader:
                x, target = batch['x'], batch['y']
                for v in x.keys():
                    x[v] = x[v].to(self.device)
                target = target.to(self.device)
                view_e, fusion_e, loss = model(x, target, epoch)
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
                train_loss += loss.mean().item() * len(target)
                correct += torch.sum(fusion_e.argmax(dim=-1).eq(target)).item()
                num_samples += len(target)
            self.scheduler.step()
            train_loss = train_loss / num_samples
            train_acc = correct / num_samples
            valid_acc = self.validate()
            if best_valid_acc < valid_acc:
                best_valid_acc = valid_acc
                best_model_wts = copy.deepcopy(model.state_dict())  # save the best model
            print(f'Epoch {epoch:2d}; train loss {train_loss:.4f}, train acc {train_acc:.4f}; val acc: {valid_acc:.4f}')

        model.load_state_dict(best_model_wts)
        print('Validation Accuracy:', self.validate())
        if saving_path is not None:
            os.makedirs(os.path.dirname(saving_path), exist_ok=True)
            torch.save(model, saving_path)
        return model

    def validate(self, loader=None):
        if loader is None:
            loader = self.valid_loader
        model = self.model
        model.eval()
        with torch.no_grad():
            correct, num_samples = 0, 0
            for batch in loader:
                x, y = batch['x'], batch['y']
                for v in x.keys():
                    x[v] = x[v].to(self.device)
                view_e, fusion_e, loss = model(x)
                correct += torch.sum(fusion_e.cpu().argmax(dim=-1).eq(y)).item()
                num_samples += len(y)
        acc = correct / num_samples
        return acc


if __name__ == '__main__':
    Experiment().train()
