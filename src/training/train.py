import torch
import torch.nn as nn

from src.models.lstm import LSTMModel
from src.models.classifier import LSTMClassifier


def train_model(
    X_train, y_reg_train, y_cls_train,
    X_test, y_reg_test, y_cls_test,
    scaler_y
):

    # ===== CONVERT =====
    X_train = torch.tensor(X_train).float()
    y_reg_train = torch.tensor(y_reg_train).float()
    y_cls_train = torch.tensor(y_cls_train).long()

    X_test = torch.tensor(X_test).float()
    y_reg_test = torch.tensor(y_reg_test).float()
    y_cls_test = torch.tensor(y_cls_test).long()

    # ===== MODELS =====
    model_reg = LSTMModel(input_size=X_train.shape[2])
    model_cls = LSTMClassifier(input_size=X_train.shape[2])

    opt_reg = torch.optim.Adam(model_reg.parameters(), lr=0.005)
    opt_cls = torch.optim.Adam(model_cls.parameters(), lr=0.005)

    loss_reg_fn = nn.MSELoss()
    loss_cls_fn = nn.CrossEntropyLoss()

    # ===== TRAIN LOOP =====
    for epoch in range(20):

        model_reg.train()
        model_cls.train()

        pred_reg = model_reg(X_train)
        pred_cls = model_cls(X_train)

        loss_reg = loss_reg_fn(pred_reg, y_reg_train)
        loss_cls = loss_cls_fn(pred_cls, y_cls_train)

        loss = loss_reg + 0.5 * loss_cls

        opt_reg.zero_grad()
        opt_cls.zero_grad()
        loss.backward()

        opt_reg.step()
        opt_cls.step()

        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

    return model_reg, model_cls