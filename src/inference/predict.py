import torch

def ensemble_predict(model_reg, model_cls, X, scaler_y):

    model_reg.eval()
    model_cls.eval()

    # ensure tensor
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X).float()

    with torch.no_grad():
        reg_out = model_reg(X)
        cls_out = model_cls(X)

    # handle batch safely
    reg = reg_out.detach().cpu().numpy()[0]
    cls = cls_out.argmax(dim=1).detach().cpu().numpy()[0]

    # inverse transform
    reg_real = scaler_y.inverse_transform([reg])[0]

    # adjust using classification
    if cls == 2:        # high calorie
        reg_real[0] *= 1.2
    elif cls == 0:      # low calorie
        reg_real[0] *= 0.8

    return reg_real