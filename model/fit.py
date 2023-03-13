import torch
import numpy as np

def fit_model(model, train_loader, test_loader, opt, loss_fn, epochs, device, early_stopping):
    optimizer = opt
    train_loss_hist = []
    train_acc_hist = []
    test_loss_hist = []
    test_acc_hist = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_acc = 0
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_batch = torch.reshape(y_batch, (-1, 1))
            optimizer.zero_grad()

            output = model(X_batch)
            
            loss = loss_fn(output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()
            output = int(output>=0.5)
            train_acc += np.mean(output==int(y_batch))
        train_loss /= batch_idx + 1
        train_acc /= batch_idx + 1

        model.eval()
        test_loss = 0
        test_acc = 0
        for batch_idx, (X_batch, y_batch) in enumerate(test_loader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_batch = torch.reshape(y_batch,(-1,1))
            output = model(X_batch)
            loss = loss_fn(output, y_batch)
            test_loss += loss.detach().item()
            output = int(output >= 0.5)
            test_acc += np.mean(output == int(y_batch))
        test_loss /= batch_idx + 1
        test_acc /= batch_idx + 1
        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)
        test_loss_hist.append(test_loss)
        test_acc_hist.append(test_acc)
        # if epoch % 5 == 0:
        print(
            'Epoch : {}, Training loss = {:.2f}, Training Acc = {:.2f}, Validation loss = {:.2f}, Validation Acc = {:.2f}'.format(
                epoch, train_loss, train_acc, test_loss, test_acc))
        # 早停止
        early_stopping(test_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break #跳出迭代，结束训练
    return train_loss_hist, train_acc_hist, test_loss_hist, test_acc_hist