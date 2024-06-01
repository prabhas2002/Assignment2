import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import os

def train(teacher_model,student_model,train_loader, optimizer, criterion, device,mode):
    running_loss = 0.0
    correct= 0
    total = 0
    if mode == 'LoRA':
        teacher_model.train()
        for data in train_loader:
            inputs,mask, labels = data
            inputs,mask, labels = inputs.to(device),mask.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = teacher_model(inputs,mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pred = torch.argmax(outputs, dim=1)
            correct += (pred == labels).sum().item()
            total += len(labels)
      
    
    elif mode == 'rnn':
        student_model.train()
        for data in train_loader:
            inputs,mask, labels = data
            inputs,labels = inputs.to(device),labels.to(device)
            optimizer.zero_grad()
            outputs = student_model(inputs,None)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pred = torch.argmax(outputs, dim=1)
            correct += (pred == labels).sum().item()
            total += len(labels)
        
    elif mode == 'distil':
        temp= 2
        loss1_frac = 0.5
        teacher_model.eval()
        student_model.train()
        for data in train_loader:
            inputs,mask, labels = data
            inputs,mask, labels = inputs.to(device),mask.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_outputs = teacher_model(inputs,mask)
            student_outputs = student_model(inputs)
            
            soft_targets = nn.functional.softmax(teacher_outputs / temp, dim=-1)
            soft_prob = nn.functional.log_softmax(student_outputs / temp, dim=-1)
            
            loss1 = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (temp**2)
            loss2= criterion(student_outputs, labels)
            
            loss = loss1_frac*loss1 + (1-loss1_frac)*loss2
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pred = torch.argmax(student_outputs, dim=1)
            correct += (pred == labels).sum().item()
            total += len(labels)
            
    accuracy = correct / total
    loss = running_loss / len(train_loader)
    return loss, accuracy
    


def evaluate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in val_loader:
            inputs,mask, labels = data
            inputs, mask ,labels = inputs.to(device), mask.to(device), labels.to(device)
            outputs = model(inputs,mask)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            pred = torch.argmax(outputs, dim=1)
            correct += (pred == labels).sum().item()
            total += len(labels)
    accuracy = correct / total
    loss = running_loss / len(val_loader)
    
    return loss, accuracy


def plot_loss(model_name,train_loss, eval_loss, epochs):
    
    plt.plot(range(epochs), train_loss, label="Train Loss")
    plt.plot(range(epochs), eval_loss, label="Eval Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    
    path = '/raid/home/gnaneswaras/prabhas/Assignment2/plots'
    if not os.path.exists(path):
        os.makedirs(path)

    plt.savefig(f'{path}/{model_name}_loss_plot.png')
    plt.show()
    plt.close()



def plot_accuracy(model_name,train_accuracy, eval_accuracy, epochs):
    plt.plot(range(epochs), train_accuracy, label="Train Accuracy")
    plt.plot(range(epochs), eval_accuracy, label="Eval Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    
    path = '/raid/home/gnaneswaras/prabhas/Assignment2/plots'
    if not os.path.exists(path):
        os.makedirs(path)

    plt.savefig(f'{path}/{model_name}_accuracy_plot.png')
    plt.show()
    plt.close()