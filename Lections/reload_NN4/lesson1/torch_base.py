import torch
import torch.nn as nn


tensor = torch.tensor([1, 2, 3]).float()

tensor.size() # эквивалент tensor.shape
#(3)
tensor.dim() # число измерений
# 1

zeros = torch.zeros((3,2,4)) #zeros.size() (3,2,4), zeros.dims() 3

#скоринговая карта
#xi = [1, 5, 5, 4]
#batch = [xi] 

#задача 1 в домашке
class MyLayer(nn.Module):
    def __init__(self, tensor):
        super().__init__()
	self.params = nn.Parameter(tensor, requres_grad=Fasle)

    def forward(self, x):
        return x * self.params

    def backward(self, grad):
        pass #TODO расчет градиента c с возвратом

        
tensor.detach() #отсоединяет граф градиентов
tensor.detach().numpy()
torch.from_numpy()
device_id = -1
device = 'cpu' if device_id == -1 else f'cuda:{device_id}' # 'cpu'
tensor.to(device) # tensor.cuda()



# velocity = momentum (0.9-0.99)* velocity - lr*gradient
# w = w + velocity (скорость-вектор)


#Задача 2 в домашке
class SGDMomentum:
    def __init__(self, model_weights, momentum: float=0.99, lr: float = 0.001):
        self.momentum = momentum
        self.lr = lr
        self.velocity = torch.zeros_like(model_weights)
        self.weights = model_weights

    def step(self, grad):
        self.velocity = self.momentum * self.velocity - self.lr * grad
        self.weights = self.weights + self.velocity




model = MyModel()
loss_func = MyLoss()
optim = Optimizer(model.parameters())

batch

pred = model(batch.data)
loss = loss_func(pred, batch.target)
loss.bacward() # расчет градиентов
optim.step()
















