import torch
from torch.autograd import Variable
N, D_in, H, D_out = 4, 2, 2, 1

def init_weights(m):
   # print(m)
    if type(m) == torch.nn.Linear:
        m.weight.data.fill_(1)
      #  print(m)

x = Variable(torch.Tensor([[0, 0],[1, 0],[0, 1],[1, 1]]), requires_grad = True)
y = Variable(torch.Tensor([[0],[1],[1],[0]]))

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H,D_out)
    )
#model.apply(init_weights)

loss_fn = torch.nn.BCEWithLogitsLoss()

alpha = 1e-1
optimizer = torch.optim.Adam(model.parameters(), lr = alpha)
             
for t in range(1000):
    y_pred = model(x)
    #print(y_pred)
    loss = loss_fn(y_pred,y)
    print(t, loss.data[0])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
