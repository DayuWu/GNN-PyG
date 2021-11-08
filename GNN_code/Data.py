import  torch
from torch_geometric.data import Data

#NODE DATA
#eigenvetcor on each node
x = torch.tensor([[2,1], [5,6], [3,7], [12,0]], dtype=torch.float)
#label of each node
y = torch.tensor([0, 1, 0, 1], dtype=torch.float)

#EDGE DATA
#the first column is labels of starting nodes
#the second column is labels of ending nodes
edge_index = torch.tensor([[0, 1, 2, 0, 3],
                           [1, 0, 1, 3, 2]], dtype=torch.long)

data = Data(x=x, y=y, edge_index=edge_index)
