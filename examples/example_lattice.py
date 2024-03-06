# %%
import torch
from egnn_pytorch import EGNN, EGNN_Network
# %%
model = EGNN(
    dim = 512,
    edge_dim = 4
)# .cuda()
# %%
feats = torch.randn(1, 16, 512)
coors = torch.randn(1, 16, 3, requires_grad=True) * 10
edges = torch.randn(1, 16, 16, 4)
# %%
out_feats, out_coors = model(feats, coors, edges)
# %%

model_lattice = EGNN(
    dim = 512,
    edge_dim=4,
    # valid_radius=3,
    use_pbc=True
)# .cuda()
# %%
# Example of Lattice & PBC

feats = torch.randn(1, 16, 512)
coors = torch.randn(1, 16, 3, requires_grad=True) * 10.0
edges = torch.randn(1, 16, 16, 4)


lattice_vectors = torch.tensor([[10.0, 0, 0], [0, 5.0, 0], [0, 0, 10]],requires_grad=True).float()

coors.retain_grad()
out_feats, out_coors = model_lattice(feats, coors, edges,
                        lattice_vectors=lattice_vectors)
out_sum = torch.sum(out_feats, dim=[-1,-2])
out_sum.backward()

force = coors.grad
stress = lattice_vectors.grad
print('force', force.detach().cpu().numpy())
print('stress', stress.detach().cpu().numpy())
# %%

import torch
from egnn_pytorch import EGNN_Network

net = EGNN_Network(
    num_tokens = 21,
    dim = 32,
    depth = 3,
    num_adj_degrees = 3,           # fetch up to 3rd degree neighbors
    adj_dim = 8,                   # pass an adjacency degree embedding to the EGNN layer, to be used in the edge MLP
    only_sparse_neighbors = True
)

feats = torch.randint(0, 21, (1, 1024))
coors = torch.randn(1, 1024, 3)
mask = torch.ones_like(feats).bool()

# naive adjacency matrix
# assuming the sequence is connected as a chain, with at most 2 neighbors - (1024, 1024)
i = torch.arange(1024)
adj_mat = (i[:, None] >= (i[None, :] - 1)) & (i[:, None] <= (i[None, :] + 1))

feats_out, coors_out = net(feats, coors, mask = mask, adj_mat = adj_mat) # (1, 1024, 32), (1, 1024, 3)
# %%

net = EGNN_Network(
    num_tokens = 21,
    dim = 32,
    depth = 3,
    num_adj_degrees = 3,           # fetch up to 3rd degree neighbors
    adj_dim = 8,                   # pass an adjacency degree embedding to the EGNN layer, to be used in the edge MLP
    only_sparse_neighbors = True,
    use_pbc=True
)

feats = torch.randint(0, 21, (1, 1024))
coors = torch.randn(1, 1024, 3)
mask = torch.ones_like(feats).bool()


# naive adjacency matrix
# assuming the sequence is connected as a chain, with at most 2 neighbors - (1024, 1024)
i = torch.arange(1024)
adj_mat = (i[:, None] >= (i[None, :] - 1)) & (i[:, None] <= (i[None, :] + 1))

feats_out, coors_out = net(feats, coors, mask = mask, adj_mat = adj_mat, lattice_vectors=lattice_vectors) # (1, 1024, 32), (1, 1024, 3)

# %%
# no lattice_vector error should be raised 
net = EGNN_Network(
    num_tokens = 21,
    dim = 32,
    depth = 3,
    num_adj_degrees = 3,           # fetch up to 3rd degree neighbors
    adj_dim = 8,                   # pass an adjacency degree embedding to the EGNN layer, to be used in the edge MLP
    only_sparse_neighbors = True,
    use_pbc=True
)

feats = torch.randint(0, 21, (1, 1024))
coors = torch.randn(1, 1024, 3)
mask = torch.ones_like(feats).bool()

# naive adjacency matrix
# assuming the sequence is connected as a chain, with at most 2 neighbors - (1024, 1024)
i = torch.arange(1024)
adj_mat = (i[:, None] >= (i[None, :] - 1)) & (i[:, None] <= (i[None, :] + 1))

try:
    feats_out, coors_out = net(feats, coors, mask = mask, adj_mat = adj_mat) # (1, 1024, 32), (1, 1024, 3)
except Exception as e:
    print(e)
# %%
