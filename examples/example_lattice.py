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

N = 16
feats = torch.randn(1, N, 512)
coors = torch.randn(1, N, 3, requires_grad=True) * 10.0
edges = torch.randn(1, N, N, 4)


lattice_vectors = torch.tensor([[[10.0, 0, 0], [0, 5.0, 0], [0, 0, 10]]],requires_grad=True).float()

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
    only_sparse_neighbors = False,
    use_pbc=True
)

# lattice_vectors has [1,3,3]
# lattice_vectors_batch has [batch_size, 3,3] which is the same as lattice_vectors
batch_size = 4



feats = torch.randint(0, 21, (batch_size, 1024))
coors = torch.randn(batch_size, 1024, 3, requires_grad=True)
mask = torch.ones_like(feats).bool()
lattice_vectors_batch = torch.rand((batch_size, 3, 3), requires_grad=True)* 10
coors.retain_grad()
lattice_vectors_batch.retain_grad()

# naive adjacency matrix
# assuming the sequence is connected as a chain, with at most 2 neighbors - (1024, 1024)
i = torch.arange(1024)
adj_mat = (i[:, None] >= (i[None, :] - 1)) & (i[:, None] <= (i[None, :] + 1))

feats_out, coors_out = net(feats, coors, mask = mask, adj_mat = adj_mat, lattice_vectors=lattice_vectors_batch) # (1, 1024, 32), (1, 1024, 3)
print("No error raised with PBC")

print('feat_out', feats_out.shape)
print('coors_out', coors_out.shape)
energy = torch.sum(feats_out, dim= -1)
print('energy', energy.shape)
force = torch.autograd.grad(energy, coors, torch.ones_like(energy), create_graph=True)[0]
print('force', force.shape)
stress = torch.autograd.grad(energy, lattice_vectors_batch, torch.ones_like(energy), create_graph=True)[0]
print('stress', stress.shape)
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
    print("Error raised as expected:")
    print(e)
# %%

# %% PBC test code
# Test with lattice_vectors code
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import matplotlib.pyplot as plt

batch_size = 4
coors = ( torch.rand((batch_size, 5,3)))* 10
rel_coors = rearrange(coors, 'b i d -> b i () d') - rearrange(coors, 'b j d -> b () j d')
lattice_vectors = torch.rand((batch_size,3,3))*10
# lattice_vectors = torch.tensor([[[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]]]) of repeated batch_size
# lattice_vectors = (torch.tensor([[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]])).unsqueeze(0).repeat(batch_size, 1, 1)

inv_lattice_vectors = torch.linalg.inv(lattice_vectors)

# lattice unit test multiply lattice_vectors with inv_lattice_vectors
lattice_unit_test = torch.einsum('bij,bjk->bik', lattice_vectors, inv_lattice_vectors)
# plot the lattice_unit_test[0]
fig = plt.figure()
plt.imshow(lattice_unit_test[0].cpu().detach().numpy())
plt.colorbar()

lattice_unit_test2 = torch.einsum('bij,bjk->bik', inv_lattice_vectors, lattice_vectors)
fig = plt.figure()
plt.imshow(lattice_unit_test2[0].cpu().detach().numpy())
plt.colorbar()
# %%

fractional_coors = torch.einsum('bik,bkj->bij', coors, inv_lattice_vectors)
coors2 = torch.einsum('bik,bkj->bij', fractional_coors, lattice_vectors)
print(torch.sum(coors - coors2))


# %%

frac_rel_pos = torch.einsum('bijk,bkl->bijl', rel_coors, inv_lattice_vectors) # torch.matmul(rel_coors, torch.linalg.inv(lattice_vectors))
rel_coors2 = torch.einsum('bijk,bkl->bijl', frac_rel_pos, lattice_vectors)
frac_rel_pos_pbc = frac_rel_pos  -  torch.round(frac_rel_pos)

rel_pos_pbc  =   torch.einsum('bijk,bkl->bijl', frac_rel_pos_pbc, lattice_vectors)  #torch.matmul(frac_rel_pos_pbc, lattice_vectors)


# 

print(torch.sum(rel_coors - rel_coors2))
print(rel_coors[0].cpu().detach().numpy())
print(rel_coors2[0].cpu().detach().numpy())

# 2D plot of rel_coors[0] and rel_coors2[0] using subplots colorbar for each
fig, ax = plt.subplots(1, 3)
ax[0].imshow(rel_coors[0].cpu().detach().numpy())
ax[1].imshow(rel_coors2[0].cpu().detach().numpy())
ax[2].imshow(rel_pos_pbc[0].cpu().detach().numpy())
# add colorbar
fig.colorbar(ax[0].imshow(rel_coors[0].cpu().detach().numpy()), ax=ax[0])
fig.colorbar(ax[1].imshow(rel_coors2[0].cpu().detach().numpy()), ax=ax[1])
fig.colorbar(ax[2].imshow(rel_pos_pbc[0].cpu().detach().numpy()), ax=ax[2])
plt.show()
# %%
# calculate adj_mat from rel_coors and rel_pos_pbc
# radius = 4
adj_mat = torch.sum(rel_coors ** 2, dim=-1) < 4
adj_mat_pbc = torch.sum(rel_pos_pbc ** 2, dim=-1) < 4

fig, ax = plt.subplots(1,2)
ax[0].imshow(adj_mat[0].cpu().detach().numpy())
ax[1].imshow(adj_mat_pbc[0].cpu().detach().numpy())
