# %%
import torch
from egnn_pytorch import EGNN
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
