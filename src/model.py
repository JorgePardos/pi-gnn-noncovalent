"""A minimal message-passing network in plain PyTorch (no torch_geometric
- these graphs are tiny, 6-8 atoms, so a dense padded representation is
simpler and just as fast, and avoids PyG's CUDA/wheel-matching install
fragility for no real benefit at this scale).

Graphs are padded to MAX_ATOMS with a validity mask. One message-passing
layer aggregates edge features (which atom-pair distance/PI a node is
touching) into each node, then a second layer refines it, then mean
pooling over real atoms + an MLP predicts the graph-level energy.

Two edge feature configurations are compared elsewhere (train.py):
  - none:     edge_attr = [distance]
  - pi_only:  edge_attr = [penetration_index], no distance at all
Everything else (architecture, hidden size, training procedure) is kept
identical between the two, so any difference in performance is
attributable to the edge feature swap, not incidental architecture
changes.
"""

import torch
import torch.nn as nn

MAX_ATOMS = 9  # largest complex in the dataset (SiF4/GeF4/SnF4 electrophile,
                # 5 atoms, + NH3 probe, 4 atoms); smaller complexes just get
                # more zero-padding, harmless given the edge_mask
N_ELEMENTS = 54  # covers up to I (Z=53); embedding table sized 0..N_ELEMENTS


def _mlp(in_dim, hidden, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.SiLU(),
        nn.Linear(hidden, out_dim),
    )


class MessagePassingNet(nn.Module):
    def __init__(self, n_edge_features: int, hidden: int = 32):
        super().__init__()
        self.z_embed = nn.Embedding(N_ELEMENTS + 1, hidden)
        self.msg_mlp1 = _mlp(2 * hidden + n_edge_features, hidden, hidden)
        self.update_mlp1 = _mlp(2 * hidden, hidden, hidden)
        self.msg_mlp2 = _mlp(2 * hidden + n_edge_features, hidden, hidden)
        self.update_mlp2 = _mlp(2 * hidden, hidden, hidden)
        self.readout = _mlp(hidden, hidden, 1)

    def _mp_layer(self, h, edge_attr, edge_mask, msg_mlp, update_mlp):
        b, n, hidden = h.shape
        h_i = h.unsqueeze(2).expand(b, n, n, hidden)
        h_j = h.unsqueeze(1).expand(b, n, n, hidden)
        msg_input = torch.cat([h_i, h_j, edge_attr], dim=-1)
        msg = msg_mlp(msg_input) * edge_mask.unsqueeze(-1)
        agg = msg.sum(dim=2)  # sum over neighbors j
        return update_mlp(torch.cat([h, agg], dim=-1))

    def forward(self, z, edge_attr, edge_mask, atom_mask):
        """z: (B,N) long; edge_attr: (B,N,N,F); edge_mask: (B,N,N);
        atom_mask: (B,N). Returns (B,) predicted energies."""
        h = self.z_embed(z)
        h = self._mp_layer(h, edge_attr, edge_mask, self.msg_mlp1, self.update_mlp1)
        h = self._mp_layer(h, edge_attr, edge_mask, self.msg_mlp2, self.update_mlp2)

        atom_mask_f = atom_mask.unsqueeze(-1).float()
        pooled = (h * atom_mask_f).sum(dim=1) / atom_mask_f.sum(dim=1).clamp(min=1.0)
        return self.readout(pooled).squeeze(-1)


def examples_to_tensors(examples, pi_mode: str = 'none'):
    """Pads every GraphExample to MAX_ATOMS and stacks into batch tensors.

    pi_mode:
      'none'     - edge_attr = [distance] only (baseline).
      'pi_only'  - edge_attr = [PI] for every atom pair, with NO distance
                   channel at all. Isolates whether PI alone (no raw
                   geometry, so no way for the model to infer angle) can
                   compete with 'none' - the direct test of whether PI's
                   angle-blindness is fatal once distance is taken away
                   as a crutch.
    """
    assert pi_mode in ('none', 'pi_only')
    b = len(examples)
    z = torch.zeros(b, MAX_ATOMS, dtype=torch.long)
    edge_attr = torch.zeros(b, MAX_ATOMS, MAX_ATOMS, 1)
    edge_mask = torch.zeros(b, MAX_ATOMS, MAX_ATOMS)
    atom_mask = torch.zeros(b, MAX_ATOMS, dtype=torch.bool)
    y = torch.zeros(b)

    for k, ex in enumerate(examples):
        n = len(ex.z)
        z[k, :n] = torch.tensor(ex.z, dtype=torch.long)
        atom_mask[k, :n] = True

        if pi_mode == 'pi_only':
            edge_attr[k, :n, :n, 0] = torch.tensor(ex.pi, dtype=torch.float32) / 100.0
        else:
            edge_attr[k, :n, :n, 0] = torch.tensor(ex.dist, dtype=torch.float32) / 5.0

        mask = (~torch.eye(n, dtype=torch.bool))
        edge_mask[k, :n, :n] = mask.float()
        y[k] = ex.energy_kcal_mol

    return z, edge_attr, edge_mask, atom_mask, y
