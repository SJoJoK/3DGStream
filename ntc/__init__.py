import torch
class NeuralTransformationCache(torch.nn.Module):
    def __init__(self, model, xyz_bound_min, xyz_bound_max):
        super(NeuralTransformationCache, self).__init__()
        self.model = model
        self.register_buffer('xyz_bound_min',xyz_bound_min)
        self.register_buffer('xyz_bound_max',xyz_bound_max)
        
    def dump(self, path):
        torch.save(self.state_dict(),path)
        
    def get_contracted_xyz(self, xyz):
        with torch.no_grad():
            contracted_xyz=(xyz-self.xyz_bound_min)/(self.xyz_bound_max-self.xyz_bound_min)
            return contracted_xyz
        
    def forward(self, xyz:torch.Tensor):
        contracted_xyz=self.get_contracted_xyz(xyz)                          # Shape: [N, 3]
        
        mask = (contracted_xyz >= 0) & (contracted_xyz <= 1)
        mask = mask.all(dim=1)
        
        ntc_inputs=torch.cat([contracted_xyz[mask]],dim=-1)
        resi=self.model(ntc_inputs)
        
        masked_d_xyz=resi[:,:3]
        masked_d_rot=resi[:,3:7]
        # masked_d_opacity=resi[:,7:None]
        
        d_xyz = torch.full((xyz.shape[0], 3), 0.0, dtype=torch.half, device="cuda")
        d_rot = torch.full((xyz.shape[0], 4), 0.0, dtype=torch.half, device="cuda")
        d_rot[:, 0] = 1.0
        # d_opacity = self._origin_d_opacity.clone()

        d_xyz[mask] = masked_d_xyz
        d_rot[mask] = masked_d_rot
        
        return mask, d_xyz, d_rot
        
        
