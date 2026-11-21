"""Differentiable principal SO(3) logarithms, including 180 degree rotations.

The rotation vector axis is intrinsically non-unique exactly at pi. A stable
quaternion branch gives a deterministic axis there and preserves the correct
pi magnitude; this is not a claim of differentiability across the cut locus.
"""


def rotation_quaternion(matrix):
    """Convert proper [B,3,3] matrices to normalized WXYZ quaternions.

    Pick the largest quaternion component branch so the selected denominator
    stays well away from zero. Unselected branches also remain numerically safe
    to prevent 0*NaN contamination in autograd.
    """
    import torch
    r = matrix
    if r.ndim != 3 or r.shape[1:] != (3,3):
        raise ValueError('Expected batched 3x3 rotations')
    diagonal = torch.stack((1+r[:,0,0]+r[:,1,1]+r[:,2,2],
                            1+r[:,0,0]-r[:,1,1]-r[:,2,2],
                            1-r[:,0,0]+r[:,1,1]-r[:,2,2],
                            1-r[:,0,0]-r[:,1,1]+r[:,2,2]),-1)
    magnitude = diagonal.clamp_min(1e-12).sqrt()
    candidates = torch.stack((
        torch.stack((diagonal[:,0],r[:,2,1]-r[:,1,2],r[:,0,2]-r[:,2,0],r[:,1,0]-r[:,0,1]),-1),
        torch.stack((r[:,2,1]-r[:,1,2],diagonal[:,1],r[:,1,0]+r[:,0,1],r[:,0,2]+r[:,2,0]),-1),
        torch.stack((r[:,0,2]-r[:,2,0],r[:,1,0]+r[:,0,1],diagonal[:,2],r[:,2,1]+r[:,1,2]),-1),
        torch.stack((r[:,1,0]-r[:,0,1],r[:,0,2]+r[:,2,0],r[:,2,1]+r[:,1,2],diagonal[:,3]),-1)),1)
    candidates = candidates/(2*magnitude[:,:,None].clamp_min(.1))
    ids = magnitude.argmax(-1)
    q = candidates[torch.arange(len(r),device=r.device),ids]
    q = q/q.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    return torch.where(q[:,:1] < 0,-q,q)


def so3_log(matrix):
    import torch
    q = rotation_quaternion(matrix)
    vector = q[:,1:]
    norm = torch.linalg.vector_norm(vector,dim=-1)
    angle = 2*torch.atan2(norm,q[:,0])
    scale = torch.where(norm > 1e-6,angle/norm.clamp_min(1e-6),2+norm.square()/3)
    return vector*scale[:,None]


def rotation_velocity(rotation,dt):
    return so3_log(rotation[1:]@rotation[:-1].transpose(-1,-2))/dt
