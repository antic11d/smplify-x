import torch
from argparse import ArgumentParser
from pathlib import Path
from pytorch3d.io import load_obj, load_ply

def load_vertices(path, pattern, device):
    meshes = []
    for p in path.glob(pattern):
        if ('011' in p.stem or '013' in p.stem or '054' in p.stem):
            continue
        if p.suffix == '.ply':
            verts, _ = load_ply(p)
            verts = verts.to(device)
        elif p.suffix == '.obj':
            verts, _, _ = load_obj(p, device=device)
            if verts.isnan().any().item() == True:
                print(p)
        meshes.append(verts)

    return torch.stack(meshes)

def main(args):
    device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')
    print(f'Running on device: {device}')
    # Load predictions
    pred_vertices = load_vertices(Path(args.pred_folder), '**/*.obj', device)

    # Load ground-truth
    gt_vertices = load_vertices(Path(args.gt_folder), '*.obj', device)

    # Calculate v2v loss
    v2v = ((pred_vertices - gt_vertices).pow(2).sum(dim=-1).sqrt().mean()).item()

    print(f'GT:\t{args.gt_folder}\npred:\t{args.pred_folder}\n\tv2v={v2v:5f}')




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gt_folder', required=True)
    parser.add_argument('--pred_folder', required=True)
    parser.add_argument('--use_cuda', action='store_true', default=False)
    args = parser.parse_args()

    main(args)