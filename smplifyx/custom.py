import yaml
from pathlib import Path

from networks import VAEBuilder
import torch

def read_yaml(path):
    with open(path, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data


def load_model(f, model_rel):
    if not isinstance(f, Path):
        f = Path(f)

    cfg = read_yaml(f / 'config.yml')
    model = VAEBuilder.from_cfg(cfg)
    ckp = CheckpointIO(f, model, None, cfg)
    _ = ckp.load(f / model_rel)

    return model


class CheckpointIO:
    def __init__(self, checkpoint_dir, model, optimizer, cfg):
        self.module_dict_params = {
            f"{cfg['model']}_model": model,
            f'optimizer': optimizer,
            f"{cfg['model']}_config": cfg['model'],
        }
        self.checkpoint_dir = checkpoint_dir
        self.cfg = cfg

    def save(self, filename, **kwargs):
        out_dict = kwargs

        for k, v in self.module_dict_params.items():
            out_dict[k] = v
            if hasattr(v, 'state_dict'):
                out_dict[k] = v.state_dict()

        torch.save(out_dict, self.checkpoint_dir / filename)

    def load(self, filename):
        state_dict = torch.load(filename, map_location=torch.device(self.cfg['device']))

        for k, v in self.module_dict_params.items():
            if hasattr(v, 'load_state_dict') and v is not None:
                v.load_state_dict(state_dict[k])

        scalars = {k: v for k, v in state_dict.items() if k not in self.module_dict_params}
        return scalars
