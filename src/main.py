from data import TextData
from model import SequenceModel
from algorithm import Trainer

import torch
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf


def setup(config):
    # random seed
    torch.manual_seed(42)

    # device
    requested_device_id = config.system.device_id
    if requested_device_id.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested device '{requested_device_id}' but CUDA not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device_id)

    # wandb initialization
    wandb.init(
        project=config.wandb.project,
        name=config.wandb.name,
        config=OmegaConf.to_container(
            config, resolve=True, throw_on_missing=True
        ),
        settings=wandb.Settings(start_method="thread")
    )
    return device

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config : DictConfig) -> None:
    ## SETUP ##
    device = setup(config)

    ## DATA ##
    data = TextData(config)
    vocab_size = data.vocab_size
    print('Data Loaded.')

    ## MODEL ##
    model = SequenceModel(config, vocab_size, device)
    print('Model Created.')

    ## ALGORITHM ##
    print('Running Algorithm.')
    alg = Trainer(data, model, config, device)
    alg.run()
    print('Done!')

if __name__ == "__main__":
    main()