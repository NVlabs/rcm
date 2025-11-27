import glob
import webdataset as wds
import torch
from torch.utils.data import DataLoader


def dict_collation_fn(samples):
    if not samples:
        return {}

    keys = samples[0].keys()
    batched_dict = {key: [] for key in keys}

    for sample in samples:
        for key in keys:
            batched_dict[key].append(sample[key])

    for key in keys:
        if isinstance(batched_dict[key][0], torch.Tensor):
            batched_dict[key] = torch.stack(batched_dict[key])

    return batched_dict


def create_dataloader(
    tar_path_pattern,  # e.g., "/path/to/dataset/shard_*.tar"
    batch_size,
    num_workers=8,
    shuffle_buffer=1000,
    prefetch_factor=2,
):
    shards = glob.glob(tar_path_pattern)
    if not shards:
        raise FileNotFoundError(f"No files found with pattern '{tar_path_pattern}'")

    dataset = wds.DataPipeline(
        wds.SimpleShardList(shards),
        # this shuffles the shards
        wds.shuffle(1000),
        wds.split_by_node,
        wds.split_by_worker,
        wds.tarfile_to_samples(),
        # this shuffles the samples in memory
        wds.shuffle(shuffle_buffer),
        wds.decode(wds.handle_extension("pt", wds.torch_loads)),
        wds.rename(latents="latent.pt", t5_text_embeddings="embed.pt", prompts="prompt.txt"),
        wds.batched(batch_size, partial=False, collation_fn=dict_collation_fn),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
    )

    return dataloader
