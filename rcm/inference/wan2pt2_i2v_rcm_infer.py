import argparse
import math

import torch
from einops import rearrange, repeat
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.v2 as T
import numpy as np

from imaginaire.utils.io import save_image_or_video
from imaginaire.lazy_config import LazyCall as L, LazyDict, instantiate
from imaginaire.utils import log

from rcm.datasets.utils import VIDEO_RES_SIZE_INFO
from rcm.utils.umt5 import clear_umt5_memory, get_umt5_embedding
from rcm.utils.model_utils import init_weights_on_device, load_state_dict
from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface
from rcm.networks.wan2pt2 import WanModel

torch._dynamo.config.suppress_errors = True

tensor_kwargs = {"device": "cuda", "dtype": torch.bfloat16}

WAN2PT2_A14B_I2V: LazyDict = L(WanModel)(
    dim=5120,
    eps=1e-06,
    ffn_dim=13824,
    freq_dim=256,
    in_dim=36,
    model_type="i2v",
    num_heads=40,
    num_layers=40,
    out_dim=16,
    text_len=512,
)

dit_configs = {"A14B": WAN2PT2_A14B_I2V}


def load_dit_model(model_path, model_config):
    """Instantiates, loads state dict, and moves a DiT model to the correct device."""
    with init_weights_on_device():
        model = instantiate(model_config).eval()

    state_dict = load_state_dict(model_path)
    prefix_to_load = "net."
    state_dict_dit_compatible = {k[len(prefix_to_load) :] if k.startswith(prefix_to_load) else k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict_dit_compatible, strict=True, assign=True)
    del state_dict, state_dict_dit_compatible
    log.success(f"Successfully loaded DiT from {model_path}")
    return model


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="rCM inference script for Wan2.2 I2V with High/Low Noise models")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image for I2V generation.")
    parser.add_argument(
        "--high_noise_model_path", type=str, default="assets/checkpoints/Wan2.2-I2V-A14B-high-rCM-merged.pth", help="Path to the high-noise model."
    )
    parser.add_argument(
        "--low_noise_model_path", type=str, default="assets/checkpoints/Wan2.2-I2V-A14B-low-rCM-merged.pth", help="Path to the low-noise model."
    )
    parser.add_argument("--boundary", type=float, default=0.9, help="Timestep boundary for switching from high to low noise model.")

    parser.add_argument("--model_size", choices=["A14B"], default="A14B", help="Size of the model to use")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--num_steps", type=int, choices=[1, 2, 3, 4], default=4, help="1~4 for timestep-distilled inference")
    parser.add_argument("--sigma_max", type=float, default=200, help="Initial sigma for rCM")
    parser.add_argument("--vae_path", type=str, default="assets/checkpoints/Wan2.1_VAE.pth", help="Path to the Wan2.1 VAE.")
    parser.add_argument(
        "--text_encoder_path", type=str, default="assets/checkpoints/models_t5_umt5-xxl-enc-bf16.pth", help="Path to the umT5 text encoder."
    )
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames to generate")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for video generation")
    parser.add_argument("--resolution", default="720p", type=str, help="Resolution of the generated output")
    parser.add_argument("--aspect_ratio", default="16:9", type=str, help="Aspect ratio of the generated output (width:height)")
    parser.add_argument(
        "--adaptive_resolution",
        action="store_true",
        help="If set, adapts the output resolution to the input image's aspect ratio, "
        "using the area defined by --resolution and --aspect_ratio as a target.",
    )
    parser.add_argument("--ode", action="store_true", help="Use ODE for sampling (sharper but less robust than SDE)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument(
        "--save_path", type=str, default="output/generated_video.mp4", help="Path to save the generated video (include file extension)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    model_config = dit_configs[args.model_size]
    high_noise_model = load_dit_model(args.high_noise_model_path, model_config).to(**tensor_kwargs).cpu()
    low_noise_model = load_dit_model(args.low_noise_model_path, model_config).to(**tensor_kwargs).cpu()
    torch.cuda.empty_cache()

    tokenizer = Wan2pt1VAEInterface(vae_pth=args.vae_path)

    log.info(f"Loading and preprocessing image from: {args.image_path}")
    input_image = Image.open(args.image_path).convert("RGB")
    if args.adaptive_resolution:
        log.info("Adaptive resolution mode enabled.")
        base_w, base_h = VIDEO_RES_SIZE_INFO[args.resolution][args.aspect_ratio]
        max_resolution_area = base_w * base_h
        log.info(f"Target area is based on {args.resolution} {args.aspect_ratio} (~{max_resolution_area} pixels).")

        orig_w, orig_h = input_image.size
        image_aspect_ratio = orig_h / orig_w

        ideal_w = np.sqrt(max_resolution_area / image_aspect_ratio)
        ideal_h = np.sqrt(max_resolution_area * image_aspect_ratio)

        stride = tokenizer.spatial_compression_factor * 2
        lat_h = round(ideal_h / stride)
        lat_w = round(ideal_w / stride)
        h = lat_h * stride
        w = lat_w * stride

        log.info(f"Input image aspect ratio: {image_aspect_ratio:.4f}. Adaptive resolution set to: {w}x{h}")
    else:
        log.info("Fixed resolution mode.")
        w, h = VIDEO_RES_SIZE_INFO[args.resolution][args.aspect_ratio]
        log.info(f"Resolution set to: {w}x{h}")
    F = args.num_frames
    lat_h = h // tokenizer.spatial_compression_factor
    lat_w = w // tokenizer.spatial_compression_factor
    lat_t = tokenizer.get_latent_num_frames(F)

    log.info(f"Preprocessing image to {w}x{h}...")
    image_transforms = T.Compose(
        [
            T.ToImage(),
            T.Resize(size=(h, w), antialias=True),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    image_tensor = image_transforms(input_image).unsqueeze(0).to(device=tensor_kwargs["device"], dtype=torch.float32)

    with torch.no_grad():
        frames_to_encode = torch.cat(
            [image_tensor.unsqueeze(2), torch.zeros(1, 3, F - 1, h, w, device=image_tensor.device)], dim=2
        )  # -> B, C, T, H, W
        encoded_latents = tokenizer.encode(frames_to_encode)  # -> B, C_lat, T_lat, H_lat, W_lat

    msk = torch.zeros(1, 4, lat_t, lat_h, lat_w, device=tensor_kwargs["device"], dtype=tensor_kwargs["dtype"])
    msk[:, :, 0, :, :] = 1.0

    y = torch.cat([msk, encoded_latents.to(**tensor_kwargs)], dim=1)
    y = y.repeat(args.num_samples, 1, 1, 1, 1)

    log.info(f"Computing embedding for prompt: {args.prompt}")
    text_emb = get_umt5_embedding(checkpoint_path=args.text_encoder_path, prompts=args.prompt).to(dtype=torch.bfloat16).cuda()
    clear_umt5_memory()

    log.info(f"Generating with prompt: {args.prompt}")
    condition = {"crossattn_emb": repeat(text_emb.to(**tensor_kwargs), "b l d -> (k b) l d", k=args.num_samples), "y_B_C_T_H_W": y}

    to_show = []

    state_shape = [tokenizer.latent_ch, lat_t, lat_h, lat_w]

    generator = torch.Generator(device=tensor_kwargs["device"])
    generator.manual_seed(args.seed)

    init_noise = torch.randn(
        args.num_samples,
        *state_shape,
        dtype=torch.float32,
        device=tensor_kwargs["device"],
        generator=generator,
    )

    mid_t = [1.5, 1.4, 1.0][: args.num_steps - 1]

    t_steps = torch.tensor(
        [math.atan(args.sigma_max), *mid_t, 0],
        dtype=torch.float64,
        device=init_noise.device,
    )

    # Convert TrigFlow timesteps to RectifiedFlow
    t_steps = torch.sin(t_steps) / (torch.cos(t_steps) + torch.sin(t_steps))

    x = init_noise.to(torch.float64) * t_steps[0]
    ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
    total_steps = t_steps.shape[0] - 1
    high_noise_model.cuda()
    net = high_noise_model
    switched = False
    for i, (t_cur, t_next) in enumerate(tqdm(list(zip(t_steps[:-1], t_steps[1:])), desc="Sampling", total=total_steps)):
        if t_cur.item() < args.boundary and not switched:
            high_noise_model.cpu()
            low_noise_model.cuda()
            net = low_noise_model
            switched = True
            log.info("Switched to low noise model.")
        with torch.no_grad():
            v_pred = net(x_B_C_T_H_W=x.to(**tensor_kwargs), timesteps_B_T=(t_cur.float() * ones * 1000).to(**tensor_kwargs), **condition).to(
                torch.float64
            )
            if args.ode:
                x = x - (t_cur - t_next) * v_pred
            else:
                x = (1 - t_next) * (x - t_cur * v_pred) + t_next * torch.randn(
                    *x.shape,
                    dtype=torch.float32,
                    device=tensor_kwargs["device"],
                    generator=generator,
                )
    samples = x.float()

    video = tokenizer.decode(samples)

    to_show.append(video.float().cpu())

    to_show = (1.0 + torch.stack(to_show, dim=0).clamp(-1, 1)) / 2.0

    save_image_or_video(rearrange(to_show, "n b c t h w -> c t (n h) (b w)"), args.save_path, fps=16)
