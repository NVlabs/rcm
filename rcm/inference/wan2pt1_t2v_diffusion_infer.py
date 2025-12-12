from tqdm import tqdm
import argparse
import torch
from einops import rearrange, repeat

from imaginaire.utils.io import save_image_or_video
from imaginaire.lazy_config import LazyCall as L, LazyDict, instantiate
from imaginaire.utils import log

from rcm.datasets.utils import VIDEO_RES_SIZE_INFO
from rcm.utils.umt5 import clear_umt5_memory, get_umt5_embedding
from rcm.utils.model_utils import init_weights_on_device, load_state_dict
from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface
from rcm.networks.wan2pt1 import WanModel
from rcm.samplers.euler import FlowEulerSampler
from rcm.samplers.unipc import FlowUniPCMultistepSampler

_DEFAULT_NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
_DEFAULT_PROMPT = "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about."

tensor_kwargs = {"device": "cuda", "dtype": torch.bfloat16}

WAN2PT1_1PT3B_T2V: LazyDict = L(WanModel)(
    dim=1536,
    eps=1e-06,
    ffn_dim=8960,
    freq_dim=256,
    in_dim=16,
    model_type="t2v",
    num_heads=12,
    num_layers=30,
    out_dim=16,
    text_len=512,
)

WAN2PT1_14B_T2V: LazyDict = L(WanModel)(
    dim=5120,
    eps=1e-06,
    ffn_dim=13824,
    freq_dim=256,
    in_dim=16,
    model_type="t2v",
    num_heads=40,
    num_layers=40,
    out_dim=16,
    text_len=512,
)

dit_configs = {"1.3B": WAN2PT1_1PT3B_T2V, "14B": WAN2PT1_14B_T2V}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diffusion inference script for Wan2.1 T2V")
    parser.add_argument("--model_size", choices=["1.3B", "14B"], default="14B", help="Size of the model to use")
    parser.add_argument("--num_samples", type=int, default=4, help="Number of samples to generate")
    parser.add_argument("--num_steps", type=int, default=50, help="Number of sampling steps")
    parser.add_argument("--sigma_max", type=int, default=5000, help="Initial timestep represented by EDM sigma")
    parser.add_argument("--sampler", choices=["Euler", "UniPC"], default="UniPC", help="Sampler")
    parser.add_argument("--guidance_scale", type=float, default=5.0, help="CFG scale")
    parser.add_argument("--timestep_shift", type=float, default=5.0, help="Timestep shift as in Wan")
    parser.add_argument("--dit_path", type=str, default="assets/checkpoints/Wan2.1-T2V-14B.pth", help="Path to the video diffusion model.")
    parser.add_argument("--vae_path", type=str, default="assets/checkpoints/Wan2.1_VAE.pth", help="Path to the Wan2.1 VAE.")
    parser.add_argument(
        "--text_encoder_path", type=str, default="assets/checkpoints/models_t5_umt5-xxl-enc-bf16.pth", help="Path to the umT5 text encoder."
    )
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames to generate")
    parser.add_argument("--prompt", type=str, default=_DEFAULT_PROMPT, help="Text prompt for video generation")
    parser.add_argument("--negative_prompt", type=str, default=_DEFAULT_NEGATIVE_PROMPT, help="Negative text prompt for video generation")
    parser.add_argument("--resolution", default="480p", type=str, help="Resolution of the generated output")
    parser.add_argument("--aspect_ratio", default="16:9", type=str, help="Aspect ratio of the generated output (width:height)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument(
        "--save_path", type=str, default="output/generated_video.mp4", help="Path to save the generated video (include file extension)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    with init_weights_on_device():
        net = instantiate(dit_configs[args.model_size]).eval()  # inference

    state_dict = load_state_dict(args.dit_path)
    prefix_to_load = "net."
    # drop net. prefix
    state_dict_dit_compatible = dict()
    for k, v in state_dict.items():
        if k.startswith(prefix_to_load):
            state_dict_dit_compatible[k[len(prefix_to_load) :]] = v
        else:
            state_dict_dit_compatible[k] = v
    net.load_state_dict(state_dict_dit_compatible, strict=False, assign=True)
    del state_dict, state_dict_dit_compatible
    log.success(f"Successfully loaded DiT from {args.dit_path}")

    net.to(**tensor_kwargs).cpu()
    torch.cuda.empty_cache()

    tokenizer = Wan2pt1VAEInterface(vae_pth=args.vae_path)

    w, h = VIDEO_RES_SIZE_INFO[args.resolution][args.aspect_ratio]

    log.info(f"Computing embedding for prompt: {args.prompt}")
    text_emb = get_umt5_embedding(checkpoint_path=args.text_encoder_path, prompts=args.prompt).to(dtype=torch.bfloat16).cuda()
    neg_text_emb = get_umt5_embedding(checkpoint_path=args.text_encoder_path, prompts=args.negative_prompt).to(dtype=torch.bfloat16).cuda()
    clear_umt5_memory()

    log.info(f"Generating with prompt: {args.prompt}")
    condition = {"crossattn_emb": repeat(text_emb.to(**tensor_kwargs), "b l d -> (k b) l d", k=args.num_samples)}
    uncondition = {"crossattn_emb": repeat(neg_text_emb.to(**tensor_kwargs), "b l d -> (k b) l d", k=args.num_samples)}

    to_show = []

    state_shape = [
        tokenizer.latent_ch,
        tokenizer.get_latent_num_frames(args.num_frames),
        h // tokenizer.spatial_compression_factor,
        w // tokenizer.spatial_compression_factor,
    ]

    generator = torch.Generator(device=tensor_kwargs["device"])
    generator.manual_seed(args.seed)

    init_noise = torch.randn(
        args.num_samples,
        *state_shape,
        dtype=torch.float32,
        device=tensor_kwargs["device"],
        generator=generator,
    )

    x = init_noise.to(torch.float64)

    sigma_max = args.sigma_max / (args.sigma_max + 1)
    unshifted_sigma_max = sigma_max / (args.timestep_shift - (args.timestep_shift - 1) * sigma_max)
    # log.info(unshifted_sigma_max)

    samplers = {"Euler": FlowEulerSampler, "UniPC": FlowUniPCMultistepSampler}
    sampler = samplers[args.sampler](num_train_timesteps=1000, sigma_max=unshifted_sigma_max, sigma_min=0.0)
    sampler.set_timesteps(num_inference_steps=args.num_steps, device=tensor_kwargs["device"], shift=args.timestep_shift)

    # log.info(sampler.timesteps)

    ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
    net.cuda()
    for _, t in enumerate(tqdm(sampler.timesteps)):
        timesteps = t * ones

        with torch.no_grad():
            v_cond = net(x_B_C_T_H_W=x.to(**tensor_kwargs), timesteps_B_T=timesteps.to(**tensor_kwargs), **condition).float()
            v_uncond = net(x_B_C_T_H_W=x.to(**tensor_kwargs), timesteps_B_T=timesteps.to(**tensor_kwargs), **uncondition).float()

        v_pred = v_uncond + args.guidance_scale * (v_cond - v_uncond)

        x = sampler.step(v_pred, t, x)

    samples = x.float()

    video = tokenizer.decode(samples)

    to_show.append(video.float().cpu())

    to_show = (1.0 + torch.stack(to_show, dim=0).clamp(-1, 1)) / 2.0

    save_image_or_video(rearrange(to_show, "n b c t h w -> c t (n h) (b w)"), args.save_path, fps=16)
