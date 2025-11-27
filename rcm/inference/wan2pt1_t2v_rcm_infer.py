# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import math

import torch
from einops import rearrange, repeat
from tqdm import tqdm

from imaginaire.utils.io import save_image_or_video
from imaginaire.lazy_config import LazyCall as L, LazyDict, instantiate
from imaginaire.utils import log

from rcm.datasets.utils import VIDEO_RES_SIZE_INFO
from rcm.utils.umt5 import clear_umt5_memory, get_umt5_embedding
from rcm.utils.model_utils import init_weights_on_device, load_state_dict
from rcm.modules.denoiser_scaling import RectifiedFlow_TrigFlowWrapper
from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface
from rcm.networks.wan2pt1 import WanModel

torch._dynamo.config.suppress_errors = True

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

video_prompts = [
    "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about.",
    "A dramatic and dynamic scene in the style of a disaster movie, depicting a powerful tsunami rushing through a narrow alley in Bulgaria. The water is turbulent and chaotic, with waves crashing violently against the walls and buildings on either side. The alley is lined with old, weathered houses, their facades partially submerged and splintered. The camera angle is low, capturing the full force of the tsunami as it surges forward, creating a sense of urgency and danger. People can be seen running frantically, adding to the chaos. The background features a distant horizon, hinting at the larger scale of the tsunami. A dynamic, sweeping shot from a low-angle perspective, emphasizing the movement and intensity of the event.",
    "Animated scene features a close-up of a short fluffy monster kneeling beside a melting red candle. The art style is 3D and realistic, with a focus on lighting and texture. The mood of the painting is one of wonder and curiosity, as the monster gazes at the flame with wide eyes and open mouth. Its pose and expression convey a sense of innocence and playfulness, as if it is exploring the world around it for the first time. The use of warm colors and dramatic lighting further enhances the cozy atmosphere of the image.",
    "The camera follows behind a white vintage SUV with a black roof rack as it speeds up a steep dirt road surrounded by pine trees on a steep mountain slope, dust kicks up from it’s tires, the sunlight shines on the SUV as it speeds along the dirt road, casting a warm glow over the scene. The dirt road curves gently into the distance, with no other cars or vehicles in sight. The trees on either side of the road are redwoods, with patches of greenery scattered throughout. The car is seen from the rear following the curve with ease, making it seem as if it is on a rugged drive through the rugged terrain. The dirt road itself is surrounded by steep hills and mountains, with a clear blue sky above with wispy clouds.",
    "A close up view of a glass sphere that has a zen garden within it. There is a small dwarf in the sphere who is raking the zen garden and creating patterns in the sand.",
    "The camera rotates around a large stack of vintage televisions all showing different programs — 1950s sci-fi movies, horror movies, news, static, a 1970s sitcom, etc, set inside a large New York museum gallery.",
    "A playful raccoon is seen playing an electronic guitar, strumming the strings with its front paws. The raccoon has distinctive black facial markings and a bushy tail. It sits comfortably on a small stool, its body slightly tilted as it focuses intently on the instrument. The setting is a cozy, dimly lit room with vintage posters on the walls, adding a retro vibe. The raccoon's expressive eyes convey a sense of joy and concentration. Medium close-up shot, focusing on the raccoon's face and hands interacting with the guitar.",
    "A close-up shot of a ceramic teacup slowly pouring water into a glass mug. The water flows smoothly from the spout of the teacup into the mug, creating gentle ripples as it fills up. Both cups have detailed textures, with the teacup having a matte finish and the glass mug showcasing clear transparency. The background is a blurred kitchen countertop, adding context without distracting from the central action. The pouring motion is fluid and natural, emphasizing the interaction between the two cups.",
    "A dynamic and chaotic scene in a dense forest during a heavy rainstorm, capturing a real girl frantically running through the foliage. Her wild hair flows behind her as she sprints, her arms flailing and her face contorted in fear and desperation. Behind her, various animals—rabbits, deer, and birds—are also running, creating a frenzied atmosphere. The girl's clothes are soaked, clinging to her body, and she is screaming and shouting as she tries to escape. The background is a blur of greenery and rain-drenched trees, with occasional glimpses of the darkening sky. A wide-angle shot from a low angle, emphasizing the urgency and chaos of the moment.",
    "A close-up shot captures a steaming hot pot brimming with vegetables and dumplings, set on a rustic wooden table. The camera focuses on the bubbling broth as a woman, dressed in a light, patterned blouse, reaches in with chopsticks to lift a tender leaf of cabbage from the simmering mixture. Steam rises around her as she leans back slightly, her warm smile reflecting satisfaction and joy. Her movements are smooth and deliberate, showcasing her comfort and familiarity with the dining process. The background includes a small bowl of dipping sauce and a clay pot, adding to the cozy, communal dining atmosphere.",
    "In an urban outdoor setting, a man dressed in a black hoodie and black track pants with white stripes walks toward a wooden bench situated near a modern building with large glass windows. He carries a black backpack slung over one shoulder and holds a stack of papers in his hand. As he approaches the bench, he bends down, places the papers on it, and then sits down. Shortly after, a woman wearing a red jacket with yellow accents and black pants joins him. She stands beside the bench, facing him, and appears to engage in a conversation. The man continues to review the papers while the woman listens attentively. In the background, other individuals can be seen walking by, some carrying bags, adding to the bustling yet casual atmosphere of the scene. The overall mood suggests a moment of focused discussion or preparation amidst a busy environment.",
    "A video featuring a woman introducing the iPhone 15, available for purchase on Shopee. The woman has a friendly and engaging demeanor, speaking clearly and confidently about the phone's features and benefits. She demonstrates the phone's camera capabilities, display quality, and user interface. The background includes subtle animations of the Shopee app and product listings. The woman wears casual, modern clothing and maintains a neutral facial expression as she interacts with the phone. The video opens with a close-up of the woman’s face, then transitions to medium shots of her handling the phone. The camera occasionally zooms in on specific features of the iPhone 15.",
    "A dynamic time-lapse video showing the rapidly moving scenery from the window of a speeding train. The camera captures various elements such as lush green fields, towering trees, quaint countryside houses, and distant mountain ranges passing by quickly. The train window frames the view, adding a sense of speed and motion as the landscape rushes past. The camera remains static but emphasizes the fast-paced movement outside. The overall atmosphere is serene yet exhilarating, capturing the essence of travel and exploration. Medium shot focusing on the train window and the rushing scenery beyond.",
    "A Minecraft player character holding a torch enters a massive underground cave. The torchlight flickers against jagged stone walls, illuminating patches of iron and diamond ores embedded in the rock. Stalactites hang from the ceiling, lava flows in glowing streams nearby, and the faint sound of water dripping echoes through the cavern.",
]

_DEFAULT_PROMPT = video_prompts[0]

tensor_kwargs = {"device": "cuda", "dtype": torch.bfloat16}

scaling = RectifiedFlow_TrigFlowWrapper(t_scaling_factor=1000)


def denoise(
    net,
    xt_B_C_T_H_W: torch.Tensor,
    time: torch.Tensor,
    condition: dict[str, torch.Tensor],
):
    if time.ndim == 1:
        time_B_T = rearrange(time, "b -> b 1")
    elif time.ndim == 2:
        time_B_T = time
    else:
        raise ValueError(f"Time shape {time.shape} is not supported")
    time_B_1_T_1_1 = rearrange(time_B_T, "b t -> b 1 t 1 1")
    # get precondition for the network
    c_skip_B_1_T_1_1, c_out_B_1_T_1_1, c_in_B_1_T_1_1, c_noise_B_1_T_1_1 = scaling(trigflow_t=time_B_1_T_1_1)

    # forward pass through the network
    net_output_B_C_T_H_W = net(
        x_B_C_T_H_W=(xt_B_C_T_H_W * c_in_B_1_T_1_1).to(**tensor_kwargs),
        timesteps_B_T=c_noise_B_1_T_1_1.squeeze(dim=[1, 3, 4]).to(**tensor_kwargs),
        **condition,
    ).float()

    x0_pred_B_C_T_H_W = c_skip_B_1_T_1_1 * xt_B_C_T_H_W + c_out_B_1_T_1_1 * net_output_B_C_T_H_W

    return x0_pred_B_C_T_H_W


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="rCM inference script for Wan2.1 T2V")
    parser.add_argument("--model_size", choices=["1.3B", "14B"], default="1.3B", help="Size of the model to use")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--num_steps", type=int, choices=[1, 2, 3, 4], default=4, help="1~4 for timestep-distilled inference")
    parser.add_argument("--sigma_max", type=float, default=80, help="Initial sigma for rCM")
    parser.add_argument("--dit_path", type=str, default="", help="Custom path to the DiT model checkpoint for distilled models.")
    parser.add_argument("--vae_path", type=str, default="assets/checkpoints/Wan2.1_VAE.pth", help="Path to the Wan2.1 VAE.")
    parser.add_argument("--text_encoder_path", type=str, default="assets/checkpoints/models_t5_umt5-xxl-enc-bf16.pth", help="Path to the umT5 text encoder.")
    parser.add_argument("--num_frames", type=int, default=77, help="Number of frames to generate")
    parser.add_argument("--prompt", type=str, default=_DEFAULT_PROMPT, help="Text prompt for video generation")
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

    net = net.to(**tensor_kwargs)
    torch.cuda.empty_cache()

    tokenizer = Wan2pt1VAEInterface(vae_pth=args.vae_path)

    w, h = VIDEO_RES_SIZE_INFO[args.resolution][args.aspect_ratio]

    log.info(f"Computing embedding for prompt: {args.prompt}")
    text_emb = get_umt5_embedding(checkpoint_path=args.text_encoder_path, prompts=args.prompt).to(dtype=torch.bfloat16).cuda()
    clear_umt5_memory()

    log.info(f"Generating with prompt: {args.prompt}")
    condition = {"crossattn_emb": repeat(text_emb.to(**tensor_kwargs), "b l d -> (k b) l d", k=args.num_samples)}

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

    # mid_t = [1.3, 1.0, 0.6][: args.num_steps - 1]
    # For better visual quality
    mid_t = [1.5, 1.4, 1.0][: args.num_steps - 1]

    t_steps = torch.tensor(
        [math.atan(args.sigma_max), *mid_t, 0],
        dtype=torch.float64,
        device=init_noise.device,
    )

    # Sampling steps
    x = init_noise.to(torch.float64)
    ones = torch.ones(x.size(0), device=x.device, dtype=x.dtype)
    total_steps = t_steps.shape[0] - 1
    for i, (t_cur, t_next) in enumerate(tqdm(list(zip(t_steps[:-1], t_steps[1:])), desc="Sampling", total=total_steps)):
        with torch.no_grad():
            x = denoise(net, x.float(), t_cur.float() * ones, condition=condition).to(torch.float64)
            x = torch.cos(t_next) * x + torch.sin(t_next) * torch.randn(
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
