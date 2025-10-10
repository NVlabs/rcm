<h1 align="center"> rCM: Score-Regularized Continuous-Time Consistency Model <br>🚀SOTA Diffusion Distillation & Few-Step Video Generation </h1>
<div align="center">
  <a href='https://arxiv.org/abs/2510.08431'><img src='https://img.shields.io/badge/Paper%20(arXiv)-2510.08431-red?logo=arxiv'></a>  &nbsp;
  <a href='https://research.nvidia.com/labs/dir/rcm'><img src='https://img.shields.io/badge/Website-green?logo=homepage&logoColor=white'></a> &nbsp;
</div>

## Overview

rCM is the first work that:
- Scales up continuous-time consistency distillation (e.g., sCM/MeanFlow) to 10B+ parameter video diffusion models.
- Provides open-sourced FlashAttention-2 Jacobian-vector product (JVP) kernel with support for parallelisms like FSDP/CP.
- Identifies the quality bottleneck of sCM and overcomes it via a forward–reverse divergence joint distillation framework.
- Delivers models that generate videos with both high quality and strong diversity in only 2~4 steps.

#### Comparison with Other Diffusion Distillation Methods on Wan2.1 T2V 1.3B (4-step)

| sCM | DMD2 | rCM (Ours) |
| --- | --- | --- |
| <video src="https://github.com/user-attachments/assets/50693577-9a32-4b98-86ad-d4e1be4affdc" alt="sCM" controls></video> | <video src="https://github.com/user-attachments/assets/3f1ad494-9f13-4b2f-bf3e-b99ef98dbae4" alt="DMD2" controls></video> | <video src="https://github.com/user-attachments/assets/3da35a11-8ce6-4232-9aa2-6b3bc8b7cabf" alt="rCM" controls></video> |

rCM achieves both high quality and strong diversity.

#### Performance under Fewer (1~2) Steps

| 1-step | 2-step | 4-step |
| --- | --- | --- |
| <video src="https://github.com/user-attachments/assets/fffab30d-de3f-4b86-b3b6-54208761d18b" alt="1-step" controls></video> | <video src="https://github.com/user-attachments/assets/e5477835-861f-4333-a99e-040b99186de5" alt="2-step" controls></video> | <video src="https://github.com/user-attachments/assets/8c39b50e-72df-411b-8c8e-ef69a5d3431f" alt="4-step" controls></video> |

#### 5 Random Videos with Distilled Wan2.1 T2V 14B (4-step)

<video src="https://github.com/user-attachments/assets/b1e3b786-134b-429d-b859-840646502c9b" controls></video>

## Getting Started
This codebase is built on top of [Cosmos-Predict2](https://github.com/nvidia-cosmos/cosmos-predict2). Please follow its environment setup instructions.

## Inference

This is an unofficial but faithfully reproduced rCM model with Wan, available at [rcm-Wan · Hugging Face](https://huggingface.co/worstcoder/rcm-Wan). Below is an example inference script for running rCM on T2V:

```bash
# Basic usage:
#   PYTHONPATH=. python rcm/inference/wan2pt1_t2v_rcm_infer.py [arguments]

# Arguments:
# --model_size         Model size: "1.3B" or "14B" (default: 1.3B)
# --num_samples        Number of videos to generate (default: 1)
# --num_steps          Sampling steps, 1–4 (default: 4)
# --sigma_max          Initial sigma for rCM (default: 80); larger choices (e.g., 1600) reduce diversity but may enhance quality
# --dit_path           Path to the distilled DiT model checkpoint (REQUIRED for inference)
# --vae_path           Path to Wan2.1 VAE (default: checkpoints/Wan2.1_VAE.pth)
# --text_encoder_path  Path to umT5 text encoder (default: checkpoints/models_t5_umt5-xxl-enc-bf16.pth)
# --prompt             Text prompt for video generation (default: A stylish woman walks down a Tokyo street...)
# --resolution         Output resolution, e.g. "480p", "720p" (default: 480p)
# --aspect_ratio       Aspect ratio in W:H format (default: 16:9)
# --seed               Random seed for reproducibility (default: 0)
# --save_path          Output file path including extension (default: output/generated_video.mp4)


# Example
PYTHONPATH=.  python rcm/inference/wan2pt1_t2v_rcm_infer.py \
    --dit_path checkpoints/rCM_Wan2.1_T2V_1.3B_480p.pt \
    --num_samples 5 \
    --prompt "A cinematic shot of a snowy mountain at sunrise"
```

See [Wan examples](Wan.md) for additional usage examples.

## Training
The full distillation pipeline still requires refactoring. We provide essential reference code for key components:
- FlashAttention-2 JVP kernel: `rcm/utils/flash_attention_jvp_triton.py`
- JVP-adapted Wan2.1 student network: `rcm/networks/wan2pt1_jvp.py`
- Training: `rcm/models/t2v_model_distill_rcm.py`

## Future Directions

There are promising directions to explore based on rCM. For example:
- Few-step distilled models lag behind the teacher in aspects such as physical consistency; this can potentially be improved via reward-based post-training.
- The forward–reverse divergence joint distillation framework of rCM could be extended to autoregressive video diffusion. 

## Acknowledgement
We thank the [Cosmos-Predict2](https://github.com/nvidia-cosmos/cosmos-predict2) project for providing the awesome open-source video diffusion training codebase.

## Citation
```
@article{zheng2025rcm,
  title={Large Scale Diffusion Distillation via Score-Regularized Continuous-Time Consistency},
  author={Zheng, Kaiwen and Wang, Yuji and Ma, Qianli and Chen, Huayu and Zhang, Jintao and Balaji, Yogesh and Chen, Jianfei and Liu, Ming-Yu and Zhu, Jun and Zhang, Qinsheng},
  journal={arXiv preprint arXiv:2510.08431},
  year={2025}
}
```
