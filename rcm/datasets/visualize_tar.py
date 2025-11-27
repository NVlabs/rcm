import os
import io
import argparse
import tarfile
import torch
from einops import rearrange
from tqdm import tqdm

from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface
from imaginaire.utils.io import save_image_or_video


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading VAE tokenizer...")
    try:
        tokenizer = Wan2pt1VAEInterface(vae_pth=args.vae_path)
    except Exception as e:
        print(f"Error loading VAE from {args.vae_path}: {e}")
        print("Please ensure the VAE model path is correct.")
        return

    print(f"Opening tar file: {args.tar_path}")
    if not os.path.exists(args.tar_path):
        print(f"Error: Tar file not found at {args.tar_path}")
        return

    latent_files = []
    with tarfile.open(args.tar_path, "r") as tar:
        all_files = tar.getnames()
        latent_files = sorted([f for f in all_files if f.endswith(".latent.pt")])

    if not latent_files:
        print("Error: No '.latent.pt' files found in the tar archive.")
        return

    print(f"Found {len(latent_files)} latent files in the archive.")

    if args.num_samples > 0:
        latent_files = latent_files[: args.num_samples]
        print(f"Decoding the first {len(latent_files)} samples.")

    decoded_videos = []
    print("Decoding samples...")
    with tarfile.open(args.tar_path, "r") as tar:
        for member_name in tqdm(latent_files, desc="Decoding"):
            member_file = tar.extractfile(member_name)
            if member_file is None:
                print(f"Warning: Could not extract {member_name}. Skipping.")
                continue

            latent_tensor = torch.load(io.BytesIO(member_file.read()), map_location=device)

            if latent_tensor.dim() == 4:
                samples = latent_tensor.unsqueeze(0)
            else:
                samples = latent_tensor

            video = tokenizer.decode(samples.to(device, dtype=torch.bfloat16))

            decoded_videos.append(video.float().cpu())

    if not decoded_videos:
        print("No videos were decoded. Exiting.")
        return

    print("Stacking videos into a grid...")
    to_show = torch.stack(decoded_videos, dim=0)

    to_show = (1.0 + to_show.clamp(-1, 1)) / 2.0

    num_videos = to_show.shape[0]
    grid_cols = args.grid_cols
    grid_rows = (num_videos + grid_cols - 1) // grid_cols  # 向上取整

    if num_videos != grid_rows * grid_cols:
        print(f"Warning: The number of videos ({num_videos}) doesn't fit a perfect {grid_rows}x{grid_cols} grid. Rearranging may be imperfect.")

    to_show = rearrange(to_show, "(rows cols) b c t h w -> c t (rows h) (cols b w)", cols=grid_cols)

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    print(f"Saving video grid to: {args.save_path}")
    save_image_or_video(to_show, args.save_path, fps=args.fps)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode latent samples from a .tar file into a video grid.")

    parser.add_argument("--tar_path", type=str, required=True, help="Path to the input .tar file containing latent samples.")
    parser.add_argument("--save_path", type=str, default="preview.mp4", help="Path to save the output video file.")
    parser.add_argument("--vae_path", type=str, default="assets/checkpoints/Wan2.1_VAE.pth", help="Path to the Wan2.1 VAE model weights.")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of samples to decode from the tar file. Set to 0 to decode all.")
    parser.add_argument("--grid_cols", type=int, default=4, help="Number of columns in the output video grid.")
    parser.add_argument("--fps", type=int, default=16, help="Frames per second for the output video.")

    args = parser.parse_args()
    main(args)
