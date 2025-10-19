# Wan Examples for rCM

## Checkpoints Downloading

The Wan2.1 VAE and umT5 text encoder can be obtained from the official [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) repository. To download all required checkpoints into the `checkpoints/` directory, run:

```bash
mkdir checkpoints
cd checkpoints

# Wan2.1 VAE and text encoder
wget https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/Wan2.1_VAE.pth
wget https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/models_t5_umt5-xxl-enc-bf16.pth

cd ..
```

## Generation Examples

### 14B 480p

#### Example 1
Command:
```bash
PYTHONPATH=.  python rcm/inference/wan2pt1_t2v_rcm_infer.py \
    --dit_path checkpoints/rCM_Wan2.1_T2V_14B_480p.pt \
    --model_size 14B \
    --num_samples 5 \
    --prompt "an alarm clock"
```
Output video:
<video src="https://github.com/user-attachments/assets/231e4563-9e06-425a-86a1-09fee8004549" controls></video>

#### Example 2
Command:
```bash
PYTHONPATH=.  python rcm/inference/wan2pt1_t2v_rcm_infer.py \
    --dit_path checkpoints/rCM_Wan2.1_T2V_14B_480p.pt \
    --model_size 14B \
    --num_samples 5 \
    --prompt "A dynamic time-lapse video showing the rapidly moving scenery from the window of a speeding train. The camera captures various elements such as lush green fields, towering trees, quaint countryside houses, and distant mountain ranges passing by quickly. The train window frames the view, adding a sense of speed and motion as the landscape rushes past. The camera remains static but emphasizes the fast-paced movement outside. The overall atmosphere is serene yet exhilarating, capturing the essence of travel and exploration. Medium shot focusing on the train window and the rushing scenery beyond."
```
Output video:
<video src="https://github.com/user-attachments/assets/a8033bbc-23dd-4f64-8316-f6d8e15221d9" controls></video>

#### Example 3
Command:
```bash
PYTHONPATH=.  python rcm/inference/wan2pt1_t2v_rcm_infer.py \
    --dit_path checkpoints/rCM_Wan2.1_T2V_14B_480p.pt \
    --model_size 14B \
    --num_samples 5 \
    --prompt "A video featuring a woman introducing the iPhone 15, available for purchase on Shopee. The woman has a friendly and engaging demeanor, speaking clearly and confidently about the phone's features and benefits. She demonstrates the phone's camera capabilities, display quality, and user interface. The background includes subtle animations of the Shopee app and product listings. The woman wears casual, modern clothing and maintains a neutral facial expression as she interacts with the phone. The video opens with a close-up of the woman’s face, then transitions to medium shots of her handling the phone. The camera occasionally zooms in on specific features of the iPhone 15."
```
Output video:
<video src="https://github.com/user-attachments/assets/04e33fa2-311f-41e2-99d5-9b3204ab653d" controls></video>

#### Example 4
Command:
```bash
PYTHONPATH=.  python rcm/inference/wan2pt1_t2v_rcm_infer.py \
    --dit_path checkpoints/rCM_Wan2.1_T2V_14B_480p.pt \
    --model_size 14B \
    --num_samples 5 \
    --prompt "A Minecraft player character holding a torch enters a massive underground cave. The torchlight flickers against jagged stone walls, illuminating patches of iron and diamond ores embedded in the rock. Stalactites hang from the ceiling, lava flows in glowing streams nearby, and the faint sound of water dripping echoes through the cavern."
```
Output video:
<video src="https://github.com/user-attachments/assets/8195b3d6-2d4a-46c6-b313-606c526d7c6a" controls></video>

#### Example 5
Command:
```bash
PYTHONPATH=.  python rcm/inference/wan2pt1_t2v_rcm_infer.py \
    --dit_path checkpoints/rCM_Wan2.1_T2V_14B_480p.pt \
    --model_size 14B \
    --num_samples 5 \
    --prompt "In an urban outdoor setting, a man dressed in a black hoodie and black track pants with white stripes walks toward a wooden bench situated near a modern building with large glass windows. He carries a black backpack slung over one shoulder and holds a stack of papers in his hand. As he approaches the bench, he bends down, places the papers on it, and then sits down. Shortly after, a woman wearing a red jacket with yellow accents and black pants joins him. She stands beside the bench, facing him, and appears to engage in a conversation. The man continues to review the papers while the woman listens attentively. In the background, other individuals can be seen walking by, some carrying bags, adding to the bustling yet casual atmosphere of the scene. The overall mood suggests a moment of focused discussion or preparation amidst a busy environment."
```
Output video:
<video src="https://github.com/user-attachments/assets/83660db1-3341-4c6d-a03e-cb5ff2709d30" controls></video>

#### Example 6
Command:
```bash
PYTHONPATH=.  python rcm/inference/wan2pt1_t2v_rcm_infer.py \
    --dit_path checkpoints/rCM_Wan2.1_T2V_14B_480p.pt \
    --model_size 14B \
    --num_samples 5 \
    --aspect_ratio 9:16 \
    --prompt "A close-up shot captures a steaming hot pot brimming with vegetables and dumplings, set on a rustic wooden table. The camera focuses on the bubbling broth as a woman, dressed in a light, patterned blouse, reaches in with chopsticks to lift a tender leaf of cabbage from the simmering mixture. Steam rises around her as she leans back slightly, her warm smile reflecting satisfaction and joy. Her movements are smooth and deliberate, showcasing her comfort and familiarity with the dining process. The background includes a small bowl of dipping sauce and a clay pot, adding to the cozy, communal dining atmosphere."
```
Output video:
<video src="https://github.com/user-attachments/assets/9376b429-75be-4ba2-8b58-dcd2fd2e7b83" controls></video>

### 14B 720p

#### Example 1
Command:
```bash
PYTHONPATH=.  python rcm/inference/wan2pt1_t2v_rcm_infer.py \
    --dit_path checkpoints/rCM_Wan2.1_T2V_14B_720p.pt \
    --model_size 14B \
    --num_samples 4 \
    --resolution 720p \
    --sigma_max 120 \
    --prompt "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about."
```
Output video:
<video src="https://github.com/user-attachments/assets/b7b742a1-2f84-417c-b3ce-f99f5ffaeaa5" controls></video>

#### Example 2
Command:
```bash
PYTHONPATH=.  python rcm/inference/wan2pt1_t2v_rcm_infer.py \
    --dit_path checkpoints/rCM_Wan2.1_T2V_14B_720p.pt \
    --model_size 14B \
    --num_samples 4 \
    --resolution 720p \
    --sigma_max 120 \
    --prompt "A close up view of a glass sphere that has a zen garden within it. There is a small dwarf in the sphere who is raking the zen garden and creating patterns in the sand."
```
Output video:
<video src="https://github.com/user-attachments/assets/d4387c5f-8e10-47ea-a050-912d9802e3e5" controls></video>

#### Example 3
Command:
```bash
PYTHONPATH=.  python rcm/inference/wan2pt1_t2v_rcm_infer.py \
    --dit_path checkpoints/rCM_Wan2.1_T2V_14B_720p.pt \
    --model_size 14B \
    --num_samples 4 \
    --resolution 720p \
    --sigma_max 120 \
    --prompt "an alarm clock"
```
Output video:
<video src="https://github.com/user-attachments/assets/86eb6a3e-4ce6-42e6-8110-02def012f47a" controls></video>

#### Example 4
Command:
```bash
PYTHONPATH=.  python rcm/inference/wan2pt1_t2v_rcm_infer.py \
    --dit_path checkpoints/rCM_Wan2.1_T2V_14B_720p.pt \
    --model_size 14B \
    --num_samples 4 \
    --resolution 720p \
    --sigma_max 120 \
    --prompt "Low drone footage over a bustling Tuscan town square, people laughing, dogs running around"
```
Output video:
<video src="https://github.com/user-attachments/assets/c47231e3-521d-4fd9-bc59-806450cdb8ed" controls></video>

#### Example 5
Command:
```bash
PYTHONPATH=.  python rcm/inference/wan2pt1_t2v_rcm_infer.py \
    --dit_path checkpoints/rCM_Wan2.1_T2V_14B_720p.pt \
    --model_size 14B \
    --num_samples 4 \
    --resolution 720p \
    --sigma_max 120 \
    --prompt "A video featuring a woman introducing the iPhone 15, available for purchase on Shopee. The woman has a friendly and engaging demeanor, speaking clearly and confidently about the phone's features and benefits. She demonstrates the phone's camera capabilities, display quality, and user interface. The background includes subtle animations of the Shopee app and product listings. The woman wears casual, modern clothing and maintains a neutral facial expression as she interacts with the phone. The video opens with a close-up of the woman’s face, then transitions to medium shots of her handling the phone. The camera occasionally zooms in on specific features of the iPhone 15."
```
Output video:
<video src="https://github.com/user-attachments/assets/7979a411-644e-4956-aee9-98d9adc1f1f5" controls></video>