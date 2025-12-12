# Wan Examples for rCM

## Checkpoints Downloading

The Wan2.1 VAE and umT5 text encoder can be obtained from the official [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) repository. To download all required checkpoints into the `assets/checkpoints/` directory, run:

```bash
mkdir assets/checkpoints
cd assets/checkpoints

# Wan2.1 VAE and text encoder
wget https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/Wan2.1_VAE.pth
wget https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/models_t5_umt5-xxl-enc-bf16.pth

cd ..
```

## 4-Step Generation Examples

### Wan2.1 T2V 14B 480p

#### Example 1
Command:
```bash
PYTHONPATH=.  python rcm/inference/wan2pt1_t2v_rcm_infer.py \
    --dit_path assets/checkpoints/rCM_Wan2.1_T2V_14B_480p.pt \
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
    --dit_path assets/checkpoints/rCM_Wan2.1_T2V_14B_480p.pt \
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
    --dit_path assets/checkpoints/rCM_Wan2.1_T2V_14B_480p.pt \
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
    --dit_path assets/checkpoints/rCM_Wan2.1_T2V_14B_480p.pt \
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
    --dit_path assets/checkpoints/rCM_Wan2.1_T2V_14B_480p.pt \
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
    --dit_path assets/checkpoints/rCM_Wan2.1_T2V_14B_480p.pt \
    --model_size 14B \
    --num_samples 5 \
    --aspect_ratio 9:16 \
    --prompt "A close-up shot captures a steaming hot pot brimming with vegetables and dumplings, set on a rustic wooden table. The camera focuses on the bubbling broth as a woman, dressed in a light, patterned blouse, reaches in with chopsticks to lift a tender leaf of cabbage from the simmering mixture. Steam rises around her as she leans back slightly, her warm smile reflecting satisfaction and joy. Her movements are smooth and deliberate, showcasing her comfort and familiarity with the dining process. The background includes a small bowl of dipping sauce and a clay pot, adding to the cozy, communal dining atmosphere."
```
Output video:
<video src="https://github.com/user-attachments/assets/9376b429-75be-4ba2-8b58-dcd2fd2e7b83" controls></video>

### Wan2.1 T2V 14B 720p

#### Example 1
Command:
```bash
PYTHONPATH=.  python rcm/inference/wan2pt1_t2v_rcm_infer.py \
    --dit_path assets/checkpoints/rCM_Wan2.1_T2V_14B_720p.pt \
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
    --dit_path assets/checkpoints/rCM_Wan2.1_T2V_14B_720p.pt \
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
    --dit_path assets/checkpoints/rCM_Wan2.1_T2V_14B_720p.pt \
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
    --dit_path assets/checkpoints/rCM_Wan2.1_T2V_14B_720p.pt \
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
    --dit_path assets/checkpoints/rCM_Wan2.1_T2V_14B_720p.pt \
    --model_size 14B \
    --num_samples 4 \
    --resolution 720p \
    --sigma_max 120 \
    --prompt "A video featuring a woman introducing the iPhone 15, available for purchase on Shopee. The woman has a friendly and engaging demeanor, speaking clearly and confidently about the phone's features and benefits. She demonstrates the phone's camera capabilities, display quality, and user interface. The background includes subtle animations of the Shopee app and product listings. The woman wears casual, modern clothing and maintains a neutral facial expression as she interacts with the phone. The video opens with a close-up of the woman’s face, then transitions to medium shots of her handling the phone. The camera occasionally zooms in on specific features of the iPhone 15."
```
Output video:
<video src="https://github.com/user-attachments/assets/7979a411-644e-4956-aee9-98d9adc1f1f5" controls></video>

### Wan2.2 I2V A14B 720p

While rCM is only trained for Wan2.1 T2V, the weight delta can be directly applied to Wan2.2 I2V through model merging (like LoRA) and produce similar speed-up. This is because Wan2.1-14B and Wan2.2-14B share almost the same network architecture.

#### Example 1
Command:
```bash
PYTHONPATH=.  python rcm/inference/wan2pt2_i2v_rcm_infer.py \
    --high_noise_model_path assets/checkpoints/Wan2.2-I2V-A14B-high-rCM4.0-merged.pt \
    --low_noise_model_path assets/checkpoints/Wan2.2-I2V-A14B-low-rCM1.0-merged.pt \
    --num_samples 2 \
    --prompt "POV selfie video, ultra-messy and extremely fast. A white cat in sunglasses stands on a surfboard with a neutral look when the board suddenly whips sideways, throwing cat and camera into the water; the frame dives sharply downward, swallowed by violent bursts of bubbles, spinning turbulence, and smeared water streaks as the camera sinks. Shadows thicken, pressure ripples distort the edges, and loose bubbles rush upward past the lens, showing the camera is still sinking. Then the cat kicks upward with explosive speed, dragging the view through churning bubbles and rapidly brightening water as sunlight floods back in; the camera races upward, water streaming off the lens, and finally breaks the surface in a sudden blast of light and spray, snapping back into a crooked, frantic selfie as the cat resurfaces." \
    --image_path examples/i2v_input_1.jpg \
    --adaptive_resolution \
    --ode
```
Output video:
<video src="https://github.com/user-attachments/assets/916c4cd6-2e52-4c83-a7e4-e0a73f9e7925" controls></video>

#### Example 2
Command:
```bash
PYTHONPATH=.  python rcm/inference/wan2pt2_i2v_rcm_infer.py \
    --high_noise_model_path assets/checkpoints/Wan2.2-I2V-A14B-high-rCM4.0-merged.pt \
    --low_noise_model_path assets/checkpoints/Wan2.2-I2V-A14B-low-rCM1.0-merged.pt \
    --num_samples 2 \
    --prompt "A colorless, rugged, six-wheeled lunar rover—with exposed suspension arms, roll-cage framing, and broad low-gravity tires—glides into view from left to right, kicking up billowing plumes of moon dust that drift slowly in the vacuum. Astronauts in white spacesuits perform light, bouncing lunar strides as they hop aboard the rover’s open chassis. In the far distance, a VTOL lander with a vertical, thruster-based descent profile touches down silently on the gray surface. Above it all, vast aurora-like plasma ribbons ripple across the star-filled sky, casting shimmering green, blue, and purple light over the barren lunar plains, giving the entire scene an otherworldly, magical glow." \
    --image_path examples/i2v_input_2.jpg \
    --adaptive_resolution
```
Output video:
<video src="https://github.com/user-attachments/assets/4a040cc3-0b1c-40fd-afe9-48141e9e81b1" controls></video>

#### Example 3
Command:
```bash
PYTHONPATH=.  python rcm/inference/wan2pt2_i2v_rcm_infer.py \
    --high_noise_model_path assets/checkpoints/Wan2.2-I2V-A14B-high-rCM4.0-merged.pt \
    --low_noise_model_path assets/checkpoints/Wan2.2-I2V-A14B-low-rCM1.0-merged.pt \
    --num_samples 2 \
    --prompt "Uma Thurman’s Beatrix Kiddo holds her razor-sharp katana blade steady in the cinematic lighting. Without warning, the entire metal piece loses rigidity at once, its material trembling like unstable liquid. The surface destabilizes completely—chunks sag off in slow folds, turning into streams of molten silver that ooze downward in drops. Within moments, the object becomes a collapsing, formless metallic mass, with no edges, and no structure remaining. Thick liquid metal spills from her grip, followed by sheets of shimmering fluid that tear away and fall to the floor. What she holds now is only a quivering blob of mercury-like liquid, constantly sagging and dripping. Her expression shifts from calm readiness to shock and confusion as the last remnants of solidity dissolve, tear apart, and pour through her fingers, leaving her defenseless and disoriented." \
    --image_path examples/i2v_input_3.jpg \
    --adaptive_resolution
```
Output video:
<video src="https://github.com/user-attachments/assets/c0a09764-40af-4e8d-839d-16d097fa2894" controls></video>

#### Example 4
Command:
```bash
PYTHONPATH=.  python rcm/inference/wan2pt2_i2v_rcm_infer.py \
    --high_noise_model_path assets/checkpoints/Wan2.2-I2V-A14B-high-rCM4.0-merged.pt \
    --low_noise_model_path assets/checkpoints/Wan2.2-I2V-A14B-low-rCM1.0-merged.pt \
    --num_samples 2 \
    --prompt "Close-up on an elderly sailor in a weathered yellow raincoat, seated on the sun-lit deck of a gently rocking catamaran. With each small rise and dip of the hull, the shadows on his face shift subtly. He draws from his pipe, the ember brightening, and the exhale sends a thin ribbon of smoke that wavers and bends as the boat sways. His cat lies beside him, eyes half-closed, its body adjusting with soft, instinctive shifts whenever the deck tilts. Sunlight glints off the polished wood in flickering patterns as the surface of the water rolls beneath, causing brief flares of moving reflections. A pair of seabirds glide overhead, dipping slightly as the wind gusts. As the camera eases into a slow push-in, every motion becomes more pronounced—the smoke trembling, the cat’s fur fluttering, the deck creaking with each gentle sway—turning the peaceful moment into a dynamically living scene afloat at sea." \
    --image_path examples/i2v_input_4.jpg \
    --adaptive_resolution
```
Output video:
<video src="https://github.com/user-attachments/assets/13fa9a91-f5f0-4f45-a39e-1ea61a5e3be2" controls></video>

#### Example 5
Command:
```bash
PYTHONPATH=.  python rcm/inference/wan2pt2_i2v_rcm_infer.py \
    --high_noise_model_path assets/checkpoints/Wan2.2-I2V-A14B-high-rCM4.0-merged.pt \
    --low_noise_model_path assets/checkpoints/Wan2.2-I2V-A14B-low-rCM1.0-merged.pt \
    --num_samples 2 \
    --prompt "Watercolor style. Wet suminagashi inks surge and spread rapidly across the paper, swirling outward as they form island-like shapes with actively shifting, bleeding edges. A tiny paper boat is pulled forward by a faster-moving stream of pigment, gliding swiftly toward the still-wet areas. The flow pushes it in small sudden bursts, creating sharper, overlapping ripples that distort its reflection. Ink currents twist around the boat, forming brief vortices and drifting streaks that keep redirecting its path. Soft natural side-light catches the moving sheen on the wet paper, enhancing the sense of continuous, fluid motion across the painted landscape." \
    --image_path examples/i2v_input_5.jpg \
    --adaptive_resolution
```
Output video:
<video src="https://github.com/user-attachments/assets/7714bdce-6558-4db5-bad6-efe78dbcc931" controls></video>

#### Example 6
Command:
```bash
PYTHONPATH=.  python rcm/inference/wan2pt2_i2v_rcm_infer.py \
    --high_noise_model_path assets/checkpoints/Wan2.2-I2V-A14B-high-rCM4.0-merged.pt \
    --low_noise_model_path assets/checkpoints/Wan2.2-I2V-A14B-low-rCM1.0-merged.pt \
    --num_samples 2 \
    --prompt "she looks up, and then looks back" \
    --image_path examples/i2v_input_6.jpg \
    --adaptive_resolution
```
Output video:
<video src="https://github.com/user-attachments/assets/ce70f22b-db4b-429c-b3a5-b9cbf2cdc656" controls></video>

#### Example 7
Command:
```bash
PYTHONPATH=.  python rcm/inference/wan2pt2_i2v_rcm_infer.py \
    --high_noise_model_path assets/checkpoints/Wan2.2-I2V-A14B-high-rCM4.0-merged.pt \
    --low_noise_model_path assets/checkpoints/Wan2.2-I2V-A14B-low-rCM1.0-merged.pt \
    --num_samples 2 \
    --prompt "A man in a trench coat holding a black umbrella moves at a rapid, urgent pace through the streets of Tokyo on a rainy night, splashing hard through puddles. A handheld follow-cam tracks him from the side and slightly behind with quick, jittery motion, as if struggling to keep up. The focus stays locked on the man while neon signs streak into long, colorful bokeh trails from the fast movement. The scene has a cyberpunk, film-noir mood—mysterious, lonely, and restless. The slick pavement reflects vibrant neon light; raindrops cut sharply through the frame; a thin fog shifts as the man pushes forward with fast, determined steps." \
    --image_path examples/i2v_input_7.jpg \
    --adaptive_resolution \
    --ode
```
Output video:
<video src="https://github.com/user-attachments/assets/0abf6e62-8021-419b-a858-d1a2e3567b9c" controls></video>