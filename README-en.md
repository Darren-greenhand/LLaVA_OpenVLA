[中文](README.md)

## Objective:

The goal is to investigate whether using a more powerful MLLM (such as LLaVA-OneVision) combined with OpenVLA data will yield better results compared to the RT series and OpenVLA. Additionally, the project aims to flexibly adjust data structures to study the impact of different data mixes on performance, as well as scaling laws.

All the code has been executed successfully, but the results are unknown (due to some reasons, the robot is unavailable QWQ).



Comprehensive detailed notes (notion, Chinese) 

https://darren-dong.notion.site/OpenVLA-LLaVA-11a471fbaea480839ee6ca55f122a187?pvs=4darren-dong.notion.site/OpenVLA-LLaVA-11a471fbaea480839ee6ca55f122a187?pvs=4

include the following main content:

- Some architecture and implementation details from the OpenVLA paper
- The general structure of the Prismatic library (on which OpenVLA is based)
- Detailed understanding of the modifications OpenVLA made compared to the Prismatic library
- Detailed modification details of the LLaVA-OV library:
  - Data download (supports downloading by proportion without needing to download everything, as there is too much)
  - Data conversion (modified based on OpenVLA training, with black-box processing to ensure accuracy, and removal of image resizing operations)
  - Training with the LLaVA library (implemented an action tokenizer, which can output action tensors after training with new data)



## Code Repository

Downloading the data is approximately 450GB, and converting it into LLaVA format is about 600GB.

The second step, converting OpenVLA data into LLaVA format, takes approximately 12 hours.

The third step, training, uses 8 A100 GPUs, with batch size maximized, and DeepSpeed ZeRO Stage 2, taking approximately 130 hours.

GitHub - Darren-greenhand/LLaVA_OpenVLA: Converted the training data of OpenVLA into the general form of multimodal training instructions and then used with LLaVA-OneVision. [GitHub Repository](https://github.com/Darren-greenhand/LLaVA_OpenVLA)



Due to conflicts between several environments that were too difficult to resolve, three separate environments are used (conda makes this convenient):

Environment/part 1, for downloading data:

[Darren-greenhand/rlds_dataset_part: LLaVA_OpenVLA part 1, supports downloading a custom percentage of Open-X data and performing simple processing](https://github.com/Darren-greenhand/rlds_dataset_part)

Environment 2, for converting data into the MLLM standard format:

[Darren-greenhand/OpenVLA: LLaVA_OpenVLA part 2, Generate MLLM general training data](https://github.com/Darren-greenhand/OpenVLA)

Environment 3, for training LLaVA into a strong VLA model:

[Darren-greenhand/LLaVA-Next: LLaVA_OpenVLA part 3, Use LLaVA to train a stronger VLA model](https://github.com/Darren-greenhand/LLaVA-Next)

---

## Usage

`rlds_data` is the environment for part1, `openvla` is the environment for part2, and `llavaov` is the environment for part3.

The paths are too difficult to change, so I just used the ones on the server.

Step 1 (part1): Download and preprocess data, requires VPN

1. Modify the dataset selection and ratio in `prepare_open_x.sh`.

2. ```shell
   conda activate rlds_data
   cd /data/jcy/project/rlds_dataset_part
   ./prepare_open_x.sh
   # bridge is ready-made, copy a part and execute the following 👇
   ./prepare_bridge.sh
   # Note that dobbe's data is disordered, the condition for judging train_file in the sh file needs modification
   # Note: If you execute modify_rlds_dataset.py separately, you must use a VPN!!! Otherwise, an abstract bug will occur
   ```

Step 2 (part2):

1. Modify `data_mix` in `generate_llavadata.sh`.

2. Register a mix in `/data/jcy/project/openvla/prismatic/vla/datasets/rlds/oxe/mixtures.py`.

3. ```shell
   conda activate openvla
   cd /data/jcy/project/openvla
   ./generate_llavadata.sh
   # For unknown reasons, it will get stuck at the end, but it has actually been generated
   ```

4. Run `python /data/jcy/project/openvla/shuffle_reid_rename.py` # Process the sequence number, shuffle, and change to llava relative path format. Rename is direct renaming without retention, cp is for backup (used for debugging).

Step 3 (part3):

1. Modify `json_path` and `image_dir` (generated in the previous step) in `/data/jcy/project/LLaVA-NeXT/scripts/train/vla.yaml`.

2. ```shell
   conda activate llavaov
   cd /data/jcy/project/LLaVA-NeXT
   ./scripts/train/finetune_ov_vla.sh
   ```

OK, the training is done.

To use, go to part3 in the llava library, and run `python inference_action.py`.



LLaVA is a well-regarded library in the open-source multimodal project space (because it releases data, although version 1.6 has been delayed for a long time). The early code structure is relatively easy to understand (later updates introduced a lot of redundant and hard-to-use code, which is quite frustrating to review).

OpenVLA is a great implementation of VLA, but it uses relatively outdated models. Updating the models could potentially lead to significant performance improvements.

Default MIX:





| Registered Dataset Name                               | # Episodes | ratio | File Size (GB) |
| ----------------------------------------------------- | ---------- | ----- | -------------- |
| fractal20220817_data                                  | 73,499     | 0.15  | 111.06         |
| kuka                                                  | 580,392    | 0.07  | 778.02         |
| bridge                                                | 25,460     | 0.2   | 387.49         |
| taco_play                                             | 3,242      | 0.2   | 47.77          |
| jaco_play                                             | 976        | 0.3   | 9.24           |
| berkeley_cable_routing                                | 1,482      | 0.3   | 4.67           |
| roboturk                                              | 2,144      | 0.2   | 45.39          |
| viola                                                 | 135        | 0.5   | 10.4           |
| berkeley_autolab_ur5                                  | 896        | 0.3   | 76.39          |
| toto                                                  | 901        | 0.3   | 127.66         |
| language_table                                        | 442,226    | 0.1   | 399.22         |
| stanford_hydra_dataset_converted_externally_to_rlds   | 550        | 0.4   | 72.48          |
| austin_buds_dataset_converted_externally_to_rlds      | 50         | 0.5   | 1.49           |
| nyu_franka_play_dataset_converted_externally_to_rlds  | 456        | 0.3   | 5.18           |
| furniture_bench_dataset_converted_externally_to_rlds  | 5100       | 0.15  | 110            |
| ucsd_kitchen_dataset_converted_externally_to_rlds     | 150        | 0.5   | 1.33           |
| austin_sailor_dataset_converted_externally_to_rlds    | 250        | 0.5   | 18.85          |
| austin_sirius_dataset_converted_externally_to_rlds    | 600        | 0.4   | 6.55           |
| bc_z                                                  | 39,350     | 0.2   | 80.54          |
| dlr_edan_shared_control_converted_externally_to_rlds  | 100        | 0.5   | 3.09           |
| iamlab_cmu_pickup_insert_converted_externally_to_rlds | 520        | 0.4   | 50.29          |
| utaustin_mutex                                        | 1,500      | 0.2   | 20.79          |
| berkeley_fanuc_manipulation                           | 415        | 0.4   | 8.85           |
| cmu_stretch                                           | 135        | 0.5   | 0.71           |
| dobbe                                                 | 5208       | 0.1   | 21.1           |
| fmb                                                   | 1804       | 0.2   | 356.5          |

现在：113,178 trajs,  450G in memory