# LLaVA_OpenVLA
## 目的：

研究用更强的MLLM（如LLaVA-OneVision）配上OpenVLA的数据会不会相比RT系列和OpenVLA有更好的效果，同时可以灵活调整数据结构，可以研究不同数据mix对性能的影响，以及scaling law

全部代码都跑通了，效果未知（由于某些原因机器人木有了 QWQ

完整详细笔记（notion），主要内容包括：

- OpenVLA论文里的一些架构和实现细节
- Prismatic库（OpenVLA基于这个库改的）的大致结构
- OpenVLA相比Prismatic库修改了哪些，细致理解
- LLaVA-OV库修改的详细细节：
- 下载数据（支持按比例下，不用全部下下来选，太多了）
- 转换数据（基于OpenVLA训练改的，黑盒处理保证无误，去掉了对图片resize的操作）
- 用LLaVA库进行训练（实现了action tokenizer，用新数据训练后可以输出action tensor）

https://darren-dong.notion.site/OpenVLA-LLaVA-11a471fbaea480839ee6ca55f122a187?pvs=4darren-dong.notion.site/OpenVLA-LLaVA-11a471fbaea480839ee6ca55f122a187?pvs=4

## 代码库

下载数据大概是450G，转换成llava格式大概600G

第二步OpenVLA转换数据成llava格式，大概12h

第三步训练，8*A100，bs拉满，deepspeed zero2，大约是130H

GitHub - Darren-greenhand/LLaVA_OpenVLA: Converted the training data of OpenVLA  into general form of multimodal training instructions and then used with LLaVA-OneVisiongithub.com/Darren-greenhand/LLaVA_OpenVLA

因为几个环境相互冲突太难改了，干脆就还是用三个环境（conda很方便）：

环境1，下载数据用：

[Darren-greenhand/rlds_dataset_part: LLaVA_OpenVLA part 1, supports downloading custom percentage of Open-X data and performing simple processing](https://github.com/Darren-greenhand/rlds_dataset_part)

环境2，转换数据成MLLM标准格式

[Darren-greenhand/OpenVLA: LLaVA_OpenVLA part 2, Generate MLLM general training data](https://github.com/Darren-greenhand/OpenVLA)

环境3，训练LLaVA成strong VLA模型

[Darren-greenhand/LLaVA-Next: LLaVA_OpenVLA part 3, Use LLaVA to train a stronger VLA model](https://github.com/Darren-greenhand/LLaVA-Next)

\---

LLaVA是多模态开源项目里做的很好的库（因为会开源数据，虽然1.6拖了很久），早期代码结构比较通俗易懂（后面更新以后有很多冗余，基本很难调用的代码，看的头疼（

OpenVLA是一个实现VLA的很好的工作，但是里面用的模型比较老旧，如果更新模型，性能可能会有较大的提升。

默认的MIX：

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

现在：113,178条轨迹  450G
