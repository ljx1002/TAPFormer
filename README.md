# TAPFormer: Robust Arbitrary Point Tracking via Transient Asynchronous Fusion of Frames and Events (CVPR 2026)

<p align="center">
  <a href="https://arxiv.org/abs/2602.04877"><img src="https://img.shields.io/badge/arXiv-FETAP-b31b1b" alt="arXiv"></a>
  <a href="https://arxiv.org/pdf/2603.04989"><img src="https://img.shields.io/badge/arXiv-TAPFormer-b31b1b" alt="arXiv"></a>
  <a href="https://tapformer.github.io/"><img src="https://img.shields.io/badge/🌐-Project_Page-orange" alt="Project Page"></a>
  <a href="https://drive.google.com/file/d/1gvbQgS8tbVSaOtgAIryYcbDabzsDoUaE/view?usp=drive_link"><img src="https://img.shields.io/badge/📦-Dataset-blue" alt="Dataset"></a>
  <a href="https://huggingface.co/papers/2603.04989"><img src="https://img.shields.io/badge/🤗-Demo-yellow" alt="Hugging Face Page"></a>
</p>

<p align="center">
  Jiaxiong Liu<sup>1</sup>, Zhen Tan<sup>1</sup>, Jinpu Zhang<sup>1</sup>, Yi Zhou<sup>2</sup>, Hui Shen<sup>1</sup>, Xieyuanli Chen<sup>1</sup>, Dewen Hu<sup>1</sup>
</p>

<p align="center">
  <sup>1</sup>National University of Defense Technology &nbsp;&nbsp; <sup>2</sup>Hunan University
</p>

<p align="center">
  <img src="assets/teaser.png" alt="TAPFormer teaser omage" width="70%"/>
</p>

### Key Features

- The first real-world TAP benchmark covering challenging conditions with synchronized frame–event data.
- A novel asynchronous fusion paradigm that explicitly models temporal continuity between frames and events via Transient Asynchronous Fusion (TAF).
- State-of-the-art performance across multiple datasets in TAP and feature point tracking tasks.

### Example Predictions

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="assets/indoor_fruit_410_510.gif" alt="Example 1" width="200"/>
      </td>
      <td align="center">
        <img src="assets/indoor_fruit_guobao2_5_155.gif" alt="Example 2" width="200"/>
      </td>
      <td align="center">
        <img src="assets/indoor_hand_move_dynamic2_300_400.gif" alt="Example 3" width="200"/>
      </td>
    </tr>
    <tr>
      <td align="center">
        <img src="assets/outdoor_day2-1_317_417.gif" alt="Example 4" width="200"/>
      </td>
      <td align="center">
        <img src="assets/toulan_360_460.gif" alt="Example 5" width="200"/>
      </td>
      <td align="center">
        <img src="assets/11_180_380.gif" alt="Example 6" width="200"/>
      </td>
    </tr>
    <tr>
      <td align="center">
        <img src="assets/3_143_243.gif" alt="Example 7" width="200"/>
      </td>
      <td align="center">
        <img src="assets/peanuts_light_160_386.gif" alt="Example 8" width="200"/>
      </td>
      <td align="center">
        <img src="assets/peanuts_running_2360_2460.gif" alt="Example 9" width="200"/>
      </td>
    </tr>
  </table>
</div>

## Installation

### Requirements

- Python 3.7+
- CUDA-capable GPU (recommended) or CPU
- PyTorch 1.8+

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd tapformer
```

2. Set up the environment:
```bash
conda create --name TAPFormer python=3.10
conda activate TAPFormer
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. (Optional) Install flow-vis for optical flow visualization:
```bash
pip install git+https://github.com/tomrunia/OpticalFlow_Visualization.git
```

Note: If you encounter issues with `flow-vis`, it's only needed for optical flow visualization mode and can be skipped if you don't use that feature.

## Quick Start

### 1. Test Sequences and Pretrained Weights
We build the first benchmark for multimodal arbitrary point tracking, including a synthetic frame–event training set and manually annotated real-world [test sequences](https://drive.google.com/file/d/1gvbQgS8tbVSaOtgAIryYcbDabzsDoUaE/view?usp=drive_link), providing a comprehensive platform for future research. We also evaluate our model on the feature point tracking benchmarks [EDS](https://download.ifi.uzh.ch/rpg/CVPR23_deep_ev_tracker/eds_subseq.zip) and [EC](https://download.ifi.uzh.ch/rpg/CVPR23_deep_ev_tracker/ec_subseq.zip). The updated EDS datasets ground truth annotations can be downloaded [here](https://drive.google.com/file/d/1w7uQm8AK1HVtNOnBg-wxoDw3QUPn8OVs/view?usp=drive_link).

Furthermore, we also provide the [network weights](https://drive.google.com/file/d/1vklq9pCnRBMeevhGfmmHd4i8KveZM9sT/view?usp=drive_link) trained on the FE-FastKub dataset


### 2. Prepare Your Data

To generate input event representations, run the following script file to generate event representations for the corresponding dataset:  

```
data_pretation\real\InivTAP\genarate_event_represent_InivTAP.py
data_pretation\real\DrivTAP\genarate_event_represent_DrivTAP.py
data_pretation\real\genrate_EFrame_for_EDS_EC.py
```

Ensure your dataset is organized in the following structure:
```
dataset_dir/
├── eds_subseq/
│   └── sequence_name/
│       ├── events/
│       ├── images_corrected/
│       └── sequence_name.gt.txt
├── ec_subseq/
│   └── sequence_name/
│       ├── events/
│       ├── images_corrected/
│       └── track.gt.txt
├── InivTAP/
│   └── sequence_name/
│       ├── events/
│       ├── images_corrected/
│       └── annotations.npy
└── DrivTAP/
    └── sequence_name/
        ├── events/
        ├── images_corrected/
        └── annotations.npy
```

### 3. Configure Your Settings

Edit the YAML configuration files in the `config/` directory:

- `config/config_eds_ec.yaml` - For EDS and EC datasets
- `config/config_InivTAP_DrivTAP.yaml` - For InivTAP and DrivTAP dataset

Key configuration options:
- `dataset_dir`: Path to your dataset directory
- `ckpt_root`: Path to model checkpoint
- `eval_datasets_*`: List of sequences to evaluate
- `visualization.enable`: Enable/disable visualization
- `output.save_results`: Save evaluation results
- `output.save_trajectory`: Save trajectory files

### 4. Run Evaluation

#### EDS and EC Datasets
```bash
python test_EDS_EC.py --config config/config_eds_ec.yaml
```

#### TAPFormer Dataset
```bash
python test_InivTAP_DrivTAP.py --config config/config_InivTAP_DrivTAP.yaml
```

## Output

When enabled, the evaluation script generates:
- **Visualization videos**: Tracked points overlaid on input frames
- **Trajectory files**: Predicted trajectories in text format
- **Result files**: Evaluation metrics (mean error, age, etc.)

Output files are saved in:
- EDS: `output/eval_eds_subseq/{sequence_name}/{model_name}/`
- EC: `output/eval_ec_subseq/{sequence_name}/{model_name}/`
- InivTAP and DrivTAP: `output/eval_InivTAP_DrivTAP_subseq/{sequence_name}/{model_name}/`


## Citation

If you use this code in your research, please cite:

```bibtex
@article{liu2026tapformer,
  title={TAPFormer: Robust Arbitrary Point Tracking via Transient Asynchronous Fusion of Frames and Events},
  author={Liu, Jiaxiong and Tan, Zhen and Zhang, Jinpu and Zhou, Yi and Shen, Hui and Chen, Xieyuanli and Hu, Dewen},
  journal={arXiv preprint arXiv:2603.04989},
  year={2026}
}
@inproceedings{liu2025tracking,
  title={Tracking any point with frame-event fusion network at high frame rate},
  author={Liu, Jiaxiong and Wang, Bo and Tan, Zhen and Zhang, Jinpu and Shen, Hui and Hu, Dewen},
  booktitle={2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={18834--18840},
  year={2025},
  organization={IEEE}
}
```

## Acknowledgments

We gratefully appreciate the following repositories and thank the authors for their excellent work:

- [CoTracker](https://github.com/facebookresearch/co-tracker)
- [ETAP](https://github.com/tub-rip/ETAP)
- [DeepEvT](https://github.com/uzh-rpg/deep_ev_tracker)


## License

See the [LICENSE](LICENSE) file for details about the license under which this code is made available.

