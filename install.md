# Installation of ACDC

> ***Note**: We provide a more detailed installation process based on the official guide, along with a summary of the issues we encountered during our own installation process.*

The official documentation provides two installation methods: the first is a one-step installation using the `install.sh` script, and the second is a step-by-step manual installation. When using the script, ensure that your network connection is stable and provide a CUDA path with a version of 12.1 or higher. If, like me, you installed the CUDA driver via Ubuntu's package manager, you might not be able to locate the CUDA path. In that case, I recommend opting for the step-by-step installation method.

The following additional content is primarily based on the step-by-step installation method.

## Installation Step by Step

1. Create a new conda environment to be used for this repo and activate the repo:

   ```bash
   conda create -y -n acdc python=3.10
   conda activate acdc
   ```

2. Install ACDC

   ```bash
   conda install conda-build
   pip install -r requirements.txt
   pip install -e .

   # Additional Installation: Install a CUDA version that matches the versions of torch and torchvision using Conda
   conda install cuda # After the installation, please verify that the versions are compatible. I am using torch==2.5.1, torchvision==0.20.1, and cuda==12.4.1
   ```

3. Install the following key dependencies used in our pipeline. **NOTE**: Make sure to install in the exact following order:

   - Make sure we're in dependencies directory

     ```bash
     mkdir -p deps && cd deps
     ```

   - [dinov2](https://github.com/facebookresearch/dinov2)

     ```bash
     git clone https://github.com/facebookresearch/dinov2.git && cd dinov2
     conda-develop . && cd ..      # Note: Do NOT run 'pip install -r requirements.txt'!!
     ```

   - [segment-anything-2](https://github.com/facebookresearch/segment-anything-2)

     ```bash
     git clone https://github.com/facebookresearch/segment-anything-2.git && cd segment-anything-2
     pip install -e . && cd ..
     ```

   - [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)

     ```bash
     git clone https://github.com/IDEA-Research/GroundingDINO.git && cd GroundingDINO
     # export CUDA_HOME=/PATH/TO/cuda-12.3   # Make sure to set this!
     # The official documentation emphasizes setting the CUDA path; however, since we installed CUDA via Conda, it will be invoked automatically, and there is no need to set the path manually
     pip install --no-build-isolation -e . && cd ..
     ```

   - [PerspectiveFields](https://github.com/jinlinyi/PerspectiveFields)

     ```bash
     git clone https://github.com/jinlinyi/PerspectiveFields.git && cd PerspectiveFields
     pip install -e . && cd ..
     ```

   - [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)

     ```bash
     git clone https://github.com/DepthAnything/Depth-Anything-V2.git && cd DepthAnything-V2
     pip install -r requirements.txt
     conda-develop . && cd ..
     ```

   - [CLIP](https://github.com/openai/CLIP)

     ```bash
     pip install git+https://github.com/openai/CLIP.git
     ```

   - [faiss-gpu](https://github.com/facebookresearch/faiss/tree/main)

     ```bash
     conda install -c pytorch -c nvidia faiss-gpu=1.8.0
     ```

   - [robomimic](https://github.com/ARISE-Initiative/robomimic)

     ```bash
     git clone https://github.com/ARISE-Initiative/robomimic.git --branch diffusion-updated --single-branch && cd robomimic
     pip install -e . && cd ..
     ```

   - [OmniGibson](https://github.com/StanfordVL/OmniGibson)

     ```bash
     git clone https://github.com/StanfordVL/OmniGibson.git && cd OmniGibson
     pip install -e . && python -m omnigibson.install --no-install-datasets && cd ..
     ```

## Assets

In order to use this repo, we require both the asset image and BEHAVIOR datasets used to match digital cousins, as well as relevant checkpoints used by underlying foundation models. Use the following commands to install each:

> [!tip]
>
> The following links may have slow download speeds. You can copy the links and download them via a browser. Once the download is complete, move the files to the appropriate folders.

1. Asset image and BEHAVIOR datasets

   ```bash
   python -m omnigibson.utils.asset_utils --download_assets --download_og_dataset --accept_license
   python -m digital_cousins.utils.dataset_utils --download_acdc_assets
   ```

2. Model checkpoints

   ```bash
   # Make sure you start in the root directory of ACDC
   mkdir -p checkpoints && cd checkpoints
   wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
   wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
   wget https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large/resolve/main/depth_anything_v2_metric_hypersim_vitl.pth
   wget https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Large/resolve/main/depth_anything_v2_metric_vkitti_vitl.pth
   cd ..
   ```

3. Policy checkpoints

   ```bash
   mkdir -p training_results && cd training_results
   wget https://huggingface.co/RogerDAI/ACDC/resolve/main/cousin_ckpt.pth
   wget https://huggingface.co/RogerDAI/ACDC/resolve/main/twin_ckpt.pth
   cd ..
   ```

## Installation of Isaac Sim(4.0.0)

The official documentation does not mention Isaac Sim, but it is actually required to install this software. There are two methods to install it. The first method is to install it directly via pip, but it requires the glibc version to be 2.34 or higher. You can check the glibc version by running the command `ldd --version`. If the glibc version is too low, updating it is not recommended. In this case, you can use the second installation method.


For the second installation method, please refer to the [official Isaac Sim documentation](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_workstation.html). We have provided a summary of some issues we encountered during the installation process and their solutions in the "Issue Summary" section.

## Testing

To validate that the entire installation process completed successfully, please run our set of unit tests:

```bash
python tests/test_models.py --gpt_api_key <KEY> --gpt_version 4o
```

- `--gpt_api_key` specifies the GPT API key to use for GPT queries. Must be compatible with `--gpt_version`
- `--gpt_version` (optional) specifies the GPT version to use. Default is 4o

## Issue Summary

### Issue 1: Failed to download the model from https://huggingface.co.

- Solution: Use the huggingface-cli command-line tool provided by Hugging Face,which is an official command-line tool from Hugging Face that comes with a comprehensive download feature.

  1. Install dependencies:

     ```bash
     pip install -U huggingface_hub
     ```

  2. Set up the path:

     ```bash
     export HF_ENDPOINT=https://hf-mirror.com
     ```

     It is recommended to add this line to the `.bashrc` file.

- Reference article: https://blog.csdn.net/weixin_43431218/article/details/135403324

### Issue 2: After installing Isaac Sim through Omniverse, clicking the Launch button causes it to load for a while but then fail to start.

- Solution：

  1. Please try launching Isaac Sim via the terminal first by running the following command: `<your IsaacSim installation path>/isaac-sim.sh -v`。

  2. Check the error message. The issue I encountered here is:

     ```bash
     [948ms] [Info] [carb.graphics-vulkan.plugin] Instance extension VK_KHR_get_physical_device_properties2 required by NGX.
     X Error of failed request:  GLXBadFBConfig
       Major opcode of failed request:  150 (GLX)
       Minor opcode of failed request:  0 ()
       Serial number of failed request:  224
       Current serial number in output stream:  224
     ```

  3. Execute `export MESA_GL_VERSION_OVERRIDE=4.6`, It is recommended to add this line to the `.bashrc` file.

- Reference article: https://forums.developer.nvidia.com/t/isaac-sim-in-omniverse-not-launching/184360

  https://forum.winehq.org/viewtopic.php?f=8&t=34889

### Issue 3: After installing Isaac Sim, the ISAAC_PATH still shows an error.

- Solution:

  1. Please make sure you are in the conda environment:

     ```bash
     conda activate acdc
     ```

  2. Set up the environment：

     ```bash
     source <your IsaacSim installation path>/setup_conda_env.sh
     ```

  3. After setting the path for the first time, torch, torchvision, and torchaudio may have been modified. Please use `pip show` to check the installed versions. If they do not match the previous versions, you can use `pip uninstall` to remove the installation. After that, querying again should restore the previous versions.