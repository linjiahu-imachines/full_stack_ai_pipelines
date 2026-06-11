
# IMINN Performance Evaluator/Profiler

## Overview

The goal of this repository is to simplify AI/ML performance evaluation across different models, ai frameworks, compilers, architectures, and configurations. In addition, this project provides a straightforward way to setup and maintain an environment for emulating, simulating, and running (both remotely and locally) AI/ML frameworks.

## Prerequisite installations

### Python

As a first prerequisite, you will need to install python 3.11.10 or higher. Other versions of python may be compatible, but they have not been tested and therefore it is not recommended to use earlier versions. If you need to install python, it is highly recommended to install [pyenv](https://github.com/pyenv/pyenv), which is a tool for managing python installations, including python packages. You can follow the instructions included in that github for setup.

### Git setup
The different projects/frameworks used in this are primarily internal forks or private repos which requires git setup. In particular, the git clone commands will use the https protocol. If you typically use ssh as your preferred cloning protocol, there are two ways you can fix this issue:
**Option 1: Creating a Personal Access Token** 
1. [Create a personal access token](https://docs.github.com/en/enterprise-server@3.9/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-personal-access-token), and store the token somewhere secure.
2. Make sure to [authorize SAML](https://docs.github.com/en/enterprise-cloud@latest/authentication/authenticating-with-saml-single-sign-on/authorizing-a-personal-access-token-for-use-with-saml-single-sign-on) use.
3. To avoid being prompted for your username and token/password everytime you clone, run the following command on the machine you are trying to clone to:

```bash
$ git config --global credential.helper store
```
4. Now, clone either this repository or another, and when prompted for your username and password, input your github username, and for the password _input your newly created personal access token_. The next time you run a clone command, you will no longer be prompted for this particular machine.

**Option 2: Configure git to use ssh instead of https**

The alternative is to set your git config to replace all instances of https clones with the equivalent ssh url by running the following command:

```bash
git config --global url.git@github.com:.insteadOf https://github.com/
```

### LLVM/Clang
For native builds, this project requires clang for compilation. You can easily install clang by running the following command:

```bash
bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"
```

-----------------------------------------
### Clone and install

You will also need to clone this repository along with it's submodules:

```bash
$ git clone https://github.com/I-Machines/iminn-tools && cd iminn-tools 
```

 Install the pip package:
```bash
cd iminn-tools
# the `-e` argument allows for editing the iminn-tools code without requiring reinstallation to use the newly edited code
pip3 install -e .
```

You should now have the `iminnt` command on your `PATH`. You can verify this by running the following:
```bash
# This will fail if `iminnt` is not properly installed
iminnt --help 
```

## Using iminn-tools
`iminn-tools` works by setting up a local environment for building, updating, and running/simulating/emulating code. All of the code and the resulting binaries are stored in the `dev_env` directory in this repository. This allows you to make direct changes to the source code if necessary, and reuse the binaries in a straightforward way.

`iminn-tools` works by specifying a target project/framework via `-t`, followed by the action you want to take. The following are the actions you can use:
| Action   | Description                                                                                                                           |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| init     | Initialize project by either cloning or copying source code to `dev_env`. |
| pull     | Run `git pull` and update to the latest source code. |
| default  | For a given target, list the default run/simul options, such as the binary used. |
| checkout | Run `git checkout` for a specific branch, tag, or commit. |
| sync     | For forked targets such as `XNNPACK` or `llama.cpp`, sync the `upstream` branch with it's public counterpart. |
| build    | Build/compile the code. If it was built/installed previously, this will delete the old build/insall directories to start from scratch.|
| rebuild  | Recompile the code. This will recompile code, and requires the code to have already been built.                                       |
| run      | Run either emulation or direct execution for a given project.                                                                         |
| sim      | Simulate/measure execution of a given binary and commands, storing output to the `results` directory. |
| simpoint | Simpoint a given binary and commands, storing output to the `results` directory. |
| sweep    | Run a sweep of benchmarks for a given target.                                                                                         |

Command-line options for each of these commands can be found below.

In addition, below are the supported target names and what framework/hardware they correspond to:

**NOTE** The targets denoted `Compound` below represent aliases for multiple targets. That means when you run any command with a compound target specified, it will iteratively run that command for each of the sub-targets listed in the `Framework` column.

| Target Name          | Framework                                    | Compilation Target     | Runnable | Notes                                                             |
| -------------------- | -------------------------------------------- | -------------------    | -------- |------------------------------------------------------------------ |
| xnnpack_imi          | XNNPACK                                      | RISCV                  | Yes      |                                                                   |
| xnnpack_x86          | XNNPACK                                      | x86 (native)           | Yes      |                                                                   |
| xnnpack_amx          | XNNPACK                                      | x86 (emeraldrapids)    | Yes      |                                                                   |
| xnnpack_neoverse     | XNNPACK                                      | ARM (neoverse-n2)      | Yes      |                                                                   |
| llama_x86            | LLama.cpp                                    | x86 (native)           | Yes      |                                                                   |
| llama_x86_bench      | LLama.cpp                                    | x86 (native)           | Yes      | Does not build ref backend.                                       |
| llama_imi            | LLama.cpp                                    | RISCV                  | Yes      |                                                                   |
| llama_imi_bench      | LLama.cpp                                    | RISCV                  | Yes      | Does not build ref backend.                                       |
| litert_x86           | TFLite (LiteRT)                              | x86 (native)           | Yes      |                                                                   |
| litert_imi           | TFLite (LiteRT)                              | RISCV                  | Yes      |                                                                   |
| qemu                 | QEMU                                         | x86 (native)           | No       | Sim/emulation infra.                                              |
| isaextgen            | isaextgen                                    | x86 (native)           | No       | Sim/emulation infra.                                              |
| arctic               | Arctic                                       | x86 (native)           | No       | Sim/emulation infra.                                              |
| pilos                | Pilos                                        | x86 (native)           | No       | Sim/emulation infra.                                              |
| permafrost           | Permafrost                                   | x86 (native)           | No       | Sim/emulation infra.                                              |
| spike                | Spike                                        | x86 (native)           | No       | Sim/emulation infra.                                              |
| riscv_toolchain      | riscv-gnu-toolchain                          | x86 (native)           | No       | Sim/emulation infra.                                              |
| llvm                 | LLVM                                         | x86 (native)           | No       | Sim/emulation infra.                                              |
| riscv-env (Compound) | riscv_toolchain, llvm                        | x86 (native)           | No       | Sim/emulation infra.                                              |
| simpoint             | simpoint                                     | x86 (native)           | No       | Sim/emulation infra.                                              |
| psim (Compound)      | qemu,arctic,pilos,permafrost,spike,riscv-env,simpoint | x86 (native)  | No       | Sim/emulation infra.                                              |
| rvv-tests            | N/A (custom)                                 | RISCV                  | Yes      | This is custom extension testing code and misc. micro benchmarks. |
| linux-kernel         | Linux Kernel                                 | RISCV                  | No       | NPU emulation infrastructure.                                     |
| npu-driver           | VSI NPU Driver                               | RISCV                  | No       | NPU emulation infrastructure.                                     |
| npu-env              | linux-kernel,npu-driver                      | RISCV                  | No       | NPU emulation infrastructure.                                     |
| timvx_x86            | TIM-VX                                       | x86 (native)           | No       |                                                                   |
| timvx_riscv          | TIM-VX                                       | RISCV                  | No       |                                                                   |
| playground-x86       | Misc.                                        | RISCV                  | Yes      | Collection of single-file C code for testing purposes             |
| playground-riscv     | Misc.                                        | x86 (native)           | Yes      | Collection of single-file C code for testing purposes             |


## Commands


### init

* `[-r|--reinit]`: Reinitialize by deleting the previous source code and reinitializing. Default: `False`
* `[-e|--extras]`: Only update additional scripts. Default: `False`

### default

* `[-n|--names-only]`: Only print the default names instead of names and binary commands. Default: `False`

### checkout

* `[-b|--branch] <name>`: Branch name to checkout. Required option.


### pull/build/rebuild/sync

The `pull`/`build`/`rebuild`/`sync` commands do not support any additional options.

### run

* `[-b|--bin] <relative_path_to_bin>`: Binary to use for execution, if different than the default. The binary must be located in a subdirectory for the given target. Defaults to the target's default binary.
* `[-a|--bin-args] <arg0> <arg1> ... <argn>`: Arguments to pass to the executable. Defaults to no arguments being passed.
* `[-d|--default-cmd] <default_command_name>`: Run one of several different default commands from a given target. You can list possible default command names using the `default` iminnt command. When the binary or arguments are specified, this is invalid.

### sim/simpoint

* `[-b|--bin] <relative_path_to_bin>`: Binary to use for execution, if different than the default. The binary must be located in a subdirectory for the given target. Defaults to the target's default binary.
* `[-a|--bin-args] <arg0> <arg1> ... <argn>`: Arguments to pass to the executable. Defaults to no arguments being passed.
* `[-d|--default-cmd] <default_command_name>`: Run one of several different default commands from a given target. You can list possible default command names using the `default` iminnt command. When the binary or arguments are specified, this is invalid.
* `[-o|--output-dir] <output_dir_path>`: Specify output directory for results. Defaults to `results/<target_name>`.
* `[-p|--print-cfg]`: Print simulation configuration to stdout. Default: `False`
* `[-c|--custom-cfg] <path_to_custom_cfg>`: Specify custom config to use for simulation. When using this option, the custom config cannot include definitions for the following config variables: `func::live_trace`, `func::memory`, `func::execution`, `perf`. In addition, When specifying `func::imi_spike::no_args::m_pk` and `func::argv`, users should not specify the path to `libcosim.so`, and should instead use the `$carbon<COSIM_LIB>` placeholder variable. In addition, the simulated binary should be specified with `$carbon<SIM_BIN>`. Lastly, arguments to `$carbon<SIM_BIN>` should be specified as `$SIM_ARGS`, both of which will be populated by `iminnt`. Defaults to the `iminnt` config, which will be stored in the output directory in a file called `iminnt.cfg` for viewing.
* `[-k|--keep-retires]`: Preserve retired instructions for debugging purposes. Defaults to `False`.
* `[-i|--isolate-program]`: **NOTE: This is currently not functional. Work in progress.** Constrain simpointing to the start and end of the program (exclude initialization instructions). Defaults to `False`.
* `[-g|--fn-perf-graph]`: Generate a trace file representing a graph of function call performance. Defaults to `False`.

### sweep
**NOTE: Currently the `sweep` command only supports xnnpack targets.**
* `[-o|--output-dir] <dir_path>`: Specify output directory for all results. Defaults to `results/sweep_<target_name>`.
* `[-t|--test-num] <num>` : As an alternative to specifying an output directory, this specifies a number to append to the results folder. The resulting output path will be: `results/sweep_<target_name>_<num>`.
* `[-i|--iters] <num>`: Specify number of iterations per benchmark. Defaults to `1`.
* `[-n|--num-bench] <num>`: Specify number benchmarks to run. Defaults to running all benchmarks, which is target-dependent.
* `[-s|--start-bench]`: Specify the benchmark to start from. Defaults to starting from the first benchmark.
* `[-k|--kernels] <kernel0> <kernel1>...<kerneln>`: Specify the kernels to run. Defaults to running all kernels.
* `[-ow|--overwrite]`: When set, will overwrite existing directories. Defaults to `False`.

## Initial setup (Recommended)

After cloning and installing the `iminn-tools` code, it is highly recommended that you run the following commands to get the initial simulation and emulation infrastructure up and running.

1. First, you will need to initialize the simulation/emulation infrastructure by cloning each of the repositories. As mentioned above, `psim` is a compound target which is an alias for all of the simulation/emulation infrastructure required for running AI/ML frameworks.:

```bash 
iminnt -t psim init
```

2. Having Initialized the infrastructure repositories, you will now need to build each of them. The `psim` target builds each subtarget in a specific order so that all dependencies of the sub-target have been built prior to compilation. **NOTE** This step can take a very long time because each of the code bases are substantially large.:

```bash 
iminnt -t psim build
```
Alternatively, you can build each of the `psim` sub-targets individually if you prefer. Be sure to run the build commands for each individual sub-target in this order to satisfy dependencies:

```bash
iminnt -t riscv-env build
iminnt -t arctic build
iminnt -t permafrost build
iminnt -t pilos build
iminnt -t spike build
iminnt -t qemu build
```

3. Having initialized and built simulation infrastructure, you can make sure it works correctly by trying to run/simulate a test from XNNPACK:

```bash
# Initialize the xnnpack repo for compilation
iminnt -t xnnpack_imi init
# Compile the repository
iminnt -t xnnpack_imi build
# Emulate a test case
iminnt -t xnnpack_imi run -d gemm_1x4c16_imi_v1_bench0
# Simulate a test case. This may take a couple minutes.
iminnt -t xnnpack_imi sim -d gemm_1x4c16_imi_v1_bench0
```
Assuming these all successfully ran, you can repeat these steps for other AI/ML frameworks and/or different backends.



## Target Models for Benchmarking

Below is a list of MLCommons benchmarks for [Edge inference](https://mlcommons.org/benchmarks/inference-edge/), [Mobile inference](https://mlcommons.org/benchmarks/inference-mobile/), and [client inference](https://mlcommons.org/benchmarks/client/). Please note that the MLCommons website includes older models which are currently deprecated, and are no longer accepted by the latest submission. 

**NOTE** Currently, ResNet-18 and ResNet-50 are the only supported models as proofs of concept for running `iminn-tools`. However, the remaining models will be added in fairly quick succession once other more pressing components of `iminn-tools`are stabilized.

| Model Name | Dataset | Framework | Model Weights | Source Implementation | Python Package | Category |
| ---------- | ------- | --------- | ------------- | --------------------- | -------------- | -------- |
| Resnet18-v1.5 | imagenet2012 (3x224x224) | torch |  [TorchVision](https://github.com/pytorch/vision/blob/v0.8.2/torchvision/models/resnet.py) | [TorchVision](https://github.com/pytorch/vision/blob/v0.8.2/torchvision/models/resnet.py) | torchvision | N/A |
| Resnet50-v1.5 | imagenet2012 (3x224x224) | torch | [Zenodo](https://zenodo.org/record/4588417/files/resnet50-19c8e357.pth) | [TorchVision](https://github.com/pytorch/vision/blob/v0.8.2/torchvision/models/resnet.py) | torchvision | Edge |
| Retinanet | OpenImages (800x800) | torch | [Zenodo](https://zenodo.org/record/6617981/files/resnext50_32x4d_fpn.pth) | [MLPerf](https://github.com/mlcommons/training/tree/master/single_stage_detector/ssd/model) | mlcommons/infernece | Edge |
| 3D UNET | KiTS 2019 | torch | [Zenodo](https://zenodo.org/record/5597155) | [MLPerf](https://github.com/mlcommons/training/tree/master/image_segmentation/pytorch)  | mlcommons/inference repo | Edge |
| GPT-J 6B | CNN-Daily Mail | torch | [Transformers](https://huggingface.co/EleutherAI/gpt-j-6b) | [MLPerf](https://github.com/mlcommons/inference/tree/master/language/gpt-j)  | transformers | Edge |
| Bert-Large | Squad-1.1 | torch | [Zenodo](https://zenodo.org/record/3733896) | [Custom MLPerf Script](https://github.com/mlcommons/inference/blob/master/language/bert/bert_tf_to_pytorch.py)  | transformer,mlperf | Edge |
| SDXL 1.0 | COCO 2014 | torch | [HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) | same as weights  | transformers | Edge |
| MobileDETs | MS-COCO 2017 | tensorflow | [Tensorflow Site](http://download.tensorflow.org/models/object_detection/ssdlite_mobiledet_edgetpu_320x320_coco_2020_05_19.tar.gz) | [MlPerf](https://github.com/mlcommons/mobile_open/blob/main/vision/mobiledet/README.md) | N/A | Mobile |
| MOSAIC | ADE20k | tensorflow | [MLPerf](https://github.com/mlcommons/mobile_open/tree/main/vision/mosaic/models_and_checkpoints/R4) | [MLPerf Repo](https://github.com/mlcommons/mobile_open/blob/main/vision/mosaic/README.md)  | N/A | Mobile |
| Mobile-Bert | SQUAD 1.1 | tensorflow | [MLPerf](https://github.com/mlcommons/mobile_open/tree/main/language/bert/models_and_code) | [MLPerf](https://github.com/mlcommons/mobile_open/blob/main/language/bert/README.md)  | N/A | Mobile |
| EDSR F32B5 | OpenSR | tensorflow | [MLPerf](https://github.com/mlcommons/mobile_open/tree/main/vision/edsr/models_and_checkpoints) | [MLPerf](https://github.com/mlcommons/mobile_open/blob/main/vision/edsr/README.md)  | N/A | Mobile |
| MobileNetV4 | ImageNet | tensorflow | [MLPerf](https://github.com/mlcommons/mobile_open/blob/main/vision/mobilenetV4/README.md) | [MLPerf](https://github.com/mlcommons/mobile_open/tree/main/vision/mobilenetV4)  | N/A | Mobile |
| MobileNetEdge TPU | ImageNet | Tensorflow | [MLPerf](https://github.com/mlcommons/mobile_open/tree/main/vision/mobilenet/models_and_code/checkpoints/mobilenet_edgetpu_224_1.0) | [MLPerf](https://github.com/mlcommons/mobile_open/blob/main/vision/mobilenet/README.md)  | N/A | Mobile |
| DeepLabV3+ (MobileNetV2) | ADE20k | tensorflow | [MLPerf](https://github.com/mlcommons/mobile_open/blob/main/vision/deeplab/models_and_code) | [MLPerf](https://github.com/mlcommons/mobile_open/blob/main/vision/deeplab/README.md) | N/A | Mobile |
| Stable Diffusion 1.5 | MS-COCO 2014 | torch | [Huggingface](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) | same as weights  | transformers | Mobile |
| Llama 2 7B | OpenOrca | torch | [Hugginface](https://huggingface.co/meta-llama/Llama-2-7b) (Requires account signup) | same as weights  | N/A | Client |
| SSD-MobileNetV2 | MS-COCO 2017 | tensorflow | Deprecated | N/A  | N/A | Mobile |
| DLRM | 1TB Click Logs | torch | Deprecated | N/A  | N/A | Edge |
| SSD-ResNet34 | COCO (1200×1200) | torch | Deprecated | N/A | N/A | Edge |
| MobileNet-v1 (1x3x224x224) | imagenet2012 (3x224x224) | torch | Deprecated | N/A  | N/A | Edge |
| SSD-MobileNets-v1 | imagenet2012 (3x224x224) | torch | Deprecated | N/A  | N/A | Edge |
| RNNT | Librispeech dev-clean | torch | Deprecated | N/A  | N/A | Edge |
| 3D UNET (224x224x160) | BraTS 2019 (3x224x224) | torch | Deprecated | N/A | N/A | Edge |

