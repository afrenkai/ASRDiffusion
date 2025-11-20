# ASRDiffusion


requirements:

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Recommended: CLI Version of [hf auth](https://huggingface.co/docs/huggingface_hub/en/guides/cli). Note that this can be done programatically
- Cuda Compute Capable GPU [what does that mean?](https://developer.nvidia.com/cuda-gpus)
- like 50 GB of free space or something (number updated soon)
- For linux: [nvcc-cu12](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) and [cuda toolkit](https://developer.nvidia.com/cuda-downloads)
- For windows: will get back to you
- For mac: N/A, might be able to get away with mps but untested


## Auth Example:
- `uvx --from huggingface_hub hf auth login` if using UV for the CLI
- ```from huggingface_hub import login \n login()``` for the programatic solution

