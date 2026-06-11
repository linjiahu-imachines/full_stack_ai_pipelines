# Jetson: `torch.cuda.is_available()` is False (NvRm / nvmap permission errors)

## If you do not have `sudo`

You **cannot** add yourself to `video` / `render` or change device permissions. Options:

1. **Ask an administrator** to run:  
   `sudo usermod -aG video,render YOUR_USERNAME`  
   then you **log out completely** (or reboot) and sign in again.
2. **Use another Linux account** that is already in `video` (and `render` if present), if your organization allows it.
3. **Run the GPU benchmark suite on a different host** where you have a CUDA-capable user environment.

Copy-paste for IT:

> User `USERNAME` needs GPU access for CUDA/PyTorch on Jetson. Please add them to groups `video` and `render`, then have them re-login or reboot. Symptom before fix: `torch … +cu130` but `torch.cuda.is_available()` is False; logs show NvRmMemInitNvmap permission denied.

The sections below are for **you** (if you have sudo) or for an **administrator** fixing the same symptom.

If `install_jetson_gpu_deps.sh` installs **`torch … +cu130`** but you still see **NvRmMemInitNvmap permission denied** and **`torch.cuda.is_available()` is False**, PyTorch is fine; **this user cannot open the GPU device nodes** until groups/permissions are fixed.

---

## 1. Confirm the symptom

```bash
cd /home/linhu/projects/vllm_exploration/vllm_gpu_test
source venv/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

- **`2.x.x+cu130` + `False`** → driver/GPU access issue (this document).
- **`2.x.x+cpu` + `False`** → wrong wheel; re-run install or use CUDA index (see README).

---

## 2. Put your user in the `video` (and `render`) groups

On Ubuntu/Jetson this is the most common fix:

```bash
sudo usermod -aG video "$USER"
sudo usermod -aG render "$USER" 2>/dev/null || true   # render may not exist on all images
```

Then **log out completely** (or **reboot**) and sign in again so the new group membership applies.

Check:

```bash
groups
```

You should see **`video`** in the list.

---

## 3. Check device nodes

```bash
ls -l /dev/nvidia* /dev/nvhost-* 2>/dev/null | head -30
```

If `/dev/nvidia0` or related nodes are root-only and not group-readable, group membership alone may not be enough until udev rules match your image; search for **Jetson udev nvidia** in [NVIDIA Jetson forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems/) for your JetPack version.

---

## 4. Run from a normal desktop session (if applicable)

Some setups only expose full GPU access from the **local graphical login**. If you use SSH, try running the same `python -c "import torch; ..."` from a terminal **on the device desktop** after fixing groups.

---

## 5. Jetson AI Lab PyPI DNS (`Name or service not known`)

That hostname is **optional**. The install script uses **`download.pytorch.org/whl/cu130`** when Jetson PyPI is unreachable. You can also set:

```bash
export VLLM_TRY_JETSON_PYPI_FIRST=0   # default: prefer official PyTorch first (see install script)
```

---

## 6. `cuda-python` version changes during `pip install -r requirements.txt`

The resolver may **downgrade** `cuda-python` (e.g. 13.2 → 13.0.x) to match **torch 2.10 + cu130** and **vLLM** dependencies. That is normal if the install completes without errors.

---

After `torch.cuda.is_available()` is **True**, run:

```bash
./run_gpu_tests.sh compare
```
