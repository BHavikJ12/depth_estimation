# Installing on another machine

Everything here is self-contained: a folder of Python files, a virtualenv, and
model weights that download themselves on first run. There is nothing to
register, no API key, and nothing outside the folder to uninstall later.

For driving it once installed, see [GETTING_STARTED.md](GETTING_STARTED.md).

---

## 1. Will it run on that machine?

| | GPU (fast) | CPU (works, slow) |
|---|---|---|
| **Linux x86_64 + NVIDIA** | yes | yes |
| **Windows x64 + NVIDIA** | yes | yes |
| macOS (Intel or Apple Silicon) | **no** | yes |
| Linux ARM64 — Jetson, Raspberry Pi | **no**, not from PyPI | yes |
| Any x86 laptop, no NVIDIA card | — | yes |

The limit is ONNX Runtime: it publishes GPU wheels for **Linux x86_64 and
Windows x64 only**. Everywhere else you get the CPU build, which runs the same
code and gives the same answers, just slower. Measured on this project's
machine, same model, 1280x720 in:

| `--res` | GPU (GTX 1650) | CPU (4-core laptop) |
|---:|---:|---:|
| 252 | 50 fps | **11.5 fps** |
| 364 | 32 fps | 6.9 fps |
| 504 *(default)* | 19 fps | 3.6 fps |

So the CPU build is usable live at `--res 252`, and fine at any resolution for
photos and for processing clips offline.

Requirements:

- **Python 3.10 or newer** (`python3 --version`)
- **NVIDIA driver 525+** for the GPU path — `nvidia-smi` should print a table.
  You do **not** need the CUDA toolkit installed; the CUDA 12 and cuDNN 9
  runtimes arrive as pip wheels inside the venv.
- **Disk:** measured — **440 MB** for the CPU install, **3.7 GB** for the GPU
  install (the bundled CUDA/cuDNN wheels are most of it), plus ~100 MB per
  model you actually use.
- A webcam, if you want live video. Any UVC camera works.

---

## 2. Get the code

```bash
git clone https://github.com/BHavikJ12/depth_estimation.git
cd depth_estimation/monocular_depth_anything
```

The repository is ~315 KB: no virtualenv, no model weights, no test footage. All
three are rebuilt or re-fetched below. See the repo [README](../README.md) for
where each one comes from.

> **Never copy a `venv/` between machines.** It is built for one CPU
> architecture and every script inside hard-codes an absolute path, so it breaks
> as soon as the folder moves — renaming the folder is enough. Rebuild it with
> `./setup.sh`.

## 3. Install

```bash
./setup.sh
```

It detects whether an NVIDIA GPU is present, installs the matching ONNX Runtime,
builds `venv/`, and runs the self-test. Override the detection if you need to:

```bash
./setup.sh --cpu        # force the CPU build (smaller; good on a laptop on battery)
./setup.sh --gpu        # force the GPU build
./setup.sh --metric     # also torch + transformers, for the V2-Metric models (~2.5 GB)
./setup.sh --cpu --metric
```

It should end with:

```
14/14 passed

Done.  Try it:
    ./venv/bin/python run.py --source 0
```

**Anything other than `14/14 passed` means stop and fix it** before going
further — see [Troubleshooting](#6-troubleshooting).

On a GPU machine with nothing downloaded yet, the runtime check reports:

```
[  ok  ] onnxruntime providers  CUDA provider listed; no weights on disk to load it with
```

That is expected, not a warning. Confirming the GPU provider *loads* needs a
real model, so the check defers until you have one — step 4 is where you see it
for certain.

### Windows

No `setup.sh`; run the same four steps in PowerShell:

```powershell
python -m venv venv
venv\Scripts\pip install -r requirements.txt
venv\Scripts\pip install -r requirements-gpu.txt    # or requirements-cpu.txt
venv\Scripts\python selftest.py
```

Then use `venv\Scripts\python` everywhere this documentation says
`./venv/bin/python`.

---

## 4. First run

```bash
./venv/bin/python run.py --source 0
```

The first run downloads the default model (~100 MB) into `models/` and prints
progress. After that it works offline.

Confirm the header of the window says what you expect:

```
[run] v3-small on cuda (CUDAExecutionProvider, CPUExecutionProvider)
```

`on cuda` is the GPU path. `on cpu` means you are on the slow path — intentional
on a machine without NVIDIA, a problem on one with it.

On a CPU-only machine, drop the resolution so it stays interactive — this is
the difference between 3.6 and 11.5 fps:

```bash
./venv/bin/python run.py --source 0 --res 252
```

`[` and `]` change it live, so you can find the trade-off you like while
watching the result.

---

## 5. Installing without internet

The new machine needs the network twice: once for pip, once for the weights.
Both can be carried across instead.

**Weights** — copy the `models/` folder from a machine that has already run it.
It is a plain cache; drop it next to the `.py` files and nothing will download.

```bash
rsync -a old-machine:path/to/depth_estimation/monocular_depth_anything/models/ ./models/
```

**Packages** — download the wheels on a connected machine with the *same* OS,
architecture and Python version, then install from the folder:

```bash
# connected machine
pip download -r requirements.txt -r requirements-cpu.txt -d wheels/

# air-gapped machine
python3 -m venv venv
./venv/bin/pip install --no-index --find-links=wheels/ -r requirements.txt -r requirements-cpu.txt
./venv/bin/python selftest.py
```

Do not copy `venv/` between machines — it hard-codes absolute paths and will
break in ways that are annoying to diagnose.

---

## 6. Troubleshooting

**`python3: command not found`, or Python is older than 3.10.**
On Ubuntu/Debian: `sudo apt install python3 python3-venv python3-pip`. The
`python3-venv` package is separate and `setup.sh` fails without it.

**Self-test says `CPU only (onnxruntime-gpu not installed?)` on a machine that
has an NVIDIA card.** Either the GPU build was not installed, or it could not
load. Reinstall it explicitly:

```bash
./venv/bin/pip install --force-reinstall "onnxruntime-gpu[cuda,cudnn]>=1.19"
./venv/bin/python selftest.py
```

If it still says CPU, check `nvidia-smi` runs at all and reports driver 525 or
newer. Force the issue with `run.py --device cuda`, which fails loudly instead
of silently falling back.

**Self-test says `CUDA provider listed but refused to load`.** The wheels are
there but the driver is too old, or the card is not visible (common inside
containers and WSL). `nvidia-smi` is the test: if that fails, this will too.

**pip hangs for a minute on every package, then fails.** A `pip.conf` somewhere
is pointing at an unreachable extra index — `pypi.ngc.nvidia.com` is the usual
culprit on machines where NVIDIA PyIndex has been installed. `setup.sh` detects
that specific case and works around it. To check by hand:

```bash
pip config list
PIP_CONFIG_FILE=/dev/null ./venv/bin/pip install ...   # bypass it
```

**`cannot open source: 0`.** No camera at index 0, or another program holds it.
Try `--source 1`. On Linux, `ls /dev/video*` lists what exists and
`fuser -v /dev/video0` shows who is using it; you may also need to be in the
`video` group (`sudo usermod -aG video $USER`, then log out and back in).

**No window appears** — headless server, SSH without X forwarding, or a bare
Wayland session. Run without a display:

```bash
./venv/bin/python run.py --source 0 --no-display --snapshot       # one capture
./venv/bin/python run.py --source clip.mp4 --no-display --save out.mp4
```

**`is a torch model and 'torch' is not installed`.** Run `./setup.sh --metric`,
or use the ONNX models and get metres from the floor instead
(`--scale ground --camera-height 1.2`).

**Out of GPU memory.** `v3-large` needs more than a 4 GB card has. Use
`v3-small` (the default) or `v3-base`, or add `--device cpu`.

---

## 7. Uninstalling

Delete the folder. Nothing is installed system-wide, no services are registered,
and no files are written outside it. The only external footprint is Hugging
Face's download cache under `models/`, which goes with it.
