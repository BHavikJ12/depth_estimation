# Setting this up on Windows — complete beginner's guide

This walks you from a Windows PC with nothing installed to a working drone
detector that draws a box around a drone in a video and tells you how far away
it is.

**No prior Python experience needed.** Every command is copy-and-paste. If a
step says something will look odd, it will look odd, and that is fine.

* **Time:** about 20 minutes, most of it waiting for downloads
* **Disk space:** about 1 GB
* **Needs:** Windows 10 or 11, 64-bit, and an internet connection
* **You do NOT need:** a graphics card, or anything from NVIDIA

---

## What you are about to install

| | |
|---|---|
| **Python 3.12** | the programming language this is written in |
| **The code** | from GitHub, about 500 KB |
| **Some Python packages** | maths, image handling, and the thing that runs the AI model — about 400 MB |
| **The AI model** | 7 MB, downloaded automatically the first time you run it |
| **ffmpeg** (optional) | only needed if you want to save annotated videos |

Everything goes in one folder and one hidden cache folder. Section 10 removes
all of it.

---

## Step 1 — Install Python 3.12

1. Go to **<https://www.python.org/downloads/release/python-3120/>**
2. Scroll to the bottom, under **Files**, click
   **Windows installer (64-bit)**
3. Run the downloaded file.

> ### ⚠ The one thing that trips everybody up
>
> On the **first** installer screen there is a checkbox at the bottom:
>
> **☐ Add python.exe to PATH**
>
> **Tick it.** It is not ticked by default. If you miss it, every command in
> this guide fails with `'python' is not recognized`, and the fix is to run the
> installer again and choose *Modify*.

4. Then click **Install Now** and wait.
5. At the end, if you see **"Disable path length limit"**, click it. It
   prevents a class of confusing errors later.

### Why 3.12 and not the newest

The packages this needs publish ready-made Windows versions for Python 3.11,
3.12 and 3.13. Brand-new Python releases often have no ready-made packages yet,
and pip then tries to build them from source, which fails on a machine without
a C++ compiler. **3.12 is the safe choice.** 3.11 and 3.13 also work; 3.10 and
older do not.

### Do not use the Microsoft Store version

If you type `python` and the Microsoft Store opens, Windows has a placeholder
instead of real Python. Install from python.org as above. (If it keeps opening
the Store afterwards: Settings → Apps → Advanced app settings → App execution
aliases → turn **off** `python.exe` and `python3.exe`.)

### Check it worked

Press **Windows key**, type `cmd`, press Enter. A black window opens. Type:

```
python --version
```

You should see `Python 3.12.0` or similar. If you instead see
`'python' is not recognized`, PATH was not ticked — reinstall, or see section 9.

---

## Step 2 — Choose a folder

Make a folder directly on your C: drive:

```
mkdir C:\drone
```

**Use this, not Documents or Desktop.** Those are often synced by OneDrive,
which locks files mid-write and produces errors that look like bugs in the code.
A short path like `C:\drone` also avoids Windows' 260-character path limit, and
it has no spaces in it, which keeps commands simple.

---

## Step 3 — Get the code

### The easy way: download a ZIP

1. Go to **<https://github.com/BHavikJ12/depth_estimation>**
2. Green **Code** button → **Download ZIP**
3. Open the downloaded ZIP, and drag the folder inside it into `C:\drone`
4. You should end up with `C:\drone\depth_estimation\monoculer_camera_yolo\`

> If the repository page asks you to sign in, it is private — ask Bhavik for
> access, or for the ZIP directly.

### Or with Git, if you would rather

Install Git from <https://git-scm.com/download/win> (all defaults are fine),
then in a Command Prompt:

```
cd /d C:\drone
git clone https://github.com/BHavikJ12/depth_estimation.git
```

Git makes later updates a single `git pull` instead of downloading a new ZIP,
so it is worth it if you expect the code to change.

---

## Step 4 — Run the installer

Open the `monoculer_camera_yolo` folder in File Explorer and **double-click
`setup.bat`**.

A black window opens and prints its progress. It takes a few minutes; most of
that is downloading packages.

> **If Windows shows a blue "Windows protected your PC" box:** click
> **More info**, then **Run anyway**. Windows shows this for any script it has
> not seen before. You can read `setup.bat` in Notepad first if you want to see
> what it does.

If you would rather type the commands than double-click, open a Command Prompt
and run:

```
cd /d C:\drone\depth_estimation\monoculer_camera_yolo
setup.bat
```

### What it does

1. Checks Python is installed and new enough
2. Creates a **virtual environment** in `venv\` — a private copy of Python for
   this project, so installing things here cannot break anything else on your PC
3. Installs the four packages it needs
4. Runs two self-tests

### What success looks like

```
[4/4] Running the self-tests (no API key or camera needed) ...

  [ok  ] hfov round-trips   fx=914.01px
  ...
all checks passed

  stride 1
    58/60 frames locked, track ids ['1']
    mean error 2.32%   inside 20 m: 1.39%
pipeline ok

==================================================================
 Setup finished. Next:
```

Those tests use a computer-generated video, so they prove the maths, the
tracking and the drawing all work **before** any camera or internet is involved.
If they pass, your installation is sound.

---

## Step 5 — Get your free API key

The AI model lives on a website called Roboflow. Downloading it once needs a
free account.

1. Go to **<https://app.roboflow.com/settings/api>**
2. Sign up (free) if you have not already
3. Copy your **Private API Key** — a string of random letters and numbers

Now store it so Windows remembers it. In a Command Prompt:

```
setx ROBOFLOW_API_KEY "paste_your_key_here"
```

Keep the quotes. You should see `SUCCESS: Specified value was saved.`

> **`setx` only affects NEW windows.** Close this Command Prompt and open a
> fresh one before the next step, or the key will appear to be missing.

Check it took:

```
echo %ROBOFLOW_API_KEY%
```

That should print your key. If it prints `%ROBOFLOW_API_KEY%` instead, you are
in the old window — open a new one.

---

## Step 6 — Your first run

Open a **new** Command Prompt:

```
cd /d C:\drone\depth_estimation\monoculer_camera_yolo

venv\Scripts\python.exe run.py --source 0 --model-id drone-detection-rchy7/8 --keep-classes 1 --conf 0.40 --hfov 70 --target-size 0.5
```

The first run pauses for a few seconds while it downloads the 7 MB model. Then a
window opens showing your webcam with a telemetry readout in the corner.

**Press `q` to quit.** Closing the window with the X can leave the program
running; `q` is cleaner.

> **Which camera is `0`?** If your PC has more than one (a built-in webcam and
> a USB one, say), run this first:
>
> ```
> venv\Scripts\python.exe run.py --list-cameras
> ```
>
> It lists every camera with its resolution and name, and saves one photo from
> each into `camera_snapshots\`. Open those images, see which is which, and use
> that number as `--source`.

> **If Windows asks for camera permission, allow it.** If no window appears and
> you get a camera error, go to Settings → Privacy & security → Camera, and turn
> on both **Camera access** and **Let desktop apps access your camera**. This is
> off by default on many machines and is the most common reason `--source 0`
> fails.

### Note on `venv\Scripts\python.exe`

You will see this on every command. It means "use this project's private
Python". You may read elsewhere about "activating" the environment first —
you do not need to, and skipping it avoids a PowerShell security setting that
blocks activation scripts by default.

---

## Step 7 — Run it on a video file

A webcam pointed at your room will not find any drones. To try a real video, put
an `.mp4` in the project folder and:

```
venv\Scripts\python.exe run.py --source myvideo.mp4 --model-id drone-detection-rchy7/8 --keep-classes 1 --conf 0.40 --hfov 70 --target-size 0.5
```

If the file is elsewhere, give the full path in quotes:

```
venv\Scripts\python.exe run.py --source "C:\Users\You\Videos\drone clip.mp4" ...
```

Quotes matter whenever a path contains spaces.

`--source` also accepts a single image, a folder of images, or a camera stream
URL. See [README.md](README.md).

---

## Step 8 — Saving an annotated video (optional)

Add `--save`:

```
venv\Scripts\python.exe run.py --source myvideo.mp4 --model-id drone-detection-rchy7/8 --keep-classes 1 --conf 0.40 --hfov 70 --save out.mp4 --save-csv track.csv --no-display
```

* `--save out.mp4` — the video with boxes and distances drawn on
* `--save-csv track.csv` — a spreadsheet of every detection, openable in Excel
* `--no-display` — skip the preview window, which makes it finish faster

### Install ffmpeg first, or the video may look black

Without ffmpeg, the video is written in an old format that many Windows players
show as a black screen. Install it once:

```
winget install Gyan.FFmpeg
```

Then **close and reopen** the Command Prompt. Check it worked:

```
ffmpeg -version
```

If `winget` is not available (older Windows 10), download the "essentials" build
from <https://www.gyan.dev/ffmpeg/builds/>, unzip it to `C:\ffmpeg`, and add
`C:\ffmpeg\bin` to your PATH (Windows key → "environment variables" → Edit the
`Path` entry under *User variables* → New → paste the path → OK).

The program tells you which one it used:

```
writing out.mp4 as h264 (ffmpeg)      <- good, plays anywhere
writing out.mp4 as mp4v (opencv)      <- ffmpeg missing, may look black
```

---

## Step 8b — Watching from your phone (optional)

`--web-port` serves the annotated feed over your local network, so you can watch
it in a browser on a phone or another laptop instead of the desktop window:

```
venv\Scripts\python.exe run.py --source 0 --model-id drone-detection-rchy7/8 --keep-classes 1 --conf 0.40 --hfov 70 --web-port 8000 --no-display
```

**It prints the address for you** when it starts — you do not have to work it
out:

```
web stream, open from any device on this network:
    http://192.168.1.42:8000/
```

Type that into a browser on your phone. Both devices must be on the same Wi-Fi.
(If you ever need to find it yourself: `ipconfig`, look for **IPv4 Address**.)

**Windows Firewall will ask for permission the first time.** Tick **Private
networks** and click *Allow access*. If you dismissed it and nothing loads, the
firewall is blocking it — re-run and allow, or add an inbound rule for that port.

---

## Step 9 — If something breaks

| What you see | What it means | Fix |
|---|---|---|
| `'python' is not recognized` | PATH was not ticked in Step 1 | Re-run the Python installer, choose **Modify** → Next → tick **Add Python to environment variables** |
| The Microsoft Store opens when you type `python` | Windows placeholder, not real Python | Install from python.org; then Settings → Apps → Advanced app settings → App execution aliases → turn off `python.exe` |
| `Python 3.x is too old` | You have 3.10 or older | Install 3.12 from python.org |
| `error: No Roboflow API key` | Key not set, or you are in an old window | `setx ROBOFLOW_API_KEY "..."` then **open a new Command Prompt** |
| `... is not cached ... has to be downloaded` | Same as above, or no internet | As above; the model needs one download |
| `cannot open source '0'` | No camera, or Windows is blocking it | Settings → Privacy & security → Camera → allow desktop apps. Or try `--source 1` |
| Camera window is black, no error | Another app is using the camera, or the privacy shutter is closed | Close Teams, Zoom, Skype, the Camera app. Run `--list-cameras`; it flags a near-black feed |
| Wrong camera opens | You have more than one | `--list-cameras`, look at the snapshots, use the right number |
| Saved video plays black | ffmpeg not installed | Step 8 |
| `Windows protected your PC` on setup.bat | Normal for new scripts | **More info** → **Run anyway** |
| `running scripts is disabled on this system` | PowerShell blocking activation | Use `venv\Scripts\python.exe` directly as this guide does; do not activate |
| pip fails, mentions `Microsoft Visual C++ 14.0` | Your Python has no ready-made packages | You are on a too-new or too-old Python. Install 3.12 |
| Errors mentioning OneDrive, or files vanishing | Project is in a synced folder | Move it to `C:\drone` |
| `--web-port` page will not load from another device | Windows Firewall, or the wrong address | Allow the prompt for **Private networks**; check the IPv4 address with `ipconfig`; both devices must be on the same Wi-Fi |
| `The system cannot find the path specified` | Typo, or you are in the wrong folder | `cd /d C:\drone\depth_estimation\monoculer_camera_yolo` then `dir` — you should see `run.py` |

### Starting over

Deleting `venv\` and running `setup.bat` again is safe and fixes most
installation problems. Nothing outside the project folder is touched.

---

## Step 10 — Uninstalling

Three things to delete:

1. The project: delete `C:\drone`
2. The model cache: delete `C:\Users\<your name>\.cache\roboflow-onnx`
   (paste `%USERPROFILE%\.cache` into File Explorer's address bar)
3. The saved key: `setx ROBOFLOW_API_KEY ""`

Python and ffmpeg can be removed from Settings → Apps if you want them gone too.

---

## Command cheat sheet

Every command assumes you are in the project folder:

```
cd /d C:\drone\depth_estimation\monoculer_camera_yolo
```

| Goal | Command |
|---|---|
| Webcam, live | `venv\Scripts\python.exe run.py --source 0 --model-id drone-detection-rchy7/8 --keep-classes 1 --conf 0.40 --hfov 70` |
| A video file | `venv\Scripts\python.exe run.py --source clip.mp4 --model-id drone-detection-rchy7/8 --keep-classes 1 --conf 0.40 --hfov 70` |
| Save the result | add `--save out.mp4 --save-csv track.csv --no-display` |
| One image | `--source photo.jpg` |
| A folder of images | `--source images\ --save out\` |
| Watch from a phone | add `--web-port 8000 --no-display`, then browse to `http://<this-pc-ip>:8000/` |
| Which camera is which? | `venv\Scripts\python.exe run.py --list-cameras` |
| See all options | `venv\Scripts\python.exe run.py --help` |
| Check it still works | `venv\Scripts\python.exe selftest.py` |

`--hfov 70` is a guess about your camera's lens and **every distance depends on
it**. Before believing any number on screen, read "Getting a range you can
trust" in [README.md](README.md).

---

## Two honest warnings

**The distances are estimates built on an assumption.** The program works out
range from how large the drone looks, which requires knowing how big the drone
really is. `--target-size 0.5` means "half a metre across". A drone twice that
size will read at twice the distance. README.md explains how to measure the
right number for your drone.

**The detector is only reliable for drones against sky.** Measured on real
footage: 98% of frames when the drone is against clouds, but it misses drones
near the ground and against trees, and it sometimes boxes clouds confidently.
Look at the boxes rather than trusting the count. The "What this model actually
does, measured" section of [README.md](README.md) has the numbers.

---

## A note on testing

This guide was written and checked carefully, but **it has not been run on a
Windows machine** — the project was developed on Linux. The Python code itself
is platform-neutral and its self-tests pass, and the Windows-specific parts
(`setup.bat`, the paths, `setx`) follow standard practice. If a step does not
behave as described, that is worth reporting so this file can be corrected.
