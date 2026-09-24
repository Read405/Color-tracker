# Color-tracker
A simple program that tracks a given color and runs locally on your device. It uses your webcam and OpenCV. Nothing gets uploaded anywhere.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage
```bash
python color_tracker.py --color blue                 # preset: red, orange, yellow, green, blue, purple, pink
python color_tracker.py --hsv 35 80 60 85 255 255    # custom HSV range (lower H S V, upper H S V; H is 0-179)
python color_tracker.py                              # then click any object to track its color
```

Options: `--camera N` (which webcam), `--min-area PX` (ignore blobs smaller than this), `--trail N` (length of the motion trail).

### Controls
| Key | Action |
| --- | --- |
| Left click | Sample the color under the cursor and track it |
| `c` | Clear the motion trail |
| `m` | Toggle the mask window, handy for tuning HSV ranges |
| `q` / `Esc` | Quit |

## How it works
1. Each frame gets blurred and converted to HSV.
2. Pixels inside the target HSV range become a binary mask. Red wraps around the hue wheel, so it uses two ranges.
3. Morphological open and close operations remove speckle noise.
4. The largest contour is the target. The program draws its centroid, enclosing circle, and a fading trail.

## Tests
```bash
pip install pytest
python -m pytest
```
