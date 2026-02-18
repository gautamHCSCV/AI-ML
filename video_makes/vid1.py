import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from moviepy import (
    ImageClip,
    AudioFileClip,
    CompositeVideoClip,
    concatenate_videoclips,
    VideoClip,
    vfx,
)
# =============================
# CONFIGURATION
# =============================

IMAGE_FOLDER = "/home/gautam/Documents/my_images3/"  # Folder containing images
OUTPUT_VIDEO = "/home/gautam/Documents/my_images3/animated_slideshow.mp4"

TITLE_TEXT = "Happy Birthday\nMy Love!"
TITLE_DURATION = 3.0  # seconds

BACKGROUND_MUSIC = "/home/gautam/Documents/my_images3/song.mp3"   # Set to None if no music
AUDIO_START_MIN = 2
AUDIO_START_SEC = 30

DURATION_PER_IMAGE = 4
VIDEO_SIZE = (1080, 1920)  # (W, H)
FPS = 30

TRANSITION_DUR = 0.8 
MODERN_TRANSITIONS = True
SPARKLE_TRANSITIONS = True

# Performance knobs
BLUR_DOWNSCALE = 0.25   # blur at 25% resolution then upscale (big speedup)
JPEG_QUALITY = 92       # for intermediate conversions (not critical)
SPARKLE_PRESET_SEED = 123

# =====================================================
# HELPERS
# =====================================================

def mmss_to_seconds(m, s):
    return int(m) * 60 + int(s)

def pil_to_np(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))

def np_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8), "RGB")

def cover_fit_pil(img: Image.Image, size):
    """Resize to cover, then crop center to exact size."""
    W, H = size
    iw, ih = img.size
    scale = max(W / iw, H / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    img2 = img.resize((nw, nh), Image.LANCZOS)
    left = (nw - W) // 2
    top = (nh - H) // 2
    return img2.crop((left, top, left + W, top + H))

def blur_pil_fast(img: Image.Image, downscale=0.25):
    """Fast blur: downscale -> gaussian blur -> upscale."""
    w, h = img.size
    dw, dh = max(1, int(w * downscale)), max(1, int(h * downscale))
    small = img.resize((dw, dh), Image.BILINEAR)
    arr = np.array(small.convert("RGB"))
    arr = cv2.GaussianBlur(arr, (0, 0), sigmaX=7, sigmaY=7)
    out = Image.fromarray(arr).resize((w, h), Image.BILINEAR)
    return out

def add_soft_overlay(img: Image.Image, strength=0.18):
    """Slight warm overlay for 'wedding/birthday' vibe."""
    W, H = img.size
    overlay = Image.new("RGB", (W, H), (255, 235, 220))
    return Image.blend(img, overlay, strength)

def make_title_card(size, text):
    W, H = size
    img = Image.new("RGB", (W, H), (12, 10, 16))
    draw = ImageDraw.Draw(img)

    # subtle gradient
    for y in range(H):
        v = int(12 + (y / H) * 25)
        draw.line([(0, y), (W, y)], fill=(v, v, v + 10))

    # font
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    font = None
    for p in font_paths:
        if os.path.exists(p):
            font = ImageFont.truetype(p, 92)
            break
    if font is None:
        font = ImageFont.load_default()

    # center text
    lines = text.split("\n")
    line_sizes = [draw.textbbox((0, 0), ln, font=font) for ln in lines]
    heights = [(b[3] - b[1]) for b in line_sizes]
    total_h = sum(heights) + 20 * (len(lines) - 1)

    y = int(H * 0.42) - total_h // 2
    for i, ln in enumerate(lines):
        # glow
        draw.text((W//2 + 2, y + 2), ln, font=font, fill=(255, 220, 190))
        draw.text((W//2, y), ln, font=font, fill=(255, 245, 238), anchor="mm")
        y += heights[i] + 20

    return pil_to_np(img)

def make_static_slide_np(img_path, size):
    """Precompute ONE static frame per slide (fast)."""
    W, H = size
    img = Image.open(img_path).convert("RGB")

    # background: cover + blur + warm overlay
    bg = cover_fit_pil(img, size)
    bg = blur_pil_fast(bg, downscale=BLUR_DOWNSCALE)
    bg = add_soft_overlay(bg, strength=0.16)

    # foreground: fit height, keep aspect, paste center
    fg_h = int(H * 0.86)
    fg = img.copy()
    fg = fg.resize((int(fg.size[0] * (fg_h / fg.size[1])), fg_h), Image.LANCZOS)

    canvas = bg.copy()
    x = (W - fg.size[0]) // 2
    y = (H - fg.size[1]) // 2
    canvas.paste(fg, (x, y))

    return pil_to_np(canvas)

def make_sparkle_overlay_clip(size, duration, fps=30, seed=0):
    """
    FAST sparkles: vectorized mask generation (no heavy loops per particle pixel region).
    Still per-frame, but only for short duration and reused.
    """
    W, H = size
    rng = np.random.default_rng(seed)

    n = 120
    x0 = rng.uniform(0, W, n)
    y0 = rng.uniform(0, H, n)
    r = rng.uniform(2, 5, n)
    t0 = rng.uniform(0, duration * 0.8, n)
    life = rng.uniform(0.2, 0.5, n)

    # Precompute grid for cheap distance checks on small tiles
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    def make_mask(t):
        tt = float(t)
        alive = (t0 <= tt) & (tt <= (t0 + life))
        if not np.any(alive):
            return np.zeros((H, W), dtype=np.float32)

        prog = (tt - t0[alive]) / life[alive]
        a = np.sin(np.pi * prog) ** 1.5  # smoother
        xs = x0[alive]
        ys = y0[alive]
        rs = r[alive]

        mask = np.zeros((H, W), dtype=np.float32)
        # Additive compositing of circles
        for xi, yi, ri, ai in zip(xs, ys, rs, a):
            dist2 = (xx - xi) ** 2 + (yy - yi) ** 2
            mask = np.maximum(mask, (dist2 <= (ri * ri)).astype(np.float32) * ai)

        return mask

    def make_rgb(t):
        m = make_mask(t)
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        frame[m > 0] = 255
        return frame

    sparkle = VideoClip(make_rgb, duration=duration).with_fps(fps)
    mask_clip = VideoClip(make_mask, duration=duration).with_fps(fps)
    sparkle = sparkle.with_mask(mask_clip).with_opacity(0.9)
    return sparkle

# =====================================================
# BUILD: Title + Slides (precomputed)
# =====================================================

# Title clip
title_np = make_title_card(VIDEO_SIZE, TITLE_TEXT)
title_clip = ImageClip(title_np).with_duration(TITLE_DURATION).with_fps(FPS)
title_clip = title_clip.with_effects([vfx.FadeIn(0.6), vfx.FadeOut(0.6)])

# Slides: precompute composite images ONCE (huge speed win)
image_files = sorted([
    os.path.join(IMAGE_FOLDER, f)
    for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])

slides = []
for p in image_files:
    slide_np = make_static_slide_np(p, VIDEO_SIZE)
    slide = ImageClip(slide_np).with_duration(DURATION_PER_IMAGE).with_fps(FPS)

    # Lightweight Ken Burns zoom (cheap)
    slide = slide.with_effects([
        vfx.Resize(lambda t: 1 + 0.02 * t),
        vfx.FadeIn(0.6),
        vfx.FadeOut(0.6)
    ])

    slides.append(slide)

clips = [title_clip] + slides

# =====================================================
# TRANSITIONS (fast overlap + optional sparkle reused)
# =====================================================

sparkle_clip = None
if MODERN_TRANSITIONS and SPARKLE_TRANSITIONS and TRANSITION_DUR > 0:
    # Pre-render sparkle once and reuse for all transitions
    sparkle_clip = make_sparkle_overlay_clip(VIDEO_SIZE, TRANSITION_DUR, fps=FPS, seed=SPARKLE_PRESET_SEED)

if MODERN_TRANSITIONS and TRANSITION_DUR > 0:
    result = clips[0]
    for i in range(1, len(clips)):
        prev = result
        cur = clips[i]

        prev_fx = prev.with_effects([vfx.FadeOut(TRANSITION_DUR)])
        cur_fx = cur.with_effects([vfx.FadeIn(TRANSITION_DUR)]).with_start(prev.duration - TRANSITION_DUR)

        layers = [prev_fx, cur_fx]
        if sparkle_clip is not None:
            layers.append(sparkle_clip.with_start(prev.duration - TRANSITION_DUR))

        result = CompositeVideoClip(layers, size=VIDEO_SIZE).with_duration(prev.duration + cur.duration - TRANSITION_DUR)

    final_video = result
else:
    final_video = concatenate_videoclips(clips, method="compose")

final_video = final_video.with_fps(FPS)

# =====================================================
# AUDIO (trim from mm:ss)
# =====================================================

if BACKGROUND_MUSIC and os.path.exists(BACKGROUND_MUSIC):
    audio = AudioFileClip(BACKGROUND_MUSIC)
    start = mmss_to_seconds(AUDIO_START_MIN, AUDIO_START_SEC)
    end = start + final_video.duration
    final_video = final_video.with_audio(audio.subclipped(start, end))

# =====================================================
# EXPORT (use preset for speed)
# =====================================================

final_video.write_videofile(
    OUTPUT_VIDEO,
    fps=FPS,
    codec="libx264",
    audio_codec="aac",
    preset="veryfast",     # <-- big speedup
    threads=0              # auto threads
)