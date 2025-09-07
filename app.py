import io, os, time, zipfile, numpy as np, pandas as pd
import torch, torch.nn.functional as F
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
from torchvision import transforms

# ---------- pipeline ----------
def to_tensor_255(pil_img):
    t = transforms.ToTensor()(pil_img) * 255.0  # [C,H,W] float in 0..255
    if t.shape[0] == 4:   # RGBA -> RGB
        t = t[:3, :, :]
    if t.shape[0] == 1:   # grayscale -> 3ch for consistency
        t = t.repeat(3,1,1)
    return t.float()

def resize_keep_aspect(t, target_h, target_w):
    # t: [C,H,W] -> keep aspect via shortest side, then center-crop
    img = transforms.functional.resize(t, min(target_h, target_w), antialias=True)
    img = transforms.functional.center_crop(img, (target_h, target_w))
    return img

def grayscale_filter(img):
    if img.dim() == 4:
        r,g,b = img[:,0], img[:,1], img[:,2]
        gray = 0.299*r + 0.587*g + 0.114*b
        return gray.unsqueeze(1)
    else:
        r,g,b = img[0], img[1], img[2]
        gray = 0.299*r + 0.587*g + 0.114*b
        return gray.unsqueeze(0)

def _gaussian_kernel(k=5, sigma=1.0, device="cpu", dtype=torch.float32):
    ax = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2
    xx, yy = torch.meshgrid(ax, ax, indexing="xy")
    k2d = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return k2d / k2d.sum()

def gaussian_blur_filter(img, kernel_size=5, sigma=1.0):
    added_batch = False
    if img.dim() == 3:
        img = img.unsqueeze(0)
        added_batch = True
    c = img.shape[1]
    k = _gaussian_kernel(kernel_size, sigma, img.device, img.dtype).view(1,1,kernel_size,kernel_size)
    out = F.conv2d(img, k.repeat(c,1,1,1), padding=kernel_size//2, groups=c)
    return out.squeeze(0) if added_batch else out

def sobel_edge_filter(img):
    added_batch = False
    if img.dim() == 3 and img.shape[0] == 1:
        img = img.unsqueeze(0)
        added_batch = True
    sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=img.dtype, device=img.device).view(1,1,3,3)
    sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=img.dtype, device=img.device).view(1,1,3,3)
    gx = F.conv2d(img, sobel_x, padding=1)
    gy = F.conv2d(img, sobel_y, padding=1)
    edges = torch.sqrt(gx**2 + gy**2).clamp(0,255)
    return edges.squeeze(0) if added_batch else edges

def run_pipeline(batch, use_gpu):
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        batch = batch.pin_memory().to(device, non_blocking=True)
    else:
        device = torch.device('cpu')
        batch = batch.to(device)

    t0 = time.time()
    gray = grayscale_filter(batch)
    blur = gaussian_blur_filter(gray, kernel_size=st.session_state.ksize, sigma=st.session_state.sigma)
    edges = sobel_edge_filter(blur)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    return edges.detach().cpu(), elapsed

def tensor_to_png_bytes(t):
    # t: [1,H,W] or [H,W]; normalize and save to PNG bytes
    if t.dim() == 3 and t.shape[0] == 1:
        t = t.squeeze(0)
    arr = t.numpy()
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    arr = (arr*255).astype(np.uint8)
    im = Image.fromarray(arr, mode='L')
    buf = io.BytesIO()
    im.save(buf, format='PNG')
    buf.seek(0)
    return buf

# ------------------------------ UI ------------------------------
st.set_page_config(page_title="Edge Pipeline Benchmark", layout="wide")
st.title("CPU vs GPU Edge-Detection Benchmark")

colA, colB, colC = st.columns(3)
with colA:
    height = st.number_input("Height", 256, 4096, 1024, step=64)
with colB:
    width  = st.number_input("Width", 256, 4096, 1024, step=64)
with colC:
    batch_size = st.number_input("Batch Size", 1, 64, 8, step=1)

st.session_state.ksize = st.slider("Gaussian kernel size", 3, 21, 5, step=2)
st.session_state.sigma = st.slider("Gaussian sigma", 0.1, 5.0, 1.0, step=0.1)

uploads = st.file_uploader("Upload one or more images", type=["png","jpg","jpeg","bmp","tif","tiff"], accept_multiple_files=True)

use_gpu = st.checkbox("Use GPU (if available)", value=True)
run = st.button("Run Benchmark")

if run:
    if not uploads:
        st.warning("Please upload at least one image.")
        st.stop()

    # Prepare batch (resize to same HxW)
    imgs = []
    names = []
    for f in uploads[:batch_size]:
        pil = Image.open(f).convert("RGBA")
        t = to_tensor_255(pil)
        t = resize_keep_aspect(t, height, width)
        imgs.append(t)
        names.append(f.name)
    batch = torch.stack(imgs)  # [B,3,H,W]

    st.write(f"Running on **{'GPU' if (use_gpu and torch.cuda.is_available()) else 'CPU'}** with batch size {len(imgs)} …")

    # CPU
    edges_cpu, t_cpu = run_pipeline(batch.clone(), use_gpu=False)

    # GPU (optional)
    edges_gpu, t_gpu = None, None
    if use_gpu and torch.cuda.is_available():
        edges_gpu, t_gpu = run_pipeline(batch.clone(), use_gpu=True)

    # Results table
    df = pd.DataFrame({
        "Batch Size":[len(imgs)],
        "CPU Time (s)":[round(t_cpu,4)],
        "GPU Time (s)":[round(t_gpu,4) if t_gpu is not None else None],
        "Speedup (CPU/GPU)":[round(t_cpu/t_gpu,2) if t_gpu else None],
    })
    st.dataframe(df)

    # Show per-image outputs + download
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, name in enumerate(names):
            cbuf = tensor_to_png_bytes(edges_cpu[i])
            zf.writestr(f"cpu_edges_{os.path.splitext(name)[0]}.png", cbuf.getvalue())
            st.image(cbuf, caption=f"CPU edges — {name}", use_column_width=True)
            if edges_gpu is not None:
                gbuf = tensor_to_png_bytes(edges_gpu[i])
                zf.writestr(f"gpu_edges_{os.path.splitext(name)[0]}.png", gbuf.getvalue())
                st.image(gbuf, caption=f"GPU edges — {name}", use_column_width=True)

    zip_buf.seek(0)
    st.download_button("Download all outputs (zip)", data=zip_buf, file_name="edge_results.zip", mime="application/zip")

    st.caption("Tip: increase batch size and image size to stress-test performance.")
