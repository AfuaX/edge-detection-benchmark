import torch
import torch.nn.functional as F
import time
import numpy as np
import pandas as pd
from torchvision import transforms
from torchvision.io import read_image
import os
import gc


import matplotlib  
matplotlib.use("Agg")  
import matplotlib.pyplot as plt  

OUTPUT_DIR = os.path.dirname(__file__)  
INPUT_DIR = os.path.join(OUTPUT_DIR, "images")  

# ------------------ Image Loading / Generation ------------------

def generate_synthetic_image(height=1024, width=1024, channels=3):
    return torch.rand(channels, height, width) * 255  # [C,H,W]

def load_image_or_synthetic(height=1024, width=1024, channels=3, filename='input.jpg'):
    if os.path.exists(filename):
        try:
            img = read_image(filename).float()  # [C,H,W]
            if img.shape[0] != channels:
                raise ValueError(f"Expected {channels} channels, but got {img.shape[0]} for {filename}")
            if img.shape[1:] != (height, width):
                img = transforms.Resize((height, width))(img)
            return img
        except Exception as e:
            print(f"Error loading {filename}: {e}. Ensure the file is a valid image with {channels} channels. Falling back to synthetic image.")
    return generate_synthetic_image(height, width, channels)

#list images in a folder
def list_image_files(folder, exts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
    if not os.path.isdir(folder):
        return []
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]
    files.sort()
    return files

# ------------------ Filters ------------------

def grayscale_filter(img):
    if img.dim() == 4:
        if img.shape[1] != 3:
            raise ValueError(f"Batch grayscale filter expects [B, 3, H, W], got {img.shape}")
        r, g, b = img[:, 0], img[:, 1], img[:, 2]
        gray = 0.299*r + 0.587*g + 0.114*b
        return gray.unsqueeze(1)
    elif img.dim() == 3:
        if img.shape[0] != 3:
            raise ValueError(f"Grayscale filter expects [3, H, W], got {img.shape}")
        r, g, b = img[0], img[1], img[2]
        gray = 0.299*r + 0.587*g + 0.114*b
        return gray.unsqueeze(0)
    else:
        raise ValueError(f"Unsupported image dimensions: {img.shape}")
    return img  # not reached

# --------- CHANGED: batch-safe Gaussian blur using depthwise conv ---------
def _gaussian_kernel(kernel_size=5, sigma=1.0, device="cpu", dtype=torch.float32):
    ax = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2
    xx, yy = torch.meshgrid(ax, ax, indexing="xy")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel

def gaussian_blur_filter(img, kernel_size=5, sigma=1.0):
    # Accept [C,H,W] or [B,C,H,W]; return same rank
    added_batch = False
    if img.dim() == 3:
        img = img.unsqueeze(0)
        added_batch = True
    if img.dim() != 4:
        raise ValueError(f"Gaussian blur expects [C,H,W] or [B,C,H,W], got {img.shape}")

    b, c, h, w = img.shape
    kernel = _gaussian_kernel(kernel_size, sigma, img.device, img.dtype)
    weight = kernel.view(1,1,kernel_size,kernel_size).repeat(c,1,1,1)  # depthwise
    pad = kernel_size // 2
    out = F.conv2d(img, weight, padding=pad, groups=c)
    return out.squeeze(0) if added_batch else out
# -------------------------------------------------------------------------

# --------- CHANGED: sobel accepts 3D or 4D, wraps to 4D for conv2d --------
def sobel_edge_filter(img):
    if img.dim() == 3 and img.shape[0] == 1:
        img = img.unsqueeze(0)   # [1,1,H,W]
        added_batch = True
    elif img.dim() == 4 and img.shape[1] == 1:
        added_batch = False
    else:
        raise ValueError(f"Sobel filter expects [1, H, W] or [B, 1, H, W], got {img.shape}")

    sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=img.dtype, device=img.device).view(1,1,3,3)
    sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=img.dtype, device=img.device).view(1,1,3,3)
    gx = F.conv2d(img, sobel_x, padding=1)
    gy = F.conv2d(img, sobel_y, padding=1)
    edges = torch.sqrt(gx**2 + gy**2).clamp(0,255)
    return edges.squeeze(0) if added_batch else edges
# -------------------------------------------------------------------------

# ------------------ Pipelines ------------------

def cpu_pipeline(img):
    device = torch.device('cpu')
    img = img.to(device)
    return sobel_edge_filter(gaussian_blur_filter(grayscale_filter(img)))

def gpu_pipeline(img):
    if not torch.cuda.is_available():
        print("CUDA not available, running on CPU")
        return cpu_pipeline(img)
    device = torch.device('cuda')
    try:
        img = img.pin_memory().to(device, non_blocking=True)
        return sobel_edge_filter(gaussian_blur_filter(grayscale_filter(img)))
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"GPU out of memory: {e}. Falling back to CPU.")
        else:
            print(f"GPU error: {e}. Falling back to CPU.")
        torch.cuda.empty_cache()
        gc.collect()
        return cpu_pipeline(img)

# ------------------ Visualization & Saving ------------------

def visualize_results(original, result, title="Edge Detected", batch_idx=0):
    plt.ioff()
    fig, axes = plt.subplots(1,2,figsize=(10,5))
    orig_img = original[batch_idx].cpu() if original.dim()==4 else original.cpu()
    res_img = result[batch_idx].cpu() if result.dim()==4 else result.cpu()

    # Dynamic normalization
    orig_min, orig_max = orig_img.min(), orig_img.max()
    res_min, res_max = res_img.min(), res_img.max()
    orig_img = (orig_img - orig_min) / (orig_max - orig_min + 1e-8) if orig_max > orig_min else orig_img
    res_img = (res_img - res_min) / (res_max - res_min + 1e-8) if res_max > res_min else res_img

    axes[0].imshow(orig_img.permute(1,2,0).numpy() if orig_img.shape[0] in (1,3) else orig_img.squeeze(), cmap='gray' if orig_img.shape[0] == 1 else None)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(res_img.squeeze().numpy(), cmap='gray')
    axes[1].set_title(title)
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "preview.png"), dpi=150)  
    plt.show(block=True)

def save_image(tensor, filename, batch_idx=0):
    img = tensor[batch_idx].detach().cpu()
    img = img.squeeze(0) if img.dim()==3 and img.shape[0]==1 else img
    img = img.permute(1,2,0).numpy() if img.dim()==3 and img.shape[0]==3 else img.numpy()
    img = np.clip(img,0,255).astype(np.uint8)
    out_path = os.path.join(OUTPUT_DIR, filename)  
    plt.imsave(out_path, img, cmap='gray' if img.ndim==2 else None)

# ------------------ Benchmarking & Profiling ------------------

def benchmark_pipeline(batch, batch_size, height, width):
    results = {}
    # CPU
    torch.cuda.empty_cache(); gc.collect()
    start = time.time()
    result_cpu = cpu_pipeline(batch.clone())
    results['cpu_time'] = time.time()-start
    # GPU
    torch.cuda.empty_cache(); gc.collect()
    if torch.cuda.is_available():
        start = time.time()
        result_gpu = gpu_pipeline(batch.clone())
        torch.cuda.synchronize()
        results['gpu_time'] = time.time()-start
        results['match'] = torch.allclose(result_cpu, result_gpu.cpu(), atol=1e-4)
        return results, result_cpu, result_gpu
    return results, result_cpu, None

def scaling_experiment(batch_sizes=[1,8,16,24], height=1024, width=1024):
    cpu_times, gpu_times = [], []
    for bs in batch_sizes:
        img = load_image_or_synthetic(height, width, 3)
        batch = torch.stack([img.clone() for _ in range(bs)])
        res, _, _ = benchmark_pipeline(batch, bs, height, width)
        cpu_times.append(res['cpu_time'])
        gpu_time = res.get('gpu_time', None)
        gpu_times.append(gpu_time)
        print(f"Batch {bs}: CPU {res['cpu_time']:.4f}s, GPU {gpu_time if gpu_time is not None else 'N/A'}")

    # Save CSV
    df = pd.DataFrame({
        "Batch Size": batch_sizes,
        "CPU Time (s)": cpu_times,
        "GPU Time (s)": gpu_times
    })
    df.to_csv(os.path.join(OUTPUT_DIR, "scaling_results.csv"), index=False)  # CHANGED
    print(f"Scaling results saved to {os.path.join(OUTPUT_DIR, 'scaling_results.csv')}")

    # Plot results
    plt.figure(figsize=(8,5))
    plt.plot(batch_sizes, cpu_times, '-o', label="CPU")
    if any(g is not None for g in gpu_times):
        valid_batch_sizes = [bs for bs, gt in zip(batch_sizes, gpu_times) if gt is not None]
        valid_gpu_times = [gt for gt in gpu_times if gt is not None]
        plt.plot(valid_batch_sizes, valid_gpu_times, '-o', label="GPU")
    plt.xlabel("Batch Size")
    plt.ylabel("Time (seconds)")
    plt.title("CPU vs GPU Scaling Performance")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(OUTPUT_DIR, "scaling_results.png"), dpi=200)  # CHANGED
    plt.show()
    print(f"Scaling plot saved to {os.path.join(OUTPUT_DIR, 'scaling_results.png')}")

# ------------------ Main ------------------

def main():
    # ---- Configure your dataset & sizes here ----
    batch_size, height, width = 8, 1024, 1024  # CHANGED: smaller default batch to fit typical GPUs/CPUs
    print("Saving outputs to:", OUTPUT_DIR)
    print("Reading images from:", INPUT_DIR)

    files = list_image_files(INPUT_DIR)  
    if not files:
        print("No images found; running single synthetic sample (use the 'images' folder to process real files).")
        img = load_image_or_synthetic(height, width, 3)
        batch = torch.stack([img.clone() for _ in range(batch_size)])
        # Benchmark & visualize once
        res, cpu_out, gpu_out = benchmark_pipeline(batch, batch_size, height, width)
        print(f"CPU Time: {res['cpu_time']:.4f}s")
        if 'gpu_time' in res:
            print(f"GPU Time: {res['gpu_time']:.4f}s  |  Results match: {res['match']}")
        save_image(cpu_out, "cpu_edges.png")
        visualize_results(batch, cpu_out, "CPU Edges")
        if gpu_out is not None:
            save_image(gpu_out, "gpu_edges.png")
            visualize_results(batch, gpu_out, "GPU Edges")
    else:
        print(f"Found {len(files)} image(s). Processing in batches of {batch_size}...")
        # Process real images in batches
        for start in range(0, len(files), batch_size):
            chunk = files[start:start+batch_size]
            imgs = [load_image_or_synthetic(height, width, 3, f) for f in chunk]
            batch = torch.stack(imgs)  # [B,3,H,W]
            res, cpu_out, gpu_out = benchmark_pipeline(batch, len(chunk), height, width)
            print(f"[{start+1}-{start+len(chunk)}] CPU {res['cpu_time']:.4f}s"
                  + (f", GPU {res['gpu_time']:.4f}s (match={res['match']})" if 'gpu_time' in res else ""))

            # Save per-image outputs with source-based names
            for i, f in enumerate(chunk):
                base = os.path.splitext(os.path.basename(f))[0]
                save_image(cpu_out, f"cpu_edges_{base}.png", batch_idx=i)
                if gpu_out is not None:
                    save_image(gpu_out, f"gpu_edges_{base}.png", batch_idx=i)

    # Optional: keep your scaling test
    scaling_experiment([1,4,8,16,24,32], height, width)

if __name__=="__main__":
    main()
