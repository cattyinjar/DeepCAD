#!/usr/bin/env python3
"""
lightfield_reconstruction_multi_gpu.py

Multi-GPU version of lightfield reconstruction. Place configuration in the
CONFIG dict below. This script distributes (u,v) views across multiple
physical GPUs and accumulates residual updates into the global volume.

Assumptions:
- PSF stored in a .mat file with shape (Z, U, V, X, Y)
- View filenames: reset_{u}_{v}.tiff under `views_dir`

Adjust `CONFIG` for your paths and devices.
"""
import os
import glob
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from scipy.io import loadmat
from scipy.signal import fftconvolve as fftconvolve_np
import threading

try:
    import h5py
except Exception:
    h5py = None

try:
    import torch
    have_torch = True
except Exception:
    have_torch = False

try:
    import tifffile
    have_tifffile = True
except Exception:
    from imageio import imread as _imread
    have_tifffile = False


CONFIG = {
    'psf': '/mnt/nas00/DFN/RUSH3D/IdealLF_3Dpsf_M7.85_NA0.5_zmin-0.00029999_zmax0.00030001_zspacing1e-05.mat',
    'views_dir': '/mnt/nas00/DFN/RUSH3D/neuron_20X_C2/capture/shiftscan/neuron_20X_S1_C2_100',
    'U': 15,
    'V': 15,
    'iter_views': 1,
    'iter_rl': 8,
    # list of physical GPU ids to use, e.g. [0,1]
    'gpu_devices': [0, 1],
    # if True use multiple GPUs in parallel; if False, single-device/CPU path
    'use_multi_gpu': True,
    'out': '/mnt/nas00/DFN/RUSH3D/results/h1_multi.tiff'
}


def find_mat_psf(matpath):
    try:
        data = loadmat(matpath)
        for k, v in data.items():
            if isinstance(v, np.ndarray) and v.ndim == 5:
                return v
    except NotImplementedError:
        pass
    except Exception:
        pass

    if h5py is None:
        raise RuntimeError('Cannot read v7.3 MAT file: h5py not available')

    with h5py.File(matpath, 'r') as f:
        candidates = []

        def visitor(name, obj):
            try:
                if isinstance(obj, h5py.Dataset) and len(obj.shape) == 5:
                    candidates.append((name, obj.shape, obj))
            except Exception:
                pass

        f.visititems(visitor)
        if not candidates:
            raise ValueError('No 5D dataset found in MAT v7.3 file')
        best = max(candidates, key=lambda t: np.prod(t[1]))
        _, _, dataset = best
        arr = np.array(dataset[()])
        return arr


def pad_psf_xy_to_shape(psf_xy_z, target_shape):
    h, w, z = psf_xy_z.shape
    th, tw = target_shape
    if (h, w) == (th, tw):
        return psf_xy_z.copy()
    padded = np.zeros((th, tw, z), dtype=psf_xy_z.dtype)
    start_h = (th - h) // 2
    start_w = (tw - w) // 2
    padded[start_h:start_h + h, start_w:start_w + w, :] = psf_xy_z
    return padded


def _fftconv_torch(a, b, device):
    sa = a.shape
    sb = b.shape
    out_shape = [sa[i] + sb[i] - 1 for i in range(3)]
    A = torch.fft.rfftn(a, out_shape, dim=(0, 1, 2))
    B = torch.fft.rfftn(b, out_shape, dim=(0, 1, 2))
    C = A * B
    conv = torch.fft.irfftn(C, out_shape, dim=(0, 1, 2))
    start = [(out_shape[i] - sa[i]) // 2 for i in range(3)]
    return conv[start[0]:start[0] + sa[0], start[1]:start[1] + sa[1], start[2]:start[2] + sa[2]]


def _fftconv_choose(a_np, b_np, use_gpu=False, device_index=0):
    if use_gpu and have_torch:
        dev = torch.device(f'cuda:{device_index}')
        a = torch.from_numpy(a_np.astype(np.float32)).to(dev)
        b = torch.from_numpy(b_np.astype(np.float32)).to(dev)
        c = _fftconv_torch(a, b, dev)
        return c.cpu().numpy()
    else:
        return fftconvolve_np(a_np, b_np, mode='same')


def rl_deconvolution_3d(observed, psf, iterations=10, clip_min=1e-8, use_gpu=False, device_index=0):
    psf_sum = psf.sum()
    if psf_sum <= 0:
        raise ValueError('PSF sum must be positive')
    psf = psf / psf_sum

    if use_gpu and have_torch:
        dev = torch.device(f'cuda:{device_index}')
        obs_t = torch.from_numpy(np.maximum(observed, 1e-6).astype(np.float32)).to(dev)
        est = obs_t.clone()
        psf_t = torch.from_numpy(psf.astype(np.float32)).to(dev)
        psf_flip_t = psf_t.flip(dims=(0, 1, 2))

        for i in range(iterations):
            conv = _fftconv_torch(est, psf_t, dev)
            conv = torch.clamp(conv, min=clip_min)
            relative = obs_t / conv
            correction = _fftconv_torch(relative, psf_flip_t, dev)
            est = est * correction
            est = torch.clamp(est, min=1e-12)
        return est.cpu().numpy()
    else:
        estimate = np.maximum(observed, 1e-6).astype(np.float64)
        psf_flip = psf[::-1, ::-1, ::-1]
        for i in range(iterations):
            conv = fftconvolve_np(estimate, psf, mode='same')
            relative_blur = observed / np.maximum(conv, clip_min)
            correction = fftconvolve_np(relative_blur, psf_flip, mode='same')
            estimate *= correction
            estimate = np.maximum(estimate, 1e-12)
        return estimate


def load_view_image(views_dir, u, v):
    fname = os.path.join(views_dir, f'reset_{u}_{v}.tiff')
    if not os.path.exists(fname):
        raise FileNotFoundError(fname)
    if have_tifffile:
        return tifffile.imread(fname).astype(np.float32)
    else:
        return _imread(fname).astype(np.float32)


def process_views_on_device(device_index, view_pairs, CONFIG, psf5, H_img, W_img, Z):
    """Process a list of (u,v) pairs on a specific GPU and return residual sum."""
    residual_acc = np.zeros((H_img, W_img, Z), dtype=np.float64)
    for (u, v) in view_pairs:
        try:
            view_img = load_view_image(CONFIG['views_dir'], u, v)
        except Exception:
            continue
        if view_img.ndim == 3:
            view_img = view_img.mean(axis=2)
        view_img = np.maximum(view_img.astype(np.float64), 0.0)
        observed3d = np.repeat(view_img[:, :, np.newaxis], Z, axis=2)

        if u < psf5.shape[1] and v < psf5.shape[2]:
            psf_zxy = psf5[:, u, v, :, :]
        else:
            cu = min(u, psf5.shape[1] - 1)
            cv = min(v, psf5.shape[2] - 1)
            psf_zxy = psf5[:, cu, cv, :, :]

        psf_uv = np.transpose(psf_zxy, (1, 2, 0))
        psf_padded = pad_psf_xy_to_shape(psf_uv, (H_img, W_img))
        psf_padded = np.maximum(psf_padded, 0.0)

        try:
            estimate = rl_deconvolution_3d(observed3d, psf_padded, iterations=CONFIG['iter_rl'], use_gpu=True, device_index=device_index)
        except Exception:
            continue

        reproj = _fftconv_choose(estimate, psf_padded, use_gpu=True, device_index=device_index)
        residual = observed3d - reproj
        residual_acc += residual

    return residual_acc


def main():
    print('Loading PSF...')
    psf5 = find_mat_psf(CONFIG['psf'])
    if psf5.ndim != 5:
        raise ValueError('Loaded PSF is not 5D')
    # PSF shape: (Z, U, V, X, Y)
    Z, U_psf, V_psf, x_psf, y_psf = psf5.shape
    print('PSF shape (Z,U,V,X,Y):', psf5.shape)

    U = min(U_psf, CONFIG['U'])
    V = min(V_psf, CONFIG['V'])

    # detect sample view
    sample_img = None
    for u in range(U):
        for v in range(V):
            try:
                sample_img = load_view_image(CONFIG['views_dir'], u+1, v+1)
                break
            except Exception:
                continue
        if sample_img is not None:
            break
    if sample_img is None:
        raise FileNotFoundError('No sample view image found')
    H_img, W_img = sample_img.shape[:2]
    print(f'Detected view image size: {H_img}x{W_img}, Z layers from PSF: {Z}')

    volume = np.zeros((H_img, W_img, Z), dtype=np.float64)

    # build list of view pairs
    all_views = [(u, v) for u in range(U) for v in range(V)]

    if CONFIG['use_multi_gpu'] and have_torch and torch.cuda.is_available():
        # split views across devices
        devices = CONFIG['gpu_devices']
        ndev = len(devices)
        chunks = [all_views[i::ndev] for i in range(ndev)]

        for pass_i in range(CONFIG['iter_views']):
            print(f'Global pass {pass_i+1}/{CONFIG['iter_views']}')
            with ThreadPoolExecutor(max_workers=ndev) as ex:
                futures = []
                for di, dev in enumerate(devices):
                    fut = ex.submit(process_views_on_device, dev, chunks[di], CONFIG, psf5, H_img, W_img, Z)
                    futures.append(fut)
                for fut in futures:
                    res = fut.result()
                    volume += res
            volume = np.maximum(volume, 0.0)
    else:
        # single-device path (GPU if available and device specified, else CPU)
        use_gpu = have_torch and torch.cuda.is_available() and len(CONFIG.get('gpu_devices', [])) > 0
        device_index = CONFIG.get('gpu_devices', [0])[0] if use_gpu else 0
        for pass_i in range(CONFIG['iter_views']):
            print(f'Global pass {pass_i+1}/{CONFIG['iter_views']}')
            res = process_views_on_device(device_index if use_gpu else 0, all_views, CONFIG, psf5, H_img, W_img, Z) if use_gpu else process_views_on_device(0, all_views, CONFIG, psf5, H_img, W_img, Z)
            volume += res
            volume = np.maximum(volume, 0.0)

    # save
    out = CONFIG['out']
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    if out.lower().endswith('.tif') or out.lower().endswith('.tiff'):
        if have_tifffile:
            tifffile.imwrite(out, volume.astype(np.float32))
        else:
            np.save(out + '.npy', volume)
    else:
        np.save(out, volume)
    print('Saved volume to', out)


if __name__ == '__main__':
    if CONFIG['use_multi_gpu'] and not have_torch:
        raise RuntimeError('PyTorch is required for multi-GPU execution')
    main()
