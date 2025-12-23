#!/usr/bin/env python3
"""
lightfield_reconstruction.py

对重排后的各视角光场图像进行三维重建的示例脚本。

方法概要：
- 读取保存为.mat的光场PSF（形状应为 (x, y, U, V, Z) ）
- 对每个视角 (u,v)：读取二维图像（500x500），在 z 方向上复制为与 PSF 的 Z 层数相同的体栈
- 将 PSF 的 (x,y) 维度填充到视图图像尺寸（中心对齐），获得 (X_img, Y_img, Z)
- 对每个视角执行若干次 3D Richardson-Lucy (RL) 迭代
- 计算该视角的重投影误差并累加到整体三维体积中，重复若干轮

依赖：numpy, scipy, imageio, tifffile(可选)

用法示例：
python3 lightfield_reconstruction.py \
  --psf mat/psf.mat \
  --views-dir views/ \
  --out results/volume.npy

注意：脚本做了较多可配置的简化假设，便于在你的数据上快速适配。
"""
import os
import cv2
import glob
import numpy as np
from scipy.io import loadmat
from scipy.signal import fftconvolve as fftconvolve_np
import torch
from imageio import imread
import tifffile


def find_mat_psf(matpath):
    """Load a 5D PSF from a .mat file.

    Supports classic MATLAB files via `scipy.io.loadmat` and v7.3 HDF5-based
    files via `h5py` (searched recursively for a 5D dataset).
    """
    # Try scipy loadmat first (works for non-v7.3 .mat)
    try:
        data = loadmat(matpath)
        # choose first 5D ndarray found
        for k, v in data.items():
            if isinstance(v, np.ndarray) and v.ndim == 5:
                return v
    except NotImplementedError:
        # scipy raises NotImplementedError for v7.3 MAT files
        pass
    except Exception:
        # other loadmat errors -> fallback to h5py
        pass

    # Fallback: try HDF5 reader for v7.3 MAT files
    try:
        import h5py
    except Exception as e:
        raise RuntimeError('scipy.loadmat failed and h5py is not available to read v7.3 MAT files') from e

    with h5py.File(matpath, 'r') as f:
        candidates = []

        def visitor(name, obj):
            try:
                if isinstance(obj, h5py.Dataset):
                    shape = obj.shape
                    if len(shape) == 5:
                        candidates.append((name, shape, obj))
            except Exception:
                pass

        f.visititems(visitor)

        if not candidates:
            raise ValueError('No 5D dataset found in MAT v7.3 file')

        # pick largest candidate (by total elements)
        best = max(candidates, key=lambda t: np.prod(t[1]))
        name, shape, dataset = best
        arr = dataset[()]
        arr = np.array(arr)
        return arr


def pad_psf_xy_to_shape(psf_xy_z, target_shape):
    """Center-pad (Hx, Wy, Z) -> (target_h, target_w, Z)
    psf_xy_z: ndarray (h, w, z)
    target_shape: (target_h, target_w)
    """
    # h, w, z = psf_xy_z.shape
    # th, tw = target_shape
    # if (h, w) == (th, tw):
    #     return psf_xy_z.copy()
    # padded = np.zeros((th, tw, z), dtype=psf_xy_z.dtype)
    # start_h = (th - h) // 2
    # start_w = (tw - w) // 2
    # padded[start_h:start_h + h, start_w:start_w + w, :] = psf_xy_z
    # return padded
    return psf_xy_z


def _fftconv_torch(a, b, device):
    # a, b: torch tensors (H,W,Z) float32
    # compute output with FFT-based convolution and return 'same' shape as a
    sa = a.shape
    sb = b.shape
    out_shape = [sa[i] + sb[i] - 1 for i in range(3)]
    # perform rfftn to save memory on last axis
    fsize = out_shape
    A = torch.fft.fftn(a, fsize, dim=(0,1,2))
    B = torch.fft.fftn(b, fsize, dim=(0,1,2))
    C = A * B
    conv = torch.fft.irfftn(C, fsize, dim=(0,1,2))
    # center-crop to a.shape (same)
    start = [(out_shape[i] - sa[i]) // 2 for i in range(3)]
    return conv[start[0]:start[0]+sa[0], start[1]:start[1]+sa[1], start[2]:start[2]+sa[2]]


def rl_deconvolution_3d(vol_t, obs_t, psf, one_t, omega = 0.05, clip_min=1e-12, use_gpu=False, device='cuda'):
    """Richardson-Lucy deconvolution with optional GPU acceleration via PyTorch FFT.

    observed: (H,W,Z) numpy
    psf: (H,W,Z) numpy
    returns: estimate numpy
    """
    if use_gpu:
        dev = torch.device(device)
        psf_t = torch.from_numpy(psf.astype(np.float32)).to(dev)
        psf_flip_t = psf_t.flip(dims=(0,1,2))

        conv = _fftconv_torch(vol_t, psf_t, dev)
        conv = torch.clamp(conv, min=clip_min)
        relative = obs_t / conv
        correction = _fftconv_torch(relative, psf_flip_t, dev)
        convone = _fftconv_torch(one_t, psf_flip_t, dev)
        convone = torch.clamp(convone, min=clip_min)
        lbd = vol_t / convone
        err = lbd * correction
        vol_t = (1 - omega) * vol_t + omega * err
        vol_t = torch.clamp(vol_t, min=clip_min)
        return vol_t
    else:
        return


def load_view_image(views_dir, u, v, x_start=0, x_end=200, y_start=0, y_end=200):
    """Load view image for view indices (u,v).

    Expects filenames in `views_dir` to be exactly: reset_{u}_{v}.tiff
    """
    fname = os.path.join(views_dir, f'reset_{u}_{v}.tiff')
    if not os.path.exists(fname):
        raise FileNotFoundError(f'Cannot find view image for u={u}, v={v}: {fname}')
    # prefer tifffile for full TIFF support
    img = tifffile.imread(fname).astype(np.float32)
    img = img[x_start:x_end, y_start:y_end]
    return img


def main():
    # -------- Configuration (modify values here) --------
    CONFIG = {
        # path to .mat file that contains 5D PSF (x, y, U, V, Z)
        'psf': '/mnt/nas00/DFN/RUSH3D/IdealLF_3Dpsf_M7.85_NA0.5_zmin-0.00029999_zmax0.00030001_zspacing1e-05.mat',
        # directory containing rearranged view images (expects UxV views inside)
        'views_dir': '/mnt/nas00/DFN/RUSH3D/neuron_100ms_g70_20X_C2/neuron_100ms_g70_20X_S1_C2_125/',
        # angular dims (default 15x15)
        'U': 15,
        'V': 15,
        'U_start': 4,
        'U_end': 12,
        'V_start': 4,
        'V_end': 12,
        'x_start': 413,
        'x_end': 613,
        'y_start': 923,
        'y_end': 1123,
        # number of full passes over all views
        'iter_views': 4,
        'lambda': 0.05,
        # enable GPU acceleration (requires PyTorch + CUDA)
        'use_gpu': True,
        'device': 'cuda:3',
        'upsample_rate': 5,
        # output path for reconstructed volume (npy or tiff if tifffile installed)
        'out': '/mnt/nas00/DFN/RUSH3D/results/hhvuxyz.tiff'
    }

    print('Loading PSF...')
    psf5 = find_mat_psf(CONFIG['psf'])
    # expected shape now: (Z, V, U, Y, X)
    if psf5.ndim != 5:
        raise ValueError('Loaded PSF is not 5D')
    Z, _, _, _, _ = psf5.shape
    print(f'PSF shape (Z,V,U,Y,X): {psf5.shape}')

    # load a sample view to determine image size
    sample_img = None
    for u in range(CONFIG['U_start'], CONFIG['U_end'] + 1):
        for v in range(CONFIG['V_start'], CONFIG['V_end'] + 1):
            try:
                sample_img = load_view_image(CONFIG['views_dir'], u, v, CONFIG['x_start'], CONFIG['x_end'], CONFIG['y_start'], CONFIG['y_end'])
                break
            except Exception:
                continue
        if sample_img is not None:
            break
    if sample_img is None:
        raise FileNotFoundError('No sample view image found in views dir')

    # Note: update view loading to use the in-code config
    H_img, W_img = sample_img.shape[:2]
    print(f'Detected view image size: {H_img}x{W_img}, Z layers from PSF: {Z}')
    
    # initialize global volume (float64)
    dev = torch.device(CONFIG['device'])
    center_view = load_view_image(CONFIG['views_dir'], (CONFIG['U']+1)//2, (CONFIG['V']+1)//2)
    center_view = cv2.resize(center_view, (CONFIG['upsample_rate']*H_img, CONFIG['upsample_rate']*W_img), interpolation=cv2.INTER_LINEAR)
    center_view = np.repeat(center_view[:, :, np.newaxis], Z, axis=2)
    vol_t = torch.from_numpy(np.maximum(center_view, 1e-6).astype(np.float32)).to(dev)
    one_t = torch.from_numpy(np.ones_like(center_view).astype(np.float32)).to(dev)

    # main loop
    for pass_i in range(CONFIG['iter_views']):
        print(f'Global pass {pass_i+1}/{CONFIG['iter_views']}')
        for u in range(CONFIG['U_start'], CONFIG['U_end'] + 1):
            for v in range(CONFIG['V_start'], CONFIG['V_end'] + 1):
                try:
                    view_img = load_view_image(CONFIG['views_dir'], u+1, v+1, CONFIG['x_start'], CONFIG['x_end'], CONFIG['y_start'], CONFIG['y_end'])
                except Exception as e:
                    print(f'skip view u={u} v={v}: {e}')
                    continue

                # convert to grayscale if needed
                if view_img.ndim == 3:
                    # simple average
                    view_img = view_img.mean(axis=2)

                # normalize to positive floats
                view_img = np.maximum(view_img.astype(np.float64), 0.0)

                # expand into z by stacking
                view_img = cv2.resize(view_img, (CONFIG['upsample_rate']*H_img, CONFIG['upsample_rate']*W_img), interpolation=cv2.INTER_LINEAR)
                observed3d = np.repeat(view_img[:, :, np.newaxis], Z, axis=2)
                obs_t = torch.from_numpy(np.maximum(observed3d, 1e-6).astype(np.float32)).to(dev)

                # extract PSF for this angular view
                # psf5 has shape (Z, V, U, Y, X)
                psf_zyx = psf5[:, v, u, :, :]


                # transpose to (X, Y, Z) for downstream functions
                psf_uv = np.transpose(psf_zyx, (2, 1, 0))

                # pad PSF xy to image size
                psf_padded = pad_psf_xy_to_shape(psf_uv, (CONFIG['upsample_rate']*H_img, CONFIG['upsample_rate']*W_img))

                # ensure PSF non-negative
                psf_padded = np.maximum(psf_padded, 0.0)

                # run RL deconvolution per-view
                try:
                    vol_t = rl_deconvolution_3d(vol_t, obs_t, psf_padded, one_t,  omega = CONFIG['lambda'], use_gpu=CONFIG['use_gpu'], device=CONFIG['device'])
                except Exception as e:
                    print(f'RL failed for u={u},v={v}: {e}')
                    continue
                print(f'Processed view u={u} v={v}')
        
    volume = vol_t.cpu().numpy()
    v_max = volume.max()
    v_min = volume.min()
    volume = (volume - v_min) / (v_max - v_min)
    volume[volume > 1.0] = 1.0
    volume[volume < 0.0] = 0.0
    volume[np.isnan(volume)] = 0.0

    # save output
    out = CONFIG['out']
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    # Convert volume (H, W, Z) -> (Z, H, W)
    vol_zhw = np.transpose(volume, (2, 0, 1))

    def center_crop_or_pad(slice_hw, target_h, target_w):
        h, w = slice_hw.shape
         # crop if larger
        start_h = max(0, (h - target_h) // 2)
        start_w = max(0, (w - target_w) // 2)
        cropped = slice_hw[start_h:start_h + target_h, start_w:start_w + target_w]
        # pad if smaller
        ch, cw = cropped.shape
        pad_h1 = max(0, (target_h - ch) // 2)
        pad_h2 = target_h - ch - pad_h1
        pad_w1 = max(0, (target_w - cw) // 2)
        pad_w2 = target_w - cw - pad_w1
        if pad_h1 or pad_h2 or pad_w1 or pad_w2:
            cropped = np.pad(cropped, ((pad_h1, pad_h2), (pad_w1, pad_w2)), mode='constant')
        return cropped

    # apply to all z slices
    Z_out = vol_zhw.shape[0]
    vol_zhw_resized = np.zeros((Z_out, CONFIG['upsample_rate']*H_img, CONFIG['upsample_rate']*W_img), dtype=vol_zhw.dtype)
    for zi in range(Z_out):
        vol_zhw_resized[zi] = center_crop_or_pad(vol_zhw[zi], CONFIG['upsample_rate']*H_img, CONFIG['upsample_rate']*W_img)

    # save
    if out.lower().endswith('.npy') or out.lower().endswith('.npz'):
        np.save(out, vol_zhw_resized)
        print('Saved volume to', out)
    elif (out.lower().endswith('.tif') or out.lower().endswith('.tiff')):
        tifffile.imwrite(out, vol_zhw_resized.astype(np.float32))
        print('Saved volume (tiff) to', out)
    else:
        out2 = out + '.npy'
        np.save(out2, vol_zhw_resized)
        print('Saved volume to', out2)


if __name__ == '__main__':
    main()
