#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_trans_matrix.py

功能：
1. 从文件夹读取按波长排列的散斑图像。
2. 【修复】支持读取 Dark 文件夹，自动去重并计算平均底噪，在处理前减去背景。
3. 【新增】支持等间距抽样（Downsampling），例如从 80001 张中抽取 16001 张。
4. 可选中心裁剪 -> 可选降采样 -> 拉平 -> 拼接为 T 矩阵。

用法示例：
# 从 80001 张图中抽取 16001 张并处理：
python build_trans_matrix.py --img_dir ./data/images --dark_dir ./data/dark --save_dir ./output --target_count 16001
"""
import os
import re
import argparse
import glob
from PIL import Image
import numpy as np
from tqdm import tqdm
from typing import Optional, Tuple, List


# ---------------- helpers ----------------
def natural_sort_key(s: str):
    """从字符串中提取数字（包括小数），用于按数值排序；找不到数字则按字符串排序"""
    nums = re.findall(r'[-+]?\d*\.\d+|\d+', s)
    if nums:
        primary = float(nums[0])
        rest = [float(x) for x in nums[1:]] if len(nums) > 1 else []
        return (primary, rest, s)
    else:
        return (float('inf'), [], s)


def center_crop(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """PIL.Image img 中心裁剪到 target_w x target_h"""
    w, h = img.size
    if target_w > w or target_h > h:
        raise ValueError(f"目标裁剪尺寸 ({target_w},{target_h}) 大于图像尺寸 ({w},{h})")
    left = (w - target_w) // 2
    top = (h - target_h) // 2
    return img.crop((left, top, left + target_w, top + target_h))


def load_average_dark(dark_dir: str) -> Optional[np.ndarray]:
    """
    读取 dark_dir 下的所有图像，计算平均底噪。
    返回: float32 类型的 numpy 二维数组 (H, W)，若目录为空或不存在则返回 None
    """
    if not dark_dir or not os.path.isdir(dark_dir):
        return None

    # 支持常见图像格式
    exts = ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp')
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(dark_dir, ext)))
        # Windows下无需再次查找 upper，Linux下需要。
        # 为了兼容性保留两行，但在下面进行 set 去重。
        files.extend(glob.glob(os.path.join(dark_dir, ext.upper())))

    # 【修复】去重操作，解决 Windows 下大小写重复匹配导致找到 20 张图的问题
    files = list(set(files))
    files.sort()

    if not files:
        print(f"[WARN] 指定了 dark_dir={dark_dir} 但未找到图像，将跳过底噪扣除。")
        return None

    print(f"[INFO] 正在计算平均底噪，共找到 {len(files)} 张暗场图...")
    stack = []
    ref_shape = None

    for f in tqdm(files, desc="Loading Dark Frames"):
        try:
            img = Image.open(f).convert('L')
            arr = np.asarray(img, dtype=np.float32)
            if ref_shape is None:
                ref_shape = arr.shape
            elif arr.shape != ref_shape:
                print(f"[WARN] 暗场图 {os.path.basename(f)} 尺寸 {arr.shape} 与参考尺寸 {ref_shape} 不一致，跳过。")
                continue
            stack.append(arr)
        except Exception as e:
            print(f"[WARN] 读取暗场图 {f} 失败: {e}")

    if not stack:
        return None

    # 计算平均值
    avg_dark = np.mean(stack, axis=0)
    print(f"[INFO] 平均底噪计算完成。Shape: {avg_dark.shape}, Mean Level: {avg_dark.mean():.2f}")
    return avg_dark


def process_image_file_return_pils(path: str,
                                   crop_size: Optional[int],
                                   final_out_size: Optional[int],
                                   dark_array: Optional[np.ndarray] = None) -> Tuple[
    Image.Image, Optional[Image.Image], Image.Image, np.ndarray]:
    """
    读取图像 -> (减去底噪) -> 裁剪 -> 缩放 -> 返回结果
    """
    img = Image.open(path)
    if img.mode != 'L':
        img = img.convert('L')

    # --- 关键步骤：减去底噪 (Background Subtraction) ---
    if dark_array is not None:
        img_arr = np.asarray(img, dtype=np.float32)
        if img_arr.shape != dark_array.shape:
            raise RuntimeError(f"图像尺寸 {img_arr.shape} 与底噪尺寸 {dark_array.shape} 不匹配！")

        subtracted = img_arr - dark_array
        subtracted[subtracted < 0] = 0
        subtracted[subtracted > 255] = 255
        img = Image.fromarray(subtracted.astype(np.uint8), mode='L')
    # ------------------------------------------------

    w, h = img.size
    cropped = None
    work_img = img

    # 裁剪
    if crop_size is not None:
        if crop_size > w or crop_size > h:
            raise ValueError(f"Cannot center-crop to {crop_size}x{crop_size}: image {path} size {w}x{h}")
        cropped = center_crop(img, crop_size, crop_size)
        work_img = cropped

    # resize
    if final_out_size is not None:
        if work_img.size != (final_out_size, final_out_size):
            final = work_img.resize((final_out_size, final_out_size), resample=Image.LANCZOS)
        else:
            final = work_img
    else:
        final = work_img

    arr = np.asarray(final, dtype=np.float32)
    return img, cropped, final, arr


# ---------------- main function ----------------
def build_T_from_folder_and_save_examples(
        img_dir: str,
        save_dir: str,
        dark_dir: Optional[str] = None,
        crop_size: Optional[int] = None,
        out_size: Optional[int] = None,
        exts: tuple = ('png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp'),
        normalize: Optional[str] = 'max',
        transpose: bool = False,
        save_prefix: str = 'T',
        examples: int = 5,
        target_count: Optional[int] = None  # 【新增】目标图片数量
) -> Tuple[Optional[np.ndarray], List[str], List[Optional[float]]]:
    # 1. 准备底噪
    avg_dark = None
    if dark_dir:
        avg_dark = load_average_dark(dark_dir)

    # 2. collect files
    if not os.path.isdir(img_dir):
        raise RuntimeError(f"img_dir 不存在: {img_dir}")
    files = [os.path.join(img_dir, f) for f in os.listdir(img_dir)
             if os.path.isfile(os.path.join(img_dir, f)) and f.split('.')[-1].lower() in exts]

    if len(files) == 0:
        raise RuntimeError(f"在目录 {img_dir} 未找到图像文件")

    # 原始排序
    files_sorted = sorted(files, key=lambda p: natural_sort_key(os.path.basename(p)))
    total_found = len(files_sorted)

    # 3. 【新增】等间距抽样逻辑
    if target_count is not None and target_count < total_found:
        print(f"[SAMPLING] 原始图片 {total_found} 张，目标 {target_count} 张。正在执行等间距抽样...")
        # 生成等间距索引，保证首尾都被选中
        indices = np.linspace(0, total_found - 1, target_count, dtype=int)
        # 重新筛选文件列表
        files_sorted = [files_sorted[i] for i in indices]
        print(f"[SAMPLING] 抽样完成。当前处理图片数量: {len(files_sorted)}")
    else:
        print(f"[INFO] 处理所有 {total_found} 张图片 (未指定抽样或数量不足)。")

    num = len(files_sorted)

    # 解析波长
    wavelengths = []
    for p in files_sorted:
        fname = os.path.basename(p)
        nums = re.findall(r'[-+]?\d*\.\d+|\d+', fname)
        wavelengths.append(float(nums[0]) if nums else None)

    # 确定输出尺寸
    if crop_size is None and out_size is None:
        final_out_size = None
    elif crop_size is not None and out_size is None:
        final_out_size = int(crop_size)
    elif crop_size is None and out_size is not None:
        final_out_size = int(out_size)
    else:
        final_out_size = int(out_size)

    filenames = []
    arrays = []
    shapes = []

    examples_saved = 0
    os.makedirs(save_dir, exist_ok=True)
    examples_dir = os.path.join(save_dir, "examples")
    os.makedirs(examples_dir, exist_ok=True)

    # 4. process images
    pbar = tqdm(files_sorted, desc="Processing images")
    for i, p in enumerate(pbar):
        try:
            orig_pil, cropped_pil, final_pil, arr = process_image_file_return_pils(
                p, crop_size, final_out_size, dark_array=avg_dark
            )
        except Exception as e:
            raise RuntimeError(f"处理文件 {p} 时出错: {e}")

        filenames.append(os.path.basename(p))
        arrays.append(arr)
        shapes.append(arr.shape)

        # save examples
        if examples_saved < examples:
            safe_fname = filenames[-1].replace(os.sep, "_").replace(":", "_")
            orig_save = os.path.join(examples_dir, f"ex_{examples_saved + 1:02d}_orig_{safe_fname}.png")
            final_save = os.path.join(examples_dir, f"ex_{examples_saved + 1:02d}_final_{safe_fname}.png")
            try:
                orig_pil.convert('L').save(orig_save)
                final_pil.convert('L').save(final_save)
            except Exception as e:
                print(f"[WARN] 保存示例图失败: {e}")
            examples_saved += 1

    # 5. Build Matrix
    unique_shapes = list({s for s in shapes})
    uniform = (len(unique_shapes) == 1)

    meta = {
        'filenames': np.array(filenames, dtype=object),
        'wavelengths': np.array(wavelengths, dtype=np.float32),
        'crop_size': np.int32(crop_size) if crop_size is not None else np.int32(-1),
        'out_size': np.int32(final_out_size) if final_out_size is not None else np.int32(-1),
        'dark_subtracted': bool(avg_dark is not None),
        'sampling_total_original': total_found,
        'sampling_target': target_count if target_count else -1
    }

    out_npy = os.path.join(save_dir, f"{save_prefix}_ori.npy")
    out_npz = os.path.join(save_dir, f"{save_prefix}_meta.npz")

    if uniform:
        H, W = unique_shapes[0]
        pixels = H * W
        T = np.zeros((pixels, num), dtype=np.float32)
        for i, arr in enumerate(arrays):
            T[:, i] = arr.reshape(-1)

        # normalization
        if normalize == 'max':
            col_max = T.max(axis=0, keepdims=True)
            col_min = T.min(axis=0, keepdims=True)
            col_max[col_max == 0] = 1.0
            T = (T - col_min) / (col_max - col_min)
        elif normalize == 'l2':
            norms = np.linalg.norm(T, axis=0, keepdims=True)
            norms[norms == 0] = 1.0
            T = T / norms

        Tout = T.T if transpose else T
        print(f"[INFO] Matrix shape: {Tout.shape}")
        np.save(out_npy, Tout)

        np.savez_compressed(out_npz,
                            T_shape=Tout.shape,
                            filenames=meta['filenames'],
                            wavelengths=meta['wavelengths'],
                            crop_size=meta['crop_size'],
                            out_size=meta['out_size'],
                            dark_subtracted=meta['dark_subtracted'],
                            image_shape=np.array([H, W], dtype=np.int32))
        print(f"[DONE] T saved to {out_npy}")
        return Tout, filenames, wavelengths
    else:
        print("[INFO] 图像尺寸不一致，保存为 concat 模式...")
        flats = [arr.reshape(-1) for arr in arrays]
        lengths = [f.size for f in flats]
        offsets = np.cumsum([0] + lengths[:-1]).astype(np.int64)
        total_len = sum(lengths)
        concat = np.empty((total_len,), dtype=np.float32)

        for off, f in zip(offsets, flats):
            concat[off:off + f.size] = f

        # 简单处理非均匀归一化 (略去复杂逻辑以保持清晰，如需可加回)

        out_concat = os.path.join(save_dir, f"{save_prefix}_concat.npy")
        np.save(out_concat, concat)
        np.savez_compressed(out_npz,
                            concat_name=os.path.basename(out_concat),
                            filenames=meta['filenames'],
                            wavelengths=meta['wavelengths'],
                            offsets=offsets,
                            lengths=np.array(lengths))
        print(f"[DONE] Concat array saved to {out_concat}")
        return None, filenames, wavelengths


# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成传输矩阵，支持底噪扣除与等间距抽样")
    parser.add_argument("--img_dir", type=str, default="./experiment/12-9-4same-80um-1.4um/ccd/images",
                        help="包含按波长排序的散斑图像的文件夹")

    parser.add_argument("--dark_dir", type=str, default="./experiment/",
                        help="包含无光底噪图像的文件夹（可选）")

    parser.add_argument("--save_dir", type=str, default="./experiment/12-9-4same-80um-1.4um",
                        help="保存目录")

    # 【新增】抽样参数
    parser.add_argument("--target_count", type=int, default=3801,
                        help="从总文件中抽取的图像数量（例如从80001中抽16001），等间距抽取。")

    parser.add_argument("--crop_size", type=int, default=None, help="中心裁剪尺寸")
    parser.add_argument("--out_size", type=int, default=None, help="降采样/输出尺寸")
    parser.add_argument("--normalize", type=str, choices=[None, 'max', 'l2'], default=None)
    parser.add_argument("--transpose", action="store_true", help="输出转置")
    parser.add_argument("--save_prefix", type=str, default="T", help="文件前缀")

    args = parser.parse_args()

    build_T_from_folder_and_save_examples(
        img_dir=args.img_dir,
        save_dir=args.save_dir,
        dark_dir=args.dark_dir,
        crop_size=args.crop_size,
        out_size=args.out_size,
        normalize=args.normalize,
        transpose=args.transpose,
        save_prefix=args.save_prefix,
        target_count=args.target_count  # 传入新参数
    )