"""
STFT spectrogram extraction for heart sound signals.
Extracts magnitude spectrograms from both noisy and clean audio,
segments them into fixed-size blocks for model training,
and saves them along with metadata and phase information.

Input:
    - Noisy audio: synthesized noisy heart sound samples
    - Clean audio: corresponding clean heart sound samples
    - Recipe CSV: mapping between noisy and clean samples

Output:
    - blocks/noisy/: segmented noisy spectrogram blocks (.npy)
    - blocks/clean/: segmented clean spectrogram blocks (.npy)
    - meta/: phase spectra and metadata (.npy, .json)

Before running, please modify the paths below to match your local environment.
"""

import numpy as np
import librosa
import os
import json
import csv
import soundfile as sf
from pathlib import Path
import glob
from tqdm import tqdm

# ==================== User Configuration ====================
clean_root = r"F:\graduate\data\clean"
noisy_root = r"F:\HSM\noised_data"
output_root = r"F:\HSM\spectrograms"
recipe_csv = r"F:\HSM\noised_data\recipes\dataset_recipe.csv"
# ============================================================

# ==================== STFT Parameters ====================
N_FFT = 256
WIN_LENGTH = 128
HOP_LENGTH = 32
SEGMENT_FRAMES = 64
TARGET_SR = 2000

# ==================== Processing Modes ====================
MODES = ["train", "val", "test"]


def compute_stft(audio, sr=TARGET_SR):
    window = np.hanning(WIN_LENGTH)
    stft_matrix = librosa.stft(
        audio, n_fft=N_FFT, hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH, window=window, center=True
    )
    return np.abs(stft_matrix), np.angle(stft_matrix)


def compute_istft(mag, phase, length=None):
    stft_matrix = mag * np.exp(1j * phase)
    window = np.hanning(WIN_LENGTH)
    return librosa.istft(
        stft_matrix, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        window=window, n_fft=N_FFT, length=length, center=True
    )


def segment_spectrogram(mag, segment_frames=SEGMENT_FRAMES):
    freq_bins, orig_frames = mag.shape
    pad_frames = (segment_frames - (orig_frames % segment_frames)) % segment_frames

    if pad_frames > 0:
        mag_padded = np.pad(mag, ((0, 0), (0, pad_frames)), mode='constant')
    else:
        mag_padded = mag

    n_blocks = mag_padded.shape[1] // segment_frames
    segments = np.split(mag_padded, n_blocks, axis=1)
    return list(segments), orig_frames


def merge_spectrogram_segments(segments, orig_frames):
    mag_padded = np.concatenate(segments, axis=1)
    return mag_padded[:, :orig_frames]


def preprocess_for_model(mag_segment):
    return np.expand_dims(np.log1p(mag_segment), axis=-1)


def postprocess_from_model(output):
    return np.expm1(np.squeeze(output, axis=-1))


def load_recipe(csv_path):
    recipe = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            base_name = os.path.splitext(os.path.basename(row['output_file']))[0]
            recipe[base_name] = {
                'clean_source': row['clean_source'],
                'snr': float(row['snr'])
            }
    return recipe


def process_pair(noisy_path, clean_path, snr, base_name, blocks_dir, meta_dir):
    noisy_audio, _ = librosa.load(noisy_path, sr=TARGET_SR)
    clean_audio, _ = librosa.load(clean_path, sr=TARGET_SR)

    noisy_mag, noisy_phase = compute_stft(noisy_audio)
    clean_mag, _ = compute_stft(clean_audio)

    noisy_segments, orig_frames = segment_spectrogram(noisy_mag)
    clean_segments, _ = segment_spectrogram(clean_mag)

    noisy_processed = [preprocess_for_model(seg) for seg in noisy_segments]
    clean_processed = [preprocess_for_model(seg) for seg in clean_segments]

    info = {
        "audio_length": len(noisy_audio),
        "orig_frames": orig_frames,
        "n_blocks": len(noisy_segments),
        "snr": snr
    }

    noisy_blocks_dir = os.path.join(blocks_dir, "noisy")
    os.makedirs(noisy_blocks_dir, exist_ok=True)
    for i, seg in enumerate(noisy_processed):
        np.save(os.path.join(noisy_blocks_dir, f"{base_name}_block{i}.npy"), seg)

    clean_blocks_dir = os.path.join(blocks_dir, "clean")
    os.makedirs(clean_blocks_dir, exist_ok=True)
    for i, seg in enumerate(clean_processed):
        np.save(os.path.join(clean_blocks_dir, f"{base_name}_block{i}.npy"), seg)

    os.makedirs(meta_dir, exist_ok=True)
    np.save(os.path.join(meta_dir, f"{base_name}_phase.npy"), noisy_phase)
    with open(os.path.join(meta_dir, f"{base_name}_info.json"), 'w') as f:
        json.dump(info, f)

    return info


def collect_files(noisy_root, recipe):
    files = []
    for root, _, filenames in os.walk(noisy_root):
        for f in filenames:
            if f.endswith(('.wav', '.flac')):
                base_name = os.path.splitext(f)[0]
                if base_name in recipe:
                    files.append({
                        'noisy_path': os.path.join(root, f),
                        'clean_path': recipe[base_name]['clean_source'],
                        'snr': recipe[base_name]['snr'],
                        'base_name': base_name,
                        'rel_path': os.path.relpath(root, noisy_root)
                    })
    return files


def process_dataset(noisy_root, output_root, recipe):
    files = collect_files(noisy_root, recipe)
    if not files:
        return []

    blocks_root = os.path.join(output_root, "blocks")
    results = []

    for item in tqdm(files, desc=f"Processing {os.path.basename(noisy_root)}"):
        try:
            meta_dir = os.path.join(output_root, "meta", item['rel_path'])
            info = process_pair(
                item['noisy_path'], item['clean_path'], item['snr'],
                item['base_name'], blocks_root, meta_dir
            )
            results.append(info)
        except Exception as e:
            tqdm.write(f"Error on {item['base_name']}: {e}")

    return results


def reconstruct_audio(blocks_dir, meta_dir, base_name, output_path=None):
    info_path = os.path.join(meta_dir, f"{base_name}_info.json")
    with open(info_path, 'r') as f:
        info = json.load(f)

    phase = np.load(os.path.join(meta_dir, f"{base_name}_phase.npy"))

    block_files = sorted(glob.glob(os.path.join(blocks_dir, f"{base_name}_block*.npy")))
    segments = [postprocess_from_model(np.load(bf)) for bf in block_files]

    mag = merge_spectrogram_segments(segments, info["orig_frames"])
    audio = compute_istft(mag, phase, length=info["audio_length"])

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, audio, TARGET_SR)

    return audio


if __name__ == "__main__":
    print("=" * 50)
    print("STFT Spectrogram Extraction")
    print("=" * 50)
    print(f"Clean root: {clean_root}")
    print(f"Noisy root: {noisy_root}")
    print(f"Output root: {output_root}\n")

    recipe = load_recipe(recipe_csv)
    print(f"Recipe loaded: {len(recipe)} entries\n")

    total_files = 0
    total_blocks = 0

    for mode in MODES:
        noisy_folder = os.path.join(noisy_root, mode)
        if not os.path.exists(noisy_folder):
            continue

        output_folder = os.path.join(output_root, mode)
        results = process_dataset(noisy_folder, output_folder, recipe)

        if results:
            blocks = sum(r["n_blocks"] for r in results)
            print(f"  {mode}: {len(results)} files, {blocks} blocks\n")
            total_files += len(results)
            total_blocks += blocks

    print("=" * 50)
    print(f"Done! Total: {total_files} files, {total_blocks} blocks")
    print("=" * 50)