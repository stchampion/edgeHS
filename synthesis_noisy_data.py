"""
Synthesize noisy heart sound samples by mixing clean heart sounds with ambient noise.

Before running, please modify the following paths to match your local environment:
    - clean_root: root directory of clean heart sound data
    - output_root: root directory for output noisy data
    - noise_folder: root directory of noise data
"""

import os
import csv
import random
import numpy as np
import librosa
import soundfile as sf

# ==================== User Configuration ====================
clean_root = 
output_root = 
noise_folder = 
# ============================================================

TARGET_SR = 2000
FIXED_SEED = 42
CATEGORIES = ["N", "AS", "MR", "MS", "MVP", "AR"]
TEST_SNR_LIST = [-5, 0, 5]
NOISE_PER_CLEAN = 10
FADE_LENGTH = int(0.01 * TARGET_SR)

random.seed(FIXED_SEED)
np.random.seed(FIXED_SEED)


def load_and_resample(file_path, target_sr):
    audio, orig_sr = librosa.load(file_path, sr=None)
    if orig_sr != target_sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    return audio


def get_noise_segment(noise_path, target_length, target_sr):
    noise = load_and_resample(noise_path, target_sr)
    if len(noise) >= target_length:
        start = random.randint(0, len(noise) - target_length)
        return noise[start:start + target_length]
    repeats = int(np.ceil(target_length / len(noise)))
    segments = [noise.copy() for _ in range(repeats)]
    result = np.concatenate(segments)
    return result[:target_length]


def mix_with_snr(clean_audio, noise_audio, snr_db):
    clean_rms = np.sqrt(np.mean(clean_audio ** 2))
    noise_rms = np.sqrt(np.mean(noise_audio ** 2))
    if noise_rms < 1e-8:
        return clean_audio.copy()
    target_noise_rms = clean_rms / (10 ** (snr_db / 20))
    scale = target_noise_rms / noise_rms
    return clean_audio + noise_audio * scale


def safe_normalize(audio):
    max_val = np.max(np.abs(audio))
    if max_val > 1.0:
        return audio / max_val
    return audio


def collect_files(root_folder):
    files_by_category = {}
    for cat in CATEGORIES:
        cat_folder = os.path.join(root_folder, cat)
        if os.path.exists(cat_folder):
            files = [os.path.join(cat_folder, f) for f in os.listdir(cat_folder)
                     if f.endswith(('.wav', '.flac', '.mp3'))]
            files_by_category[cat] = sorted(files)
    return files_by_category


def collect_noise_files(noise_folder):
    noise_files = []
    for root, _, files in os.walk(noise_folder):
        for f in files:
            if f.endswith(('.wav', '.flac', '.mp3')):
                noise_files.append(os.path.join(root, f))
    return noise_files


def generate_dataset(input_folder, output_folder, noise_pool, mode):
    files_by_category = collect_files(input_folder)
    recipe = []
    file_counter = {}

    for category in CATEGORIES:
        clean_files = files_by_category.get(category, [])
        if not clean_files:
            continue

        out_cat_folder = os.path.join(output_folder, category)
        os.makedirs(out_cat_folder, exist_ok=True)

        for clean_path in clean_files:
            base_name = os.path.splitext(os.path.basename(clean_path))[0]
            if base_name not in file_counter:
                file_counter[base_name] = 0

            clean_audio = load_and_resample(clean_path, TARGET_SR)
            clean_length = len(clean_audio)

            for _ in range(NOISE_PER_CLEAN):
                noise_file = random.choice(noise_pool)
                noise_segment = get_noise_segment(noise_file, clean_length, TARGET_SR)

                if mode == 'train':
                    snr_list = [random.uniform(-5, 5) for _ in range(5)]
                else:
                    snr_list = TEST_SNR_LIST

                for snr in snr_list:
                    noisy_audio = mix_with_snr(clean_audio, noise_segment, snr)
                    noisy_audio = safe_normalize(noisy_audio)

                    file_counter[base_name] += 1
                    output_filename = f"{base_name}_{file_counter[base_name]}.wav"
                    output_path = os.path.join(out_cat_folder, output_filename)
                    sf.write(output_path, noisy_audio, TARGET_SR)

                    recipe.append({
                        "output_file": os.path.join(category, output_filename),
                        "clean_source": clean_path,
                        "noise_source": noise_file,
                        "snr": round(snr, 4),
                        "mode": mode
                    })

    return recipe


def main():
    print("=" * 50)
    print("Noisy Heart Sound Dataset Generation")
    print("=" * 50)

    noise_pool = collect_noise_files(noise_folder)
    print(f"Noise files found: {len(noise_pool)}")

    if len(noise_pool) == 0:
        print("Error: No noise files found.")
        return

    all_recipes = []

    for mode in ['train', 'val', 'test']:
        input_folder = os.path.join(clean_root, mode)
        output_folder = os.path.join(output_root, mode)

        if not os.path.exists(input_folder):
            print(f"Warning: {input_folder} does not exist, skipping.")
            continue

        print(f"Processing {mode} set...")
        recipe = generate_dataset(input_folder, output_folder, noise_pool, mode)
        all_recipes.extend(recipe)
        print(f"  Generated {len(recipe)} samples.")

    recipe_folder = os.path.join(output_root, "recipes")
    os.makedirs(recipe_folder, exist_ok=True)

    csv_path = os.path.join(recipe_folder, "dataset_recipe.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ["output_file", "clean_source", "noise_source", "snr", "mode"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_recipes)

    print(f"Recipe saved to: {csv_path}")
    print("=" * 50)
    print("Generation Summary:")
    mode_counts = {}
    for r in all_recipes:
        mode_counts[r['mode']] = mode_counts.get(r['mode'], 0) + 1
    for mode, count in mode_counts.items():
        print(f"  {mode}: {count} files")
    print(f"  Total: {len(all_recipes)} files")
    print("=" * 50)


if __name__ == "__main__":
    main()
