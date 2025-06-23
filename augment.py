import torchaudio
import torchaudio.transforms as T
import math
import matplotlib.pyplot as plt
import numpy as np
import librosa
import random, glob
import pandas as pd
import argparse

'''
PLAN:
1. Calculate the number of audio we have, then randomly choose 20%
2. Randomly choose noise / music? Definitely not speech
3. Add the noisy speech into the folder
4. Automatically add a new line into the train.tsv file with the new noisy speech audio
5. Only augment original files, not previously augmented ones
'''

def is_augmented_file(filename):
    """Check if a file is already an augmented version"""
    return '_augmented_' in filename and 'db.wav' in filename

def get_original_files_only(file_list):
    """Filter out augmented files from the file list"""
    return [f for f in file_list if not is_augmented_file(f)]

def add_noise(wave_file, noise_file, snr_dbs, df):
    waveform, sr = torchaudio.load(wave_file)
    noise, noise_sr = torchaudio.load(noise_file)
    cleaned_wave_file = wave_file[:-4]

    # Resample NOISE to match waveform sample rate, not the opposite
    resampler = T.Resample(noise_sr, sr)  # From noise_sr TO sr
    noise = resampler(noise)  # Resample the noise, not the waveform

    waveform_length = waveform.shape[1]
    noise_length = noise.shape[1]
    num_repeats = math.ceil(waveform_length / noise_length)

    noise_repeated = noise.repeat(1, num_repeats)
    noise_repeated = noise_repeated[:, :waveform_length]

    waveform_power = waveform.norm(p=2)
    noise_power = noise_repeated.norm(p=2)

    for snr_db in snr_dbs:
        snr = math.exp(snr_db / 10)
        scale = snr * noise_power / waveform_power
        noisy_waveform = (scale * waveform + noise_repeated) / 2
        path = cleaned_wave_file + '_augmented_' + str(snr_db) + 'db.wav'

        # Find the original row based on the path column containing the filename
        original_filename = wave_file.split("/")[-1]
        matching_rows = df[df['filename'] == original_filename]
        
        if not matching_rows.empty:
            torchaudio.save(path, noisy_waveform, sr)
            print(f'saved at {path}')
            
            # Get the original row data
            original_row = matching_rows.iloc[0].copy()
            
            # Update the path column with the new augmented filename
            original_row['filename'] = path.split("/")[-1]
            
            # Add the new row to the dataframe
            new_row_df = pd.DataFrame([original_row])
            df = pd.concat([df, new_row_df], ignore_index=True)
            
            print(f'Added row for {path.split("/")[-1]} with data from original file')
        else:
            print(f'Warning: Could not find original row for {original_filename}')
    
    return df

def augment(split, percentage=0.2, snr_dbs=[5, 15, 25], cv_path="./common_voice/cv-corpus-21.0-2025-03-14/id", noise_path="../whisper/musan/noise/**/*.wav"):
    df = pd.read_json(f'{cv_path}/{split}_metadata.json')

    wave_pattern = f"{cv_path}/{split}_augment/*.wav"
    
    # Get all files and filter out augmented ones
    all_files = glob.glob(wave_pattern, recursive=True)
    original_files_only = get_original_files_only(all_files)
    
    file_count = len(original_files_only)
    file_to_augment = math.ceil(file_count * percentage)
    augmented_wave = set()
    
    print(f'Total files found: {len(all_files)}')
    print(f'Original (non-augmented) files: {file_count}')
    print(f'Files to augment: {file_to_augment}')
    
    # Check if we have enough original files to augment
    if file_count == 0:
        print("No original files found to augment!")
        return
    
    if file_to_augment > file_count:
        print(f"Warning: Requested to augment {file_to_augment} files, but only {file_count} original files available.")
        file_to_augment = file_count

    for i in range(file_to_augment):
        # Pick a random original file
        available_files = [f for f in original_files_only if f not in augmented_wave]
        
        if not available_files:
            print(f"No more original files available to augment. Stopping after {i} augmentations.")
            break
            
        wave_file = random.choice(available_files)
        augmented_wave.add(wave_file)
        
        # Pick a random noise
        noise_file = random.choice(glob.glob(noise_path, recursive=True))
        print(f"augmenting {wave_file} with noise {noise_file}")

        # Augment
        df = add_noise(wave_file, noise_file, snr_dbs, df)
    
    # Save the dataframe as TSV
    output_file = f'{cv_path}/{split}_augment.json'
    df.to_json(output_file, orient='records', indent=2, index=False)
    print(f"Saved updated TSV with {len(df)} total rows to {output_file}")

def parse_snr_list(snr_string):
    """Parse comma-separated SNR values"""
    try:
        return [int(x.strip()) for x in snr_string.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError("SNR values must be comma-separated integers (e.g., '5,15,25')")

def main():
    parser = argparse.ArgumentParser(
        description='Audio augmentation script for adding noise to speech data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
        Examples:
        python script.py --split dev --percentage 0.3 --snr 10,20,30
        python script.py --split train --percentage 0.1 --snr 15
        python script.py --cv-path /path/to/common_voice --noise-path /path/to/noise/*.wav
        '''
    )
    
    parser.add_argument(
        '--split', 
        type=str, 
        default='dev',
        help='Dataset split to augment (default: dev)'
    )
    
    parser.add_argument(
        '--percentage', 
        type=float, 
        default=0.2,
        help='Percentage of files to augment (default: 0.2)'
    )
    
    parser.add_argument(
        '--snr', 
        type=parse_snr_list, 
        default=[10],
        help='SNR values in dB as comma-separated list (default: 10). Example: 5,15,25'
    )
    
    parser.add_argument(
        '--cv-path',
        type=str,
        default='./data/cv-corpus-21.0-2025-03-14/id',
        help='Path to Common Voice dataset directory'
    )
    
    parser.add_argument(
        '--noise-path',
        type=str,
        default='../whisper/musan/noise/**/*.wav',
        help='Path pattern for noise files (default: ../whisper/musan/noise/**/*.wav)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (optional)'
    )
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Random seed set to: {args.seed}")
    
    # Validate percentage
    if not 0 < args.percentage <= 1:
        parser.error("Percentage must be between 0 and 1")
    
    # Validate SNR values
    if not all(isinstance(snr, int) for snr in args.snr):
        parser.error("All SNR values must be integers")
    
    print(f"Configuration:")
    print(f"  Split: {args.split}")
    print(f"  Percentage: {args.percentage}")
    print(f"  SNR values: {args.snr}")
    print(f"  CV path: {args.cv_path}")
    print(f"  Noise path: {args.noise_path}")
    print()
    
    augment(
        split=args.split,
        percentage=args.percentage, 
        snr_dbs=args.snr,
        cv_path=args.cv_path,
        noise_path=args.noise_path
    )
        
if __name__ == "__main__":
    main()