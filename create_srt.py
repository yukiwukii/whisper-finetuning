import pandas as pd
import json
from mutagen import File
import os
import argparse
from datetime import timedelta

def transcribe_from_tsv(tsv_path, audio_dir, output_dir):
    # Read the TSV file
    df = pd.read_csv(tsv_path, sep='\t')
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    created_files = []
    
    for index, row in df.iterrows():
        filename = row['path']
        text = row['sentence']
        
        # Get audio duration and create SRT
        srt_path = create_srt_file(filename, text, audio_dir, output_dir)
        if srt_path:
            created_files.append(srt_path)
    
    return created_files

def transcribe_from_json(json_path, audio_dir, output_dir):
    # Read the JSON file
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    created_files = []
    
    for item in data:
        filename = item['filename']
        text = item['text']
        
        # Get audio duration and create SRT
        srt_path = create_srt_file(filename, text, audio_dir, output_dir)
        if srt_path:
            created_files.append(srt_path)
    
    return created_files

def create_srt_file(filename, text, audio_dir, output_dir):
    # Get audio duration
    audio_path = os.path.join(audio_dir, filename)
    try:
        audio = File(audio_path)
        duration = audio.info.length
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None
    
    # Format timestamps for SRT
    startTime = "00:00:00,000"
    endTime = str(timedelta(seconds=int(duration))).zfill(8) + ",000"
    if len(endTime.split(':')) == 2:  # Handle cases where duration < 1 hour
        endTime = "00:" + endTime
    
    # Create SRT content (single segment)
    segmentId = 1
    srt_content = f"{segmentId}\n{startTime} --> {endTime}\n{text}\n\n"
    
    # Create SRT filename (replace .mp3 with .srt)
    srt_filename = os.path.splitext(filename)[0] + '.srt'
    srt_path = os.path.join(output_dir, srt_filename)
    
    # Write SRT file
    with open(srt_path, 'w', encoding='utf-8') as srt_file:
        srt_file.write(srt_content)
    
    print(f"Created: {srt_filename}")
    return srt_path

def main():
    parser = argparse.ArgumentParser(description='Convert TSV or JSON file to SRT subtitle files')
    
    # Create mutually exclusive group for input file type
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--tsv-file', '-t', 
                            help='Path to the TSV file containing transcriptions')
    input_group.add_argument('--json-file', '-j', 
                            help='Path to the JSON file containing transcriptions')
    
    parser.add_argument('--input-audio', '-i', 
                        required=True,
                        help='Directory containing the audio files')
    
    parser.add_argument('--output-directory', '-o', 
                        required=True,
                        help='Directory to save the SRT files')
    
    args = parser.parse_args()
    
    # Call appropriate function based on input type
    if args.tsv_file:
        created_files = transcribe_from_tsv(args.tsv_file, args.input_audio, args.output_directory)
    elif args.json_file:
        created_files = transcribe_from_json(args.json_file, args.input_audio, args.output_directory)
    
    print(f"\nProcessing complete. Created {len(created_files)} SRT files.")

if __name__ == "__main__":
    main()