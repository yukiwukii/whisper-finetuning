#!/bin/sh

# Variables
DATASET='gg_en'
LANGUAGE='en'
AUDIO_DIR="data/${DATASET}"
TSV_TRAIN_DIR="data/${DATASET}/train_metadata.json"
TSV_DEV_DIR="data/${DATASET}/dev_metadata.json"
TSV_TEST_DIR="data/${DATASET}/test_metadata.json"
SRT_DIR="work/timestamp/srt/${DATASET}"
OUTPUT_TRAIN_JSON="work/timestamp/json/${DATASET}/train.json"
OUTPUT_DEV_JSON="work/timestamp/json/${DATASET}/dev.json"
OUTPUT_MODEL="work/timestamp/expt/giga-vanilla"

echo This is for model without timestamp, augmented at snr DB

export CUDA_VISIBLE_DEVICES='2'

# pip install evaluate
# pip install jiwer

# pip install -r requirements.txt
# pip install openai-whisper
# conda install -c conda-forge ffmpeg
# pip install ffmpeg
# pip install torchaudio soundfile

# STEP 0: create augmented files (optional)
# echo Creating augmented files
# python work/timestamp/augment.py --split dev --percentage 0.001 --snr 10 --cv-path 'data/gigaspeech' --noise-path 'data/musan/noise/**/*.wav'

# STEP 1: create SRT files (done)
echo Creating SRT files
# python work/timestamp/create_srt.py --json-file ${TSV_TRAIN_DIR} --input-audio ${AUDIO_DIR}/train --output-directory ${SRT_DIR}/train
# python work/timestamp/create_srt.py --json-file ${TSV_DEV_DIR} --input-audio ${AUDIO_DIR}/dev --output-directory ${SRT_DIR}/dev
# python work/timestamp/create_srt.py --json-file ${TSV_TEST_DIR} --input-audio ${AUDIO_DIR}/test --output-directory ${SRT_DIR}/test

# STEP 2: create JSON files
echo Creating JSON files
# python work/timestamp/create_data.py --with-timestamps --audio-dir ${AUDIO_DIR}/dev --transcript-dir ${SRT_DIR}/dev --output ${OUTPUT_DEV_JSON}
# python work/timestamp/create_data.py --with-timestamps --audio-dir ${AUDIO_DIR}/train --transcript-dir ${SRT_DIR}/train --output ${OUTPUT_TRAIN_JSON}

# STEP 3: run finetuning
echo Running finetuning
# python work/timestamp/run_finetuning.py --train-json ${OUTPUT_TRAIN_JSON} --dev-json ${OUTPUT_DEV_JSON} --model small --save-dir ${OUTPUT_MODEL}

# STEP 4: transcription
echo Transcribing
# python work/timestamp/transcribe.py --audio-dir ${AUDIO_DIR}/test --save-dir ${OUTPUT_MODEL}/res/${DATASET} --model ${OUTPUT_MODEL}/best_model.pt --task transcribe --language ${LANGUAGE}
# python work/timestamp/transcribe.py --audio-dir ${AUDIO_DIR}/test --save-dir ${OUTPUT_MODEL}/res/${DATASET} --model small --task transcribe --language en


# STEP 5: calculate wer
echo Calculating WER
python work/timestamp/calculate_metric.py --recognized-dir ${OUTPUT_MODEL}/res/${DATASET} --transcript-dir ${SRT_DIR}/test --metric WER