import os
from argparse import Namespace


LOCAL_DATA_PATH =  os.path.dirname(os.path.dirname(__file__))

# PATHS LJSPEECH
PATH_LJ_AUDIOS = f"{LOCAL_DATA_PATH}/raw_data/LJSpeech/wavs"
PATH_LJ_CSV = f"{LOCAL_DATA_PATH}/raw_data/LJSpeech/metadata.csv"
PATH_PHONES_MAPPING_LJSPEECH = f"{LOCAL_DATA_PATH}/processed_data/phonem_mapping.json"
PATH_TRANSCRIPT_TEXTGRID = f"{LOCAL_DATA_PATH}/processed_data/LJSPeech_TextGrids"
PATH_PHONES_MAPPING_NEW = f"{LOCAL_DATA_PATH}/processed_data/phonem_mapping_new.json"

# PATH_TRANSCRIPT_PHONEMIZED_G2P = f"{LOCAL_DATA_PATH}/processed_data/phonemized_transcripts"
# PATH_TRANSCRIPT_PHONEMIZED_G2P_file = f"{PATH_TRANSCRIPT_PHONEMIZED_G2P}/phonemized_transcripts.txt"
PATH_TRANSCRIPT_PHONEMIZED_GITHUB_FILE= f"{LOCAL_DATA_PATH}/raw_data/phonemized_transcripts_github.txt"
PATH_TRANSCRIPT_PHONEMIZED_GITHUB_DIR= f"{LOCAL_DATA_PATH}/raw_data"

# PRE-PROCESSING PARAMS
TOKEN_PADDING_VALUE = 0
MEL_SPEC_PADDING_VALUE = 1
DURATION_PADDING_VALUE = 0

# FOR DURATION PREDICTION
MAX_DURATION = 9
MIN_DURATION = 2

# MODEL INPUTS DIRECTORIES 
PATH_PADDED_DURATIONS = f"{LOCAL_DATA_PATH}/processed_data/phone_durations"
PATH_PADDED_TOKENS = f"{LOCAL_DATA_PATH}/processed_data/tokens"
PATH_PADDED_MELSPECS = f"{LOCAL_DATA_PATH}/processed_data/melspectrograms"

# PATH MODEL
PATH_MODEL_CHECKPOINTS = f"{LOCAL_DATA_PATH}/app/saved_models/checkpoints"
PATH_FULL_MODEL = f"{LOCAL_DATA_PATH}/app/saved_models/final_models"

PATH_PREDICTED_MELSPEC = f"{LOCAL_DATA_PATH}/processed_data/predicted_melspecs"

# PATHS BARK
PATH_Tacatron2_WAV = f"{LOCAL_DATA_PATH}/Tacotron2_model/wav"
PATH_Tacatron2_DUMMY_WAV = f"{LOCAL_DATA_PATH}/Tacotron2_model/dummy_wav"
PATH_SINGLE_MELSPEC = f"{LOCAL_DATA_PATH}/drafts/mel_pour_antoine.npy"

# AUDIO PARAMS
SAMPLE_RATE=22050
N_FFT=1024
HOP_LENGTH=256
N_MELS=80
MEL_FRAMES=870

# INPUT OUTPUT PARAMS
MELSPEC_SHAPE= (80,870)
TARGET_LENGTH= 870

# MODEL TRAIN
BATCH_SIZE = 32

