DATASET_ROOT = "openslr/librispeech_asr"
DATASET_SUBSETS = ["all", "clean", "other"]
DATASET_SPLITS = [
    "test.clean",
    "test.other",
    "train.clean.100",
    "train.clean.360",
    "train.clean.500",
    "validation.clean",
    "validation.other",
]
DATASET_TEXT_COL = "text"
DATADIR = "Data/"

PAD = "[PAD]"
EOS = "[EOS]"

NUM_MELS = 128
# From NVIDIA TacoTron2 params
SAMPLE_RATE = 22050
SYMBOLS = [
    " ",  # space
    *list("abcdefghijklmnopqrstuvwxyz'-"),
    "[PAD]",
    "[EOS]",
]
NUM_FFT = 2048
NUM_STFT = int((NUM_FFT // 2) + 1)
FRAME_SHIFT = 0.0125  # seconds
HOP_LENGTH = int(NUM_FFT / 8.0)
FRAME_LENGTH = 0.05  # seconds
WINDOW_LENGTH = int(NUM_FFT / 2.0)
MAX_MEL_TIME = 1024
MAX_DECIBELS = 100
SCALE_DECIBELS = 10
REF = 4.0
POWER = 2.0
NORMALIZED_DECIBELS = 10
AMPLITUTE_MULTIPLIER = 10.0
AMPLITUDE_AMIN = 1e-10
DECIBEL_MULTIPLIER = 1.0
AMPLITUDE_REFERENCE = 1.0
AMPLITUDE_POWER = 1.0

# Improved Denoising Diffusion Probabilistic Models Param
EPS = 0.008
