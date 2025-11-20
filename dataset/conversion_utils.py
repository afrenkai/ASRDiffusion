import torchaudio
from utils.consts import (NUM_MELS, NUM_FFT, WINDOW_LENGTH, HOP_LENGTH, POWER, SAMPLE_RATE, NUM_STFT, AMPLITUTE_MULTIPLIER, AMPLITUDE_AMIN, FRAME_LENGTH, AMPLITUDE_REFERENCE, AMPLITUDE_POWER, SCALE_DECIBELS, MAX_DECIBELS, DECIBEL_MULTIPLIER)


class SpeechConverter():
    def __init__(self, num_mels:int =NUM_MELS):
        self.num_mel = num_mels
        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=NUM_FFT, 
            win_length=WINDOW_LENGTH,
            hop_length=HOP_LENGTH,
            power=POWER
        )
        self.mel_scale_transform = torchaudio.transforms.MelScale(
            n_mels=self.num_mel, 
            sample_rate=SAMPLE_RATE, 
            n_stft=NUM_STFT
        )

        self.mel_inverse_transform = torchaudio.transforms.InverseMelScale(
            n_mels=self.num_mel, 
            sample_rate=SAMPLE_RATE, 
            n_stft=NUM_STFT
        )

        self.griffnlim_transform = torchaudio.transforms.GriffinLim(
            n_fft=NUM_FFT,
            win_length=WINDOW_LENGTH,
            hop_length=HOP_LENGTH,
        )
        
    def pow_to_db_mel_spec(self,mel_spec):
        mel_spec = torchaudio.functional.amplitude_to_DB(
            mel_spec,
            multiplier = AMPLITUTE_MULTIPLIER, 
            amin = AMPLITUDE_AMIN, 
            db_multiplier = DECIBEL_MULTIPLIER, 
            top_db = MAX_DECIBELS
        )
        mel_spec = mel_spec/SCALE_DECIBELS
        return mel_spec

    def convert_to_mel_spec(self, raw_audio):
        spec = self.spec_transform(raw_audio)
        mel_spec = self.mel_scale_transform(spec)
        db_mel_spec = self.pow_to_db_mel_spec(mel_spec)
        db_mel_spec = db_mel_spec.squeeze(0)
        return db_mel_spec
    
    def inverse_mel_spec_to_wav(self, mel_spec):
        power_mel_spec = self.db_to_power_mel_spec(mel_spec)
        spectrogram = self.mel_inverse_transform(power_mel_spec)
        pseudo_wav = self.griffnlim_transform(spectrogram)
        return pseudo_wav

    def db_to_power_mel_spec(self, mel_spec):
        mel_spec = mel_spec*SCALE_DECIBELS
        mel_spec = torchaudio.functional.DB_to_amplitude(
            mel_spec,
            ref=AMPLITUDE_REFERENCE,
            power=AMPLITUDE_POWER)  
 
        return mel_spec


