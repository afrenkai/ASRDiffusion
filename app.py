import os
import torch
import torchaudio
import soundfile as sf
import traceback
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from models.mamba_denoiser import MambaDenoiser
from models.audio_encoder import AudioEncoder
from models.embed import TextEmbedding
from models.masked_diffusion_scheduler import MaskedDiffusionScheduler
from dataset.conversion_utils import SpeechConverter
from tokenizers import Tokenizer
from utils.consts import PAD, NUM_MELS, SAMPLE_RATE


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  


os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

D_MODEL = 256
N_LAYERS = 4
NUM_STEPS = 20
tokenizer = None
audio_encoder = None
text_embedder = None
model = None
scheduler = None
vocab_size = None
mask_token_id = None
speech_converter = None


class SimpleCharTokenizer:
    """
    simple fallback tokenizer to match the checkpoint's vocabulary size (43).
    Maps characters to IDs 0-42.
    """
    def __init__(self):
        self.vocab = "abcdefghijklmnopqrstuvwxyz0123456789 .,!?'"
        self.char_to_id = {c: i for i, c in enumerate(self.vocab)}
        self.id_to_char = {i: c for c, i in self.char_to_id.items()}
        self.vocab_size = len(self.vocab) # ~42

    def token_to_id(self, token):
        return self.char_to_id.get(token.lower(), 0) 

    def get_vocab_size(self):
        return 43

    def decode(self, token_ids, skip_special_tokens=True):
        chars = []
        for tid in token_ids:
            if tid < len(self.vocab):
                chars.append(self.id_to_char.get(tid, ''))
        return "".join(chars)


def load_models():
    """load model and tokenizer at startup."""
    global tokenizer, audio_encoder, text_embedder, model, scheduler
    global vocab_size, mask_token_id, speech_converter
    
    print("Loading models...")
    
    print("Using Fallback Character Tokenizer to match checkpoint dimensions...")
    tokenizer = SimpleCharTokenizer()
    
    vocab_size = 43 
    
    pad_token_id = 41
    mask_token_id = 42
    
    speech_converter = SpeechConverter(num_mels=NUM_MELS)
    
    audio_encoder = AudioEncoder(in_channels=1, dropout=0.1).to(device)
    text_embedder = TextEmbedding(vocab_size=vocab_size, embed_dim=D_MODEL, max_seq_len=512).to(device)
    
    model = MambaDenoiser(
        text_dim=D_MODEL,
        d_model=D_MODEL,
        audio_channels=128,
        n_layers=N_LAYERS,
        mask_token_id=mask_token_id
    ).to(device)
    
    scheduler = MaskedDiffusionScheduler(
        num_steps=NUM_STEPS,
        masking_schedule="cosine"
    )
    
    checkpoint_path = "checkpoint_epoch_10.pt"
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        audio_encoder.load_state_dict(checkpoint['audio_encoder_state_dict'])

        text_embedder.load_state_dict(checkpoint['text_embedder_state_dict'])
        
        print("Checkpoint loaded successfully!")
    else:
        print(f"Warning: No checkpoint found at {checkpoint_path}. Using randomly initialized weights.")
    
    model.eval()
    audio_encoder.eval()
    text_embedder.eval()
    
    print("Models loaded successfully!")


def process_audio(audio_path):
    """
    process uploaded audio file and convert to mel spectrogram.
    runs on CPU to avoid device mismatch errors with SpeechConverter.
    """
    data, sr = sf.read(audio_path)
    waveform = torch.from_numpy(data).float()
    
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)  # [1, Time]
    else:
        waveform = waveform.t()           # [Channels, Time]

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)
    
    waveform = waveform.squeeze(0)
    
    mel_spec = speech_converter.convert_to_mel_spec(waveform)

    mel_batch = mel_spec.unsqueeze(0).unsqueeze(0).to(device)
    
    return mel_batch


def run_inference(mel_batch, estimated_length=20):
    """runs reverse diffusion inference to transcribe audio."""
    global model, audio_encoder, text_embedder, scheduler, tokenizer, mask_token_id
    
    with torch.no_grad():
        audio_features = audio_encoder(mel_batch)
        
        batch_size = 1
        seq_len = estimated_length
        
        mask_token_emb = text_embedder.tok_emb(torch.tensor(mask_token_id, device=device)).view(1, 1, -1)
        x_t = mask_token_emb.expand(batch_size, seq_len, -1).clone()
        
        #reverse
        timesteps = torch.linspace(1, 0, NUM_STEPS, device=device)
        
        for i, t_val in enumerate(timesteps):
            t = torch.full((batch_size,), t_val.item(), device=device)
            
            x_0_pred = model(x_t, t, audio_features)
            
            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                t_next_batch = torch.full((batch_size,), t_next.item(), device=device)
                x_t, _ = scheduler.apply_masking(x_0_pred, t_next_batch, mask_value=mask_token_emb)
            else:
                x_t = x_0_pred
        
        emb_weights = text_embedder.tok_emb.weight
        logits = torch.matmul(x_t, emb_weights.t())
        predicted_ids = torch.argmax(logits, dim=-1)
        
        predicted_text = tokenizer.decode(predicted_ids[0].tolist(), skip_special_tokens=True)
        
    return predicted_text


@app.route('/')
def index():
    """render main page."""
    return render_template('index.html', device=device)


@app.route('/transcribe', methods=['POST'])
def transcribe():
    """handle audio upload and transcription."""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    allowed_extensions = {'.wav'}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_extensions:
        return jsonify({'error': f'Unsupported file format. Allowed: {", ".join(allowed_extensions)}'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        print(f"DEBUG: Processing file: {filepath}")
        print(f"DEBUG: Audio Encoder status: {audio_encoder}") 
        print(f"DEBUG: Model status: {model}")
        
        mel_batch = process_audio(filepath)
        
        audio_duration = mel_batch.shape[-1] * 256 / SAMPLE_RATE
        estimated_length = max(10, min(100, int(audio_duration * 10)))
        
        transcription = run_inference(mel_batch, estimated_length)
        
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'transcription': transcription,
            'device': device
        })
        
    except Exception as e:
        print("\n=== FULL ERROR TRACEBACK ===")
        traceback.print_exc()
        print("============================\n")
        #debug
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'device': device,
        'models_loaded': model is not None
    })


if __name__ == '__main__':
    load_models()
    app.run(debug=True, host='0.0.0.0', port=5000)
