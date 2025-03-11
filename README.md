# feature_extractor
base for smart feature extractor..  the future is tomorrow!

``` python
import torch
import numpy as np
import torchaudio.transforms as T
import torchaudio.functional as F
import torchaudio
from torch import Tensor
from typing import Dict, List, Optional, Union, Any, Tuple

"""
EchoFeatureExtractor is a feature extractor class for processing audio signals into log mel spectrogram features with variable context lengths and can easily be used with hugging face datasets or a pytorch dataset.

Attributes:
    feature_size (int): Number of mel filterbanks.
    sampling_rate (int): Sampling rate of the audio.
    hop_length (int): Number of samples between successive frames.
    n_fft (int): Size of FFT.
    audio_ctx (int): Context window size for audio.
    padding_value (float): Value used for padding.
    device (str): Device to run the computations on.
    return_attention_mask (bool): Whether to return attention masks.
    model_input_names (List[str]): Names of the model inputs.
    padding_side (str): Side to apply padding ('right' or 'left').
Methods:
    __call__(audio, sampling_rate=None, return_tensors="pt", padding=False, max_length=None, truncation=True, pad_to_multiple_of=None, return_attention_mask=None):
        Processes the input audio and returns the extracted features.
    _get_expected_frames(audio_length):
        Calculates the expected number of frames after processing.
    _adjust_audio_length(audio, expected_frames):
        Adjusts the audio length to match the expected frames after processing.
    _adjust_feature_length(features):
        Ensures the feature length matches audio_ctx or a multiple of it.
    pad(features, max_length=None, padding="max_length", pad_to_multiple_of=None, return_attention_mask=None):
        Pads the features similar to Hugging Face's feature extractors.
    prepare_for_model(audio, padding=True, max_length=None, truncation=True, pad_to_multiple_of=None, return_tensors="pt", return_attention_mask=None):
        Prepares the audio for the model (Hugging Face compatibility).
    batch_decode(logits, skip_special_tokens=True):
        Placeholder for compatibility - would normally decode output tokens.
    process_batch(audio_files: List[str], batch_size=4):
        Processes a batch of audio files.
    load_audio(file_path, target_sr=16000):
        Helper function to load audio files.


    # With Hugging Face datasets

    from datasets import load_dataset
    import torch

    ds = load_dataset("some/audio_dataset")
    extractor = EchoFeatureExtractor(audio_ctx=1500, device="cuda" if torch.cuda.is_available() else "cpu")

    # Process a single example
    features = extractor(ds[0]["audio"])

    # Or map over the entire dataset
    ds = ds.map(lambda x: extractor(x["audio"]), batched=True, batch_size=16)

    # With PyTorch DataLoader

    from torch.utils.data import DataLoader

    class AudioDataset(torch.utils.data.Dataset):
        def __init__(self, audio_files):
            self.audio_files = audio_files
            
        def __len__(self):
            return len(self.audio_files)
            
        def __getitem__(self, idx):
            return {"audio": load_audio(self.audio_files[idx])}

    dataset = AudioDataset(audio_files)
    loader = DataLoader(dataset, batch_size=16, collate_fn=lambda x: {"audio": [item["audio"] for item in x]})

    for batch in loader:
        features = extractor(batch["audio"])
        # Use features in your model


"""


class EchoFeatureExtractor:
    def __init__(
        self,
        feature_size=128,
        sampling_rate=16000,
        hop_length=160,
        n_fft=400,
        audio_ctx=1500,  # This parameter should match the audio_ctx in your model's Dimensions
        padding_value=0.0,
        device="cpu",
        return_attention_mask=False,
        padding=True,
        max_length="max_length",
        truncation=True
    ):
        self.feature_size = feature_size
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.audio_ctx = audio_ctx
        self.padding_value = padding_value
        self.device = device
        self.return_attention_mask = return_attention_mask
        self.model_input_names = ["input_features"]
        self.padding_side = "right"
        self.padding = padding
        self.max_length = max_length
        self.truncation = truncation
        
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sampling_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=feature_size,
            power=2.0,
            normalized=False,
        ).to(device)
        
        self.window = torch.hann_window(n_fft).to(device)
        
    def __call__(
        self,
        audio,
        sampling_rate=None,
        return_tensors="pt",
        padding=None,
        max_length=None,
        truncation=None,
        pad_to_multiple_of=None,
        return_attention_mask=None
    ):
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            raise ValueError(f"Expected sampling rate {self.sampling_rate}, got {sampling_rate}")
        
        return_attention_mask = return_attention_mask if return_attention_mask is not None else self.return_attention_mask
        padding = padding if padding is not None else self.padding
        max_length = max_length if max_length is not None else self.max_length
        truncation = truncation if truncation is not None else self.truncation
        
        # Handle different input types (HF dataset compatibility)
        if isinstance(audio, dict) and "array" in audio and "sampling_rate" in audio:
            # Handle Hugging Face dataset audio format
            audio = audio["array"]
            sampling_rate = audio["sampling_rate"]
            if sampling_rate != self.sampling_rate:
                raise ValueError(f"Expected sampling rate {self.sampling_rate}, got {sampling_rate}")
            
        # Determine if input is batched
        is_batched = False
        if isinstance(audio, list):
            is_batched = isinstance(audio[0], np.ndarray) or isinstance(audio[0], list) or isinstance(audio[0], dict)
        else:
            is_batched = isinstance(audio, torch.Tensor) and len(audio.shape) > 1
            
        if not is_batched:
            audio = [audio]
            
        # Convert all inputs to torch tensors
        audio_tensors = []
        for a in audio:
            if isinstance(a, np.ndarray):
                tensor = torch.from_numpy(a).float().to(self.device)
            elif isinstance(a, torch.Tensor):
                tensor = a.float().to(self.device)
            elif isinstance(a, list):
                tensor = torch.tensor(a, dtype=torch.float32, device=self.device)
            elif isinstance(a, dict) and "array" in a:
                tensor = torch.tensor(a["array"], dtype=torch.float32, device=self.device)
            else:
                raise ValueError("Unsupported audio format")
            
            # Ensure we have the right shape (mono audio)
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            if tensor.shape[0] > 1:
                tensor = tensor.mean(dim=0, keepdim=True)  # Convert to mono
                
            audio_tensors.append(tensor)
            
        # Process each audio sample
        features = []
        attention_masks = []
        for tensor in audio_tensors:
            # Calculate expected length in frames after processing
            expected_frames = self._get_expected_frames(tensor.size(1))
            
            # Get attention mask before padding/truncation
            if return_attention_mask:
                orig_len = min((tensor.size(1) - self.n_fft) // self.hop_length + 1, expected_frames)
                attention_mask = torch.ones(expected_frames, dtype=torch.int32, device=self.device)
                attention_mask[orig_len:] = 0
                attention_masks.append(attention_mask)
            
            # Pad or truncate audio to match expected length before feature extraction
            tensor = self._adjust_audio_length(tensor, expected_frames)
            
            # Extract log mel spectrogram
            mel = self.mel_transform(tensor)
            log_spec = torch.clamp(mel, min=1e-10).log10()
            log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
            log_spec = (log_spec + 4.0) / 4.0
            
            # Always normalize
            mean = log_spec.mean()
            std = log_spec.std()
            log_spec = (log_spec - mean) / (std + 1e-7)
            
            # Ensure output length matches audio_ctx
            log_spec = self._adjust_feature_length(log_spec)
                
            features.append(log_spec)
        
        # Stack features
        input_features = torch.stack(features) if len(features) > 1 else features[0].unsqueeze(0)
        
        # Prepare return dictionary (Hugging Face style)
        result = {"input_features": input_features}
        
        # Add attention mask if requested
        if return_attention_mask:
            if attention_masks:
                result["attention_mask"] = torch.stack(attention_masks)
            else:
                # Default attention mask if none was created
                result["attention_mask"] = torch.ones(
                    (input_features.shape[0], input_features.shape[2]), 
                    dtype=torch.int32, 
                    device=input_features.device
                )
        
        # Handle padding for batch processing if requested
        if padding and is_batched:
            result = self.pad(
                features=result,
                max_length=max_length,
                padding="max_length",
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask
            )
        
        # Convert to numpy if requested (Hugging Face compatibility)
        if return_tensors == "np":
            for key, value in result.items():
                result[key] = value.cpu().numpy()
        elif return_tensors != "pt":
            # Keep tensors as is for PyTorch
            pass
            
        return result
    
    def _get_expected_frames(self, audio_length):
        """Calculate expected number of frames after processing"""
        frames = (audio_length - self.n_fft) // self.hop_length + 1
        # Round up to nearest multiple of audio_ctx if needed
        if frames < self.audio_ctx:
            return self.audio_ctx
        elif frames % self.audio_ctx != 0:
            return ((frames // self.audio_ctx) + 1) * self.audio_ctx
        return frames
    
    def _adjust_audio_length(self, audio, expected_frames):
        """Adjust audio length to match expected frames after processing"""
        required_length = (expected_frames - 1) * self.hop_length + self.n_fft
        current_length = audio.size(1)
        
        if current_length < required_length:
            # Pad audio
            padding = torch.zeros(1, required_length - current_length, device=audio.device)
            audio = torch.cat([audio, padding], dim=1)
        elif current_length > required_length:
            # Truncate audio
            audio = audio[:, :required_length]
            
        return audio
    
    def _adjust_feature_length(self, features):
        """Ensure feature length matches audio_ctx or a multiple of it"""
        current_length = features.size(2)
        
        if current_length < self.audio_ctx:
            # Pad features
            padding = torch.full(
                (features.size(0), features.size(1), self.audio_ctx - current_length),
                self.padding_value,
                device=features.device
            )
            features = torch.cat([features, padding], dim=2)
        elif current_length > self.audio_ctx and current_length % self.audio_ctx != 0:
            # Pad to next multiple of audio_ctx
            next_multiple = ((current_length // self.audio_ctx) + 1) * self.audio_ctx
            padding = torch.full(
                (features.size(0), features.size(1), next_multiple - current_length),
                self.padding_value,
                device=features.device
            )
            features = torch.cat([features, padding], dim=2)
            
        return features
    
    def pad(self, features, max_length=None, padding="max_length", pad_to_multiple_of=None, return_attention_mask=None):
        """Pad features similar to Hugging Face's feature extractors"""
        return_attention_mask = return_attention_mask if return_attention_mask is not None else self.return_attention_mask
        
        if isinstance(features, dict):
            input_features = features["input_features"]
        else:
            input_features = features
            
        batch_size = input_features.shape[0]
        seq_length = input_features.shape[2]
        
        if max_length is None:
            if seq_length % self.audio_ctx == 0:
                max_length = seq_length
            else:
                max_length = ((seq_length // self.audio_ctx) + 1) * self.audio_ctx
        
        # If we need to pad to a multiple of something
        if pad_to_multiple_of is not None and max_length % pad_to_multiple_of != 0:
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
            
        # Check if we actually need to pad
        needs_to_be_padded = seq_length < max_length
            
        if return_attention_mask and "attention_mask" not in features:
            features["attention_mask"] = torch.ones(
                (batch_size, seq_length), 
                dtype=torch.int32, 
                device=input_features.device
            )
            
        if needs_to_be_padded:
            padding_length = max_length - seq_length
            
            # Pad input features
            if self.padding_side == "right":
                padding_shape = (batch_size, input_features.shape[1], padding_length)
                padding_tensor = torch.full(padding_shape, self.padding_value, device=input_features.device)
                padded_features = torch.cat([input_features, padding_tensor], dim=2)
                
                # Pad attention mask if needed
                if return_attention_mask:
                    padding_mask = torch.zeros((batch_size, padding_length), device=input_features.device, dtype=torch.int32)
                    features["attention_mask"] = torch.cat([features["attention_mask"], padding_mask], dim=1)
            else:
                padding_shape = (batch_size, input_features.shape[1], padding_length)
                padding_tensor = torch.full(padding_shape, self.padding_value, device=input_features.device)
                padded_features = torch.cat([padding_tensor, input_features], dim=2)
                
                # Pad attention mask if needed
                if return_attention_mask:
                    padding_mask = torch.zeros((batch_size, padding_length), device=input_features.device, dtype=torch.int32)
                    features["attention_mask"] = torch.cat([padding_mask, features["attention_mask"]], dim=1)
                    
            features["input_features"] = padded_features
            
        return features
    
    def prepare_for_model(
        self,
        audio,
        padding=True,
        max_length=None,
        truncation=True,
        pad_to_multiple_of=None,
        return_tensors="pt",
        return_attention_mask=None,
    ):
        """Prepare audio for the model (Hugging Face compatibility)"""
        return self(
            audio=audio,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            return_attention_mask=return_attention_mask
        )
    
    def batch_decode(self, logits, skip_special_tokens=True):
        """Placeholder for compatibility - would normally decode output tokens"""
        raise NotImplementedError("This feature extractor doesn't handle decoding. Use a tokenizer instead.")
    
    def process_batch(self, audio_files: List[str], batch_size=4):
        """Process a batch of audio files"""
        all_features = []
        
        for i in range(0, len(audio_files), batch_size):
            batch = audio_files[i:i+batch_size]
            batch_tensors = []
            
            for file_path in batch:
                waveform, sr = torchaudio.load(file_path)
                if sr != self.sampling_rate:
                    waveform = F.resample(waveform, sr, self.sampling_rate)
                batch_tensors.append(waveform)
                
            features = self(batch_tensors)["input_features"]
            all_features.append(features)
            
        return torch.cat(all_features, dim=0) if all_features else None

def load_audio(file_path, target_sr=16000):
    """Helper function to load audio files"""
    waveform, sr = torchaudio.load(file_path)
    if sr != target_sr:
        resampler = T.Resample(sr, target_sr)
        waveform = resampler(waveform)
    return waveform







```
