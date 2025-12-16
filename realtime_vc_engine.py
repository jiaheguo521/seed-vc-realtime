import os
import sys
import struct
import shutil
import time
import json
import warnings
import yaml
import re
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import torchaudio.transforms as tat
import sounddevice as sd
import librosa
from multiprocessing import shared_memory
from audio_modules.commons import str2bool, recursive_munch, build_model, load_checkpoint

# Add current directory to path for imports
now_dir = os.getcwd()
sys.path.append(now_dir)

# Script directory for relative path resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from hf_utils import load_custom_model_from_hf
from funasr import AutoModel

# Suppress warnings
warnings.simplefilter("ignore")

class RealtimeVCConfig:
    def __init__(self):
        self.reference_audio_path = ""
        self.diffusion_steps = 10
        self.sr_type = "sr_model"
        self.block_time = 0.25  # s
        self.threhold = -60
        self.crossfade_time = 0.05
        self.extra_time_ce = 2.5
        self.extra_time = 0.5
        self.extra_time_right = 2.0
        self.I_noise_reduce = False
        self.O_noise_reduce = False
        self.inference_cfg_rate = 0.7
        self.max_prompt_length = 3.0
        self.sg_hostapi = ""
        self.wasapi_exclusive = False
        self.sg_input_device = ""
        self.sg_output_device = ""
        self.samplerate = 44100  # default, will be updated
        self.channels = 2
        self.sr_model = True
        self.sr_device = False
        self.function = "vc"  # "vc" for voice conversion, "im" for input monitoring (passthrough)

class RealtimeVCEngine:
    def __init__(self, args):
        self.args = args
        self.config = RealtimeVCConfig()
        self.device = self._get_device(args.gpu)
        self.fp16 = args.fp16
        
        # Runtime state
        self.flag_vc = False
        self.stream = None
        self.shm = None
        self.delay_queue = []
        self.last_error_print_time = 0
        
        # Audio processing state
        self.prompt_condition = None
        self.mel2 = None
        self.style2 = None
        self.reference_wav_name = ""
        self.prompt_len = 3
        self.ce_dit_difference = 2.0
        
        # Load models
        print(f"Loading models on {self.device}...")
        self.model_set = self._load_models(args)
        self.vad_model = AutoModel(model="fsmn-vad", model_revision="v2.0.4", disable_pbar=True, log_level="ERROR", disable_update=True)
        
        # Shared Memory
        shm_name = os.environ.get("RAS_SHARED_MEM_NAME")
        if shm_name:
            try:
                self.shm = shared_memory.SharedMemory(name=shm_name)
                print(f"Connected to Shared Memory: {shm_name}")
            except Exception as e:
                print(f"Failed to connect to Shared Memory: {e}")

        # Devices
        self.hostapis = []
        self.input_devices = []
        self.output_devices = []
        self.input_devices_indices = []
        self.output_devices_indices = []
        self.update_devices()
        
        # Callbacks
        self.on_perf_update = None # Callback function(infer_time, delay_time)

    def _get_device(self, gpu_id):
        cuda_target = f"cuda:{gpu_id}" if gpu_id else "cuda"
        if torch.cuda.is_available():
            return torch.device(cuda_target)
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def _load_models(self, args):
        print(f"Using fp16: {self.fp16}")
        if args.checkpoint_path is None or args.checkpoint_path == "":
            dit_checkpoint_path, dit_config_path = load_custom_model_from_hf("Plachta/Seed-VC",
                                                                             "DiT_uvit_tat_xlsr_ema.pth",
                                                                             "config_dit_mel_seed_uvit_xlsr_tiny.yml")
        else:
            dit_checkpoint_path = args.checkpoint_path
            dit_config_path = args.config_path
            
        config = yaml.safe_load(open(dit_config_path, "r"))
        model_params = recursive_munch(config["model_params"])
        model_params.dit_type = 'DiT'
        model = build_model(model_params, stage="DiT")
        hop_length = config["preprocess_params"]["spect_params"]["hop_length"]
        sr = config["preprocess_params"]["sr"]

        # Load checkpoints
        model, _, _, _ = load_checkpoint(
            model,
            None,
            dit_checkpoint_path,
            load_only_params=True,
            ignore_modules=[],
            is_distributed=False,
        )
        for key in model:
            model[key].eval()
            model[key].to(self.device)
        model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

        # Load additional modules
        from audio_modules.campplus.DTDNN import CAMPPlus

        campplus_ckpt_path = load_custom_model_from_hf(
            "funasr/campplus", "campplus_cn_common.bin", config_filename=None
        )
        campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
        campplus_model.eval()
        campplus_model.to(self.device)

        vocoder_type = model_params.vocoder.type

        if vocoder_type == 'bigvgan':
            from audio_modules.bigvgan import bigvgan
            bigvgan_name = model_params.vocoder.name
            bigvgan_model = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=False)
            bigvgan_model.remove_weight_norm()
            bigvgan_model = bigvgan_model.eval().to(self.device)
            vocoder_fn = bigvgan_model
        elif vocoder_type == 'hifigan':
            from audio_modules.hifigan.generator import HiFTGenerator
            from audio_modules.hifigan.f0_predictor import ConvRNNF0Predictor
            hifigan_config_path = os.path.join(SCRIPT_DIR, 'configs/hifigan.yml')
            hift_config = yaml.safe_load(open(hifigan_config_path, 'r'))
            hift_gen = HiFTGenerator(**hift_config['hift'], f0_predictor=ConvRNNF0Predictor(**hift_config['f0_predictor']))
            hift_path = load_custom_model_from_hf("FunAudioLLM/CosyVoice-300M", 'hift.pt', None)
            hift_gen.load_state_dict(torch.load(hift_path, map_location='cpu'))
            hift_gen.eval()
            hift_gen.to(self.device)
            vocoder_fn = hift_gen
        elif vocoder_type == "vocos":
            vocos_config = yaml.safe_load(open(model_params.vocoder.vocos.config, 'r'))
            vocos_path = model_params.vocoder.vocos.path
            vocos_model_params = recursive_munch(vocos_config['model_params'])
            vocos = build_model(vocos_model_params, stage='mel_vocos')
            vocos_checkpoint_path = vocos_path
            vocos, _, _, _ = load_checkpoint(vocos, None, vocos_checkpoint_path,
                                             load_only_params=True, ignore_modules=[], is_distributed=False)
            _ = [vocos[key].eval().to(self.device) for key in vocos]
            _ = [vocos[key].to(self.device) for key in vocos]
            vocoder_fn = vocos.decoder
        else:
            raise ValueError(f"Unknown vocoder type: {vocoder_type}")

        speech_tokenizer_type = model_params.speech_tokenizer.type
        
        # Define semantic_fn based on type
        if speech_tokenizer_type == 'whisper':
            from transformers import AutoFeatureExtractor, WhisperModel
            whisper_name = model_params.speech_tokenizer.name
            whisper_model = WhisperModel.from_pretrained(whisper_name, torch_dtype=torch.float16).to(self.device)
            del whisper_model.decoder
            whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)

            def semantic_fn(waves_16k):
                ori_inputs = whisper_feature_extractor([waves_16k.squeeze(0).cpu().numpy()],
                                                       return_tensors="pt",
                                                       return_attention_mask=True)
                ori_input_features = whisper_model._mask_input_features(
                    ori_inputs.input_features, attention_mask=ori_inputs.attention_mask).to(self.device)
                with torch.no_grad():
                    ori_outputs = whisper_model.encoder(
                        ori_input_features.to(whisper_model.encoder.dtype),
                        head_mask=None,
                        output_attentions=False,
                        output_hidden_states=False,
                        return_dict=True,
                    )
                S_ori = ori_outputs.last_hidden_state.to(torch.float32)
                S_ori = S_ori[:, :waves_16k.size(-1) // 320 + 1]
                return S_ori
        
        elif speech_tokenizer_type == 'cnhubert':
            from transformers import Wav2Vec2FeatureExtractor, HubertModel
            hubert_model_name = config['model_params']['speech_tokenizer']['name']
            hubert_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(hubert_model_name)
            hubert_model = HubertModel.from_pretrained(hubert_model_name)
            hubert_model = hubert_model.to(self.device)
            hubert_model = hubert_model.eval()
            hubert_model = hubert_model.half()

            def semantic_fn(waves_16k):
                ori_waves_16k_input_list = [waves_16k[bib].cpu().numpy() for bib in range(len(waves_16k))]
                ori_inputs = hubert_feature_extractor(ori_waves_16k_input_list,
                                                      return_tensors="pt",
                                                      return_attention_mask=True,
                                                      padding=True,
                                                      sampling_rate=16000).to(self.device)
                with torch.no_grad():
                    ori_outputs = hubert_model(ori_inputs.input_values.half())
                S_ori = ori_outputs.last_hidden_state.float()
                return S_ori

        elif speech_tokenizer_type == 'xlsr':
            from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
            model_name = config['model_params']['speech_tokenizer']['name']
            output_layer = config['model_params']['speech_tokenizer']['output_layer']
            wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            wav2vec_model = Wav2Vec2Model.from_pretrained(model_name)
            wav2vec_model.encoder.layers = wav2vec_model.encoder.layers[:output_layer]
            wav2vec_model = wav2vec_model.to(self.device)
            wav2vec_model = wav2vec_model.eval()
            wav2vec_model = wav2vec_model.half()

            def semantic_fn(waves_16k):
                ori_waves_16k_input_list = [waves_16k[bib].cpu().numpy() for bib in range(len(waves_16k))]
                ori_inputs = wav2vec_feature_extractor(ori_waves_16k_input_list,
                                                       return_tensors="pt",
                                                       return_attention_mask=True,
                                                       padding=True,
                                                       sampling_rate=16000).to(self.device)
                with torch.no_grad():
                    ori_outputs = wav2vec_model(ori_inputs.input_values.half())
                S_ori = ori_outputs.last_hidden_state.float()
                return S_ori
        else:
             raise ValueError(f"Unknown speech tokenizer type: {speech_tokenizer_type}")

        # Generate mel spectrograms
        mel_fn_args = {
            "n_fft": config['preprocess_params']['spect_params']['n_fft'],
            "win_size": config['preprocess_params']['spect_params']['win_length'],
            "hop_size": config['preprocess_params']['spect_params']['hop_length'],
            "num_mels": config['preprocess_params']['spect_params']['n_mels'],
            "sampling_rate": sr,
            "fmin": config['preprocess_params']['spect_params'].get('fmin', 0),
            "fmax": None if config['preprocess_params']['spect_params'].get('fmax', "None") == "None" else 8000,
            "center": False
        }
        from audio_modules.audio import mel_spectrogram
        to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)

        return (
            model,
            semantic_fn,
            vocoder_fn,
            campplus_model,
            to_mel,
            mel_fn_args,
        )

    def update_devices(self, hostapi_name=None):
        self.stop_stream()
        sd._terminate()
        sd._initialize()
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        for hostapi in hostapis:
            for device_idx in hostapi["devices"]:
                devices[device_idx]["hostapi_name"] = hostapi["name"]
        self.hostapis = [hostapi["name"] for hostapi in hostapis]
        if hostapi_name not in self.hostapis:
            hostapi_name = self.hostapis[0]
        self.input_devices = [
            d["name"]
            for d in devices
            if d["max_input_channels"] > 0 and d["hostapi_name"] == hostapi_name
        ]
        self.output_devices = [
            d["name"]
            for d in devices
            if d["max_output_channels"] > 0 and d["hostapi_name"] == hostapi_name
        ]
        self.input_devices_indices = [
            d["index"] if "index" in d else d["name"]
            for d in devices
            if d["max_input_channels"] > 0 and d["hostapi_name"] == hostapi_name
        ]
        self.output_devices_indices = [
            d["index"] if "index" in d else d["name"]
            for d in devices
            if d["max_output_channels"] > 0 and d["hostapi_name"] == hostapi_name
        ]

    def get_device_samplerate(self):
        return int(sd.query_devices(device=sd.default.device[0])["default_samplerate"])

    def get_device_channels(self):
        max_input_channels = sd.query_devices(device=sd.default.device[0])["max_input_channels"]
        max_output_channels = sd.query_devices(device=sd.default.device[1])["max_output_channels"]
        return min(max_input_channels, max_output_channels, 2)

    def set_config(self, **kwargs):
        """Update configuration parameters."""
        for k, v in kwargs.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)
        
        # Handle derived config logic here if necessary
        if "sr_type" in kwargs:
             self.config.sr_model = (self.config.sr_type == "sr_model")
             self.config.sr_device = (self.config.sr_type == "sr_device")

    def _prepare_buffers(self):
        # Initialize variables based on current config
        if self.device.type == "mps":
            torch.mps.empty_cache()
        else:
            torch.cuda.empty_cache()
            
        self.reference_wav, _ = librosa.load(
            self.config.reference_audio_path, sr=self.model_set[-1]["sampling_rate"]
        )
        
        self.config.samplerate = (
            self.model_set[-1]["sampling_rate"]
            if self.config.sr_type == "sr_model"
            else self.get_device_samplerate()
        )
        self.config.channels = self.get_device_channels()
        self.zc = self.config.samplerate // 50
        self.block_frame = int(np.round(self.config.block_time * self.config.samplerate / self.zc)) * self.zc
        self.block_frame_16k = 320 * self.block_frame // self.zc
        self.crossfade_frame = int(np.round(self.config.crossfade_time * self.config.samplerate / self.zc)) * self.zc
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
        self.sola_search_frame = self.zc
        self.extra_frame = int(np.round(self.config.extra_time_ce * self.config.samplerate / self.zc)) * self.zc
        self.extra_frame_right = int(np.round(self.config.extra_time_right * self.config.samplerate / self.zc)) * self.zc
        
        self.input_wav = torch.zeros(
            self.extra_frame + self.crossfade_frame + self.sola_search_frame + self.block_frame + self.extra_frame_right,
            device=self.device, dtype=torch.float32
        )
        self.input_wav_denoise = self.input_wav.clone()
        self.input_wav_res = torch.zeros(
            320 * self.input_wav.shape[0] // self.zc,
            device=self.device, dtype=torch.float32
        )
        self.rms_buffer = np.zeros(4 * self.zc, dtype="float32")
        self.sola_buffer = torch.zeros(self.sola_buffer_frame, device=self.device, dtype=torch.float32)
        self.nr_buffer = self.sola_buffer.clone()
        self.output_buffer = self.input_wav.clone()
        self.skip_head = self.extra_frame // self.zc
        self.skip_tail = self.extra_frame_right // self.zc
        self.return_length = (self.block_frame + self.sola_buffer_frame + self.sola_search_frame) // self.zc
        
        self.fade_in_window = (torch.sin(0.5 * np.pi * torch.linspace(0.0, 1.0, steps=self.sola_buffer_frame, device=self.device, dtype=torch.float32)) ** 2)
        self.fade_out_window = 1 - self.fade_in_window
        
        self.resampler = tat.Resample(orig_freq=self.config.samplerate, new_freq=16000, dtype=torch.float32).to(self.device)
        if self.model_set[-1]["sampling_rate"] != self.config.samplerate:
            self.resampler2 = tat.Resample(orig_freq=self.model_set[-1]["sampling_rate"], new_freq=self.config.samplerate, dtype=torch.float32).to(self.device)
        else:
            self.resampler2 = None

        # VAD state
        self.vad_cache = {}
        self.vad_chunk_size = min(500, 1000 * self.config.block_time)
        self.vad_speech_detected = False
        self.vad_pos_start = False
        self.last_vad_time = 0
        self.last_vad_end_time = 0
        self.vad_input_history = np.array([], dtype=np.float32)
        self.last_vad_reset_time = time.time()

    def start_stream(self):
        if not self.flag_vc:
            self._prepare_buffers()
            self.flag_vc = True
            
            # Set devices in sd
            if self.config.sg_hostapi and self.config.sg_input_device and self.config.sg_output_device:
                 try:
                    hostapi_idx = [h for h in sd.query_hostapis() if h['name'] == self.config.sg_hostapi][0]['index']
                    # Find device indices
                    input_idx = -1
                    output_idx = -1
                    for d in sd.query_devices():
                         if d['hostapi'] == hostapi_idx:
                             if d['name'] == self.config.sg_input_device and d['max_input_channels'] > 0:
                                 input_idx = d['index']
                             if d['name'] == self.config.sg_output_device and d['max_output_channels'] > 0:
                                 output_idx = d['index']
                    
                    if input_idx >= 0 and output_idx >= 0:
                        sd.default.device = (input_idx, output_idx)
                 except Exception as e:
                     print(f"Error setting devices: {e}")

            if "WASAPI" in self.config.sg_hostapi and self.config.wasapi_exclusive:
                extra_settings = sd.WasapiSettings(exclusive=True)
            else:
                extra_settings = None
                
            self.stream = sd.Stream(
                callback=self._audio_callback,
                blocksize=self.block_frame,
                samplerate=self.config.samplerate,
                channels=self.config.channels,
                dtype="float32",
                extra_settings=extra_settings,
            )
            self.stream.start()
            
            if self.stream:
                input_latency = self.stream.latency[0]
                output_latency = self.stream.latency[1]
                self.delay_time = (
                    input_latency + output_latency
                    + self.config.block_time
                    + self.config.crossfade_time
                    + 0.01
                    + 0.5 
                )
                print(f"Audio Delay Calculated: {self.delay_time*1000:.2f}ms")

    def stop_stream(self):
        if self.flag_vc:
            self.flag_vc = False
            if self.stream is not None:
                self.stream.abort()
                self.stream.close()
                self.stream = None
            
            if self.shm:
                try:
                    self.shm.buf[0:8] = struct.pack('d', 0.0)
                    print("Audio delay reset to 0.0 in SHM")
                except Exception as e:
                    print(f"Failed to reset SHM: {e}")

    @torch.no_grad()
    def _custom_infer(self, input_wav_res):
        (model, semantic_fn, vocoder_fn, campplus_model, to_mel, mel_fn_args) = self.model_set
        sr = mel_fn_args["sampling_rate"]
        hop_length = mel_fn_args["hop_size"]
        
        cd_difference = self.config.extra_time_ce - self.config.extra_time
        max_prompt_length = self.config.max_prompt_length # assumed available in config

        if self.ce_dit_difference != cd_difference:
            self.ce_dit_difference = cd_difference
        
        if self.prompt_condition is None or self.reference_wav_name != self.config.reference_audio_path or self.prompt_len != max_prompt_length:
            self.prompt_len = max_prompt_length
            print(f"Setting max prompt length to {max_prompt_length} seconds.")
            # Re-process reference wav
            ref_wav_slice = self.reference_wav[:int(sr * self.prompt_len)]
            reference_wav_tensor = torch.from_numpy(ref_wav_slice).to(self.device)
            
            ori_waves_16k = torchaudio.functional.resample(reference_wav_tensor, sr, 16000)
            S_ori = semantic_fn(ori_waves_16k.unsqueeze(0))
            feat2 = torchaudio.compliance.kaldi.fbank(
                ori_waves_16k.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000
            )
            feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
            self.style2 = campplus_model(feat2.unsqueeze(0))
            
            self.mel2 = to_mel(reference_wav_tensor.unsqueeze(0))
            target2_lengths = torch.LongTensor([self.mel2.size(2)]).to(self.mel2.device)
            self.prompt_condition = model.length_regulator(
                S_ori, ylens=target2_lengths, n_quantizers=3, f0=None
            )[0]
            self.reference_wav_name = self.config.reference_audio_path

        converted_waves_16k = input_wav_res
        
        S_alt = semantic_fn(converted_waves_16k.unsqueeze(0))
        
        ce_dit_frame_difference = int(self.ce_dit_difference * 50)
        S_alt = S_alt[:, ce_dit_frame_difference:]
        target_lengths = torch.LongTensor([(self.skip_head + self.return_length + self.skip_tail - ce_dit_frame_difference) / 50 * sr // hop_length]).to(S_alt.device)
        
        cond = model.length_regulator(
            S_alt, ylens=target_lengths, n_quantizers=3, f0=None
        )[0]
        cat_condition = torch.cat([self.prompt_condition, cond], dim=1)
        
        with torch.autocast(device_type=self.device.type, dtype=torch.float16 if self.fp16 else torch.float32):
            vc_target = model.cfm.inference(
                cat_condition,
                torch.LongTensor([cat_condition.size(1)]).to(self.mel2.device),
                self.mel2,
                self.style2,
                None,
                n_timesteps=int(self.config.diffusion_steps),
                inference_cfg_rate=self.config.inference_cfg_rate,
            )
            vc_target = vc_target[:, :, self.mel2.size(-1) :]
            vc_wave = vocoder_fn(vc_target).squeeze()
            
        output_len = self.return_length * sr // 50
        tail_len = self.skip_tail * sr // 50
        output = vc_wave[-output_len - tail_len: -tail_len]
        return output

    def _audio_callback(self, indata, outdata, frames, times, status):
        start_time = time.perf_counter()
        indata = librosa.to_mono(indata.T)
        
        # Synchronization setup
        if self.device.type == "mps":
             torch.mps.synchronize()
        else:
             torch.cuda.synchronize()

        # VAD Processing
        indata_16k = librosa.resample(indata, orig_sr=self.config.samplerate, target_sr=16000)
        
        if time.time() - self.last_vad_reset_time > 10.0:
            self.vad_cache = {}
            if len(self.vad_input_history) > 0:
                warmup_audio = self.vad_input_history[-int(16000 * 1.5):]
                self.vad_model.generate(input=warmup_audio, cache=self.vad_cache, is_final=False, chunk_size=self.vad_chunk_size)
            self.last_vad_reset_time = time.time()

        res = self.vad_model.generate(input=indata_16k, cache=self.vad_cache, is_final=False, chunk_size=self.vad_chunk_size)
        self.vad_input_history = np.concatenate((self.vad_input_history, indata_16k))
        if len(self.vad_input_history) > 16000 * 6:
            self.vad_input_history = self.vad_input_history[-16000 * 6:]
            
        res_value = res[0]["value"]
        
        for segment in res_value:
            s_time, e_time = segment
            if s_time != -1:
                self.vad_pos_start = True
            if e_time != -1:
                self.vad_pos_start = False
                self.last_vad_end_time = time.time()
        
        if self.vad_pos_start:
            self.vad_speech_detected = True
        else:
            if self.vad_speech_detected and (time.time() - self.last_vad_end_time > 0.5):
                self.vad_speech_detected = False

        if not self.vad_pos_start and not self.vad_speech_detected:
            self.vad_cache = {}

        # Preprocessing
        self.input_wav[: -self.block_frame] = self.input_wav[self.block_frame :].clone()
        self.input_wav[-indata.shape[0] :] = torch.from_numpy(indata).to(self.device)
        
        self.input_wav_res[: -self.block_frame_16k] = self.input_wav_res[self.block_frame_16k :].clone()
        # Note: Using librosa resample on CPU then moving to GPU to match original implementation's behavior logic roughly
        # The original code had a commented out torch resampler and used librosa.
        resampled_indata = librosa.resample(self.input_wav[-indata.shape[0] - 2 * self.zc :].cpu().numpy(), orig_sr=self.config.samplerate, target_sr=16000)[320:]
        self.input_wav_res[-320 * (indata.shape[0] // self.zc + 1) :] = torch.from_numpy(resampled_indata).to(self.device)

        # Inference
        infer_wav = None
        
        # Check if we're in input monitoring mode (passthrough)
        if self.config.function == "im":
            # Input monitoring mode: directly pass through the input audio
            infer_wav = self.input_wav[self.extra_frame : self.extra_frame + self.block_frame + self.sola_buffer_frame + self.sola_search_frame].clone()
        elif self.config.extra_time_ce - self.config.extra_time < 0:
             if time.time() - self.last_error_print_time > 1.0:
                 print("Error: Content encoder extra context must be greater than DiT extra context!")
                 self.last_error_print_time = time.time()
             infer_wav = torch.zeros_like(self.input_wav[self.extra_frame :])
        else:
             try:
                 infer_wav = self._custom_infer(self.input_wav_res)
                 if self.resampler2 is not None:
                     infer_wav = self.resampler2(infer_wav)
             except Exception as e:
                 if time.time() - self.last_error_print_time > 1.0:
                     print(f"Inference error: {e}")
                     self.last_error_print_time = time.time()
                 infer_wav = torch.zeros_like(self.input_wav[self.extra_frame :])

        # Only apply VAD muting during voice conversion mode
        if self.config.function == "vc" and not self.vad_speech_detected:
             infer_wav = torch.zeros_like(self.input_wav[self.extra_frame :])

        # SOLA
        conv_input = infer_wav[None, None, : self.sola_buffer_frame + self.sola_search_frame]
        cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
        cor_den = torch.sqrt(
            F.conv1d(
                conv_input**2,
                torch.ones(1, 1, self.sola_buffer_frame, device=self.device),
            ) + 1e-8
        )
        tensor = cor_nom[0, 0] / cor_den[0, 0]
        if tensor.numel() > 1:
            if sys.platform == "darwin":
                _, sola_offset = torch.max(tensor, dim=0)
                sola_offset = sola_offset.item()
            else:
                sola_offset = torch.argmax(tensor, dim=0).item()
        else:
            sola_offset = tensor.item()
            
        infer_wav = infer_wav[int(sola_offset):]
        infer_wav[: self.sola_buffer_frame] *= self.fade_in_window
        infer_wav[: self.sola_buffer_frame] += (self.sola_buffer * self.fade_out_window)
        self.sola_buffer[:] = infer_wav[self.block_frame : self.block_frame + self.sola_buffer_frame]
        
        final_output = (
            infer_wav[: self.block_frame]
            .repeat(self.config.channels, 1)
            .t()
            .cpu()
            .numpy()
        )
        
        total_time = time.perf_counter() - start_time
        if self.on_perf_update:
            self.on_perf_update(total_time, self.delay_time)

        # Output handling with Shared Memory sync
        if self.shm:
            try:
                current_delay_ms = self.delay_time * 1000
                self.shm.buf[0:8] = struct.pack('d', current_delay_ms)
                target_delay_ms = struct.unpack('d', self.shm.buf[8:16])[0]
                extra_delay_s = max(0, (target_delay_ms - current_delay_ms) / 1000.0)
                
                self.delay_queue.append((time.perf_counter(), final_output))
                if self.delay_queue:
                    ts, data = self.delay_queue[0]
                    if time.perf_counter() - ts >= extra_delay_s:
                        outdata[:] = data
                        self.delay_queue.pop(0)
                    else:
                        outdata[:] = np.zeros_like(outdata)
                else:
                     outdata[:] = np.zeros_like(outdata)
            except Exception as e:
                if time.time() - self.last_error_print_time > 1.0:
                    print(f"SHM Error: {e}")
                    self.last_error_print_time = time.time()
                outdata[:] = final_output
        else:
            outdata[:] = final_output

