# Seed-VC Realtime

[Seed-VC](https://github.com/Plachtaa/Seed-VC) Real-time Voice Conversion separated for easier deployment and usage.

This project provides a standalone implementation of the real-time voice conversion functionality from Seed-VC, supporting zero-shot voice conversion with low latency.

## Features

-   **Real-time Voice Conversion**: Capable of cloning voices with 1~30 seconds of reference speech.
-   **Low Latency**: Algorithm delay of ~300ms and device side delay of ~100ms.
-   **Zero-shot**: No training required for new voices.

## Requirements

-   **OS**: Windows or Linux.
-   **GPU**: NVIDIA GPU with CUDA support is **strongly recommended** for real-time performance.
-   **Python**: 3.10+

## Installation & Usage



1.  **Create and activate a virtual environment**:
    ```bash
    # Windows
    python -m venv venv
    venv\Scripts\activate

    # Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The default `requirements.txt` includes CUDA 12.1 support. If you have a different CUDA version, please edit `requirements.txt` accordingly.*

3.  **Run the GUI**:
```bash
python real-time-gui.py --checkpoint-path <path-to-checkpoint> --config-path <path-to-config>
```
- `checkpoint` is the path to the model checkpoint if you have trained or fine-tuned your own model, leave to blank to auto-download default model from huggingface. (`seed-uvit-tat-xlsr-tiny`)
- `config` is the path to the model config if you have trained or fine-tuned your own model, leave to blank to auto-download default config from huggingface  


## Configuration & Performance

It is strongly recommended to use a GPU for real-time voice conversion. Below are some benchmark results on an NVIDIA RTX 3060 Laptop GPU:

| Model Configuration | Diffusion Steps | Inference CFG Rate | Max Prompt Length | Block Time (s) | Latency (ms) | Inference Time (ms) |
|---------------------|-----------------|--------------------|-------------------|----------------|--------------|---------------------|
| seed-uvit-xlsr-tiny | 10              | 0.7                | 3.0               | 0.18s          | 430ms        | 150ms               |

### Key Parameters

-   **Diffusion Steps**: 4~10 recommended for fastest real-time inference.
-   **Block Time**: The length of each audio chunk. Must be greater than inference time per block.
-   **Extra context**: Increasing this improves stability but adds latency.
-   **Virtual Cable**: Use [VB-CABLE](https://vb-audio.com/Cable/) to route GUI output to a virtual microphone for use in other applications (Discord, Zoom, etc.).

## Acknowledgements 🙏

This project is a separated version of [Seed-VC](https://github.com/Plachtaa/Seed-VC). All credit goes to the original authors.

-   [Seed-VC](https://github.com/Plachtaa/Seed-VC) - Original Project
- [Amphion](https://github.com/open-mmlab/Amphion) for providing computational resources and inspiration!
- [Vevo](https://github.com/open-mmlab/Amphion/tree/main/models/vc/vevo) for theoretical foundation of V2 model
- [MegaTTS3](https://github.com/bytedance/MegaTTS3) for multi-condition CFG inference implemented in V2 model
- [ASTRAL-quantiztion](https://github.com/Plachtaa/ASTRAL-quantization) for the amazing speaker-disentangled speech tokenizer used by V2 model
- [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) for foundationing the real-time voice conversion
- [SEED-TTS](https://arxiv.org/abs/2406.02430) for the initial idea
