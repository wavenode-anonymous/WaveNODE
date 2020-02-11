## WaveNODE : A Continuous Normalizing Flow for Speech Synthesis

We provide a PyTorch implementation of WaveNODE

### Abstract
In recent years, various flow-based generative
models have been proposed to generate highfidelity waveforms in real-time. However, these
models require either a well-trained teacher network or a number of flow steps making them
memory-inefficient. In this paper, we propose
a novel generative model called WaveNODE
which exploits a continuous normalizing flow for
speech synthesis. Unlike the conventional models,
WaveNODE places no constraint on the function
used for flow operation, thus allowing the usage
of more flexible and complex functions. Moreover, WaveNODE can be optimized to maximize
the likelihood without requiring any teacher network or auxiliary loss terms. We experimentally
show that WaveNODE achieves comparable performance in terms of mean opinion score (MOS)
and conditional log-likelihood (CLL) with fewer
parameters compared to the existing models.

### Requirements

- PyTorch 1.3.1
- Python 3.7.3
- Librosa
- torchdiffeq : https://github.com/rtqichen/torchdiffeq

### Examples

#### 1. Download Dataset

LJSpeech : https://keithito.com/LJ-Speech-Dataset/

#### 2. Preprocessing (Preparing Mel Spectrogram)

`python preprocessing.py --in_dir LJSpeech-1.1 --out_dir DATASETS/ljspeech`

#### 3. Train

`python train.py  --batch_size=20 --n_block=4 --split_period=2 --scale_init=4 --n_layer_wvn=4 --n_channel_wvn=128 --d_i=3`

#### 4. Synthesize

`python synthesize.py --load_step=27140 --tol_synth=1e-3 --num_synth=10 --batch_size=20 --n_block=4 --scale_init=4 --split_period=2 --n_layer_wvn=4 --n_channel_wvn=128 --d_i=3`

#### 5. Test Conditional Log-Likelihood

`python test_cll.py --load_step=27140 --batch_size=20 --n_block=4 --scale_init=4 --split_period=2 --n_layer_wvn=4 --n_channel_wvn=128 --d_i=3`

#### 6. Test Sampling Speed

`python test_speed.py --load_step=27140 --batch_size=20 --n_block=4 --scale_init=4 --split_period=2 --n_layer_wvn=4 --n_channel_wvn=128 --d_i=3`

### Audio Samples

Audio examples : https://wavenode-anonymous.github.io/

### Reference

- FFJORD : https://github.com/rtqichen/ffjord
- WaveGlow : https://github.com/NVIDIA/waveglow
- FloWaveNet : https://github.com/ksw0306/FloWaveNet
- PointFlow : https://github.com/stevenygd/PointFlow