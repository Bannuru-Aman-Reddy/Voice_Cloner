# Voice Cloning using SV2TTS
An implementation of [Transefer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis](https://arxiv.org/abs/1806.04558) (Sv2TTS).

SV2TTS is a comprehensive deep learning framework consisting of three distinct stages. In the initial stage, it generates a digital voice representation based on a short duration of audio data. Subsequently, in the second and third stages, this representation is used as a reference point to facilitate the generation of speech corresponding to any given text input.

## Setup
### Install Requirements
1. A GPU is recommended for fast training and for inference speed, but can also be run on CPU.
2. Python 3.7 is recommended. Python 3.5 or greater should work, but you'll probably have to tweak the dependencies' versions. Setting up a virtual environment using **venv** and using *pyenv* to use Python 3.7 is recommended.
3. Install [ffmpeg.org](https://ffmpeg.org/). FFmpeg is a multimedia framework that is used to handle various audio, video, and multimedia processing tasks.
4. Install [pytorch](https://pytorch.org/). Use the latest stable version, your operating system, your package manager and the CUDA versions if you have a GPU, otherwise pick CPU. Run the given command.
5. Install the remaining requirements with **pip install -r requirements.txt**

##### Note: Pretrained models are downloaded automatically. If not, download from [here](https://drive.google.com/drive/folders/19V-29XGl-gZrWay6wV4ehvOhA8NKSqNR?usp=sharing)

## Sample Input and Sample Output
### Sample Input Voice
https://github.com/Bannuru-Aman-Reddy/Voice_Cloner/assets/96659682/7ea87bdf-f1ba-4220-9de6-bf2ea92216a1
### Sample Output Voice
https://github.com/Bannuru-Aman-Reddy/Voice_Cloner/assets/96659682/2a4dae06-6d7c-4982-8e91-c51d7c322761





