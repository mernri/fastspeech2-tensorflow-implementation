# FastSpeech 2 TensorFlow Implementation

This repository contains an implementation of the FastSpeech2 model for text-to-speech (TTS) synthesis using TensorFlow. The project is based on the FastSpeech2 paper, [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/pdf/2006.04558) which aims to improve upon the original FastSpeech model by addressing the one-to-many mapping problem inherent in TTS tasks.

## Project Overview

FastSpeech 2 improves upon the original FastSpeech model by addressing the one-to-many mapping problem in TTS through direct training with ground-truth targets and introducing variance information such as pitch, energy, and accurate duration predictions. This repository aims to implement these improvements using TensorFlow.

## Repository Structure

- **.envrc**: Environment configuration file.
- **.gitignore**: Specifies files to be ignored by git.
- **Dockerfile**: Docker configuration for containerized setup.
- **Makefile**: Makefile for managing build tasks.
- **README.md**: Project documentation.
- **requirements.txt**: Python dependencies for development.
- **requirements_prod.txt**: Python dependencies for production.
- **setup.py**: Package setup script.

### Model Implementation

- **app/**: Main application directory with submodules and scripts.
  - `__init__.py`: Initialization of the app module.
  - `main.py`: Main script to run the application.
  - `params.py`: Configuration parameters for the model.
  - `registry.py`: Registry for model components.
  
  **model/**: Contains the model architecture components. This includes the encoder, decoder, and various utility functions specific to the FastSpeech 2 model. The model uses a feed-forward Transformer architecture, incorporating self-attention and 1D-convolution layers. The variance adaptor in the model handles duration, pitch, and energy predictions to improve the quality of synthesized speech.
  - `Attention.py`: Attention mechanism for the model.
  - `Config.py`: Contains configuration settings for the model, specifying parameters and structures.
  - `CustomLearningRateScheduler.py`: Custom learning rate scheduler to optimize the training process.
  - `CustomMelspecLoss.py`: Custom loss function tailored for mel-spectrogram prediction.
  - `Decoder.py`: The decoder component that converts hidden sequences back to mel-spectrograms.
  - `Encoder.py`: The encoder component that processes input phoneme sequences into hidden representations.
  - `MultiHeadAttention.py`: Implements the multi-head attention mechanism used to improve the model’s ability to focus on different parts of the input.
  - `Transformer.py`: Core Transformer architecture combining self-attention and feed-forward neural network layers.
  - `VarianceAdaptor.py`:  Modifies the hidden sequences by incorporating duration, pitch, and energy variations to enhance speech quality.
  - `__init__.py`: Initialization of the model module.
  
  **utils/**: Utility functions for data and audio processing.
  - `__init__.py`: Initialization of the utils module.
  - `audio.py`: Audio processing utilities.
  - `data.py`: Data processing utilities.
  - `data_checks.py`: Data validation checks.
  - `preprocess_audio.py`: Audio preprocessing scripts.
  - `save_melspecs_inputs.py`: Scripts to save mel-spectrogram inputs.
  - `save_tokens_inputs.py`: Scripts to save token inputs.
  - `text.py`: Text processing utilities.

### Additional Components

- **notebooks/**: Jupyter notebooks for exploratory data analysis and experiments.
  - `cheat_sheet.ipynb`: Jupyter notebook with cheat sheet and useful information.

- **tests/**: Unit tests for different components of the project.
  - `test_model_classes.py`: Unit tests for model classes.

## Getting Started

To get started with this project, clone the repository and install the necessary dependencies using the provided requirements files. You can use Docker for a containerized setup or set up a virtual environment manually.

```bash
git clone https://github.com/mernri/fastspeech2-tensorflow-implementation.git
cd fastspeech2-tensorflow-implementation
pip install -r requirements.txt
