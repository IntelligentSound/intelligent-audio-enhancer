# AI-Based Audio Equalizer

## Overview

An AI-driven audio equalizer that automatically adjusts audio settings based on the predicted music genre. Utilizes a Convolutional Neural Network (CNN) for genre classification and applies genre-specific equalization presets.

## Features

- **Genre Classification**: Predicts music genre using MFCC features.
- **Data Augmentation**: Enhances training data with noise addition and time stretching.
- **Equalization**: Applies predefined equalizer settings tailored to each genre.
- **Batch Processing**: Supports processing multiple audio files simultaneously.
- **Logging**: Detailed logs for monitoring and debugging.
- **Unit Tests**: Ensures reliability of feature extraction and genre prediction.

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ai-audio-equalizer.git
cd ai-audio-equalizer
