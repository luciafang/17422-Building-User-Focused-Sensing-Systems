# EmoSense: Emotion Analysis Tool
Here is the 2 mins demo for the application: https://www.youtube.com/watch?v=qg4M_2HnCCA



EmoSense is an innovative tool designed to analyze emotions by leveraging the power of cutting-edge machine learning technologies. It combines the visual emotion recognition capabilities of deepfaceML, the audio transcription accuracy of a speech-to-text ML algorithm, and the advanced understanding and processing abilities of OpenAI's GPT-3 through a command-line wrapper. This unique blend allows EmoSense to provide comprehensive emotion analysis from video recordings, making it an ideal tool for a wide range of applications including mental health assessment, user experience research, and interactive applications.

### Features
Visual Emotion Recognition: Utilizes deepfaceML to detect and analyze facial expressions in videos for emotion recognition.
Speech-to-Text Transcription: Converts spoken words in videos into text using a state-of-the-art speech-to-text ML algorithm, enhancing the emotion analysis process.
Emotion Analysis through GPT-3: Leverages a command-line wrapper for OpenAI's GPT-3 to interpret both visual and textual cues for a holistic understanding of the user's emotional state.
Mobile Compatibility: Designed for use in mobile applications, allowing users to record themselves directly within the app for real-time emotion analysis.
User-Friendly Interface: Easy-to-use interface for recording videos, with immediate feedback on emotion analysis results.

### Getting Started
##### Prerequisites
Python 3.8 or higher
Node.js (for the command-line wrapper)
OpenAI API key (for GPT-3 integration)
Access to a speech-to-text ML API

### Installation
1. Fork and clone the repository
2. Navigate to the project directory: cd emosense
3. Install the required Python dependencies: pip install -r requirements.txt
4. Set up the command-line wrapper for OpenAI's GPT-3 (follow the instructions provided in the cli-wrapper directory).

### Configuration
OpenAI API Key: Store your OpenAI API key in a .env file as follows:
OPENAI_API_KEY='your_openai_api_key_here'
Speech-to-Text API Key: Similarly, store your speech-to-text API key in the .env file.

### Running EmoSense
streamlit run main.py

### Usage
Use EmoSense to record yourself speaking about any topic. The app will analyze your facial expressions and speech to provide a comprehensive emotion analysis. This can be particularly useful for mental health tracking, user experience studies, or any application where emotional feedback is valuable.

