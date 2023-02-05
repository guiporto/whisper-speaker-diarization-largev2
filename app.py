import os
import time
import wave
import torch
import base64
import whisper
import datetime
import contextlib
import numpy as np
import pandas as pd
from io import BytesIO
from pytube import YouTube
from pyannote.audio import Audio
from pyannote.core import Segment
from sklearn.cluster import AgglomerativeClustering
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    global model_name
    global embedding_model
    #medium, large-v1, large-v2
    model_name = "large-v2"
    model = whisper.load_model(model_name)
    embedding_model = PretrainedSpeakerEmbedding(
        "speechbrain/spkrec-ecapa-voxceleb",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))


def convert_time(secs):
    return datetime.timedelta(seconds=round(secs))

def speech_to_text(audio_file_path, selected_source_lang, whisper_model, num_speakers):
    model = whisper.load_model(whisper_model)
    time_start = time.time()
    if(audio_file_path == None):
        raise ValueError("Error no audio input")
    print(audio_file_path)

    try:
        # Get duration
        with contextlib.closing(wave.open(audio_file_path,'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
        print(f"duration of audio file: {duration}")

        # Transcribe audio
        options = dict(language=selected_source_lang, beam_size=5, best_of=5)
        transcribe_options = dict(task="transcribe", **options)
        result = model.transcribe(audio_file_path, **transcribe_options)
        segments = result["segments"]
        print("starting whisper done with whisper")
    except Exception as e:
        raise RuntimeError("Error transcribing audio")

    try:
        # Create embedding
        def segment_embedding(segment):
            audio = Audio()
            start = segment["start"]
            # Whisper overshoots the end timestamp in the last segment
            end = min(duration, segment["end"])
            clip = Segment(start, end)
            waveform, sample_rate = audio.crop(audio_file_path, clip)
            return embedding_model(waveform[None])

        embeddings = np.zeros(shape=(len(segments), 192))
        for i, segment in enumerate(segments):
            embeddings[i] = segment_embedding(segment)
        embeddings = np.nan_to_num(embeddings)
        print(f'Embedding shape: {embeddings.shape}')

        # Assign speaker label
        clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
        labels = clustering.labels_
        for i in range(len(segments)):
            segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)

        # Make output
        objects = {
            'Start' : [],
            'End': [],
            'Speaker': [],
            'Text': []
        }
        text = ''
        for (i, segment) in enumerate(segments):
            if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
                objects['Start'].append(str(convert_time(segment["start"])))
                objects['Speaker'].append(segment["speaker"])
                if i != 0:
                    objects['End'].append(str(convert_time(segments[i - 1]["end"])))
                    objects['Text'].append(text)
                    text = ''
            text += segment["text"] + ' '
        objects['End'].append(str(convert_time(segments[i - 1]["end"])))
        objects['Text'].append(text)

        time_end = time.time()
        time_diff = time_end - time_start

        system_info = f"""-----Processing time: {time_diff:.5} seconds-----"""
        print(system_info)
        return pd.DataFrame(objects)

    except Exception as e:
        raise RuntimeError("Error Running inference with local model", e)


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    global model_name
    global embedding_model

    # Parse out your arguments
    audio_file_path = model_inputs.get('audio_file_path', None)
    selected_source_lang = model_inputs.get('language', "en")
    number_speakers = model_inputs.get('num_speakers', 2)

    if audio_file_path == None:
        return {'message': "No input provided"}

    # Run the model
    video_in = get_audio_file(audio_file_path)
    transcription_df = speech_to_text(video_in, selected_source_lang, model_name, number_speakers)
    # print(transcription_df)

    # Return the results as a dictionary
    return transcription_df.to_json()
