import os
import yt_dlp as ydlp
import whisper

# Constants
SAMPLE_RATE = 16000  # Assuming a sample rate of 16,000 samples/second

def file_exists(filepath):
    return os.path.exists(filepath)

# 1. Download the Video/Audio
def download_video(video_url, output_format="mp4"):
    if file_exists('downloaded_video.mp4'):
        print("Video already downloaded. Skipping download.")
        return True
    ydl_opts = {
        'format': output_format,
        'outtmpl': 'downloaded_video.%(ext)s',
    }
    with ydlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([video_url])
        except ydlp.utils.DownloadError as e:
            print(f"Error downloading video: {e}")
            return False
    return True

def extract_audio(video_file, audio_format="mp3"):
    if file_exists('output_audio.mp3'):
        print("Audio already extracted. Skipping extraction.")
        return
    os.system(f"ffmpeg -i {video_file} output_audio.{audio_format}")

# 2. Transcribe the Audio using Whisper in 30-second chunks
def transcribe_audio_in_chunks(audio_file, chunk_duration=30):
    model = whisper.load_model("base")
    audio = whisper.load_audio(audio_file)

    # Calculate the number of chunks based on the assumed sample rate
    total_duration = len(audio) / SAMPLE_RATE
    num_chunks = int(total_duration / chunk_duration)

    full_transcription = ""

    for i in range(num_chunks):
        start_sample = i * chunk_duration * SAMPLE_RATE
        end_sample = start_sample + chunk_duration * SAMPLE_RATE
        audio_chunk = audio[start_sample:end_sample]

        audio_chunk = whisper.pad_or_trim(audio_chunk)
        mel = whisper.log_mel_spectrogram(audio_chunk).to(model.device)
        options = whisper.DecodingOptions(fp16=False)
        result = whisper.decode(model, mel, options)

        full_transcription += result.text + " "

    return full_transcription

# 3. Save transcription to a text file
def save_transcription_to_file(transcription, filename="transcription.txt"):
    with open(filename, "w") as file:
        file.write(transcription)

# Main execution:
if download_video("https://www.youtube.com/watch?v=qL2GFB3mSs8"):
    extract_audio("downloaded_video.mp4")
    transcription = transcribe_audio_in_chunks("output_audio.mp3")
    if transcription:
        print("Transcription successful!")
        save_transcription_to_file(transcription)
        print(f"Transcription saved to 'transcription.txt'.")
    else:
        print("Transcription failed.")
