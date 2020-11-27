import subprocess
import os
AUDIO_DIR='audios'
OUTPUT_DIR='outputs'
VIDEO_DIR='videos'

def extract_audio_from_video(video_file):
    video_fname = os.path.basename(video_file).split('.')[0]
    audio_file = os.path.join(AUDIO_DIR, video_fname + '.mp3')
    subprocess.call('ffmpeg -i {} -f mp3 {}'.format(video_file, audio_file), shell=True)

def merge(audio_file, video_file):
    video_fname = os.path.basename(video_file).split('.')[0]
    out = os.path.join(OUTPUT_DIR , video_file.split('.')[0] + '.mp4')
    subprocess.call('ffmpeg -i {} -i {} -strict -2 -f mp4 {}'.format(video_file, audio_file, out), shell=True)

if __name__ == "__main__":
    extract_audio_from_video('teacher_ma.mp4')
    merge(os.path.join(AUDIO_DIR, 'teacher_ma.mp3'), 'teacher_ma_v2.avi')
    #merge('teacher_ma.mp3', 'teacher_ma.mp4')
