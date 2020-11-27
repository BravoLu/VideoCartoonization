import torch
import numpy as np
from skimage import img_as_ubyte
import cv2
import os
import argparse

from network import SimpleGenerator
from postprocess import *

def preprocess(raw_image):
	image = raw_image/127.5 - 1
	image = image.transpose(2, 0, 1)
	image = torch.tensor(image).unsqueeze(0)
	return image

def image_cartoonize(input, cuda=False):
	model = SimpleGenerator()
	model.load_state_dict(torch.load('weight.pth'))
	model.eval()
	if cuda:
		model.cuda()
		input = input.cuda()
	output = model(input.float()).cpu()
	output = output.squeeze(0).detach().numpy()
	output = output.transpose(1, 2, 0)
	output = (output + 1) * 127.5
	output = np.clip(output, 0, 255).astype(np.uint8)
	return output

def video_cartoonize(video_path):
	cap = cv2.VideoCapture(video_path)
	fourcc = cv2.VideoWriter_fourcc(*'MJPG')
	video_fname = os.path.basename(video_path).split('.')[0]
	out = cv2.VideoWriter('videos/{}.avi'.format(video_fname), fourcc, 25, (256, 256), True)
	cnt = 0
	with torch.no_grad():
		while(1):
			ret, frame = cap.read()
			if ret:
				frame = frame[180:180+256, 180:180+256]
				print('Frame #{}'.format(cnt))
				cnt+=1
				preprocessed_image = preprocess(frame)
				output = image_cartoonize(preprocessed_image, cuda=True)
				out.write(img_as_ubyte(output))
			else:
				break
	cap.release()
	out.release()
	print('Cartoonization finished!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Video Cartoonization')
    parser.add_argument('--video', type=str, default='videos/teacher_ma.mp4')
    args = parser.parse_args()
    video_cartoonize(args.video)
    video_fname = os.path.basename(args.video).split('.')[0]
    extract_audio_from_video(args.video)
    merge(os.path.join(AUDIO_DIR, video_fname+'.mp3'), os.path.join(VIDEO_DIR, video_fname+'.avi'))

