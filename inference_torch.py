import argparse
import copy
import os
import pickle
import platform
import subprocess
import time
import uuid
import warnings
from glob import glob
import cv2
import numpy as np
from tqdm import tqdm
import onnxruntime as ort
import audio
from utils import img_warp_back_inv_m


warnings.filterwarnings('ignore')
mel_step_size = 16


def load_model(model_path):

    sess_opt = ort.SessionOptions()
    sess_opt.intra_op_num_threads = 8
    sess = ort.InferenceSession(model_path, sess_options=sess_opt, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    # 打印当前使用的Execution Provider
    print(f"Execution Providers: {sess.get_providers()}")
    return sess


def post_process_frame(params):
    p, f, c, af, inv_m = params
    y1, y2, x1, x2 = c
    p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
    af[y1:y2, x1:x2] = p

    f = img_warp_back_inv_m(af, f, inv_m)

    return f


class Runner(object):
    def __init__(self, args):
        self.batch_size = args.batch_size

        self.img_size = (256, 256)
        self.fps = 25

        self.a_alpha = 1.25
        self.audio_smooth = args.audio_smooth

        model_a_path = 'weights/wav2lip/model_1.onnx'
        model_g_path = 'weights/wav2lip/model_2.onnx'

        self.model_a = load_model(model_a_path)
        if self.model_a is None:
            print("Failed to load model_a from the provided path.")
        self.model_g = load_model(model_g_path)
        if self.model_g is None:
            print("Failed to load model_g from the provided path.")
        print("Model loaded")

        self.avatars = {}

        avatar_info_file_list = glob(f'app/assets/*.pkl')

        for avatar_info_file in avatar_info_file_list:
            try:
                with open(avatar_info_file, "rb") as f:
                    avatar_info = pickle.load(f)
                    avatar_name, ext = os.path.splitext(os.path.basename(avatar_info_file))
                    
                    self.avatars[avatar_name] = avatar_info
            except:
                print("Error {}".format(avatar_info_file))
        print('avatar_list: ', list(self.avatars.keys()))

    def get_input_imginfo_by_index(self, idx, avatar):
        return avatar['frame_info_list'][idx]

    def get_input_mel_by_index(self, index, wav_mel):
        T = 5
        mel_idx_multiplier = 80. / self.fps
        start_idx = int((index - (T - 1) // 2) * mel_idx_multiplier)
        if start_idx < 0:
            start_idx = 0
        if start_idx + mel_step_size > len(wav_mel[0]):
            start_idx = len(wav_mel[0]) - mel_step_size
        mel = wav_mel[:, start_idx: start_idx + mel_step_size]
        return mel

    def get_intput_by_index(self, index, wav_mel, avatar):
        mel = self.get_input_mel_by_index(index, wav_mel)

        frame_num = avatar['frame_num']
        idx = index % frame_num
        idx = idx if index // frame_num % 2 == 0 else frame_num - idx - 1

        input_dict = {'mel': mel}
        input_imginfo = self.get_input_imginfo_by_index(idx, avatar)
        input_dict.update(copy.deepcopy(input_imginfo))
        return input_dict

    def run(self, audio_path, avatar_name, outfile=None):
        if outfile is None:
            aname = os.path.basename(audio_path)
            aname = os.path.splitext(aname)[0]
            outfile = f"results/result_voice_{avatar_name}_{aname}.mp4"
        temp_dir = "/tmp/"
        # 生成一个唯一的文件名，这里使用UUID以保证较高的唯一性
        unique_filename = f"{uuid.uuid4()}.wav"
        # 拼接临时文件的完整路径
        temp_file_path = os.path.join(temp_dir, unique_filename)
        temp_audio_file = open(temp_file_path, 'wb')
        
        if not audio_path.endswith('.wav'):
            print('Extracting raw audio...')
            command = 'ffmpeg -y -i {} -strict -2 {}'.format(audio_path, temp_audio_file.name)
            subprocess.call(command, shell=True)
            wav_path = temp_audio_file.name
        else:
            wav_path = audio_path

        avatar = self.avatars[avatar_name]
        fps = avatar['fps']

        wav = audio.load_wav(wav_path, 16000)
        wav_mel = audio.melspectrogram(wav)
        mel_idx_multiplier = 80. / fps
        gen_frame_num = int(len(wav_mel[0]) / mel_idx_multiplier)

        batch_size = self.batch_size

        frame_h, frame_w = avatar['frame_h'], avatar['frame_w']

        unique_filename = f"{uuid.uuid4()}.mp4"
        temp_file_path = os.path.join(temp_dir, unique_filename)
        temp_face_file = open(temp_file_path, 'wb')
        out = cv2.VideoWriter(temp_face_file.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_w, frame_h))

        from collections import defaultdict
        batch_data = defaultdict(list)

        start_infer = time.time()
        pure_model_time = 0.0

        for i in tqdm(range(gen_frame_num)):
            input_data = self.get_intput_by_index(i, wav_mel, avatar)
            # 组batch
            for k, v in input_data.items():
                batch_data[k + '_batch'].append(v)

            if len(batch_data.get('mel_batch')) == batch_size or i == gen_frame_num - 1:
                infer_size = len(batch_data['mel_batch'])

                img_batch = batch_data['img_batch']
                mel_batch = batch_data['mel_batch']
                frames = batch_data['frame_batch']
                coords = batch_data['coords_batch']
                align_frames = batch_data['align_frame_batch']
                ms = batch_data['m_batch']
                inv_ms = batch_data['inv_m_batch']

                if self.audio_smooth:
                    mel_batch.insert(0, self.get_input_mel_by_index(max(0, i - infer_size), wav_mel))
                    mel_batch.append(self.get_input_mel_by_index(min(i + 1, gen_frame_num - 1), wav_mel))

                onnxruntime_inputs = {"audio_seqs__0": mel_batch, }
                onnxruntime_names = [output.name for output in self.model_a.get_outputs()]
                embeddings = self.model_a.run(onnxruntime_names, onnxruntime_inputs)[0]
                if self.audio_smooth:
                    embeddings = 0.2 * embeddings[:-2] + 0.7 * embeddings[1:-1] + 0.1 * embeddings[2:]

                onnxruntime_inputs = {"audio_embedings__0": embeddings, "img_seqs__1": img_batch}
                onnxruntime_names = [output.name for output in self.model_g.get_outputs()]

                start_model = time.time()
                onnxruntime_output = self.model_g.run(onnxruntime_names, onnxruntime_inputs)[0]
                end_model = time.time()
                pure_model_time += (end_model - start_model)

                pred = onnxruntime_output

                for p, f, c, af, inv_m in zip(pred, frames, coords, align_frames, inv_ms):
                    f = post_process_frame((p, f, c, af, inv_m))
                    out.write(f)

                batch_data.clear()

        end_infer = time.time()
        latency_per_frame = (end_infer - start_infer) * 1000 / gen_frame_num
        latency_model = pure_model_time * 1000 / gen_frame_num
        print(f"每一帧延迟: {latency_per_frame:.3f} ms")
        print(f"每一帧延迟，纯模型: {latency_model:.3f} ms")

        out.release()

        command = 'ffmpeg -y -i "{}" -i "{}" -strict -2 -q:v 1 "{}"'.format(wav_path, temp_face_file.name, outfile)
        subprocess.call(command, shell=platform.system() != 'Windows')

        temp_face_file.close()
        temp_audio_file.close()
        return outfile


if __name__ == '__main__':
    t0 = time.time()
    parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')
    parser.add_argument('--batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=4)
    parser.add_argument('--audio_smooth', default=True, action='store_true', help='smoothing audio embedding')
    args = parser.parse_args()
    runner = Runner(args)
    t0 = time.time()
    audio_path = 'audio_pathudio'
    avatar_name = 'avatar_name'
    runner.run(audio, avatar_name)
    print('total infer time: {}'.format(time.time() - t0))

    print('total time: {}'.format(time.time() - t0))
