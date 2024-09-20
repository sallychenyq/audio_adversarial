# import contextlib
# import scipy.io.wavfile as wavfile
# import numpy as np
# import wave

import torch
import torchaudio

# device = "cuda:1"
# class merge_2channels():
#     """
#     用于将两个已经存在的单声道wav音频，合成一个双声道wav音频
#     根据两个单声道wav音频的不同之处，能够生成左右声道内容不同的双声道wav文件
#     目前还有一些限制，在后续的版本上会进行更新
#     限制：目前要求两个单声道音频的参数都相同（时间、采样频率、采样宽度，压缩格式等）

#     """

#     def __init__(self, merge_framerate, merge_samples, merge_sampwidth=2, merge_channels=2):
#         self.merge_framerate = merge_framerate # 采样率
#         self.merge_samples = merge_samples # 采样点
#         self.merge_sampwidth = merge_sampwidth # 采样宽度，默认2byte
#         self.merge_channels = merge_channels # 声道数，因为是生成双声道音频所以默认2

#     def save_file(self, filepath, data):
#         """
#         保存录音文件
#         :param filepath: 用于保存的路径(str)
#         :param data: 音频数据(2列的ndarray)
#         :return:
#         """
#         wf = wave.open(filepath, 'wb') # 打开目标文件，wb表示以二进制写方式打开，只能写文件，如果文件不存在，创建该文件；如果文件已存在，则覆盖写
#         wf.setnchannels(self.merge_channels) # 设置声道数
#         wf.setsampwidth(self.merge_sampwidth) # 设置采样宽度
#         wf.setframerate(self.merge_framerate) # 设置采样率
#         wf.writeframes(data.tobytes()) # 将data转换为二进制数据写入文件
#         wf.close() # 关闭已打开的文件

#     def merge(self, audio_left, audio_right, merge_audio):
#         """
#         基于两个单声道音频合成一个双声道音频
#         :param audio_left: 左声道音频的路径(str)
#         :param audio_right: 右声道音频的路径(str)
#         :param merge_audio: 合成音频的路径(str)
#         :return:
#         """
                
        
#         l,sl=torchaudio.load(audio_left)
#         r,sr=torchaudio.load(audio_right)
#         torch.cat((l,r),0)
#         # data_left=l.numpy()[0] #     
#         # fs_left, data_lef = wavfile.read(audio_left)  # 读取左声道音频数据
#         # data_right=r.numpy()[0]# fs_right, data_right = wavfile.read(audio_right)  # 读取右声道音频数据
#         # # print(len(data_left)) #.astype(np.int16)
#         # data = np.vstack([data_left, data_right])  # 组合左右声道
#         # # print(data,len(data[0]),len(data[1]))
#         # data = data.T  # 转置（这里我参考了双声道音频读取得到的格式）
#         # self.save_file(merge_audio, data) # 保存

# import librosa, wave
# from scipy import signal
import math
# # 导入音频及绘图显示包
# import librosa.display
# # 导入绘图工作的函数集合
# import matplotlib.pyplot as plt
# import numpy as np
# import pyAudioKits.audio as ak
# import torchaudio


# # 去除趋势化信息
# def polydetrend(x, fs, m):
#     N = len(x)  # 信号长度
#     xtrend = np.zeros(len(x))  # 创建0数组
#     i = np.arange(0, N, 1)  # 时间刻度
#     k = i / fs  # 按信号长度取时间刻度
#     a = np.polyfit(k, x, m)  # 在此m为逼近多项式系数的次数，返回逼近后的信号序列多项式拟合系数值
#     xtrend[i] = np.polyval(a, k)  # 用系数构成趋势项
#     y = x - xtrend  # 从需要去趋势信息的信号中减去趋势项
#     return (y, xtrend)

# def cheb(modified,sr=16000):
#     # y0=yy0.shape[1]  
#     # print(type(y0))
#     y0=modified[0] #.cpu().numpy()
#     times = librosa.get_duration(y=y0, sr=sr)  # 获取音频时长单位为秒
#     # with contextlib.closing(wave.open(wavf, 'r')) as f:
#     #     frames = f.getnframes()
#     #     rate = f.getframerate()
#     #     times = frames / float(rate)
#     # y0, sr = librosa.load('origin.wav', sr=8000, offset=0.0, duration=None)  # 返回音频采样数组及采样率
#     # y0,sr=torchaudio.load(wavf)
#     # print("sr=", sr)
#     # print("times=", times)
#     # 绘图
#     # PointNumbers = int(times * sr) + 1
#     # x1 = np.arange(0, PointNumbers, 1)  # 采样点刻度
#     x2 = np.arange(0, times, 1 / sr)  # 时间刻度
#     plt.figure()
#     plt.xlabel("times")
#     plt.ylabel("amplitude")
#     plt.title('origin.wav', fontsize=12, color='black')
#     plt.plot(x2, y0)
#     plt.show()
    
#     y0, xtrend = polydetrend(y0, sr, 2)
#     plt.figure()
#     plt.xlabel("times")
#     plt.ylabel("amplitude")
#     plt.title('origin.wav_Elimination_trend', fontsize=12, color='black')
#     plt.plot(x2, y0)
#     plt.show()
#     y1 = y0 - np.mean(y0)  # 消除直流分量
#     y1 = (y1 / np.max(np.abs(y1)))#.astype(np.cfloat)  # 幅值归一化
#     plt.figure()
#     plt.xlabel("times")
#     plt.ylabel("amplitude")
#     plt.title('The normalized modified_audio.wav', fontsize=12, color='black')
#     plt.plot(x2, y1)
#     plt.show()

#     # 设计切比雪夫II型滤波器
#     fp = 1000  # 滤波器通带频率
#     fs1 = 750  # 滤波器阻带频率
#     sr2 = sr/2  # 采样频率半周期每样本
#     Wp = fp/sr2  # 通带频率归一化
#     Ws = fs1/sr2  # 阻带频率归一化
#     Rp = 3  # 通带最大损耗
#     Rs = 50  # 阻带最小衰减
#     n, Wn = signal.cheb2ord(Wp, Ws, Rp, Rs, analog=False, fs=None)  #当参数未归一化时，fs=sr
#     b, a = signal.cheby2(n, Rs, Wn, btype='lowpass', analog=False, output='ba', fs=None)
#     # sos = signal.cheby2(n, Rs, Wn, btype='lowpass', analog=False, output='sos', fs=None)
#     sos=signal.butter(n, 0.9, btype='lowpass', analog=False, output='sos', fs=None)
#     # print(n,Wn)
#     w, h = signal.freqz(b, a ,fs=sr)  # 返回的w单位与参数fs相同
#     plt.plot(w, 20 * np.log10(abs(h)))
#     plt.title('Chebyshev Type II frequency response')
#     plt.xlabel('Frequency [Hz]')
#     plt.ylabel('Amplitude [dB]')
#     plt.margins(0, 0.1)  # 去除画布四周白边
#     plt.grid(which='both', axis='both')#网格
#     plt.axvline(Wn*sr2, color='green')  # 绘制竖线，低通截止频率(取归一化)
#     plt.axhline(-Rs, color='green')  # 绘制横线，阻带衰减
#     plt.fill([fs1,  fs1,  sr2,   sr2], [-Rs, 20,  20, -Rs], '0.9', lw=0)  # 阻带约束
#     plt.fill([0,  0, fp, fp], [ -100, -Rp, -Rp,   -100], '0.9', lw=0)  # 通带约束
#     plt.show()

#     # 对语音信号滤波
#     filtered = signal.sosfilt(sos, y1)
#     # mse = np.mean((filtered-y0)**2)
#     # if mse>0:
#     #     psnr=20*np.log10(255/np.sqrt(mse))
#     # else:
#     #     psnr=float('inf')
#     # print(psnr)
#     return filtered
#     # print(type(sos),type(y1),type(sos[0]),type(y1[0]))
#     fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
#     ax1.plot(x2, y1)
#     ax1.set_title('The normalized modified_audio.wav')
#     ax1.axis([0, times, -1, 1])
#     ax1.set_ylabel('Amplitude')
    
#     # # print(filtered,type(filtered))
#     # # power = np.mean(filtered ** 2)  # 平均功率
#     # # scalar = np.sqrt(10 ** (dB / 10) / (power + np.finfo(np.float32).eps))
#     # # filtered *= scalar
#     # audio = ak.Audio(filtered, sr)
#     # audio = audio.amplify(3)
#     # filtered=audio.samples #加响度
    
#     ax2.plot(x2, filtered)
#     ax2.set_title('After 500 Hz cheby2 low-pass filter')
#     ax2.axis([0, times, -1, 1])
#     ax2.set_xlabel('Time [seconds]')
#     ax2.set_ylabel('Amplitude')
#     plt.savefig('filt.png')
#     plt.show()
    

# import io
# from pydub import AudioSegment
# from pydub.playback import play
# def compress_decompress_audio(modified_audio, sample_rate=16000):
#     # 将tensor转换为numpy数组，并确保在CPU上进行操作
#     audio_numpy = modified_audio.cpu().detach().numpy()
#     # print(type(audio_numpy[0][0]))
#     # 创建AudioSegment实例
#     audio_segment = AudioSegment(
#                 audio_numpy.tobytes(), 
#                 frame_rate=sample_rate, 
#                 sample_width=audio_numpy.dtype.itemsize, 
#                 channels=1)  # 从实例到export 57psnr
#     # audio_segment.export("temp.wav", format="wav")
#     torchaudio.save("temp.wav",modified_audio.cpu(),sample_rate)
#     audio_seg=AudioSegment.from_wav("temp.wav") #从wav到export 65psnr
#     #noise*0.6 78psnr
#     # print(audio_seg.max_dBFS)
#     # play(audio_seg)
#     # audio_seg
#     # 压缩为MP3格式
#     # mp3_buffer = io.BytesIO() #
#     audio_seg.export("temp.mp3", format="mp3",bitrate='512k')
#     # print(np.array(audio_seg.get_array_of_samples()).reshape(-1,1))
#     # mp3_buffer.seek(0) #

#     # 解压缩回PCM格式
#     audio_segment_decompressed = AudioSegment.from_file("temp.mp3", format="mp3")
#     # audio_segment_decompressed.export("modified1.wav",format="wav")
#     # decompressed_.audio,s=torchaudio.load("modified1.wav")
#     # a=torch.Tensor(list(audio_segment_decompressed.raw_data))
#     # # torchaudio.save("modified1.wav",a.unsqueeze(0),sample_rate)
#     decompressed_audio = np.frombuffer(audio_segment_decompressed.raw_data, dtype=np.int16) #audio_numpy.dtype

#     # # 确保解压后的音频与输入音频具有相同的长度
#     decompressed_audio = decompressed_audio[:audio_numpy.size]
#     decompressed_audio = decompressed_audio / np.iinfo(np.int16).max  # Normalize to [-1, 1]
#     mse = np.mean((audio_numpy[0]-decompressed_audio)**2)
#     if mse>0:
#         psnr=20*np.log10(255/np.sqrt(mse))
#     else:
#         psnr=float('inf')
#     print(psnr)
#     decompressed_audio = torch.tensor(decompressed_audio, dtype=torch.float32).to(device)
#     decompressed_audio=decompressed_audio.unsqueeze(0)
#     return decompressed_audio

# import datetime
from Models.pytorch_model import PyTorchAudioModel
from Models.utils import load_model, load_decoder
from torch.utils import data
# # def generate_test_data():
# #     dev_url = "test-clean"
# #     pipeline_dataset = torchaudio.datasets.LIBRISPEECH("./data", url=dev_url, download=False)
# #     pipeline_loader = data.DataLoader(dataset=pipeline_dataset,
# #                                       batch_size=1,
# #                                       shuffle=False, )
# #     evaluation_data = []
# #     for i, y in enumerate(pipeline_loader):
# #         sound = y[0][0]
# #         label = y[2][0]
# #         if sound.shape[1] < 32000 or sound.shape[1] > 64000:
# #             continue
# #         evaluation_data.append({"wave": sound, "label": label})
# #     return evaluation_data
# def success_rate_noise_add(sound_data, noise, model):
#     i = 0
#     cer = 0
#     sound_len = len(sound_data)
#     # for sound in sound_data:
#     modified_audio = sound_data + noise[:, :sound_data.shape[1]]
#     # torchaudio.save('modified.wav', modified_audio, 16000)
#     modified_audio = modified_audio.to(device) #
#     predict = model(modified_audio, decode=True)[1] #29种
#     print(model(modified_audio, decode=True)[0][0][0],torch.argmax(predict, dim=-1))
#     # modified_audio1 = sound_data[1]['wave'] + noise[:, :sound_data[1]['wave'].shape[1]]*1.1
#     # torchaudio.save('modified1.wav', modified_audio1, 16000)
#     # modified_audio1 = modified_audio1.to(device) #
#     # predict1 = model(modified_audio1, decode=True)[0][0][0]
#     # print(predict1)
#     # creation_datetime = datetime.datetime.now()-creation_time
#     # print(creation_datetime)

# import torch.nn as nn
# from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
# processor = Wav2Vec2Processor.from_pretrained("/home/hujin/zyj/TUAP/pretrained_model/phon_recognition")
# wav2vec = Wav2Vec2ForCTC.from_pretrained("/home/hujin/zyj/TUAP/pretrained_model/phon_recognition")
# wav2vec = wav2vec.to(device)
# def naturalLoss(noise):
#     # raw_data = torch.FloatTensor(modified_audio)
#             # 音频数据
#     # modified_audio = raw_data.to(device)
#     logits=wav2vec(noise).logits #[N,T,C]
#     softmax=nn.Softmax(dim=2)
#     #p=F.softmax(,dim=0) #
#     # log=F.log_softmax(p,dim=0)
#     # entropy=-(p*log).sum() 
#     # p=torch.tensor([-6.0495e-01, -1.4675e+01, -1.4417e+01])
    
#     predicted_ids = torch.argmax(logits, dim=-1)    # torch.Size([1, 97])
#     text = processor.decode(predicted_ids[0])  # ASR的解码结果
#     print(text)
#     logits = softmax(logits)
#     # loss = torch.empty(0,dtype=torch.float).to(device)
#     loss = 0
#         # ca=Categorical(probs=logits[0][i][4:]) #
#         # print(ca.entropy(),predicted_ids.cpu().numpy()[0][i])
#         # print(logits[0][i].sum(dim=-1))
#     loss += (-torch.log(1 - logits[0][..., :4].sum(dim=-1)).mean())
#     logits = logits[..., 4:]
#     phi = logits[0] / logits[0].sum(dim=-1, keepdim = True)

#     loss += (-phi * torch.log(phi)).sum(dim = -1).mean()

#     print(loss)
#     return loss

# import librosa
# import numpy as np
# from scipy.signal import lfilter, get_window
# import matplotlib.pyplot as plt
# import matplotlib.ticker as mtick


# def func_format(x, pos):
#     return "%d" % (1000 * x)

# class RhythmFeatures:
#     """韵律学特征"""
#     def __init__(self, input_file, sr=None, frame_len=512, n_fft=None, win_step=2 / 3, window="hamming"):
#         """
#         初始化
#         :param input_file: 输入音频文件
#         :param sr: 所输入音频文件的采样率，默认为None
#         :param frame_len: 帧长，默认512个采样点(32ms,16kHz),与窗长相同
#         :param n_fft: FFT窗口的长度，默认与窗长相同
#         :param win_step: 窗移，默认移动2/3，512*2/3=341个采样点(21ms,16kHz)
#         :param window: 窗类型，默认汉明窗
#         """
#         self.input_file = input_file
#         self.frame_len = frame_len  # 帧长，单位采样点数
#         self.wave_data, self.sr = librosa.load(self.input_file, sr=sr)
#         self.window_len = frame_len  # 窗长512
#         if n_fft is None:
#             self.fft_num = self.window_len  # 设置NFFT点数与窗长相等
#         else:
#             self.fft_num = n_fft
#         self.win_step = win_step
#         self.hop_length = round(self.window_len * win_step)  # 重叠部分采样点数设置为窗长的1/3（1/3~1/2）,即帧移(窗移)2/3
#         self.window = window

#     def energy(self):
#         """
#         每帧内所有采样点的幅值平方和作为能量值
#         :return: 每帧能量值，np.ndarray[shape=(1，n_frames), dtype=float64]
#         """
#         mag_spec = np.abs(librosa.stft(self.wave_data, n_fft=self.fft_num, hop_length=self.hop_length,
#                                        win_length=self.frame_len, window=self.window))
#         pow_spec = np.square(mag_spec)
#         energy = np.sum(pow_spec, axis=0)
#         energy = np.where(energy == 0, np.finfo(np.float64).eps, energy)  # 避免能量值为0，防止后续取log出错(eps是取非负的最小值)
#         return energy

# class Spectrogram:
#     """声谱图（语谱图）特征"""
#     def __init__(self, input_file, sr=None, frame_len=512, n_fft=None, win_step=2 / 3, window="hamming", preemph=0.97):
#         """
#         初始化
#         :param input_file: 输入音频文件
#         :param sr: 所输入音频文件的采样率，默认为None
#         :param frame_len: 帧长，默认512个采样点(32ms,16kHz),与窗长相同
#         :param n_fft: FFT窗口的长度，默认与窗长相同
#         :param win_step: 窗移，默认移动2/3，512*2/3=341个采样点(21ms,16kHz)
#         :param window: 窗类型，默认汉明窗
#         :param preemph: 预加重系数,默认0.97
#         """
#         self.input_file = input_file
#         self.wave_data, self.sr = librosa.load(self.input_file, sr=sr)  # 音频全部采样点的归一化数组形式数据
#         self.wave_data = librosa.effects.preemphasis(self.wave_data, coef=preemph)  # 预加重，系数0.97
#         self.window_len = frame_len  # 窗长512
#         if n_fft is None:
#             self.fft_num = self.window_len  # 设置NFFT点数与窗长相等
#         else:
#             self.fft_num = n_fft
#         self.hop_length = round(self.window_len * win_step)  # 重叠部分采样点数设置为窗长的1/3（1/3~1/2）,即帧移(窗移)2/3
#         self.window = window

#     def get_magnitude_spectrogram(self):
#         """
#         获取幅值谱:fft后取绝对值
#         :return: np.ndarray[shape=(1 + n_fft/2, n_frames), dtype=float32]，（257，全部采样点数/(512*2/3)+1）
#         """
#         # 频谱矩阵：行数=1 + n_fft/2=257，列数=帧数n_frames=全部采样点数/(512*2/3)+1（向上取整）
#         # 快速傅里叶变化+汉明窗
#         mag_spec = np.abs(librosa.stft(self.wave_data, n_fft=self.fft_num, hop_length=self.hop_length,
#                                        win_length=self.window_len, window=self.window))
#         return mag_spec


# class QualityFeatures:
#     """声音质量特征（音质）"""

#     def __init__(self, input_file, sr=None, frame_len=512, n_fft=None, win_step=2 / 3, window="hamming"):
#         """
#         初始化
#         :param input_file: 输入音频文件
#         :param sr: 所输入音频文件的采样率，默认为None
#         :param frame_len: 帧长，默认512个采样点(32ms,16kHz),与窗长相同
#         :param n_fft: FFT窗口的长度，默认与窗长相同
#         :param win_step: 窗移，默认移动2/3，512*2/3=341个采样点(21ms,16kHz)
#         :param window: 窗类型，默认汉明窗
#         """
#         self.input_file = input_file
#         self.frame_len = frame_len  # 帧长，单位采样点数
#         self.wave_data, self.sr = librosa.load(self.input_file, sr=sr)
#         self.n_fft = n_fft
#         self.window_len = frame_len  # 窗长512
#         self.win_step = win_step
#         # 重叠部分采样点数设置为窗长的1/3（1/3~1/2）,即帧移(窗移)2/3
#         self.hop_length = round(self.window_len * win_step)
#         self.window = window

#     def formant(self, ts_e=0.01, ts_f_d=200, ts_b_u=2000):
#         """
#         LPC求根法估计每帧前三个共振峰的中心频率及其带宽
#         :param ts_e: 能量阈值：默认当能量超过0.01时认为可能会出现共振峰
#         :param ts_f_d: 共振峰中心频率下阈值：默认当中心频率超过200，小于采样频率一半时认为可能会出现共振峰
#         :param ts_b_u: 共振峰带宽上阈值：默认低于2000时认为可能会出现共振峰
#         :return: F1/F2/F3、B1/B2/B3,每一列为一帧 F1/F2/F3或 B1/B2/B3，np.ndarray[shape=(3, n_frames), dtype=float64]
#         """
#         _data = lfilter([1., 0.83], [1], self.wave_data)  # 预加重0.83：高通滤波器
#         inc_frame = self.hop_length  # 窗移
#         n_frame = int(np.ceil(len(_data) / inc_frame))  # 分帧数
#         n_pad = n_frame * self.window_len - len(_data)  # 末端补零数
#         _data = np.append(_data, np.zeros(n_pad))  # 无法整除则末端补零
#         win = get_window(self.window, self.window_len, fftbins=False)  # 获取窗函数
#         formant_frq = []  # 所有帧组成的第1/2/3共振峰中心频率
#         formant_bw = []  # 所有帧组成的第1/2/3共振峰带宽
#         rym = RhythmFeatures(self.input_file, self.sr,
#                              self.frame_len, self.n_fft, self.win_step, self.window)
#         e = rym.energy()  # 获取每帧能量值
#         e = e / np.max(e)  # 归一化
#         for i in range(n_frame):
#             f_i = _data[i * inc_frame:i * inc_frame + self.window_len]  # 分帧
#             if np.all(f_i == 0):  # 避免上面的末端补零导致值全为0，防止后续求LPC线性预测误差系数出错(eps是取非负的最小值)
#                 f_i[0] = np.finfo(np.float64).eps
#             f_i_win = f_i * win  # 加窗
#             # 获取LPC线性预测误差系数，即滤波器分母多项式，阶数为 预期共振峰数3 *2+2，即想要得到F1-3
#             a = librosa.lpc(f_i_win, order=8)
#             rts = np.roots(a)  # 求LPC返回的预测多项式的根,为共轭复数对
#             # 只保留共轭复数对一半，即虚数部分为+或-的根
#             rts = np.array([r for r in rts if np.imag(r) >= 0])
#             rts = np.where(rts == 0, np.finfo(np.float64).eps,
#                            rts)  # 避免值为0，防止后续取log出错(eps是取非负的最小值)
#             ang = np.arctan2(np.imag(rts), np.real(rts))  # 确定根对应的角(相位）
#             # F(i) = ang(i)/(2*pi*T) = ang(i)*f/(2*pi)
#             # 将以角度表示的rad/sample中的角频率转换为赫兹sample/s
#             frq = ang * (self.sr / (2 * np.pi))
#             indices = np.argsort(frq)  # 获取frq从小到大排序索引
#             frequencies = frq[indices]  # frq从小到大排序
#             # 共振峰的带宽由预测多项式零点到单位圆的距离表示: B(i) = -ln(r(i))/(pi*T) = -ln(abs(rts[i]))*f/pi
#             bandwidths = -(self.sr / np.pi) * np.log(np.abs(rts[indices]))
#             formant_f = []  # F1/F2/F3
#             formant_b = []  # B1/B2/B3
#             if e[i] > ts_e:  # 当能量超过ts_e时认为可能会出现共振峰
#                 # 采用共振峰频率大于ts_f_d小于self.sr/2赫兹，带宽小于ts_b_u赫兹的标准来确定共振峰
#                 for j in range(len(frequencies)):
#                     if (ts_f_d < frequencies[j] < self.sr/2) and (bandwidths[j] < ts_b_u):
#                         formant_f.append(frequencies[j])
#                         formant_b.append(bandwidths[j])
#                 # 只取前三个共振峰
#                 if len(formant_f) < 3:  # 小于3个，则补nan
#                     formant_f += ([np.nan] * (3 - len(formant_f)))
#                 else:  # 否则只取前三个
#                     formant_f = formant_f[0:3]
#                 formant_frq.append(np.array(formant_f))  # 加入帧列表
#                 if len(formant_b) < 3:
#                     formant_b += ([np.nan] * (3 - len(formant_b)))
#                 else:
#                     formant_b = formant_b[0:3]
#                 formant_bw.append(np.array(formant_b))
#             else:  # 能量过小，认为不会出现共振峰，此时赋值为nan
#                 formant_frq.append(np.array([np.nan, np.nan, np.nan]))
#                 formant_bw.append(np.array([np.nan, np.nan, np.nan]))
#         formant_frq = np.array(formant_frq).T
#         formant_bw = np.array(formant_bw).T
#         # print(formant_frq.shape, np.nanmean(formant_frq, axis=1))
#         # print(formant_bw.shape, np.nanmean(formant_bw, axis=1))
#         return formant_frq, formant_bw

#     def plot(self, show=True):
#         """
#         绘制语音波形曲线和log功率谱、共振峰叠加图
#         :param show: 默认最后调用plt.show()，显示图形
#         :return: None
#         """
#         plt.figure(figsize=(8, 6))
#         # 以下绘制波形图
#         plt.subplot(2, 1, 1)
#         plt.title("Wave Form")
#         plt.ylabel("Normalized Amplitude")
#         plt.xticks([])
#         audio_total_time = int(len(self.wave_data) / self.sr * 1000)  # 音频总时间ms
#         plt.xlim(0, audio_total_time)
#         plt.ylim(-1, 1)
#         x = np.linspace(0, audio_total_time, len(self.wave_data))
#         plt.plot(x, self.wave_data, c="b", lw=1)  # 语音波形曲线
#         plt.axhline(y=0, c="pink", ls=":", lw=1)  # Y轴0线
#         # 以下绘制灰度对数功率谱图
#         plt.subplot(2, 1, 2)
#         spec = Spectrogram(self.input_file, self.sr, self.frame_len,
#                            self.n_fft, self.win_step, self.window, 0.83)
#         log_power_spec = librosa.amplitude_to_db(
#             spec.get_magnitude_spectrogram(), ref=np.max)
#         librosa.display.specshow(log_power_spec[:, 1:], sr=self.sr, hop_length=self.hop_length,
#                                  x_axis="s", y_axis="linear", cmap="gray_r")
#         plt.title("Formants on Log-Power Spectrogram")
#         plt.xlabel("Time/ms")
#         plt.ylabel("Frequency/Hz")
#         plt.gca().xaxis.set_major_formatter(mtick.FuncFormatter(func_format))
#         # 以下在灰度对数功率谱图上叠加绘制共振峰点图
#         formant_frq, __ = self.formant()  # 获取每帧共振峰中心频率
#         color_p = {0: ".r", 1: ".y", 2: ".g"}  # 用不同颜色绘制F1-3点，对应红/黄/绿
#         # X轴为对应的时间轴ms 从第0帧中间对应的时间开始，到总时长结束，间距为一帧时长
#         x = np.linspace(0.5 * self.hop_length / self.sr,
#                         audio_total_time / 1000, formant_frq.shape[1])
#         for i in range(formant_frq.shape[0]):  # 依次绘制F1/F2/F3
#             plt.plot(x, formant_frq[i, :], color_p[i], label="F" + str(i + 1))
#         plt.legend(loc="upper right",framealpha=0.5, ncol=3, handletextpad=0.2, columnspacing=0.7)

#         plt.tight_layout()
#         if show:
#             plt.show()
#             plt.savefig('1.png')

import numpy as np
def generate_test_data():
    dev_url = "test-clean"
    pipeline_dataset = torchaudio.datasets.LIBRISPEECH("./data", url=dev_url, download=False)
    pipeline_loader = data.DataLoader(dataset=pipeline_dataset,
                                      batch_size=1,
                                      shuffle=False, )
    evaluation_data = []
    train_cnt = 0
    noi=np.load('out/tuap/poweroff/seed0_150.npy')
    for i, y in enumerate(pipeline_loader):
        sound = y[0][0]
        sound_path = y[1][0]
        label = y[3][0]
        #print(sound.shape[1])
        if sound.shape[1] < 32000 or sound.shape[1] > 64000:
            continue
        modified_audio = sound + noi[:, :sound.shape[1]]
        torchaudio.save('audioTest/noise/seed0.wav', modified_audio, 16000)
        break
        if train_cnt < 150:
            evaluation_data.append({"wave": sound, "label": label, "path":sound_path})
            train_cnt += 1
    # for sample in evaluation_data:
    #     os.system(f"cp {'data/LibriSpeech/'+sample['path'] } {'data/2-4s/'+sample['path']}")
    return evaluation_data

from pydub import AudioSegment
if __name__ == '__main__':
    # a=AudioSegment.from_file('录音.m4a')
    # a=torchaudio.load('data/poweroff.wav')
    
    # au=AudioSegment.from_wav("modified.wav")
    # play(au)
    # evaluation_data = generate_test_data()
#     evaluation_data,s = torchaudio.load('data/LibriSpeech/test-clean/1089/134686/1089-134686-0003.flac')
    device = "cuda:1" #if torch.cuda.is_available() else 'cpu'
#     model_path = './model/librispeech_pretrained_v3.pth.tar'
# # #误导模型
#     model = load_model(model_path=model_path)
#     decoder = load_decoder(labels=model.labels)
#     audio_model = PyTorchAudioModel(model, decoder, device)
#     w,s=torchaudio.load('audioTest/noise/seed0.wav')
#     print(audio_model(w.to(device), decode=True)[0][0][0])
    
#     uni_delta = np.load('0726/seed22imperceptible_150.npy') #/home/hujin/zyj/TUAP/empty/25.npy
#     success_rate_noise_add(evaluation_data, uni_delta, audio_model)
#     # wav,sr=torchaudio.load('addrir.wav')
    
#     w1,s=torchaudio.load('data/LibriSpeech/test-clean/1089/134686/1089-134686-0003.flac')
#     # n1=np.load('/home/hujin/zyj/TUAP/empty/22_rir.npy')
#     n1=np.load('0708/rir_add22.npy')
#     torchaudio.save('audio.wav',torch.from_numpy(n1),s)
#     data,fs=torchaudio.load('audioTest/audio/HELLO BERTIE ANY GOOD IN YOUR MIND.wav')
# # 预处理-预加重
#     # 调用声音质量特征,获取共振峰
#     quality_features = QualityFeatures('audio.wav')
#     fmt_frq, fmt_bw = quality_features.formant()  # 3个共振峰中心频率及其带宽

#     # 绘制波形图、功率谱，并叠加共振峰
#     quality_features.plot(True)
#     print(f"fmt_frq.shape:{fmt_frq.shape}")
    w1,s=torchaudio.load('data/LibriSpeech/test-clean/1089/134686/1089-134686-0003.flac')
    sigdata=np.array(w1)
    noidata = np.load('out/tuap/poweroff/rir_49stage1_l2015_composition.npy') #snr/db:seed21-0.01
    torchaudio.save('audioTest/noise/49+l2015.wav', torch.from_numpy(noidata), 16000)
    audio_seg=AudioSegment.from_wav("audioTest/noise/49+l2015.wav") #从wav到export 65psnr
    print(audio_seg.dBFS)
    # # a,s=librosa.load('origin.wav')
    # print(librosa.amplitude_to_db(librosa.feature.rms(y=a)[0]))
    # naturalLoss(torch.from_numpy(noidata).to(device))
    # w,s=torchaudio.load('audioTest/noise/seed22+natural001.wav')
    # print(AudioSegment.from_wav("audioTest/noise/seed25+natr001.wav").dBFS)
    signal_power = np.sum(sigdata ** 2) / len(sigdata)
    noise_power = np.sum(noidata ** 2) / len(noidata)

    # 计算信噪比
    snr = 10 * math.log10(signal_power / noise_power)
    print(snr)
    # print(f"fmt_frq:{fmt_frq}")

    # cheb(w1.squeeze().numpy(),s) audioTest/audio/STUFF IT INTO YOU HIS BELLY COUNSELLED HIM.wav
    # w2,s=torchaudio.load('modified1.wav')
    # print((w1-w2).numpy()[0][100:115])
    # plt.plot((w1-w2).numpy()[0])
    # plt.savefig('1.png')
    # plt.show()
    # m=torch.FloatTensor(torch.unsqueeze(cheb(wav.squeeze().numpy(),sr),0).float())
    # m=compress_decompress_audio(wav,sr).cpu()
    # torchaudio.save('modified1.wav',m,sr)

    # a = merge_2channels(merge_framerate=16000, merge_samples=2000) # 指定相关参数，目前要求单通道音频参数一致
    # a.merge('origin.wav', 'origin.wav','stereo.wav') # 合成


# # Use the transform in dataset pipeline mode
#     waveform = np.random.random([5, 8])  # 5 samples
#     numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
#     transforms = [audio.Gain(1.2)]
#     numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
#     for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
#         print(item["audio"].shape, item["audio"].dtype)
#         break


#     # Use the transform in eager mode
#     waveform = np.random.random([8])  # 1 sample
#     output = audio.Gain(1.2)(waveform)
#     print(output.shape, output.dtype)

