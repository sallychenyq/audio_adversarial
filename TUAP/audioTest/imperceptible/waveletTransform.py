import matplotlib.pyplot as plt
import numpy as np
import pywt
import torchaudio
import wave
import librosa
import math
import torch
# data = np.ones((4, 4), dtype=np.float64)

def transform(path,signal,noise):
    times = librosa.get_duration(path=path)  # 获取音频时长单位为秒
    data, sr = librosa.load(path, sr=8000, offset=0.0, duration=None)  # 返回音频采样数组及采样率
    PointNumbers = int(times * sr) + 1
    x1 = np.arange(0, PointNumbers, 1)  # 采样点刻度
    x2 = np.arange(0, times, 1 / sr)  # 时间刻度
    x3 = np.arange(0, times, 2 / sr)
    # sigdata, sr = librosa.load(signal, sr=8000, offset=0.0, duration=None)  # 返回音频采样数组及采样率
    # noidata, sr = librosa.load(noise, sr=8000, offset=0.0, duration=None)  # 返回音频采样数组及采样率
    sigdata=np.array(signal)
    noidata = np.array(noise)
    signal_power = np.sum(sigdata ** 2) / len(sigdata)
    noise_power = np.sum(noidata ** 2) / len(noidata)

    # 计算信噪比
    snr1 = 10 * math.log10(signal_power / noise_power)
    fig, axs = plt.subplots(2, 2, sharex=True)
    axs[0,0].plot(x2, data)
    axs[0,0].set_title('The normalized LibriSpeech(flac)')
    axs[0,0].axis([0, times, -1, 1])
    axs[0,0].set_ylabel('Amplitude')

    # 绘制波形

    # data,sample =torchaudio.load('../../origin.wav')
    print(data)
    coeffs = pywt.dwt(data, 'haar')
    sig,H1 = pywt.dwt(sigdata, 'haar')
    noi,H2 = pywt.dwt(noidata, 'haar')
    signal_power = np.sum(sig ** 2) / len(sig)
    noise_power = np.sum(noi ** 2) / len(noi)

    # 计算信噪比
    snr2 = 10 * math.log10(signal_power / noise_power)
    print(snr1,snr2)
    # (cA, (cH, cV, cD)) : tuple
    # Approximation, horizontal detail, vertical detail and diagonal
    # detail coefficients respectively
    cA, cH = coeffs #, cV, cD)

    print(cA,cH)
    titles = ['Approximation', ' Horizontal detail',
              'Vertical detail', 'Diagonal detail']
    # plt.imshow(data)
    # plt.colorbar(shrink=0.8)
    # fig = plt.figure(figsize=(12, 3))
    # for i, a in enumerate([cA, cH, cV, cD]):
    #     ax = fig.add_subplot(1, 4, i + 1)
    #     ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    #     ax.set_title(titles[i], fontsize=10)
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #
    # fig.tight_layout()
    # plt.figure(num='ca')
    # plt.plot(data)


    axs[0,1].plot(x3, cA)
    axs[0,1].set_title('After wavelet(ca)')
    axs[0,1].axis([0, times, -1, 1])
    # axs[0,1].set_xlabel('Time [seconds]')
    # axs[0,1].set_ylabel('Amplitude')
    axs[1,0].plot(x3, cH)
    axs[1,0].set_title('After wavelet(ch)')
    axs[1,0].axis([0, times, -1, 1])
    axs[1,0].set_xlabel('Time [seconds]')
    axs[1,0].set_ylabel('Amplitude')
    axs[1,1].plot(x3, cA+cH)
    axs[1,1].set_title('After wavelet(ca+ch)')
    axs[1,1].axis([0, times, -1, 1])
    axs[1,1].set_xlabel('Time [seconds]')
    # axs[1,1].set_ylabel('Amplitude')
    # plt.savefig('wavelet.png')
    plt.show()

if __name__=='__main__':
    transform('../../origin.wav','../../data/LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac','../../origin.wav')
