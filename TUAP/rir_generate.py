import torchaudio
import torch
import os
import torch.nn.functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def get_rir_sample(path, resample=None, processed=False):
    rir_raw, sample_rate = torchaudio.load(path)
    if rir_raw.shape[0] > 1:
        rir_raw = rir_raw[0].unsqueeze(0)
    if not processed:
        return rir_raw, sample_rate
    rir = rir_raw[:, int(sample_rate * 1.01):int(sample_rate * 1.3)]
    rir = rir / torch.norm(rir, p=2)
    rir = torch.flip(rir, [1])
    return rir, sample_rate

def rir_transform(waveform, sample_rate, rir_path):
    # with torch.no_grad():
    sample_rate = sample_rate
    rir_raw, _ = get_rir_sample(path=rir_path, resample=sample_rate)
    # 提取主脉冲，归一化信号功率，然后翻转时间轴。
    # rir = rir_raw[:, int(sample_rate * 1.01):int(sample_rate * 1.3)]
    # rir = rir_raw[:, int(sample_rate * 0.13):int(sample_rate * 0.44)]
    rir = rir_raw
    # 二阶范数就是欧几里得距离
    # rir = rir / torch.norm(rir, p=2)
    # flip 函数翻转 tensor 数组
    rir = torch.flip(rir, [1]).to(device)
    speech_ = F.pad(waveform, (rir.shape[1] - 1, 0))
    augmented = F.conv1d(speech_[None, ...], rir[None, ...])[0]

    # print(np.max(augmented[0]))
    return augmented

def generate_rir_list():
    rir_list = []
    source = 'rir_list'
    speech, sample = torchaudio.load('origin.wav')
    speech = speech.to(device)

    for rir in os.listdir(source):
        if os.path.splitext(rir)[1] == ".wav":
            # and os.path.splitext(rir)[0][:28] == 'RVB2014_type2_noise_simroom1':
            side = os.path.join(source, rir)
            try:
                # x = rir_transform(speech, 16000, side)
                rir_list.append(side)
                # if len(rir_list) > 5:
                #     break
            except:
                pass
    return rir_list