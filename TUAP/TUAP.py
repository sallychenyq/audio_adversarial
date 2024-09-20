from torch.utils import data
import torchaudio
from tqdm import tqdm
import random
import torch
import numpy as np
import torchaudio
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
from rir_generate import generate_rir_list
from Models.utils import load_decoder,load_model
import editdistance
from Models.pytorch_model import PyTorchAudioModel
from utils.deepspeech.model import DeepSpeech
from pathlib import Path
import scipy.io.wavfile as wavfile
from g2p_en import G2p
import Levenshtein
device = "cuda" if torch.cuda.is_available() else 'cpu'
model_path = './model/librispeech_pretrained_v3.pth.tar'
#误导模型
model = load_model(model_path=model_path)
decoder = load_decoder(labels=model.labels)
#训练模型
audio_model = PyTorchAudioModel(model, decoder, device)
g2p = G2p()
target_labels = [
    'power off',
    'open the door',
    'turn off lights',
    'use airplane mode',
    'visit malicious dot com'
]

phone_map=np.load('./generate_phonmap/phone_map.npy',allow_pickle=True)
phone_map = dict(phone_map)
def product(df,modifiedlist,targetlist):
    if len(modifiedlist)>len(targetlist):
        modifiedlist=modifiedlist[0:len(targetlist)]
    #print(modifiedlist,targetlist)
    newlist=[]
    for i in targetlist:
        if i=='AA0' or i=='AA1' or i=='AA2' or i=='AA':
            newlist.append('AA')
        elif i=='AE0' or i=='AE1' or i=='AE2' or i=='AE':
            newlist.append('AE')
        elif i=='AH1' or i=='AH2' or i=='AH':
            newlist.append('AH')
        elif i=='AO0' or i=='AO1' or i=='AO2' or i=='AO':
            newlist.append('AO')
        elif i=='AW0' or i=='AW1' or i=='AW2' or i=='AW':
            newlist.append('AW')
        elif i=='AY0' or i=='AY1' or i=='AY2' or i=='AY':
            newlist.append('AY')
        elif i=='EH0' or i=='EH1' or i=='EH2' or i=='EH':
            newlist.append('EH')
        elif i=='ER0' or i=='ER1' or i=='ER2' or i=='ER':
            newlist.append('ER')
        elif i=='EY0' or i=='EY1' or i=='EY2'or i=='EY':
            newlist.append('EY')
        elif i=='HH0' or i=='HH1' or i=='HH2' or i=='HH':
            newlist.append('HH')
        elif i=='IH0' or i=='IH1' or i=='IH2' or i=='IH':
            newlist.append('IH')
        elif i=='IY0' or i=='IY1' or i=='IY2' or i=='IY':
            newlist.append('IY')
        elif i=='OW0' or i=='OW1' or i=='OW2' or i=='OW':
            newlist.append('OW')
        elif i=='OY0' or i=='OY1' or i=='OY2' or i=='OY':
            newlist.append('OY')
        elif i=='UH0' or i=='UH1' or i=='UH2' or i=='UH':
            newlist.append('UH')
        elif i=='UW0' or i=='UW1' or i=='UW2' or i=='UW':
            newlist.append('UW')
        elif i==' ' or i == '\'':
            continue
        else : newlist.append(i)
    newlist1=[]
    for i in modifiedlist:
        if i=='AA0' or i=='AA1' or i=='AA2'or i=='AA':
            newlist1.append('AA')
        elif i=='AE0' or i=='AE1' or i=='AE2'or i=='AE':
            newlist1.append('AE')
        elif i=='AH1' or i=='AH2'or i=='AH':
            newlist1.append('AH')
        elif i=='AO0' or i=='AO1' or i=='AO2'or i== 'AO':
            newlist1.append('AO')
        elif i=='AW0' or i=='AW1' or i=='AW2' or i=='AW':
            newlist1.append('AW')
        elif i=='AY0' or i=='AY1' or i=='AY2' or i=='AY':
            newlist1.append('AY')
        elif i=='EH0' or i=='EH1' or i=='EH2' or i=='EH':
            newlist1.append('EH')
        elif i=='ER0' or i=='ER1' or i=='ER2' or i=='ER':
            newlist1.append('ER')
        elif i=='EY0' or i=='EY1' or i=='EY2'or i=='EY':
            newlist1.append('EY')
        elif i=='HH0' or i=='HH1' or i=='HH2'or i=='HH':
            newlist1.append('HH')
        elif i=='IH0' or i=='IH1' or i=='IH2' or i=='IH':
            newlist1.append('IH')
        elif i=='IY0' or i=='IY1' or i=='IY2' or i=='IY':
            newlist1.append('IY')
        elif i=='OW0' or i=='OW1' or i=='OW2' or i=='OW':
            newlist1.append('OW')
        elif i=='OY0' or i=='OY1' or i=='OY2'  or i=='OY':
            newlist1.append('OY')
        elif i=='UH0' or i=='UH1' or i=='UH2' or i=='UH':
            newlist1.append('UH')
        elif i=='UW0' or i=='UW1' or i=='UW2' or i=='UW':
            newlist1.append('UW')
        elif i==' ' or i == '\'':
            continue
        else : newlist1.append(i)
    sum=0
    #print(newlist)
    #print(newlist1)
    for i,j in zip(newlist1,newlist):
        if i == j:
            sum += 1 
            continue
        if j not in df[i].keys():
            sum += df[j][i]
            continue
        sum+=df[i][j]
    return (sum/len(targetlist))

def phoneLoss(modifySentence,target):
    modifyList = g2p(modifySentence)
    targetlist = g2p(target)
    #distance=product(phone_map,modifyList,targetlist)
    edit_ops = Levenshtein.editops(modifyList, targetlist)
    distance1 = len(edit_ops)
    similarity = (distance1 / max(len(modifyList), len(targetlist)))
    # similarity1 = (distance / max(len(modifyList), len(targetlist)))
    loss = similarity  # 将相似度转换为损失
    #loss1 = (1 - distance)
    return loss #+ loss1


def success_rate_noise_add(sound_data, noise, model, label):
    i = 0
    cer = 0
    sound_len = len(sound_data)
    for sound in sound_data:
        modified_audio = sound['wave'] + noise[:, :sound['wave'].shape[1]]
        modified_audio = modified_audio.to(device)
        predict = model(modified_audio, decode=True)[0][0][0]
        if predict == label:
            i += 1
        pred_cer = cel("POWER OFF", predict)
        cer += pred_cer
    print('now asr is ' + str(i / sound_len))
    print('now cer is ' + str(cer / sound_len))

def target_sentence_to_label(sentence, labels="_'ABCDEFGHIJKLMNOPQRSTUVWXYZ "):
    out = []
    for word in sentence:
        out.append(labels.index(word))
    return torch.IntTensor(out)

def _max_norm_loss(delta, max_norm_clip):
    max_norm_loss = torch.relu(torch.abs(delta) - max_norm_clip)
    max_norm_loss = torch.sum(max_norm_loss)
    return max_norm_loss

def check_loss(loss, loss_value):
    """
    Check that warp-ctc loss is valid and will not break training
    :return: Return if loss is valid, and the error in case it is not
    """
    loss_valid = True
    error = ''
    if loss_value == float("inf") or loss_value == float("-inf"):
        loss_valid = False
        error = "WARNING: received an inf loss"
    elif torch.isnan(loss).sum() > 0:
        loss_valid = False
        error = 'WARNING: received a nan loss, setting loss value to 0'
    elif loss_value < 0:
        loss_valid = False
        error = "WARNING: received a negative loss"
    return loss_valid, error

# 生成噪声
def generateTUAP(audio_set, target_phrase, out_delta_len, delta_val, audio_model):
    # 优化目标 （某种转向量操作）
    targets = target_sentence_to_label(target_phrase)
    targets = targets.view(1, -1).to(device).detach()
    target_lengths = torch.IntTensor([targets.shape[1]]).view(1, -1)
    # 初始化
    lr = 1e-3  # 学习率
    if delta_val is not None:
        delta = torch.FloatTensor(delta_val)
        delta = delta.to(device)
    else:
        delta = torch.zeros(out_delta_len, dtype=torch.float)
        delta = delta.to(device).unsqueeze(0)
    delta.requires_grad = True
    optimizer = torch.optim.Adam(params=[delta], lr=lr)  # Adam优化器
    criterion = nn.CTCLoss()  # ctc损失
    softmax = nn.Softmax(dim=2)

    delta_clipped = None
    valid_delta_val = None

    model_loss_alpha = 1.0  # 超参数 alpha

    max_norm_clip = 1.0  # 音频上下限
    max_norm_alpha = 0.1

    success = False
    uni_delta = None
    total_iters = 0  # 总迭代次数
    max_iters = 5000  # 最大迭代次数

    for cur_iter in range(max_iters):
        # shuffle the audio set each time
        succ_num = 0
        # 生成随机index 并随机取数据进行循环
        rand_idxes = np.random.permutation(len(audio_set))
        # print(rand_idxes)
        for cur_idx, audio_idx in enumerate(rand_idxes):
            # print(audio_idx)
            audio = audio_set[audio_idx]['wave']
            raw_data = torch.FloatTensor(audio)
            # 音频数据
            raw_data = raw_data.to(device)

            # 向 delta 添加小噪声以避免 nan 梯度
            noise = torch.normal(mean=0, std=1e-4, size=delta.shape).to(device)
            # noise = noise.unsqueeze(0)
            # 叠加delta和noise并限制到 【-1，1】之间  证明是使用torchaudio.load读取数据的
            delta_clipped = torch.clamp(delta + noise, -max_norm_clip, max_norm_clip)
            # 叠加原音频和噪声
            # input_specs, input_sizes = _apply_noise_to_raw_data(raw_data, delta_clipped)
            modified_audio = raw_data + delta_clipped[:, :raw_data.shape[1]]
            # modified_audio为叠加后的音频
            modified_audio = torch.clamp(modified_audio, -1.0, 1.0)
            # 运行模型预测
            trans, out, output_sizes = audio_model(modified_audio, decode=True)
            out = softmax(out)
            out = out.transpose(0, 1).log()
            model_loss = criterion(out, targets, output_sizes, target_lengths)
            phon_loss = phoneLoss(trans[0][0],target_phrase)
            # 计算损失
            max_norm_loss = _max_norm_loss(delta, max_norm_clip)
            loss = model_loss * model_loss_alpha + max_norm_loss * max_norm_alpha + phon_loss
            # 检查以确保计算了综合损失
            valid_loss, error = check_loss(loss, loss.item())
            if not valid_loss:
                loss.item()
            
            #if (not valid_loss) and ((loss.item() < 0) is False):
                #print(f"*** WARNING: the COMBINED loss is invalid. Reset delta and continue. ERROR = {error}. ***")
                #reset the delta and continue
            #     # assert np.count_nonzero(np.isnan(valid_delta_val)) == 0
            #     # delta, optimizer = _init_delta(None, valid_delta_val)
            #     continue

            # 如果优化的结果已经成功，报告成功1次
            if trans[0][0] == target_phrase:
                succ_num += 1

            print(
                "univeral --- iter = {}/{}; audio = {}({})/{}; succ_rate={}/{}; trans = {}; loss = {:.2f}; target = {}".format(
                    cur_iter + 1, max_iters, cur_idx + 1, audio_idx + 1, len(audio_set),
                    succ_num, len(audio_set),
                    trans, loss.item(), target_phrase
                ))

            # 在向后传递之前，我们保存有效的 delta vals
            valid_delta_val = delta.cpu().detach().numpy()
            assert np.count_nonzero(np.isnan(valid_delta_val)) == 0

            # 对输入数据进行梯度下降并继续
            audio_model.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 如果报告次数和数据集长度相等，报告成功
        if succ_num == len(audio_set):
            success = True
            total_iters = cur_iter + 1
            uni_delta = delta_clipped.cpu().detach().numpy()
            break

    if success is False:
        return None, None

    #assert np.count_nonzero(np.isnan(uni_delta)) == 0
    #assert len(uni_delta.shape) == 1 and uni_delta.size == out_delta_len
    # uni—delta是结果，total——iter是计算迭代的次数
    return uni_delta, total_iters

def run_stage_1(src_audio_lst, target_phrase, out_delta_len, out_delta_path):
    # 在迭代中逐渐使用更多的源音频
    uni_delta = None
    for src_used_num in range(1, len(src_audio_lst) + 1):
        audio_set = src_audio_lst[:src_used_num]  # 取子集
        # 加载现有的 UAP（如果存在）
        delta_save_path = out_delta_path.with_name(out_delta_path.stem + f"_{src_used_num}.npy")
        if delta_save_path.exists():
            print(f"Load existing delta for src_used_num = {src_used_num}")
            uni_delta = np.load(delta_save_path)
            continue
        uni_delta, total_iters = generateTUAP(audio_set, target_phrase, out_delta_len, uni_delta, audio_model)
        if uni_delta is None:
            return src_used_num - 1
        # save the universal perturbations and call back
        np.save(delta_save_path, uni_delta)
    return len(src_audio_lst)  # all src audio have been used

def generate_100_data():
    dev_url = "dev-clean"
    pipeline_dataset = torchaudio.datasets.LIBRISPEECH("./data", url=dev_url, download=True)
    pipeline_loader = data.DataLoader(dataset=pipeline_dataset,
                                      batch_size=1,
                                      shuffle=False, )
    data_list = []
    for i, y in enumerate(tqdm(pipeline_loader)):
        # print(y)
        sound = y[0][0]
        label = y[2][0]
        if sound.shape[1] < 32000 or sound.shape[1] > 64000:
            continue
        else:
            data_list.append({"wave": sound, "label": label})
    sample_list = random.sample(data_list, 100)
    np.save('out/test_data_100_2s-4s.npy', sample_list)
    
def generate_data():
    dev_url = "dev-clean"
    pipeline_dataset = torchaudio.datasets.LIBRISPEECH("./data", url=dev_url, download=False)
    pipeline_loader = data.DataLoader(dataset=pipeline_dataset,
                                      batch_size=1,
                                      shuffle=False, )
    train_data = []
    # test_data = []
    evaluation_data = []
    index = 0
    train_cnt = 0
    evaluation_cnt = 0
    for i, y in enumerate(pipeline_loader):
        sound = y[0][0]
        label = y[2][0]
        if sound.shape[1] < 32000 or sound.shape[1] > 64000:
            continue
        if train_cnt < 150 :
            train_data.append({"wave": sound, "label": label})
            train_cnt += 1
        else: break
        #index += 1
    # return train_data, test_data, evaluation_data
    return train_data, evaluation_data
def generate_test_data():
    dev_url = "test-clean"
    pipeline_dataset = torchaudio.datasets.LIBRISPEECH("./data", url=dev_url, download=False)
    pipeline_loader = data.DataLoader(dataset=pipeline_dataset,
                                      batch_size=1,
                                      shuffle=False, )
    evaluation_data = []
    for i, y in enumerate(pipeline_loader):
        sound = y[0][0]
        label = y[2][0]
        if sound.shape[1] < 32000 or sound.shape[1] > 64000:
            continue
        evaluation_data.append({"wave": sound, "label": label})
    return evaluation_data

def cel(origin, noise):
    return editdistance.eval(origin, noise) / len(origin)

def success_rate_origin(sound_data, model):
    cer = 0.0
    i = 0
    for sound in sound_data:
        i += 1
        predict = model(sound['wave'].to(device), decode=True)[0][0][0]
        pred_cer = cel(sound['label'], predict)
        cer += pred_cer
    print('now cer is ' + str(cer / i))


def torch_conv_audio(audio, impulse_res):
    audio = audio[0]
    impulse_res = impulse_res[0]
    audio_len = audio.size(0)
    impulse_len = impulse_res.size(0)

    conv_length = audio_len + impulse_len - 1
    nfft = np.power(2.0, np.ceil(np.log2(conv_length)))
    nfft = int(nfft)

    zero_array = np.zeros(nfft)
    audio_extended = torch.FloatTensor(zero_array).to(device)
    audio_extended[:audio_len] = audio

    impulse_res_extended = torch.FloatTensor(zero_array).to(device)
    impulse_res_extended[:impulse_len] = impulse_res

    audio_fft = torch.fft.rfft(audio_extended)
    ir_fft = torch.fft.rfft(impulse_res_extended)
    out_fft = audio_fft * ir_fft

    out_conved = torch.fft.irfft(out_fft)
    # get rid of boundary effect by removing (M-1)/2 samples from each side
    # !!! length of IR must be odd
    boundary_eff_len = int((impulse_len - 1) / 2)  # length of boundary effect: (M-1)/2 samples from each side

    out_conved = out_conved[boundary_eff_len: boundary_eff_len + audio_len]
    out_conved = torch.clamp(out_conved, -1.0, 1.0)

    return out_conved.unsqueeze(0)

def success_rate_rir(sound_data, model, rir):
    cer = 0.0
    i = 0
    for sound in sound_data:
        i += 1
        # a = rir_transform(sound['wave'].to(device), 16000, rir)
        rir_sound, sample = torchaudio.load(rir)
        a = torch_conv_audio(sound['wave'], rir_sound)
        predict = model(a, decode=True)[0][0][0]
        pred_cer = cel(sound['label'], predict)
        cer += pred_cer
    print('now cer is ' + str(cer / i))




def _do_minimize_maxnorm(src_audio_lst, target_phrase, init_delta_val, out_delta_path, audio_model,
                         rir_lst=None, init_max_norm=None):
    # 优化目标转向量
    targets = target_sentence_to_label(target_phrase)
    targets = targets.view(1, -1).to(device).detach()
    target_lengths = torch.IntTensor([targets.shape[1]]).view(1, -1)

    # 加载stage1噪声，初始化优化器
    lr = 1e-3
    delta = torch.FloatTensor(init_delta_val).to(device)
    delta.requires_grad = True
    optimizer = torch.optim.Adam(params=[delta], lr=lr)

    criterion = nn.CTCLoss()  # ctc损失
    softmax = nn.Softmax(dim=2)

    delta_val_saved = None
    rlt_norm = None

    delta_clipped = None
    valid_delta_val = None

    max_failed_iters = 30 if rir_lst is None else 60
    failed_iter = 0
    least_success_rate = 0.8

    model_loss_alpha = 1.0
    max_norm_alpha = 0.1

    max_norm_clip = 1.0
    max_norm_multi = 0.8

    # 定义最大噪声扰动级别
    if init_max_norm is not None:
        print(f"********* using init_max_norm = {init_max_norm} ***********")
        max_norm_clip = init_max_norm

    while True:
        # shuffle the audio set each time
        succ_num = 0
        rand_idxes = np.random.permutation(len(src_audio_lst))
        for cur_idx, audio_idx in enumerate(rand_idxes):
            audio = src_audio_lst[audio_idx]['wave']
            raw_data = torch.FloatTensor(audio)
            raw_data = raw_data.to(device)

            # 随机选择一个rir
            rand_rir = None
            rand_rir_idx = None
            if rir_lst is not None:
                rand_rir_idx = np.random.randint(len(rir_lst))
                # 随机拿到一个音频path
                rand_rir = rir_lst[rand_rir_idx]

            noise = torch.normal(mean=0, std=1e-4, size=delta.shape).to(device)
            # 叠加delta和noise并限制到 【-1，1】之间  证明是使用torchaudio.load读取数据的
            delta_clipped = torch.clamp(delta + noise, -max_norm_clip, max_norm_clip)
            # 进行rir变换
            if rand_rir is not None:
                rir_sound, rir_sample = torchaudio.load(rand_rir)
                raw_data = torch_conv_audio(raw_data.detach(), rir_sound)
                # plot_waveform(modified_audio.detach().cpu(),16000)
            # 叠加原音频和噪声
            modified_audio = raw_data + delta_clipped[:, :raw_data.shape[1]]
            # modified_audio为叠加后的音频
            modified_audio = torch.clamp(modified_audio, -1.0, 1.0)
            # 运行模型预测
            trans, out, output_sizes = audio_model(modified_audio, decode=True)
            out = softmax(out)
            out = out.transpose(0, 1).log()
            model_loss = criterion(out, targets, output_sizes, target_lengths)
            # 求损失
            max_norm_loss = _max_norm_loss(delta, max_norm_clip)
            phon_loss = phoneLoss(trans[0][0],target_phrase)
            loss = model_loss * model_loss_alpha + max_norm_loss * max_norm_alpha + phon_loss*0.3
            print("model loss",model_loss)
            print("max_norm_loss",max_norm_loss)
            print("phon_loss",phon_loss)
            # 优化成功，报告
            if trans[0][0] == target_phrase:
                succ_num += 1

            print(
                "maxnorm--- iter = {}/{}; max_norm={:.3f}; audio = {}({})/{}; rir_idx = {}; succ_rate={}/{}; trans = {}; loss = {:.2f}; target = {}".format(
                    failed_iter + 1, max_failed_iters, max_norm_clip, cur_idx + 1, audio_idx + 1, len(src_audio_lst),
                    rand_rir_idx, succ_num, len(src_audio_lst),
                    trans, loss.item(), target_phrase
                ))

            # do gradient decent on the input data and continue
            audio_model.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            delta.grad.nan_to_num_(nan=0)
            optimizer.step()

        # delta_val_saved = delta_clipped.cpu().detach().numpy()
        # np.save(out_delta_path, delta_val_saved)

        # 如果成功数量大于预设成功率 衰减噪声边界*0.8
        if succ_num > len(src_audio_lst) * least_success_rate:
            # save the clipped delta and norm
            delta_val_saved = delta_clipped.cpu().detach().numpy()
            # assert np.count_nonzero(np.isnan(delta_val_saved)) == 0
            rlt_norm = max_norm_clip
            print("************** save working delta, max_norm and lower max norm ***************")
            np.save(out_delta_path, delta_val_saved)
            np.save(out_delta_path.with_name(out_delta_path.stem + "_maxnorm_val.npy"), max_norm_clip)
            max_norm_clip *= max_norm_multi
            #if max_norm_clip < 0.2:
            #    max_norm_clip = 0.03
            failed_iter = 0
        else:
            if failed_iter + 1 >= max_failed_iters:
                delta_val_saved = delta_clipped.cpu().detach().numpy()
                np.save(out_delta_path, delta_val_saved)
                return
            failed_iter += 1

    if delta_val_saved is None:
        return None, None
    return delta_val_saved, rlt_norm


def finetune_maxnorm(src_audio_lst, delta, target_phrase, out_delta_path, rir_lst=None, init_max_norm=None):
    rlt = _do_minimize_maxnorm(src_audio_lst, target_phrase, delta, out_delta_path, audio_model, rir_lst, init_max_norm)
    return rlt

def superpose(x, v):
    # 两个向量的长度
    x = x.to(device)
    x_len = x.shape[1]
    v_len = v.shape[1]
    # 随机在噪声中寻找一个开头
    start = random.randint(0, v_len)
    # 循环播放噪声音频
    while v_len < x_len:
        v = torch.cat((v, v), dim=1)
        v_len = v.shape[1]
    # 防止噪声不够长
    v = torch.cat((v, v), dim=1)
    # 对齐噪声与音频
    v = v[:, start:x_len + start]
    v = v.to(device)
    res = x + v
    res = torch.clamp(res, -1, 1)
    return res, start

def success_rate_and_cer(sound_data, noise,rir_list = None):
    i = 0
    cer = 0.0
    sound_num = len(sound_data)
    noise = torch.Tensor(noise)
    for sound in sound_data:
        wave = sound['wave'].to(device)
        # cx = audio_model(wave, decode=True)[0][0][0]
        modified_audio = sound['wave'] + noise[:, :sound['wave'].shape[1]]
        # plot_waveform(new_sound.cpu(), 16000)
        modified_audio = modified_audio.to(device)
        new_sound, start = superpose(wave, noise)
        new_sound = new_sound.to(device)
        rir_sound, sample = torchaudio.load(rir_list)
        #a = torch_conv_audio(new_sound, rir_sound)
        #a = a.to(device)
        #new_sound, start = superpose(a, noise)
        predict = audio_model(modified_audio, decode=True)[0][0][0]
        print(predict)
        pred_cer = cel("POWER OFF", predict)
        pred_cer = cel(sound['label'], predict)
        #print('cer = ' + str(pred_cer) + ';length = ' + str(wave.shape[1]) + ';height = ' + str(sound['height']))
        cer += pred_cer
        if pred_cer >= 0.99:
            i += 1
    print('success rate is ' + str(i / sound_num))
    print('mean cer is ' + str(cer / sound_num))
    return i / sound_num
import torchaudio
from torchaudio.transforms import Resample
def normalize(spectrogram):
  # 计算谱图的均值和标准差
  mean = torch.mean(spectrogram)
  std = torch.std(spectrogram)

  # 归一化谱图
  normalized_spectrogram = (spectrogram - mean) / std

  return normalized_spectrogram
def dealTargetAudio():
    filename = "target.wav"
    wavsignal, sample_rate = torchaudio.load(filename)
  # 将采样率调整为 16 kHz
    if sample_rate != 16000:
        resample_transform = Resample(sample_rate, 16000)
        wavsignal = resample_transform(wavsignal)
  # 将位深度调整为 16 位
    wavsignal = wavsignal.float()
      # 将双声道音频转换为单声道
    if wavsignal.shape[0] > 1:
        wavsignal = torch.mean(wavsignal, dim=0, keepdim=True)
  # 转换为谱图
    #spectrogram = torchaudio.transforms.Spectrogram()(wavsignal)
  # 归一化
    #spectrogram = normalize(spectrogram)
    #plot_waveform(spectrogram,16000,"target_fig")
    return wavsignal

def plot_waveform(waveform, sample_rate,output_file):
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle("waveform")
    figure.savefig(output_file)
    plt.close(figure)
def celPhonemeLength(train_data):
    meanList = []
    for item in train_data:
        sound_len = len(item["wave"][0])
        phoneme_len = len(g2p(item['label']))
        print(g2p(item['label']))
        meanList.append(sound_len/phoneme_len)
    return np.mean(meanList)
if __name__=='__main__':
    #print(np.load())
    train_data,_ = generate_data()
    evaluation_data = generate_test_data()
    #celPhonemeLength(train_data)
    print("----data load OK")
    #rir_list_all = generate_rir_list()
    #print("----rir load OK")
    #rir_list = rir_list_all[:80]
    #success_rate_origin(evaluation_data,audio_model)
    #success_rate_rir(evaluation_data, audio_model, rir_list[2])
        #进行降噪处理
    run_stage_1(train_data,target_labels[0].upper(),64000, Path('0414/oral'))
    #uni_delta = np.load('0311/out_150.npy')
    uni_delta = np.load('0414/oral_150.npy')
    #finetune_maxnorm(train_data, uni_delta, target_labels[0].upper(), Path('0409/rir_add_phon_new'), rir_list)
    #v = np.load('0409/rir_add_phon_new.npy', allow_pickle=True)
    #success_rate_and_cer(evaluation_data,v,rir_list[2])
    success_rate_noise_add(evaluation_data, uni_delta, audio_model, 'POWER OFF')
    #uni_delta = np.load('0406/out_150.npy')
    #success_rate_noise_add(evaluation_data, v, audio_model,'POWER OFF')
    #取训练集和验证集
    #generate_100_data()
    #success_rate_origin(evaluation_data,audio_model)
    #train_data size is 150
    #uni_delta = np.load('0311/out_150.npy')
    #success_rate_noise_add(evaluation_data, uni_delta, audio_model,'POWER OFF')
    #
    
    
    
    
