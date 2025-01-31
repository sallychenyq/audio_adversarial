import scipy.io.wavfile as wav
import numpy as np
from scipy.fftpack import fft
from scipy.fftpack import ifft
from scipy import signal
import scipy
import librosa

def compute_PSD_matrix(audio, window_size):
    """
	First, perform STFT.
	Then, compute the PSD.
	Last, normalize PSD.
    """
#计算音频信号的短时傅里叶变换（STFT）。STFT将信号分成小段，然后对每段进行傅里叶变换，得到时频图。
#计算STFT的幅度平方，得到功率谱密度（PSD）。使用10 * np.log10(z * z + 0.0000000000000000001)将PSD转换为对数刻度。
    win = np.sqrt(8.0/3.) * librosa.core.stft(audio, center=False)
    z = abs(win / window_size)
    psd_max = np.max(z*z)
    psd = 10 * np.log10(z * z + 0.0000000000000000001)
    PSD = 96 - np.max(psd) + psd
    return PSD, psd_max   

def Bark(f):
    """returns the bark-scale value for input frequency f (in Hz)
    计算了输入频率f对应的Bark刻度值。Bark刻度是一种心理声学刻度，表示声音的感知频率。"""
    return 13*np.arctan(0.00076*f) + 3.5*np.arctan(pow(f/7500.0, 2))

def quiet(f):
     """returns threshold in quiet measured in SPL at frequency f with an offset 12(in Hz)
     计算了在频率f处的掩蔽阈值（threshold in quiet）"""
     thresh = 3.64*pow(f*0.001,-0.8) - 6.5*np.exp(-0.6*pow(0.001*f-3.3,2)) + 0.001*pow(0.001*f,4) - 12
     return thresh

def two_slops(bark_psd, delta_TM, bark_maskee):
    """
	returns the masking threshold for each masker using two slopes as the spread function 
    计算了每个掩蔽器的掩蔽阈值，使用了两个不同的斜率作为传播函数。"""
    Ts = []
    for tone_mask in range(bark_psd.shape[0]):
        bark_masker = bark_psd[tone_mask, 0]
        dz = bark_maskee - bark_masker
        zero_index = np.argmax(dz > 0)
        sf = np.zeros(len(dz))
        sf[:zero_index] = 27 * dz[:zero_index]
        sf[zero_index:] = (-27 + 0.37 * max(bark_psd[tone_mask, 1] - 40, 0)) * dz[zero_index:] 
        T = bark_psd[tone_mask, 1] + delta_TM[tone_mask] + sf
        Ts.append(T)
    return Ts
    
def compute_th(PSD, barks, ATH, freqs):
    """ returns the global masking threshold
    计算全局掩蔽阈值"""
    # Identification of tonal maskers寻找PSD中的局部极大值来识别音调掩蔽器
    # find the index of maskers that are the local maxima
    length = len(PSD)
    masker_index = signal.argrelextrema(PSD, np.greater)[0]
    
    
    # delete the boundary of maskers for smoothing
    if 0 in masker_index:
        masker_index = np.delete(0)
    if length - 1 in masker_index:
        masker_index = np.delete(length - 1)
    num_local_max = len(masker_index)

    # treat all the maskers as tonal (conservative way)
    # smooth the PSD 通过计算相邻点的幅度平方，得到局部最大值处的PSD
    p_k = pow(10, PSD[masker_index]/10.)    
    p_k_prev = pow(10, PSD[masker_index - 1]/10.)
    p_k_post = pow(10, PSD[masker_index + 1]/10.)
    P_TM = 10 * np.log10(p_k_prev + p_k + p_k_post)
    
    # bark_psd: the first column bark, the second column: P_TM, the third column: the index of points
    #比较Bark刻度和局部最大值处的PSD，筛选出符合条件的掩蔽器。根据Bark刻度、掩蔽阈值和频率信息计算全局掩蔽阈值
    _BARK = 0
    _PSD = 1
    _INDEX = 2
    bark_psd = np.zeros([num_local_max, 3])
    bark_psd[:, _BARK] = barks[masker_index]
    bark_psd[:, _PSD] = P_TM
    bark_psd[:, _INDEX] = masker_index
    
    # delete the masker that doesn't have the highest PSD within 0.5 Bark around its frequency 
    for i in range(num_local_max):
        next = i + 1
        if next >= bark_psd.shape[0]:
            break
            
        while bark_psd[next, _BARK] - bark_psd[i, _BARK]  < 0.5:
            # masker must be higher than quiet threshold
            if quiet(freqs[int(bark_psd[i, _INDEX])]) > bark_psd[i, _PSD]:
                bark_psd = np.delete(bark_psd, (i), axis=0)
            if next == bark_psd.shape[0]:
                break
                
            if bark_psd[i, _PSD] < bark_psd[next, _PSD]:
                bark_psd = np.delete(bark_psd, (i), axis=0)
            else:
                bark_psd = np.delete(bark_psd, (next), axis=0)
            if next == bark_psd.shape[0]:
                break        
    
    # compute the individual masking threshold 计算个体掩蔽阈值
    delta_TM = 1 * (-6.025  -0.275 * bark_psd[:, 0])
    Ts = two_slops(bark_psd, delta_TM, barks) 
    Ts = np.array(Ts)
    
    # compute the global masking threshold
    theta_x = np.sum(pow(10, Ts/10.), axis=0) + pow(10, ATH/10.) 
 
    return theta_x

def generate_th(audio, fs, window_size=2048):
    """
	returns the masking threshold theta_xs and the max psd of the audio
    """
    PSD, psd_max= compute_PSD_matrix(audio , window_size)  
    freqs = librosa.core.fft_frequencies(fs, window_size)
    barks = Bark(freqs)

    # compute the quiet threshold ATH（听觉绝对阈值）表示人耳可以检测到的最低声音级。
    ATH = np.zeros(len(barks)) - np.inf
    bark_ind = np.argmax(barks > 1)
    ATH[bark_ind:] = quiet(freqs[bark_ind:])

    # compute the global masking threshold theta_xs 
    theta_xs = []
    # compute the global masking threshold in each window
    for i in range(PSD.shape[1]):
        theta_xs.append(compute_th(PSD[:,i], barks, ATH, freqs))
    theta_xs = np.array(theta_xs)
    return theta_xs, psd_max





