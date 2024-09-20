"""
This example creates a room with reverberation time specified by inverting Sabine's formula.
This results in a reverberation time slightly longer than desired.
The simulation is pure image source method.
The audio sample with the reverb added is saved back to `examples/samples/guitar_16k_reverb.wav`.
"""
import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from scipy.io import wavfile

import pyroomacoustics as pra

methods = ["ism", "hybrid"]


def generate_rir():
    parser = argparse.ArgumentParser(
        description="Simulates and adds reverberation to a dry sound sample. Saves it into `./examples/samples`."
    )
    parser.add_argument(
        "--method",
        "-m",
        choices=methods,
        default=methods[0],
        help="Simulation method to use",
    )
    args = parser.parse_args()

    # The desired reverberation time and dimensions of the room
    random_u = random.randint(20, 50)
    u = random_u / 100
    rt60_tgt = 0.5  # seconds

    random_chang = random.randint(70, 120)
    chang = random_chang / 10
    random_kuan = random.randint(50, 100)
    kuan = random_kuan / 10
    random_gao = random.randint(25, 35)
    gao = random_gao / 10
    room_dim = [chang, kuan, gao]  # meters

    # import a mono wavfile as the source signal
    # the sampling frequency should match that of the room
    fs, audio = wavfile.read("origin.wav")

    # We invert Sabine's formula to obtain the parameters for the ISM simulator
    e_absorption, max_order = pra.inverse_sabine(u, room_dim)

    # Create the room
    if args.method == "ism":
        room = pra.ShoeBox(
            room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order
        )
    elif args.method == "hybrid":
        room = pra.ShoeBox(
            room_dim,
            fs=fs,
            materials=pra.Material(e_absorption),
            max_order=3,
            ray_tracing=True,
            air_absorption=True,
        )

    # place the source in the room
    random_sound_x = random.randint(0, random_h * 10) / 100
    random_sound_y = random.randint(0, random_h * 10) / 100
    random_sound_z = random.randint(0, 350) / 100
    room.add_source([random_sound_x, random_sound_y, random_sound_z], signal=audio, delay=0)

    random_mic_x = random.randint(0, random_h * 10) / 100
    random_mic_y = random.randint(0, random_h * 10) / 100
    random_mic_z = random.randint(0, 350) / 100

    # define the locations of the microphones
    mic_locs = np.c_[
        [random_mic_x, random_mic_y, random_mic_z]
        # [6.3, 4.93, 1.2],  # mic 1  # mic 2
    ]

    # finally place the array in the room
    room.add_microphone_array(mic_locs)

    # Run the simulation (this will also build the RIR automatically)
    room.simulate()

    rir_1_0 = room.rir[0][0]
    # print(rir_1_0)
    name = 'rir_u' + str(u) + '_h' + str(h) + '_sound' + str(random_sound_x) + '-' + str(random_sound_y) + '-' \
           + str(random_sound_z) + '_mic' + str(random_mic_x) + '-' + str(random_mic_y) + '-' + str(random_mic_z)
    torchaudio.save('rir_list/'+name+'.wav', torch.Tensor(rir_1_0).unsqueeze(0), 16000)
    #
    # room.mic_array.to_wav(
    #     f"origin_{args.method}.wav",
    #     norm=True,
    #     bitdepth=np.int16,
    # )
    #
    # # measure the reverberation time
    # rt60 = room.measure_rt60()
    # print("The desired RT60 was {}".format(rt60_tgt))
    # print("The measured RT60 is {}".format(rt60[0, 0]))
    #
    # # Create a plot
    # plt.figure()
    #
    # # plot one of the RIR. both can also be plotted using room.plot_rir()
    # rir_1_0 = room.rir[0][0]
    # print(rir_1_0)
    #
    # torchaudio.save('rir.wav',torch.Tensor(rir_1_0).unsqueeze(0),16000)
    # plt.subplot(2, 1, 1)
    # plt.plot(np.arange(len(rir_1_0)) / room.fs, rir_1_0)
    # plt.title("The RIR from source 0 to mic 1")
    # plt.xlabel("Time [s]")
    #
    # # plot signal at microphone 1
    # plt.subplot(2, 1, 2)
    # plt.plot(room.mic_array.signals[0, :])
    # plt.title("Microphone 1 signal")
    # plt.xlabel("Time [s]")
    #
    # plt.tight_layout()
    # plt.show()


if __name__ == '__main__':
    for i in range(99):
        generate_rir()
