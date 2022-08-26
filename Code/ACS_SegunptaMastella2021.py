import numpy as np
from brian2 import *
import pandas as pd
import os
from os import listdir
from os.path import isfile, join
from scipy import signal

Current_PATH = os.getcwd()
FILE_PATH = os.path.dirname(os.getcwd())
CODE_PATH = os.path.join(FILE_PATH, 'Code')

sys.path.append(os.path.join(CODE_PATH, 'utils'))
from utils_plot import *
from utils_spikes import *
from utils_misc import *


def plot_freqstimulus_vs_freqdevice(collection_of_ffts, intervals, stimuli):
    yticks = [i for i in range(collection_of_ffts.shape[0])]
    xticks = [i for i in range(collection_of_ffts.shape[1])]
    yticks_labels = Reduce_ticks_labels(ticks=intervals, ticks_steps=10)
    xticks_label = list(map(str, stimuli))

    plt.figure()
    im = plt.imshow(collection_of_ffts, cmap='BuGn', aspect='auto', vmin=0, vmax=1)
    for index, stimulus in enumerate(stimuli):
        whereismyfreq = np.where(intervals == stimulus)[0][0]
        plt.plot(index, yticks[whereismyfreq], '*', color='r')
    plt.yticks(yticks, tuple(list(map(str, yticks_labels))))
    plt.xticks(xticks, tuple(xticks_label))
    plt.xlabel('Stimulus Frequency (Hz)')
    plt.ylabel('Device Main Frequency Components(Hz)')
    plt.title('Device\'s response to frequencies')
    cbar = plt.colorbar(im)
    plt.legend(['Stimulus Freq'])
    cbar.set_label('Magnitude Normalized for each Stimulus', rotation=90)
    return

def create_intervals(freq_max, lsb):
    n = int(round(freq_max/lsb))
    intervals = np.zeros(n)
    for i in range(n):
        intervals[i] = i*lsb
    return intervals
def discretize_analogdata(data, intervals):
    bins = np.zeros(len(intervals))
    for value in data:
        indexes_chosen = np.where((intervals <= value[1]))[0][-1]
        bins[indexes_chosen] += value[0]
    return bins
def Reduce_ticks_labels(ticks,ticks_steps = 10):
    new_ticks = []
    counter = 0
    for label in ticks:
        if mod(counter,ticks_steps) == 0:
            new_ticks.append(label)
        else:
            new_ticks.append('')
        counter += 1
    return new_ticks
def create_confusion_matrix(input_analog, output):
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams['font.size'] = 20
    rcParams['axes.linewidth'] = 1

    intervals = create_intervals(input_analog['V'].max(), lsb=input_analog['V'].max() / 20)
    bins = np.zeros(len(input_analog['V']))
    for index, value in enumerate(abs(input_analog['V'])):
        indexes_chosen = np.where((intervals <= value))[0][-1]
        bins[index] = indexes_chosen

    intervals2 = create_intervals(output.max(), lsb=output.max() / 20)
    bins2 = np.zeros(len(output))
    for index, value in enumerate(output):
        indexes_chosen = np.where((intervals2 <= value))[0][-1]
        bins2[index] = indexes_chosen
    figure()
    confusion_matrix = np.zeros([len(intervals), len(intervals2)])
    for index, valuex in enumerate(bins[:len(bins2)]):
        confusion_matrix[int(valuex), int(bins2[index])] = 1
    im = imshow(confusion_matrix, cmap='Oranges', aspect='auto')
    yticks_labels = Reduce_ticks_labels(np.round(intervals, 2), ticks_steps=10)
    plt.yticks([i for i in range(len(intervals))], tuple(list(map(str, yticks_labels))))
    xticks_labels = Reduce_ticks_labels(np.round(intervals2), ticks_steps=5)
    plt.xticks([i for i in range(len(intervals2))], tuple(xticks_labels))
    plt.xlabel('Spike Count (#)')
    plt.ylabel('Sensor Voltage (V)')
    # plt.title('Neuron Rate Coding of Stretch Amplitude')
    cbar = plt.colorbar(im)
    cbar.set_label('Occurences', rotation=90)
    print('ciao')
def butter_bandstop_filter(data, lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    i, u = signal.butter(order, [low, high], btype='bandstop')
    y = signal.lfilter(i, u, data)
    return y

def create_intervals(freq_max, lsb):
    n = int(round(freq_max / lsb))
    intervals = np.zeros(n)
    for i in range(n):
        intervals[i] = i * lsb
    return intervals

def discretize_analogdata(data, intervals):
    bins = np.zeros(len(intervals))
    for value in data:
        indexes_chosen = np.where((intervals <= value[1]))[0][-1]
        bins[indexes_chosen] += value[0]
    return bins

def Reduce_ticks_labels(ticks, ticks_steps=10):
    new_ticks = []
    counter = 0
    for label in ticks:
        if mod(counter, ticks_steps) == 0:
            new_ticks.append(label)
        else:
            new_ticks.append('')
        counter += 1
    return new_ticks

def plot_the_data(input_analog):
  # otherwise the right y-label is slightly clipped
    print('eee')
def plot_tapping():
    DATA_PATH = os.path.join(FILE_PATH, 'Data',  'ACS', 'Tapping_Data')
    onlyfiles = [f for f in listdir(DATA_PATH) if isfile(join(DATA_PATH, f))]
    wanted_frequencies = 10
    intervals = create_intervals(freq_max=650, lsb=10)
    # b, a = signal.butter(4, [45, 55], 'bandstop', fs, output='ba')
    rc('font', family="Times New Roman", size=12)

    rc('')  # bold fonts are easier to see
    # rc('xlabel', weight='bold', family = "Times New Roman")
    # rc('ylabel', weight='bold', family = "Times New Roman")
    # tick labels bigger
    rc('lines', lw=3, color='k')  # thicker black lines
    # rc('grid', c='0.5', ls='-', lw=0.5)  # solid gray grid lines
    rc('savefig', dpi=300)
    stimuli = np.array([10, 60, 110, 160, 210, 310, 410, 510, 610])
    collection_of_ffts = np.zeros([len(intervals), len(stimuli)])
    for file in onlyfiles:
        # stimulus_freq = int(file.split("-")[0])
        # print(stimulus_freq)
        input_analog = pd.read_csv(os.path.join(DATA_PATH, file), delimiter='\t', sep='', header=0,
                                   error_bad_lines=False)
        dt = input_analog['s'][1] - input_analog['s'][0]
        fs = 1 / dt

        # show()
        input_analog_brian = TimedArray(input_analog['V'], dt=dt * second)
        SA_neuron_eq = Equations('''
                                     dv/dt = -v/(1*ms) +Ie_h/(0.01*pF) : volt
                                     dIe_h/dt = -(Ie_h-input_analog_brian(t)*10*pamp)/(1*ms)  : amp
                                   ''')
        SA_neuron = NeuronGroup(1, model=SA_neuron_eq, method='euler', threshold='v > 400*mV',
                                refractory='0.1*ms',
                                reset='v = 0*mV')
        SA_neuron_monitor = StateMonitor(SA_neuron, 'v', record=True)
        SA_neuron_spikes = SpikeMonitor(SA_neuron)
        wee = PopulationRateMonitor(SA_neuron)
        run(100 * second, report='stdout')

        # plot_the_data(input_analog)
    binning = [[i * 100 * ms] for i in range(900)]

    output = rate_calculator(stimulus_array=binning, Spikes=SA_neuron_spikes, nNeurons_layer2=1,
                             orientation_adimension=True, method='Spike_Count').T
    # from scipy.interpolate import interp1d
    # x = [i * 100 * ms for i in range(100)]
    # f2 = interp1d(x, output.T[:], kind='cubic')
    # y = f2([1 * ms for i in range(100 * 100)])
    # plot(y)
    # plt.rcParams["font.weight"] = "bold"
    # plt.rcParams["axes.labelweight"] = "bold"
    # plt.rcParams['font.size'] = 12
    # rcParams['axes.linewidth'] = 1
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_ylabel('Sensor Voltage (V)', family="Times New Roman")
    ax1.set_xlabel('Time (s)', family="Times New Roman")
    # we already handled the x-label with ax1
    ax1.plot(input_analog['s'], input_analog['V'] / input_analog['V'].max(), color=color)
    # ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_yticks([0.2, 0.4, 0.6, 0.8, 1])
    ax1.set_yticklabels(
        [round(0.2 * round(input_analog['V'].max(), 2)), round(0.4 * input_analog['V'].max(), 2),
         round(0.6 * input_analog['V'].max(), 2), round(0.8 * input_analog['V'].max(), 2),
         round(1 * input_analog['V'].max(), 2)])
    fig.tight_layout()  #
    fig, ax2 = plt.subplots()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Spike Count (#Spikes)', family="Times New Roman")
    ax2.set_xlabel('Time (s)',  family="Times New Roman")
    # we already handled the x-label with ax1
    ax2.plot([i * 10 * ms for i in range(len(output))], output / output.max(), color=color)
    # ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_yticks([0.2, 0.4, 0.6, 0.8, 1])
    ax2.set_yticklabels(
        [round(0.2 * output.max()), round(0.4 * output.max()), round(0.6 * output.max()), round(0.8 * output.max()),
         round(1 * output.max())])
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig, ax3 = plt.subplots()  # instantiate a second axes that shares the same x-axis

    color = 'tab:orange'
    ax3.set_ylabel('Vmem (V)', family="Times New Roman")  # we already handled the x-label with ax1
    ax3.set_xlabel('Time (s)', family="Times New Roman")
    ax3.plot(SA_neuron_monitor.t, SA_neuron_monitor.v.T, color=color)
    # ax3.tick_params(axis='y', labelcolor=color)
    # ax2.set_yticks([0.2, 0.4, 0.6, 0.8, 1])
    # ax2.set_yticklabels(
    #     [round(0.2 * output.max()), round(0.4 * output.max()), round(0.6 * output.max()), round(0.8 * output.max()),
    #      round(1 * output.max())])
    create_confusion_matrix(input_analog, output)  # otherwise the right y-label is slightly clipped

    show()
    print('eee')

def plot_numbers():
    rc('font', weight='bold', family="Times New Roman", size=12)

    rc('')  # bold fonts are easier to see
    # rc('xlabel', weight='bold', family = "Times New Roman")
    # rc('ylabel', weight='bold', family = "Times New Roman")
    # tick labels bigger
    rc('lines', lw=3, color='k')  # thicker black lines
    # rc('grid', c='0.5', ls='-', lw=0.5)  # solid gray grid lines
    rc('savefig', dpi=300)
    colors = [(127 / 255, 190 / 255, 0), (204 / 255, 0, 0 / 255), (122 / 255, 122 / 255, 1),
              (191 / 255, 191 / 255, 0), ]
    n = 5
    colors1 = plt.cm.magma(np.linspace(0, 1, n))
    colors2 = plt.cm.tab10(np.linspace(0, 1, n + 1))
    cm = 1 / 2.54
    fig, ax1 = plt.subplots(nrows = 3, ncols = 1, figsize=(12 * cm, 22 * cm), sharex = True)
    ax2 = ax1[2].twinx()
    fig1, ax3 = plt.subplots(figsize=(14 * cm, 20 * cm))
    ax4 = ax1[1].twinx()
    fig2, ax5 = plt.subplots(figsize=(14 * cm, 8 * cm))
    ax6 = ax1[0].twinx()
    DATA_PATH = os.path.join(FILE_PATH, 'Data', 'ACS', 'Count-5-1')
    onlyfiles = [f for f in listdir(DATA_PATH) if isfile(join(DATA_PATH, f))]
    fingers = ['thumb', 'index', 'middle', 'ring', 'little']
    maximum = []
    signals = np.zeros([5, 1000])
    for f_ix, file in enumerate(onlyfiles):

        # stimulus_freq = int(file.split("-")[0])
        # print(stimulus_freq)
        if 'txt' in file:
            finger_name = file.split('-')[0].lower()
            index = fingers.index(finger_name)
            print(fingers[index])
            set_device('cpp_standalone', directory='boh')

            input_analog = pd.read_csv(os.path.join(DATA_PATH, file), delimiter='\t', sep='', header=0,
                                       error_bad_lines=False)
            dt = input_analog['s'][1] - input_analog['s'][0]
            fs = 1 / dt
            signals[index, :len(input_analog['V'])] = input_analog['V']

    input_analog_brian = TimedArray(signals.T, dt=dt * second)
    SA_neuron_eq = Equations('''
                                     dv/dt = -v/(10*ms) +Ie_h/(0.01*pF) : volt
                                     dIe_h/dt = -(Ie_h-input_analog_brian(t,i)*5*pamp)/(1*ms)  : amp
                                   ''')
    SA_neuron = NeuronGroup(5, model=SA_neuron_eq, method='euler', threshold='v > 400*mV',
                            refractory='1*ms',
                            reset='v = 0*mV')
    RL_neuron_eq = Equations('''
                                     dv/dt = -v/(1*ms) +(Ie_h - Ii_h)/(0.01*pF) : volt
                                     dIe_h/dt = -(Ie_h)/(1*ms)  : amp
                                     dIi_h/dt = -(Ii_h)/(10*ms)  : amp
                                   ''')
    Recognition_layer = NeuronGroup(5, model=RL_neuron_eq, method='euler', threshold='v > 400*mV',
                                    refractory='10*ms',
                                    reset='v = 0*mV')
    WTA = NeuronGroup(1, model=RL_neuron_eq, method='euler', threshold='v > 400*mV',
                      refractory='10*ms',
                      reset='v = 0*mV')
    P = PoissonGroup(1, 100 * Hz)
    mysynapse2 = Synapses(P, Recognition_layer, on_pre='Ie_h += 10*pA')
    mysynapse2.connect('j == 4')
    synapse_model = 'w : amp'
    mysynapse = Synapses(SA_neuron, Recognition_layer, model=synapse_model, on_pre='Ie_h += 10*w')
    mysynapse.connect()

    WTA_synapse = Synapses(Recognition_layer, WTA, on_pre='Ie_h += 10000000*pA')
    WTA_synapse.connect()
    WTA_response = Synapses(WTA, Recognition_layer, on_pre='Ii_h += 10000000*pA')
    WTA_response.connect()
    myw = np.array(
        [[0.35, 0.4, 0.45, 0.6, 0], [0, 0, 0, 0, 0], [0.35, 0, 0, 0, 0], [0.35, 0.4, 0, 0, 0], [0.35, 0.4, 0.45, 0, 0]])

    mysynapse.w = myw.flatten() * pA

    SA_neuron_monitor = StateMonitor(SA_neuron, 'v', record=True)
    SA_neuron_spikes = SpikeMonitor(SA_neuron)
    RL_neuron_spikes = SpikeMonitor(Recognition_layer)
    P_neuron_spikes = SpikeMonitor(P)
    WTA_neuron_spikes = SpikeMonitor(WTA)
    run(15.5 * second, report='stdout')
    # plot(SA_neuron_spikes.t,SA_neuron_spikes.i + index,'.', color = colors[index])
    timelow = 4
    timehigh = 13
    sampling = 500 * ms
    multiplier_factor = int(1 * second / sampling)
    number = int(timehigh * second / sampling)
    multiplier_factor_analog = int(1 / dt)

    binning = [[i * sampling, 0] for i in range(int(number))]
    timestep = [i * sampling for i in range(number)]
    timestep2 = [i * second for i in range(2 * number)]
    output = rate_calculator(stimulus_array=binning, Spikes=SA_neuron_spikes, nNeurons_layer2=5,
                             orientation_adimension=True, method='Spike_Count')
    output2 = rate_calculator(stimulus_array=binning, Spikes=RL_neuron_spikes, nNeurons_layer2=5,
                              orientation_adimension=True, method='Spike_Count')
    output3 = rate_calculator(stimulus_array=binning, Spikes=P_neuron_spikes, nNeurons_layer2=1,
                              orientation_adimension=True, method='Spike_Count')
    # maximum.append(output[0].max())

    for neuron in range(5):
        ax2.plot(RL_neuron_spikes.t[
                     (RL_neuron_spikes.t > timelow * second) & (RL_neuron_spikes.t < timehigh * second) & (
                                 RL_neuron_spikes.i == neuron)],
                 RL_neuron_spikes.i[
                     (RL_neuron_spikes.t > timelow * second) & (RL_neuron_spikes.t < timehigh * second) & (
                                 RL_neuron_spikes.i == neuron)]+0.1, '.', color=colors1[neuron])
        ax1[2].plot(timestep[timelow * multiplier_factor:timehigh * multiplier_factor], output2[neuron,
                                                                                     timelow * multiplier_factor:timehigh * multiplier_factor] / output2.max() * 0.8 + neuron + 0.2,
                 color=colors1[neuron])
        ax1[0].plot([dt * i*second for i in
                  range(len(signals[neuron, timelow * multiplier_factor_analog:timehigh * multiplier_factor_analog]))]+4*second,
                 signals[neuron,
                 timelow * multiplier_factor_analog:timehigh * multiplier_factor_analog] / signals.max() * 0.8 + neuron + 0.2,
                 color=colors2[neuron])
    ax1[2].set_xlabel('Time (s)')
    ax2.set_yticks([0, 1, 2, 3, 4])
    ax2.set_yticklabels(['one', 'two', 'three', 'four', 'five'])
    ax2.set_ylim([-0.2, 5])
    ax1[2].set_ylabel('Spike Count', weight='bold')
    ax1[2].set_yticks([0.1, 0.9, 1.2, 1.9, 2.2, 2.9, 3.2, 3.9, 4.2, 4.9])
    ax1[2].set_yticklabels([0, output2.max(), 0, output2.max(), 0, output2.max(), 0, output2.max(), 0, output2.max()])
    ax1[2].set_ylim([-0.2, 5])
    ax1[0].set_yticks([0.1, 0.9, 1.2, 1.9, 2.2, 2.9, 3.2, 3.9, 4.2, 4.9])
    ax1[0].set_yticklabels(
        [0, round(signals.max(), 2), 0, round(signals.max(), 2), 0, round(signals.max(), 2), 0, round(signals.max(), 2),
         0, round(signals.max(), 2)])
    ax1[0].set_ylabel('Vsensor (V)', weight='bold')
    ax6.set_yticks([0.5, 1.5, 2.5, 3.5, 4.5])
    ax6.set_yticklabels(['thumb', 'index', 'middle', 'ring', 'little'])
    ax6.set_ylim([-0.2, 5])
    ax1[0].set_ylim([-0.2, 5])
    # figure()
    for neuron in range(5):
        ax4.plot(SA_neuron_spikes.t[
                     (SA_neuron_spikes.t > timelow * second) & (SA_neuron_spikes.t < timehigh * second) & (
                                 SA_neuron_spikes.i == neuron)],
                 SA_neuron_spikes.i[
                     (SA_neuron_spikes.t > timelow * second) & (SA_neuron_spikes.t < timehigh * second) & (
                                 SA_neuron_spikes.i == neuron)],
                 '.', color=colors2[neuron], markersize=0.1)
        ax1[1].plot(timestep[timelow * multiplier_factor:timehigh * multiplier_factor], output[neuron,
                                                                                     timelow * multiplier_factor:timehigh * multiplier_factor] / output.max() * 0.8 + neuron + 0.1,
                 color=colors2[neuron])
    ax4.plot(P_neuron_spikes.t[(P_neuron_spikes.t > timelow * second) & (P_neuron_spikes.t < timehigh * second)],
             P_neuron_spikes.i[(P_neuron_spikes.t > timelow * second) & (P_neuron_spikes.t < timehigh * second)] + 5,
             '.', color=colors2[5], markersize=0.1)
    ax1[1].plot(timestep[timelow * multiplier_factor:timehigh * multiplier_factor],
             output3[0, timelow * multiplier_factor:timehigh * multiplier_factor] / output3.max() * 0.8 + 5 + 0.2,
             color=colors2[5])
    ax4.set_yticks([0, 1, 2, 3, 4, 5])
    ax4.set_yticklabels(['thumb', 'index', 'middle', 'ring', 'little', 'poisson'])
    ax4.set_ylim([-0.2, 6])
    ax1[1].set_ylabel('Spike Count', weight='bold')
    ax1[1].set_yticks([0.1, 0.9, 1.2, 1.9, 2.2, 2.9, 3.2, 3.9, 4.2, 4.9])
    ax1[1].set_yticklabels([0, output.max(), 0, output.max(), 0, output.max(), 0, output.max(), 0, output.max()])
    ax1[1].set_ylim([-0.2, 6])
    ax1[2].set_xticks(timestep2[timelow:timehigh] / second)
    ax1[2].set_xticklabels(timestep2[timelow:timehigh] / second - timelow)
    ax2.set_xticks(timestep2[timelow:timehigh] / second)
    ax2.set_xticklabels(timestep2[timelow:timehigh] / second - timelow)
    ax1[1].set_xticks(timestep2[timelow:timehigh] / second)
    ax1[1].set_xticklabels(timestep2[timelow:timehigh] / second - timelow)
    ax4.set_xticks(timestep2[timelow:timehigh] / second)
    ax4.set_xticklabels(timestep2[timelow:timehigh] / second - timelow)


    aaa = mysynapse.w
    # plot(output.T)
    figure()
    myw = np.append(np.array([[0, 0, 0, 0, 0]]).T, myw, axis=1)
    myw = np.append(np.array([[0, 0, 0, 0, 0, 1]]), myw, axis=0)
    imshow(myw)
    title('Synapse Map')
    xticks([0, 1, 2, 3, 4], ['thumb', 'index', 'middle', 'ring', 'little'])
    yticks([0, 1, 2, 3, 4], ['one', 'two', 'three', 'four', 'five'])
    # show()
    # ax2.plot(SA_neuron_spikes.t, SA_neuron_spikes.i + index, '.', color=colors[1])
    # ax1[2].plot(input_analog['s'][(input_analog['s']>timelow) & (input_analog['s']<timehigh)], 0.1 + 0.8 * input_analog['V'][(input_analog['s']>timelow) & (input_analog['s']<timehigh)] / 1.968757 + index, '--', color=colors[0],
    #          linewidth=3)
    # ax2.plot(binning, 0.1 + 0.8 * output[0] / 44 + index, color=colors[2],
    #          linewidth=3)
    # ax2.set_yticks([0.1, 0.9, 1.1, 1.9, 2.1, 2.9, 3.1, 3.9, 4.1, 4.9])
    # # ax2.set_yticklabels([0,maximum[0],0,maximum[1],0,maximum[2],0,maximum[3],0,maximum[4]])
    # ax2.set_yticklabels([0, 44, 0, 44, 0, 44, 0, 44, 0, 44])
    # # ax1[2].set_ylabel('Raster')
    # ax2.set_ylabel('#Spikes', weight='bold', family="Times New Roman")
    # ax1[2].set_xlabel('Times(s)', weight='bold', family="Times New Roman")
    # ax1[2].set_xlim([45, 70])
    device.reinit()
    device.activate(directory='boh')
    # ax2.set_yticks([0, 1, 2, 3, 4])
    # ax2.set_yticklabels(fingers)
    # ax1[2].set_yticks([0.1, 0.9, 1.1, 1.9, 2.1, 2.9, 3.1, 3.9, 4.1, 4.9])
    # ax1[2].set_yticklabels([0, 1.97, 0, 1.97, 0, 1.97, 0, 1.97, 0, 1.97])
    # ax1[2].set_ylabel('Analog Voltage (V)', weight='bold', family="Times New Roman")
    # ax1[2].set_ylim([-0.2, 5])
    # ax2.set_ylim([-0.2, 5])
    fig.tight_layout()

def plot_gestures():
    rc('font', weight='bold', family="Times New Roman", size=12)

    rc('')  # bold fonts are easier to see
    # rc('xlabel', weight='bold', family = "Times New Roman")
    # rc('ylabel', weight='bold', family = "Times New Roman")
    # tick labels bigger
    rc('lines', lw=3, color='k')  # thicker black lines
    # rc('grid', c='0.5', ls='-', lw=0.5)  # solid gray grid lines
    rc('savefig', dpi=300)
    colors = [(127 / 255, 190 / 255, 0), (204 / 255, 0, 0 / 255), (122 / 255, 122 / 255, 1),
              (191 / 255, 191 / 255, 0), ]

    cm = 1 / 2.54
    fig, ax1 = plt.subplots(figsize=(17 * cm, 11.4 * cm))
    ax2 = ax1.twinx()
    DATA_PATH = os.path.join(FILE_PATH, 'Data', 'ACS', 'Gesture')
    onlyfiles = [f for f in listdir(DATA_PATH) if isfile(join(DATA_PATH, f))]
    fingers = ['thumb', 'index', 'middle', 'ring', 'little']
    maximum = []
    for f_ix, file in enumerate(onlyfiles):

        # stimulus_freq = int(file.split("-")[0])
        # print(stimulus_freq)
        if 'txt' in file:
            finger_name = file.split('-')[0].lower()
            index = fingers.index(finger_name)
            set_device('cpp_standalone', directory='boh')

            input_analog = pd.read_csv(os.path.join(DATA_PATH, file), delimiter='\t', sep='', header=1,
                                       error_bad_lines=False)
            dt = input_analog['s'][1] - input_analog['s'][0]
            fs = 1 / dt

            # show()
            input_analog_brian = TimedArray(input_analog['V'], dt=dt * second)
            SA_neuron_eq = Equations('''
                                         dv/dt = -v/(10*ms) +Ie_h/(0.01*pF) : volt
                                         dIe_h/dt = -(Ie_h-input_analog_brian(t)*1*pamp)/(1*ms)  : amp
                                       ''')
            SA_neuron = NeuronGroup(1, model=SA_neuron_eq, method='euler', threshold='v > 400*mV',
                                    refractory='1*ms',
                                    reset='v = 0*mV')
            SA_neuron_monitor = StateMonitor(SA_neuron, 'v', record=True)
            SA_neuron_spikes = SpikeMonitor(SA_neuron)
            run(input_analog['s'][len(input_analog) - 1] * second, report='stdout')
            # plot(SA_neuron_spikes.t,SA_neuron_spikes.i + index,'.', color = colors[index])

            binning = [[i * 100 * ms, 0] for i in range(700)]

            output = rate_calculator(stimulus_array=binning, Spikes=SA_neuron_spikes, nNeurons_layer2=1,
                                     orientation_adimension=True, method='Spike_Count')
            # maximum.append(output[0].max())
            ax2.plot(SA_neuron_spikes.t, SA_neuron_spikes.i + index, '.', markersize=0.1, color=colors[1])
            ax1.plot(input_analog['s'], 0.1 + 0.8 * input_analog['V'] / 1.968757 + index, '--', color=colors[0],
                     linewidth=3)
            ax2.plot(binning, 0.1 + 0.8 * output[0] / 44 + index, color=colors[2],
                     linewidth=3)
            device.reinit()
            device.activate(directory='boh')
    ax2.set_yticks([0, 1, 2, 3, 4])
    ax2.set_yticklabels(fingers)
    ax1.set_yticks([0.1, 0.9, 1.1, 1.9, 2.1, 2.9, 3.1, 3.9, 4.1, 4.9])
    ax1.set_yticklabels([0, 1.97, 0, 1.97, 0, 1.97, 0, 1.97, 0, 1.97])
    ax1.set_ylabel('Analog Voltage (V)', weight='bold', family="Times New Roman", size = 12)
    ax1.set_ylim([-0.2, 5])
    ax2.set_ylim([-0.2, 5])
    ax2.set_yticks([0.1, 0.9, 1.1, 1.9, 2.1, 2.9, 3.1, 3.9, 4.1, 4.9])
    # ax2.set_yticklabels([0,maximum[0],0,maximum[1],0,maximum[2],0,maximum[3],0,maximum[4]])
    ax2.set_yticklabels([0, 44, 0, 44, 0, 44, 0, 44, 0, 44])
    # ax1.set_ylabel('Raster')
    ax2.set_ylabel('#Spikes', weight='bold', family="Times New Roman", size = 12)
    ax1.set_xlabel('Times(s)', weight='bold', family="Times New Roman", size = 12)
    ax1.set_xlim([45, 70])

def plot_wrist():
    rc('font', weight='bold', family="Times New Roman", size = 12)

    rc('')  # bold fonts are easier to see
    # rc('xlabel', weight='bold', family = "Times New Roman")
    # rc('ylabel', weight='bold', family = "Times New Roman")
    # tick labels bigger
    rc('lines', lw=3, color='k')  # thicker black lines
    # rc('grid', c='0.5', ls='-', lw=0.5)  # solid gray grid lines
    rc('savefig', dpi=300)
    colors = [(127 / 255, 190 / 255, 0), (204 / 255, 0, 0 / 255), (122 / 255, 122 / 255, 1),
              (191 / 255, 191 / 255, 0), ]

    cm = 1 / 2.54
    fig, ax1 = plt.subplots(nrows = 2, ncols = 1, figsize=(17 * cm, 13 * cm))
    ax2 = ax1[1].twinx()
    # fig2, ax3 = plt.subplots(figsize=(17 * cm, 6 * cm))
    ax4 = ax1[0].twinx()
    DATA_PATH = os.path.join(FILE_PATH, 'Data', 'ACS', 'Wrist')
    onlyfiles = [f for f in listdir(DATA_PATH) if isfile(join(DATA_PATH, f))]
    fingers = ['thumb', 'index', 'middle', 'ring', 'little']
    maximum = []
    for f_ix, file in enumerate(onlyfiles):

        # stimulus_freq = int(file.split("-")[0])
        # print(stimulus_freq)
        if 'txt' in file:
            set_device('cpp_standalone', directory='boh')

            input_analog = pd.read_csv(os.path.join(DATA_PATH, file), delimiter='\t', sep='', header=1,
                                       error_bad_lines=False)
            dt = input_analog['s'][1] - input_analog['s'][0]
            fs = 1 / dt

            # show()
            input_analog_scremato = input_analog[input_analog['V'] > 0.2]
            angles = [30, 30, 45, 30, 45, 30, np.nan, 45, 45, 90, 45, 90, 45, 90]
            collection = []
            prev_t = input_analog_scremato.iloc[0, 0]
            low_bound_index = input_analog_scremato.index[0]
            counter = 0
            for i in range(len(input_analog_scremato.iloc[:, 0])):
                if input_analog_scremato.iloc[i, 0] - prev_t > 1:
                    collection.append([angles[counter], low_bound_index, prev_index])
                    # low_bound = input_analog_scremato.iloc[i, 0]
                    low_bound_index = input_analog_scremato.index[i]
                    counter += 1
                prev_t = input_analog_scremato.iloc[i, 0]
                prev_index = input_analog_scremato.index[i]
            results = np.zeros([len(collection), 2])
            for m_ix, element in enumerate(collection):
                results[m_ix, 0] = element[0]
                results[m_ix, 1] = np.average(input_analog_scremato.loc[element[1]:element[2]]['V'])
            angles_here = [30, 45, 90]
            stats = np.zeros([3, 2])
            for ang_ix, angle in enumerate(angles_here):
                myind = np.where(results[:, 0] == angle)
                stats[ang_ix, 0] = np.average(results[myind, 1])
                stats[ang_ix, 1] = np.std(results[myind, 1])

            ax1[1].fill_between([30, 45, 90], stats[:, 0] + stats[:, 1], stats[:, 0] - stats[:, 1], color=colors[1],
                             alpha=0.5)
            ax1[1].plot([30, 45, 90], stats[:, 0], '--', color=colors[1], linewidth=3)
            # ax1.set_xticks([0, 1, 2])
            # ax1.set_xticklabels([30, 45, 90])
            ax1[1].set_xlabel('Bending Angle(°)', weight='bold', family="Times New Roman")
            ax1[1].set_ylabel('Analog Voltage (V)', weight='bold', family="Times New Roman")
            ax1[0].set_xlabel('Time(s)', weight='bold', family="Times New Roman")
            ax1[0].set_ylabel('Analog Voltage (V)', weight='bold', family="Times New Roman")
            # ax3.set_yticks([0,1])
            # ax3.set_yticklabels([0,input_analog['V'].max()])
            # show()
            input_analog_brian = TimedArray(input_analog['V'], dt=dt * second)
            SA_neuron_eq = Equations('''
                                         dv/dt = -v/(10*ms) +Ie_h/(0.01*pF) : volt
                                         dIe_h/dt = -(Ie_h-input_analog_brian(t)*2*pamp)/(1*ms)  : amp
                                       ''')
            SA_neuron = NeuronGroup(1, model=SA_neuron_eq, method='euler', threshold='v > 400*mV',
                                    refractory='1*ms',
                                    reset='v = 0*mV')
            SA_neuron_monitor = StateMonitor(SA_neuron, 'v', record=True)
            SA_neuron_spikes = SpikeMonitor(SA_neuron)
            run(input_analog['s'][len(input_analog) - 1] * second, report='stdout')
            # plot(SA_neuron_spikes.t,SA_neuron_spikes.i + index,'.', color = colors[index])

            binning = [[i * 100 * ms, 0] for i in range(1600)]

            output = rate_calculator(stimulus_array=binning, Spikes=SA_neuron_spikes, nNeurons_layer2=1,
                                     orientation_adimension=True, method='Spike_Count')
            # maximum.append(output[0].max())
            collection = []
            times = np.array([i * 0.1 for i in range(1600)])
            times_scremato = np.where(output[0, :] > 1)[0]
            prev_t = times_scremato[0]
            low_bound_index = times_scremato[0]
            counter = 0
            for i in range(len(times_scremato)):
                if times_scremato[i] - prev_t > 2:
                    collection.append([angles[counter], low_bound_index, times_scremato[i - 1]])
                    # low_bound = input_analog_scremato.iloc[i, 0]
                    low_bound_index = times_scremato[i]
                    counter += 1
                prev_t = times_scremato[i]
            results = np.zeros([len(collection), 2])
            for m_ix, element in enumerate(collection):
                results[m_ix, 0] = element[0]
                temp = np.average(output[0, element[1]:element[2]])
                if np.isnan(temp):
                    results[m_ix, 0] = 0
                    results[m_ix, 1] = np.nan
                else:
                    results[m_ix, 1] = temp
            angles_here = [30, 45, 90]
            stats = np.zeros([3, 2])
            for ang_ix, angle in enumerate(angles_here):
                myind = np.where(results[:, 0] == angle)
                stats[ang_ix, 0] = np.average(results[myind, 1])
                stats[ang_ix, 1] = np.std(results[myind, 1])
            ax2.fill_between([30, 45, 90], stats[:, 0] + stats[:, 1], stats[:, 0] - stats[:, 1], color=colors[0],
                             alpha=0.5)
            ax2.plot([30, 45, 90], stats[:, 0], '--', color=colors[0], linewidth=3)

            ax4.plot(times, output.T, color=colors[0], alpha=0.9)
            ax4.set_ylabel('Spikes(#)', weight='bold', family="Times New Roman")
            ax4.set_xlabel('Times(s)', weight='bold', family="Times New Roman")
            # ax4.set_yticks([1,2])
            # ax4.set_yticklabels([0,output.max()])
            ax2.set_ylabel('Spikes(#)', weight='bold', family="Times New Roman")
            ax1[0].plot(input_analog['s'], input_analog['V'], color=colors[1], linewidth=5)

            # ax2.plot(SA_neuron_spikes.t, SA_neuron_spikes.i, '.', color=colors[1])
            # ax1[1].plot(input_analog['s'], 0.1 + 0.8 * input_analog['V'] / 1.968757, '--', color=colors[0],
            #          linewidth=3)
            # ax2.plot([i * 100 * ms for i in range(1600)], 0.1 + 0.8 * output[0] / 44, color=colors[2],
            #          linewidth=3)
            # ax1[0].set_ylim([0,1.8])
            # ax4.set_ylim([0,60])
            device.reinit()
            device.activate(directory='boh')

    # ax2.set_yticks([0, 1, 2, 3, 4])
    # ax2.set_yticklabels(fingers)
    # ax1[1].set_yticks([0.1, 0.9, 1.1, 1.9, 2.1, 2.9, 3.1, 3.9, 4.1, 4.9])
    # ax1[1].set_yticklabels([0, 1.97, 0, 1.97, 0, 1.97, 0, 1.97, 0, 1.97])
    plt.tight_layout()

def set_pub():
    rc('font', weight='bold', family="Times New Roman")
    rc('')  # bold fonts are easier to see
    # rc('xlabel', weight='bold', family = "Times New Roman")
    # rc('ylabel', weight='bold', family = "Times New Roman")
    # tick labels bigger
    rc('lines', lw=3, color='k')  # thicker black lines
    # rc('grid', c='0.5', ls='-', lw=0.5)  # solid gray grid lines
    rc('savefig', dpi=300)
    return rc


rc = set_pub()
# plot_gestures()
plot_tapping()
# plot_wrist()
# plot_numbers()
show()

print('wee')
