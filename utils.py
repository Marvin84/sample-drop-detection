from itertools import chain
import librosa
import logging
import math
import numpy as np
import os
import pickle
import random
import soundfile as sfile
import xml.etree.ElementTree as ET

from config import *
from data_structures import *
from data_io import *
from training import *
from feature_extraction import *
from online_stats import OnlineStatCompute
from datetime import datetime



####################
# CHiME-5
###################

def get_chime_wave_name(sessionId, deviceId):
    name = "/S0{:01d}/S0{:01d}_U0{:01d}.CH1.wav".format(sessionId, sessionId, deviceId)
    return name

def read_chime_segments(sessionIds, seconds=1):

    deviceObjects = []

    sampleIdPath = ("/").join([config_chimePath, "loss-id-chime.pickle"])
    sampleIds = get_pickle(sampleIdPath)
    for id in sessionIds:
        print("session started: ", id)
        for j in config_deviceIds:
            print("preparing device: ", j+1)
            deviceId = j+1
            lossInstantArray = sampleIds[id][deviceId]
            for lossInstant in lossInstantArray:
                start = int(lossInstant - config_CTX/2)
                end = int(start + config_CTX)
                name = get_chime_wave_name(id, deviceId)
                wavPath = ("/").join([config_chimePath, name])
                signal = get_read_and_normalize_utterance_for_interval(wavPath, start, end)
                refSegments = []
                for k in config_deviceIds:
                    refDeviceId = k+1
                    if refDeviceId != deviceId:
                        refName = get_chime_wave_name(id, refDeviceId)
                        refWavPath = ("/").join([config_chimePath, refName])
                        refSignal = get_read_and_normalize_utterance_for_interval(refWavPath, start, end)
                        refSegments.append(refSignal)

                deviceObjects.append(Device(deviceId, name, config_sf, signal, refSegments, [lossInstant, lossInstantArray.index(lossInstant)+1]))


    return deviceObjects


def read_chime_segments_no_shift(sessionIds, seconds=1):


    deviceObjects = []
    signals = []

    sampleIdPath = ("/").join([config_chimePath, "loss-id-chime.pickle"])
    sampleIds = get_pickle(sampleIdPath)
    tsteps = get_pickle(config_chimePath + "/timesteps.pickle")

    for id in sessionIds:
        print("session started: ", id)
        for j in config_deviceIds:
            if id == 7 and j == 0:
                continue
            print("preparing device: ", j + 1)
            deviceId = j + 1
            lossInstantArray = sampleIds[id][deviceId]

            for lossIndex, lossInstant in enumerate(lossInstantArray):
                timeKey = ("-").join([str(deviceId), str(lossIndex + 1)])
                ts = tsteps[id][timeKey]["time"]
                hypDeviceStart = ts[str(deviceId)]
                #print(hypDeviceStart)
                lossWindowStart = int(lossInstant - int((config_CTX * seconds) / 2))
                lossWindowEnd = int(lossWindowStart + config_CTX * seconds)

                name = get_chime_wave_name(id, deviceId)
                #embed()
                wavPath = ("").join([config_chimePath, name])
                signal = get_read_and_normalize_utterance_for_interval(wavPath, lossWindowStart, lossWindowEnd)
                # signals.extend(get_read_and_normalize_utterance_for_interval(wavPath, lossWindowStart, lossWindowEnd))

                refSegments = []
                for k in config_deviceIds:
                    refDeviceId = k + 1
                    if refDeviceId != deviceId:
                        timeshift = int(get_sample_difference(hypDeviceStart, ts[str(refDeviceId)]))
                        print(deviceId, ", ", refDeviceId, ", ", timeshift)
                        refName = get_chime_wave_name(id, refDeviceId)
                        refWavPath = ("/").join([config_chimePath, refName])
                        refStartTimeWindow = lossWindowStart+timeshift
                        refEndTimeWindow = lossWindowEnd+timeshift
                        refSignal = get_read_and_normalize_utterance_for_interval(refWavPath, refStartTimeWindow, refEndTimeWindow)
                        #signals.extend(get_read_and_normalize_utterance_for_interval(refWavPath, start, end))
                        refSegments.append(refSignal)

                # s = get_signal_dictionary_normalized(signals)
                # sigs = [np.array(s[i]) for i in range(6)]

                # deviceObjects.append(Device(deviceId, name, config_sf, sigs[0], sigs[1:], [lossInstant, lossInstantArray.index(lossInstant)+1]))

                deviceObjects.append(Device(deviceId, name, config_sf, signal, refSegments,
                                            [lossInstant, lossInstantArray.index(lossInstant) + 1]))
                #embed()

    return deviceObjects




def read_chime_segments_for_utterance():


    ctxWindower = DataWindower(config_windows["preprocess"], config_contextWindow, config_contextOverlap)
    frameWindower = DataWindower(config_windows["extract"], config_fftWindow, config_fftOverlap)
    statsComputer = OnlineStatCompute(config_fqValues)


    uttNames = os.listdir(config_chimeHdf)
    labels = get_labels_chime(int(15*16000 / 2), 15*16000)
    uttSegments = {}
    for uttName in uttNames:
        uttSegments[uttName] = {}

        sessId = uttName[0]
        deviceId = uttName[2:5]

        hypUtt, refUtt = read_hdf_maurizio(("/").join([config_chimeHdf,uttName]))
        hypUtt = normalize_utterance(hypUtt)


        hypUttSegments = ctxWindower.split(hypUtt, config_sf)
        refSegmentsArray = [ctxWindower.split(ref, config_sf) for ref in refUtt]
        for segInd, hypseg in enumerate(hypUttSegments):
            uttSegments[uttName][segInd] = []
            for rSeg in refSegmentsArray:
                seg = get_segment_multidevice_chime_utt(hypseg, rSeg[segInd], int(labels[segInd]), frameWindower)
                statsComputer.update_stats(seg.ref.T)
                statsComputer.update_stats(seg.hyp.T)
                uttSegments[uttName][segInd].append(seg)
        embed()

    mean = statsComputer.get_mean()
    std = statsComputer.get_std()

    return uttSegments, mean, std













####################################
#Maurizio
###################################

def read_power_spectrum_train_and_eval():



    for dataIndex in range(config_numberXmls):
        if dataIndex not in config_problematicIndices:
            deviceMetadatas = read_xml_scene(get_session_path(dataIndex, config_xmlPath))


###########################
# General
###########################

def get_sample_difference(s1, s2):

    FMT = '%H:%M:%S.%f'
    tdelta = datetime.strptime(s2, FMT) - datetime.strptime(s1, FMT)
    seconds = tdelta.total_seconds()
    return int(seconds*config_sf)


def normalize_with_statistics_train(segments, mean,std):
    normalizedSegments = []
    for s in segments:
        h = (s[0] - mean)/std
        r = (s[1] - mean)/std
        label = s[2]
        normalizedSegments.append([h, r, label])
    return normalizedSegments

def normalize_with_statistics_dev(segments, mean,std):

    for k in list(segments.keys()):
        for subk in list(segments[k].keys()):
            for s in segments[k][subk]:
                s.ref = (s.ref - mean) / std
                s.hyp = (s.hyp - mean) / std

    return segments

def get_context_window_start(maxLength, cxtWinLength):

    startIndex = config_uttMargin + config_maxLossLen
    endIndex = maxLength - cxtWinLength - config_maxLossLen - config_uttMargin
    if endIndex < startIndex:
        return None
    return np.random.randint(startIndex, endIndex)



def get_context_window_end(maxLength, cxtWinLength):

    startIndex = config_uttMargin + cxtWinLength + config_maxLossLen
    endIndex = maxLength - config_uttMargin - config_maxLossLen
    if endIndex < startIndex:
        return None
    return np.random.randint(startIndex, endIndex)


def get_pointer_within_context(contextStart, contextEnd):
    return np.random.randint(contextStart + config_cxtMargin, contextEnd - config_cxtMargin)




def get_20log10(X):
    Y = np.log10(X)
    values = np.full(X.shape, 20)
    return np.multiply(Y, values)




def is_nan_in_vector(X):
    zeros = len(np.array(np.where(X == 0)).reshape(-1))
    if zeros > 0:
        print(zeros)
    return zeros > 0

def normalize_signal(x):
    x -= np.mean(x)
    x /= np.max(np.abs(x))
    return x


def normalize_datapoint(X, max_, min_):
    return (X - min_)/(max_ - min_)

def get_hyp_utterance_with_loss_samples_eliminated(X, startLoss, endLoss):
    indices = list(range(startLoss - 1, endLoss - 1))
    return np.delete(X, indices)


def create_utterance(uttMapDictionary, labelMapDict, normHyp=False):  # mini

    utterances = []
    labelMaps = get_labels_for_segments(labelMapDict)

    for k, v in uttMapDictionary.items():
        if k in list(labelMaps.keys()):
            with open(v["ref"], 'rb') as f:
                refSignal, refSf = sfile.read(f)
            with open(v["hyp"]["path"], 'rb') as f:
                hypSignal, hypSf = sfile.read(f)

            if refSf != hypSf:
                logging("different sampling frequency between ref and hyp.")

            if normHyp:
                hyp = normalize_signal(hypSignal)
            else:
                hyp = hypSignal

            utterances.append(Utterance(k,
                                        refSf,
                                        normalize_signal(refSignal),
                                        hyp,
                                        v["hyp"]["startLoss"],
                                        v["hyp"]["endLoss"],
                                        labelMaps[k]))

    return utterances


#############################
# contaminated data with rnn
############################

def get_random_window_start(lenSegment, lenContext, startLoss, lossInterval):
    percentage = np.random.randint(2, 7) / 10
    while (startLoss - (percentage * lenContext) + lossInterval + config_minMargin) > lenSegment\
            or (startLoss - (percentage * lenContext)) < config_minMargin:
        percentage = np.random.randint(2, 7) / 10
    return int(startLoss - (percentage * lenContext))


def is_overlapping(s1, s2, e1, e2):
    return int(max(s1, s2) < min(e1, e2))


def is_overlapping_chime(start, end, lossPoint):
    return lossPoint < end and lossPoint > start

def get_labels(startWindow, startLoss, endLoss):
    indices = list(range(0, int((config_contextWindow * config_sf)), 1600))
    labels = []
    for i, stride in enumerate(indices[:-1]):
        labels.append(is_overlapping(startWindow + stride, startLoss, startWindow + indices[i + 1], endLoss))
    return labels

def get_labels_chime(lossPoint, signalLength):
    indices = list(range(0, signalLength, int((config_CTX*(100-config_contextOverlap)/100))))
    labels = []
    for i, stride in enumerate(indices[:-1]):
        labels.append(is_overlapping_chime(stride, stride+config_CTX, lossPoint))
    return labels


def get_segments_rnn(frameWindower, utt, hypContextWindow, refContextWindow, labels, isLog=config_isLog, statsComputer=None):
    # yuo can pass just name and samlfreq so that it is not data structure dependent
    hypFftValues = []
    refFftValues = []
    hypFrames = frameWindower.split(hypContextWindow, utt.sampleFreq)
    for f in hypFrames:
        hypFftValues.append(extract_magnitude_spectrum_per_frame(f, utt.sampleFreq))
    refFrames = frameWindower.split(refContextWindow, utt.sampleFreq)
    for f in refFrames:
        refFftValues.append(extract_magnitude_spectrum_per_frame(f, utt.sampleFreq))

    hypFftValues = np.array(hypFftValues)
    refFftValues = np.array(refFftValues)

    if isLog:
        hypFftValues = get_20log10(hypFftValues)
        refFftValues = get_20log10(refFftValues)

    if statsComputer is not None:
        statsComputer.update_stats(np.array(hypFftValues).T)
        statsComputer.update_stats(np.array(refFftValues).T)

    seg = Segment(utt.name, hypFftValues, refFftValues, labels)
    return seg


def get_segment_rnn_librosa(utt, hypContextWindow, refContextWindow, labels, isLog=config_isLog, statsComputer=None):


    hypFftValues = abs(librosa.stft(hypContextWindow, n_fft=config_nFFT, hop_length=config_hop)[:-1].T)
    refFftValues = abs(librosa.stft(refContextWindow, n_fft=config_nFFT, hop_length=config_hop)[:-1].T)

    if isLog:
        hypFftValues = get_20log10(hypFftValues)
        refFftValues = get_20log10(refFftValues)

        if is_nan_in_vector(hypFftValues) or is_nan_in_vector(refFftValues):
            print("zero fft values in : ", utt.name)
            return None


    if statsComputer is not None:
        statsComputer.update_stats(np.array(hypFftValues).T)
        statsComputer.update_stats(np.array(refFftValues).T)

    seg = Segment(utt.name, hypFftValues, refFftValues, labels)
    return seg






def create_segments_from_utterance_for_rnn(utterances, statsComputer=None, isLibrosa=config_isLibrosa):


    segments = []
    if not isLibrosa:
        frameWindower = DataWindower(config_windows["extract"], config_fftWindow, config_fftOverlap)

    for index, utt in enumerate(utterances):
        lossInterval = utt.endLoss - utt.startLoss
        #minContextWindow = config_contextWindow * utt.sampleFreq
        minContextWindow = config_CTX

        if lossInterval + config_uttMargin * 2 > minContextWindow:
            continue
        withLossStart = get_random_window_start(len(utt.hypTimeSamples), minContextWindow, utt.startLoss,
                                                lossInterval)
        withoutLossStart = utt.endLoss + config_uttMargin

        refContextWindow1, hypContextWindow1, labels1 = get_windows_with_loss(utt, minContextWindow, withLossStart)
        refContextWindow0, hypContextWindow0, labels0 = get_windows_without_loss(utt, minContextWindow,
                                                                                 withoutLossStart)

        if (len(hypContextWindow1) == minContextWindow and len(refContextWindow1) == minContextWindow):
            if config_isLibrosa:
                seg1 = get_segment_rnn_librosa(utt, hypContextWindow1, refContextWindow1, labels1, statsComputer=statsComputer)
            else: seg1 = get_segments_rnn(frameWindower, utt, hypContextWindow1, refContextWindow1, labels1, statsComputer=statsComputer)

            if seg1 is not None:
                segments.append(seg1)

        if (len(hypContextWindow0) == minContextWindow and len(refContextWindow0) == minContextWindow):
            if config_isLibrosa:
                seg0 = get_segment_rnn_librosa(utt, hypContextWindow0, refContextWindow0, labels0, statsComputer=statsComputer)
            else: seg0 = get_segments_rnn(frameWindower, utt, hypContextWindow0, refContextWindow0, labels0, statsComputer=statsComputer)

            if seg0 is not None:
                segments.append(seg0)

    return segments




########################
# multi-device
########################
def read_sound_file(filePath):
    with open(filePath, 'rb') as f:
        signal, samplFreq = sfile.read(f)

    return signal, samplFreq


def read_and_normalize_utterance(filePath):
    x, _ = sfile.read(filePath)  # use start and stop
    x -= np.mean(x)
    x /= np.max(np.abs(x))
    return x


def get_read_and_normalize_utterance_for_interval(filePath, start, end):
    # x, _ = sfile.read(filePath, start=start-config_normChimeContext, stop=end+config_normChimeContext)  # use start and stop
    # x -= np.mean(x)
    # x /= np.max(np.abs(x))
    # middle = int(len(x)/2)
    # marg = int(config_CTX/2)
    # newStart = middle-marg
    # newEnd = newStart+config_CTX
    # return x[newStart:newEnd]



    x, _ = sfile.read(filePath, start=start, stop= end)
    # x -= np.mean(x)
    # x /= np.max(np.abs(x))
    return x


def normalize_utterance(x):
    x -= np.mean(x)
    x /= np.max(np.abs(x))
    return x


def get_ref_and_hyp(ref, hyp):
    ref_stft = librosa.stft(ref, n_fft=config_nFFT, hop_length=config_hop)[:-1]
    hyp_stft = librosa.stft(hyp, n_fft=config_nFFT, hop_length=config_hop)[:-1]
    return ref_stft, hyp_stft


def get_wave_name(index, id_, channelId):
    name = "{:04d}/p{:04d}_U0{:1d}.CH{:1d}.wav".format(index, index, id_, channelId)
    return name


def get_session_path(index, xmlPath):
    return ("/").join([xmlPath, "session_{:1d}.xml".format(index)])


def get_all_wav_data_for_scene_and_channel(index, channelId, devices):
    signals = []
    lengths = []
    for device in devices:
        id_ = int(device.find("Kinect_Index").text.strip().split()[0])
        name = get_wave_name(index, id_, channelId)
        wavName = ("/").join([config_wavPath, name])
        s, _ = read_sound_file(wavName)
        lengths.append(len(s))
        signals.extend(s)

    return signals, lengths


def get_signal_dictionary_normalized(signals, lengths):
    sigDic = {}
    ind = 0
    for i, l in enumerate(lengths):
        sigDic[i] = signals[ind:ind + l]
        ind += l
    return sigDic



def get_wav_device_data_random_start(index, channelId, devices):

    deviceObjects = []
    contextWindow = int(config_CTX)
    #which device has loss
    noLossIds = [device.find("LostSamples") is None for device in devices]

    print("getting data from scene: ", index)
    for device in devices:
        id_ = int(device.find("Kinect_Index").text.strip().split()[0])
        name = get_wave_name(index, id_, channelId)
        wavName = ("/").join([config_wavPath, name])
        actualDeviceSignal =  read_and_normalize_utterance(wavName)
        if device.find("LostSamples") is not None:
            startLoss = int(device.find("LostSamples").find("First").text.strip())
            endLoss = int(device.find("LostSamples").find("Last").text.strip())
            if (startLoss - int(contextWindow / 2) > config_uttMargin and endLoss + int(contextWindow / 2) > config_uttMargin):
                lossDuration = endLoss-startLoss
                s = get_random_window_start(len(actualDeviceSignal), contextWindow, startLoss, lossDuration)
                deviceSegments = []
                for j in config_deviceIds:
                    if noLossIds[j]:
                        refDevName = get_wave_name(index, j, channelId)
                        refDevWavName = ("/").join([config_wavPath, refDevName])
                        devsig = read_and_normalize_utterance(refDevWavName)
                        deviceSegments.append(devsig[s: s + contextWindow])
                deviceObjects.append(Device(id_,
                                            name,
                                            config_sf,
                                            actualDeviceSignal[s: s + contextWindow],
                                            deviceSegments,
                                            [startLoss, endLoss]))

        if not len(deviceObjects):
            for k in range(config_maxnumSegMulti):
                j = np.random.randint(len(config_deviceIds))
                name = get_wave_name(index, j, channelId)
                wavName = ("/").join([config_wavPath, name])
                actualDeviceSignal = read_and_normalize_utterance(wavName)
                if (len(actualDeviceSignal) > contextWindow + 4000):
                    s = int(len(actualDeviceSignal) / 2) - int(contextWindow / 2)
                    deviceSegments = []
                    for m in config_deviceIds:
                        if m != j and noLossIds[m]: #you do not need to see if there is loss because you entered this if not len(.)
                            refDevName = get_wave_name(index, m, channelId)
                            refDevWavName = ("/").join([config_wavPath, refDevName])
                            devsig = read_and_normalize_utterance(refDevWavName)
                            deviceSegments.append(devsig[s: s + contextWindow])
                    deviceObjects.append(Device(j,
                                                name,
                                                config_sf,
                                                actualDeviceSignal[s: s + contextWindow],
                                                deviceSegments))
    return deviceObjects



def get_wav_device_data(index, channelId, devices):
    signals, lengths = get_all_wav_data_for_scene_and_channel(index, channelId, devices)
    normalizedSignals = normalize_utterance(signals)
    signals = get_signal_dictionary_normalized(normalizedSignals, lengths)
    contextWindow = int(config_CTX)


    print("getting data from scene: ", index)
    deviceObjects = []
    lossIds = [device.find("LostSamples") is None for device in devices]
    for device in devices:
        id_ = int(device.find("Kinect_Index").text.strip().split()[0])
        name = get_wave_name(index, id_, channelId)
        sig = signals[id_]
        if device.find("LostSamples") is not None:
            startLoss = int(device.find("LostSamples").find("First").text.strip())
            endLoss = int(device.find("LostSamples").find("Last").text.strip())
            if (startLoss - int(contextWindow / 2) > config_uttMargin and endLoss + int(contextWindow / 2) > config_uttMargin):
                s = startLoss - int(contextWindow / 2)
                if len(sig[s: s + contextWindow]) < contextWindow:
                    print("not enough in scene", index, " and device: ", id_)
                    continue
                deviceSegments = []
                for j in config_deviceIds:
                    if lossIds[j]:
                        devsig = signals[j]
                        deviceSegments.append(devsig[s: s + contextWindow])
                deviceObjects.append(Device(id_,
                                            name,
                                            16000,
                                            sig[s: s + contextWindow],
                                            deviceSegments,
                                            [startLoss, endLoss]))
    if not len(deviceObjects):
        for k in range(config_maxnumSegMulti):
            j = np.random.randint(len(config_deviceIds))
            name = get_wave_name(index, j, channelId)
            hypsig = signals[j]
            if (len(hypsig) > contextWindow + 4000):
                s = int(len(hypsig) / 2) - int(contextWindow / 2)
                if len(hypsig[s: s + contextWindow]) < contextWindow:
                    print("not enough in scene", index, " and device: ", j)
                deviceSegments = []
                for m in config_deviceIds:
                    if m != j:
                        deviceSegments.append(signals[m][s: s + contextWindow])
                deviceObjects.append(Device(j,
                                            name,
                                            16000,
                                            hypsig[s: s + contextWindow],
                                            deviceSegments))

    return deviceObjects

def get_wav_device_data_with_targets_eval(index, channelId, devices, targets):
    signals, lengths = get_all_wav_data_for_scene_and_channel(index, channelId, devices)
    normalizedSignals = normalize_utterance(signals)
    signals = get_signal_dictionary_normalized(normalizedSignals, lengths)
    contextWindow = int(config_CTX)


    print("getting data from scene: ", index)
    deviceObjects = []
    lossIds = [device.find("LostSamples") is None for device in devices]
    for device in devices:
        id_ = int(device.find("Kinect_Index").text.strip().split()[0])
        name = get_wave_name(index, id_, channelId)
        sig = signals[id_]
        if device.find("LostSamples") is not None:
            startLoss = int(device.find("LostSamples").find("First").text.strip())
            endLoss = int(device.find("LostSamples").find("Last").text.strip())
            if (startLoss - int(contextWindow / 2) > config_uttMargin and endLoss + int(contextWindow / 2) > config_uttMargin):
                s = startLoss - int(contextWindow / 2)
                deviceSegments = []
                for j in config_deviceIds:
                    if j != id_:
                        devsig = signals[j]
                        deviceSegments.append(devsig[s: s + contextWindow])
                deviceObjects.append(Device(id_,
                                            name,
                                            16000,
                                            sig[s: s + contextWindow],
                                            deviceSegments,
                                            [startLoss, endLoss]))
    if not len(deviceObjects):
        for j in list(targets[index].keys()):
            name = get_wave_name(index, j, channelId)
            hypsig = signals[j]
            if (len(hypsig) > contextWindow + 4000):
                s = int(len(hypsig) / 2) - int(contextWindow / 2)
                deviceSegments = []
                for m in config_deviceIds:
                    if m != j:
                        deviceSegments.append(signals[m][s: s + contextWindow])
                deviceObjects.append(Device(j,
                                            name,
                                            16000,
                                            hypsig[s: s + contextWindow],
                                            deviceSegments))

    return deviceObjects

def get_wav_device_data_with_targets(index, channelId, devices, targets, isDev=False):
    signals, lengths = get_all_wav_data_for_scene_and_channel(index, channelId, devices)
    normalizedSignals = normalize_utterance(signals)
    signals = get_signal_dictionary_normalized(normalizedSignals, lengths)
    contextWindow = int(config_CTX)


    print("getting data from scene: ", index)
    deviceObjects = []
    lossIds = [device.find("LostSamples") is None for device in devices]
    for device in devices:
        id_ = int(device.find("Kinect_Index").text.strip().split()[0])
        name = get_wave_name(index, id_, channelId)
        sig = signals[id_]
        if device.find("LostSamples") is not None:
            startLoss = int(device.find("LostSamples").find("First").text.strip())
            endLoss = int(device.find("LostSamples").find("Last").text.strip())
            if (startLoss - int(contextWindow / 2) > config_uttMargin and endLoss + int(contextWindow / 2) > config_uttMargin):
                s = startLoss - int(contextWindow / 2)
                deviceSegments = []
                for j in config_deviceIds:
                    if lossIds[j] and not isDev:
                        devsig = signals[j]
                        deviceSegments.append(devsig[s: s + contextWindow])
                    if j != id_:
                        devsig = signals[j]
                        deviceSegments.append(devsig[s: s + contextWindow])

                deviceObjects.append(Device(id_,
                                            name,
                                            16000,
                                            sig[s: s + contextWindow],
                                            deviceSegments,
                                            [startLoss, endLoss]))
    if not len(deviceObjects):
        for j in list(targets[index].keys()):
            name = get_wave_name(index, j, channelId)
            hypsig = signals[j]
            if (len(hypsig) > contextWindow + 4000):
                s = int(len(hypsig) / 2) - int(contextWindow / 2)
                deviceSegments = []
                for m in config_deviceIds:
                    if m != j:
                        deviceSegments.append(signals[m][s: s + contextWindow])
                deviceObjects.append(Device(j,
                                            name,
                                            16000,
                                            hypsig[s: s + contextWindow],
                                            deviceSegments))

    return deviceObjects


def read_xml_scene(xmlPath):
    tree = ET.parse(xmlPath)
    root = tree.getroot()
    # sampFreq = int(root.find("General").find("NominalSamplingRate").text.strip())
    # totLength = int(root.find("General").find("TotalLength_Samples").text.strip())
    devices = root.find("SpatialData").find("Room").find("Kinect_features").findall("Kinect_Sampling")

    return devices





def read_scenes_for_eval(startInd, endInd, indices):


    targets = get_pickle("shortscenes_all.pickle")

    newTargets, segments = get_scenes_segments_with_targets(startInd, endInd, targets, isDev=True)

    for i in indices:
        #modelPath = "models/lr_5e-05_isLog_True_fft_0.032_24.pth_att_noPre"

        modelPath = (".").join([("_").join([config_modelPath, str(i)]), "pth"])
        if config_isAttention:
            modelPath+="_att"
        if not config_isPretrained:
            modelPath+="_noPre"

        evaluate_model_with_dev_segments_multiple_devices(modelPath, newTargets, segments)




def read_scenes_for_train(startInd, endInd):

    targets = get_pickle("shortscenes_all.pickle")
    newTargets, segments = get_scenes_segments_with_targets(startInd, endInd, targets)


def get_scenes_segments_with_targets(startInd, endInd, targets, mean=None, std=None, isDev=False):


    allDevices = {}

    for dataIndex in range(startInd, endInd):
        if dataIndex not in config_problematicIndices:
            deviceMetadatas = read_xml_scene(get_session_path(dataIndex, config_xmlPath))
            allDevices[dataIndex] = get_wav_device_data_with_targets(dataIndex, config_selectedChannel, deviceMetadatas, targets, isDev=isDev)



    if not isDev:
        newTargets, segments, hdfSegments, devsegs = prepare_segments_and_targets_for_multiple_devices_rnn(allDevices,
                                                                                                           mean, std,
                                                                                                           isDev=isDev)
        hdfPath = ("").join([config_hdfTrainMulti, ("_").join(["/train", sceneName])])
        write_train_segments_as_hdf(hdfSegments, hdfPath)
        return newTargets, segments

    else:
        newTargets, devSegments, mean, std = prepare_segments_and_targets_for_multiple_devices_rnn(allDevices, isDev=isDev)
        for dKey in list(devSegments.keys()):
            for device in devSegments[dKey]:
                for seg in devSegments[dKey][device]:
                    seg.hyp = (seg.hyp - mean) / std
                    seg.ref = (seg.ref - mean) / std

        return newTargets, devSegments








#ok
def prepare_segments_and_targets_for_multiple_devices_rnn(allDevices, mean=None, std=None, isDev=False, isLibrosa=config_isLibrosa):

    #remember that if you do not give mean and std it will calculate, if not the data is already normalized
    # for training

    if not isLibrosa:
        frameWindower = DataWindower(config_windows["extract"], config_fftWindow, config_fftOverlap)
    else: frameWindower=None

    if mean is None or std is None:
        statsComputer = OnlineStatCompute(config_fqValues)


    trainSegments = []
    segments = {}
    targets = {}

    hdfSegments = []

    for dKey in list(allDevices.keys()):
        targets[dKey] = {}
        segments[dKey] = {}
        for device in allDevices[dKey]:
            targets[dKey][device.deviceId] = []
            segments[dKey][device.deviceId] = []
            if device.lossInterval is not None:
                labels = 1
                targets[dKey][device.deviceId].append(
                    {"label": 1, "startLoss": device.lossInterval[0], "endLoss": device.lossInterval[1]})
            else:
                labels = 0
                targets[dKey][device.deviceId].append({"label": 0})

            for refSeg in device.refSegments:
                seg = get_segment_multidevice(device, device.contextWindow, refSeg, labels, frameWindower=frameWindower)

                if seg is None or \
                    seg.ref.shape != (config_tmValues, config_fqValues) or \
                    seg.hyp.shape != (config_tmValues, config_fqValues):
                    print("segment not valid: ", dKey, " , ", device.deviceId)
                    continue
                if mean is None or std is None:
                    statsComputer.update_stats(seg.ref.T)
                    statsComputer.update_stats(seg.hyp.T)
                else:
                    seg.ref = (seg.ref - mean) / std
                    seg.hyp = (seg.hyp - mean) / std


                trainSegments.append([seg.hyp, seg.ref, seg.label])
                segments[dKey][device.deviceId].append(seg)
                hdfSegments.append(seg)

    if mean is None or std is None:
        mean = statsComputer.get_mean()
        std = statsComputer.get_std()
        if not isDev:
            write_pickle(mean, ("").join(["stats/", sceneName, "_mean.pickle"]))
            write_pickle(std, ("").join(["stats/", sceneName, "_std.pickle"]))


    if isDev:
        return targets, segments, mean, std
    return targets, trainSegments, hdfSegments, segments



def prepare_segments_and_targets_chime(devices, idSessions, mean=None, std=None):


    labelName = ("/").join([config_chimePath, "labels-chime.pickle"])
    labelDict = get_pickle(labelName)
    frameWindower = DataWindower(config_windows["extract"], config_fftWindow, config_fftOverlap)

    if mean is None or std is None:
        statsComputer = OnlineStatCompute(config_fqValues)



    segments = {}
    targets = {}



    for id in idSessions:
        segments[id] = {}
        targets[id] = {}



    for device in devices:
        sessId = int(device.name[3]) #name
        key = str(device.deviceId)+"-"+str(device.lossInterval[1])
        label = labelDict[sessId][device.deviceId][device.lossInterval[1]-1]
        targets[sessId][key] = [{"label": label}]
        segments[sessId][key] = []
        for refSeg in device.refSegments:
            seg = get_segment_multidevice(device, device.contextWindow, refSeg, label,frameWindower)
            if mean is None or std is None:
                statsComputer.update_stats(seg.ref.T)
                statsComputer.update_stats(seg.hyp.T)
            else:
                seg.ref = (seg.ref - mean) / std
                seg.hyp = (seg.hyp - mean) / std

            segments[sessId][key].append(seg)

    if mean is None or std is None:
        mean = statsComputer.get_mean()
        std = statsComputer.get_std()
        write_pickle(mean, "stats/chime-mean.pickle")
        write_pickle(std, "stats/chime-std.pickle")
        return targets, segments




    return targets, segments








def create_nested_dic_segments_from_multiple_devices_rnn_librosa(allDevices, isTrainSegs=True, isHdf=False):

    targets = {}

    if isTrainSegs:
        trainSegments = []
    else: segments = {}


    for dKey in list(allDevices.keys()):
        targets[dKey] = {}
        if not isTrainSegs:
            segments[dKey] = {}

        for device in allDevices[dKey]:
            targets[dKey][device.deviceId] = []
            if device.lossInterval is not None:
                labels = 1
                targets[dKey][device.deviceId].append(
                    {"label": 1, "startLoss": device.lossInterval[0], "endLoss": device.lossInterval[1]})
            else:
                labels = 0
                targets[dKey][device.deviceId].append({"label": 0})
            if not isTrainSegs:
                segments[dKey][device.deviceId] = []

            for refSeg in device.refSegments:
                seg = get_segment_multidevice(device, device.contextWindow, refSeg, labels)

                if seg is None or seg.ref.shape != (config_tmValues, config_fqValues) or seg.hyp.shape != (config_tmValues, config_fqValues):
                    print(seg.utteranceName)
                    continue

                if isTrainSegs:
                    if isHdf:
                        trainSegments.append(seg)
                    else: trainSegments.append([seg.hyp, seg.ref, seg.label])

                else: segments[dKey][device.deviceId].append(seg)

    if isTrainSegs:
        return targets, trainSegments

    return targets, segments






def create_nested_dic_segments_from_multiple_devices_rnn(allDevices):

    statComputer = OnlineStatCompute(config_fqValues)
    frameWindower = DataWindower(config_windows["extract"], config_fftWindow, config_fftOverlap)

    segments = {}
    trainSegments = []
    targets = {}


    for dKey in list(allDevices.keys()):
        targets[dKey] = {}
        segments[dKey] = {}
        for device in allDevices[dKey]:
            targets[dKey][device.deviceId] = []
            if device.lossInterval is not None:
                labels = 1
                targets[dKey][device.deviceId].append(
                    {"label": 1, "startLoss": device.lossInterval[0], "endLoss": device.lossInterval[1]})
            else:
                labels = 0
                targets[dKey][device.deviceId].append({"label": 0})
            segments[dKey][device.deviceId] = []

            for refSeg in device.refSegments:
                seg = get_segment_multidevice(device, device.contextWindow, refSeg, labels, frameWindower)


                if seg.ref.shape != (config_tmValues-1, config_fqValues) or seg.hyp.shape != (config_tmValues-1, config_fqValues):
                    print(seg.utteranceName)
                    continue

                statComputer.update_stats(np.array(seg.ref.T))
                statComputer.update_stats(np.array(seg.hyp.T))
                segments[dKey][device.deviceId].append(seg)
                trainSegments.append([seg.hyp, seg.ref, seg.label])

    newMean =  statComputer.get_mean()
    newStd = statComputer.get_std()

    #return targets, segments, newMean, newStd
    return targets, trainSegments, newMean, newStd












####################
# Old model
###################


def get_hyp_utterance_with_label(X, startLoss, endLoss, hasLoss=False):
    if hasLoss:
        indices = list(range(startLoss - 1, endLoss - 1))
        X = np.delete(X, indices)

    return [X, int(hasLoss)]


def create_segments_from_utterance(uttMapDictionary):  # mini
    """

    :param
    - uttMapDictionary: is a nested dictionary of paths, having as first level keys the name of the utterances
                        and as a second hierarchy keys ref and hyp. ref can have an array of paths
                        ref is/are the device(s) to compare with, hyp is the segment with/without loss.
                        it has a dictionary associated with path end and start time of loss
    - lossNames: a list with segment names which has loss
    - noLossNames: similar to lossNames but without loss


    :return: time samples referred to the CNN input segment taken from hypothesis and reference utterances
    """

    contextWindower = DataWindower(config_windows["preprocess"], config_contextWindow, config_contextOverlap)
    lossWindower = DataWindower(config_windows["preprocess"], config_lossWindow, config_lossOverlap)
    frameWindower = DataWindower(config_windows["extract"], config_fftWindow, config_fftOverlap)
    # the division into segemtns with and without loss
    uttNameLoss, uttNameNoLoss = read_utterance_names_with_and_without_loss(config_devFilenameSeparation)

    for k, v in uttMapDictionary.items():
        if k in uttNameLoss:
            hasLoss = True
        elif k in uttNameNoLoss:
            hasLoss = False
        else:
            continue

        with open(v["ref"], 'rb') as f:
            refSignal, refSf = sfile.read(f)
        with open(v["hyp"]["path"], 'rb') as f:
            hypSignal, hypSf = sfile.read(f)

        # ToDo: check if sf of both is the same
        minContextWindow = config_contextWindow * refSf
        minLossWindow = config_lossWindow * refSf

        # prepare the two time sample segments
        [hypSignal, label] = get_hyp_utterance_with_label(hypSignal,
                                                          v["hyp"]["startLoss"],
                                                          v["hyp"]["endLoss"],
                                                          hasLoss)
        refSignal = normalize_signal(refSignal)[0]
        if (len(refSignal) < minContextWindow or len(hypSignal) < minContextWindow):
            continue
        refSigWindowed = contextWindower.split(refSignal, refSf)
        hypSigWindowed = contextWindower.split(hypSignal, hypSf)

        # segment and extract the features and save to hdf fro each utterance
        timeSegments = []
        # print("working on: " + k)
        for refWindowId, refWindow in enumerate(refSigWindowed):
            if refWindowId < len(hypSigWindowed):
                if len(hypSigWindowed[refWindowId]) < minLossWindow:
                    continue
                lossSigWindows = lossWindower.split(hypSigWindowed[refWindowId], hypSf)
                for lossWindow in lossSigWindows:
                    timeSegments.append([lossWindow, refWindow])

        fftSegments = per_segment_extractor(timeSegments, frameWindower, refSf)
        write_hdf_segment_with_loss(fftSegments, label, k)


def get_labels_for_segments(labelMapDictPath):
    """
    remember that we could have different labaling strategies
    so we could prepare a-priori the labels with respect to the
    modeling task
    for now the used strategies are:
    - divide into two groups
    - take contextwindows where the loss occurs as withloss nad rest without loss

    """

    with open(labelMapDictPath, 'rb') as fp:
        sDict = pickle.load(fp)

    return sDict


def get_value_from_distribution(distribution, parameters):
    if distribution == Distribution.NORMAL:
        return np.random.normal(parameters[0], parameters[1])
    if distribution == Distribution.UNIFORM:
        return np.random.uniform(parameters[0], parameters[1])


def window_and_extract_from_utterance(hypContextWindow, refContextWindow,
                                      lossWindower, frameWindower, sampleFreq, name, labels):
    # segment and extract the features and save to hdf fro each utterance
    timeSegments = []
    segments = []
    # print("working on: " + utt.name)

    lossSigWindows = lossWindower.split(hypContextWindow, sampleFreq)
    for lossId, lossWindow in enumerate(lossSigWindows):
        timeSegments.append([lossWindow, refContextWindow])

    fftSegments = per_segment_extractor(timeSegments, frameWindower, sampleFreq)

    if len([i for i in labels if not i]) == 7:
        segments.append(Segment(name, fftSegments[0], fftSegments[1], labels[0]))
    else:
        for index in range(len(fftSegments[0])):
            segments.append(Segment(name, fftSegments[0][index], fftSegments[1][index], labels[index]))

    return segments


def get_windows_with_loss(utt, minContextWindow, start):
    hypTimeSamplesWithloss = get_hyp_utterance_with_loss_samples_eliminated(utt.hypTimeSamples,
                                                                            utt.startLoss, utt.endLoss)

    refContextWindow = utt.refTimeSamples[start:start + minContextWindow]
    hypContextWindow = hypTimeSamplesWithloss[start:start + minContextWindow]
    # labels = get_labels(start, utt.startLoss, utt.endLoss)
    labels = 1
    return refContextWindow, hypContextWindow, labels


def get_windows_without_loss(utt, minContextWindow, start):
    refContextWindow = utt.refTimeSamples[start:start + minContextWindow]
    hypContextWindow = utt.hypTimeSamples[start:start + minContextWindow]
    # labels = [0 for _ in range(7)]
    labels = 0
    return refContextWindow, hypContextWindow, labels





def create_segments_from_utterance_with_high_time_scale(utterances):
    # take just one context window where the loss occurs

    max = - math.inf
    min = math.inf
    segments = []

    lossWindower = DataWindower(config_windows["preprocess"], config_lossWindow, config_lossOverlap)
    frameWindower = DataWindower(config_windows["extract"], config_fftWindow, config_fftOverlap)

    for index, utt in enumerate(utterances):
        lossInterval = utt.endLoss - utt.startLoss
        minContextWindow = config_contextWindow * utt.sampleFreq

        if index % 2 == 0:
            margin = int((minContextWindow - lossInterval) / 2)
            start = utt.startLoss - margin
            refContextWindow = utt.refTimeSamples[start:start + minContextWindow]
            hypContextWindow = utt.hypTimeSamples[start:start + minContextWindow]
            labels = [0 for _ in range(7)]

        else:
            if lossInterval + 4000 > minContextWindow:
                continue
            hypTimeSamplesWithloss = get_hyp_utterance_with_loss_samples_eliminated(utt.hypTimeSamples,
                                                                                    utt.startLoss, utt.endLoss)
            start = utt.startLoss - int(minContextWindow / 2)
            refContextWindow = utt.refTimeSamples[start:start + minContextWindow]
            hypContextWindow = hypTimeSamplesWithloss[start:start + minContextWindow]
            labels = [1 for _ in range(7)]

        # segment and extract the features and save to hdf fro each utterance
        timeSegments = []
        # print("working on: " + utt.name)

        lossSigWindows = lossWindower.split(hypContextWindow, utt.sampleFreq)
        for lossId, lossWindow in enumerate(lossSigWindows):
            timeSegments.append([lossWindow, refContextWindow])

        fftSegments = per_segment_extractor(timeSegments, frameWindower, utt.sampleFreq)

        for index, s in enumerate(fftSegments):
            if np.max(s) > max:
                max = np.max(s)
            if np.min(s) < min:
                min = np.min(s)

            segments.append(Segment(utt.name, s, labels[index]))

    return segments, max, min


def create_segments_from_utterance_old(utterances):
    max = - math.inf
    min = math.inf
    segments = []
    contextWindower = DataWindower(config_windows["preprocess"], config_contextWindow, config_contextOverlap)
    lossWindower = DataWindower(config_windows["preprocess"], config_lossWindow, config_lossOverlap)
    frameWindower = DataWindower(config_windows["extract"], config_fftWindow, config_fftOverlap)

    for utt in utterances:
        minContextWindow = config_contextWindow * utt.sampleFreq
        minLossWindow = config_lossWindow * utt.sampleFreq

        if (len(utt.refTimeSamples) < minContextWindow or len(utt.hypTimeSamples) < minContextWindow):
            continue

        # segment and extract the features and save to hdf fro each utterance
        timeSegments = []
        labels = []
        # print("working on: " + utt.name)
        hypTimeSamplesWithloss = get_hyp_utterance_with_loss_samples_eliminated(utt.hypTimeSamples,
                                                                                utt.startLoss, utt.endLoss)
        hypSigWindowedWithoutLoss = contextWindower.split(utt.hypTimeSamples, utt.sampleFreq)
        hypSigWindowedWithLoss = contextWindower.split(hypTimeSamplesWithloss, utt.sampleFreq)
        refSigWindowed = contextWindower.split(utt.refTimeSamples, utt.sampleFreq)

        for refWindowId, refWindow in enumerate(refSigWindowed):
            if refWindowId > len(hypSigWindowedWithLoss) - 1:
                continue
            if utt.frameLabels[refWindowId][0] == 0:
                hypContextWindow = hypSigWindowedWithoutLoss[refWindowId]
            else:
                hypContextWindow = hypSigWindowedWithLoss[refWindowId]

            lossSigWindows = lossWindower.split(hypContextWindow, utt.sampleFreq)
            for lossId, lossWindow in enumerate(lossSigWindows):
                timeSegments.append([lossWindow, refWindow])
                labels.append(utt.frameLabels[refWindowId][lossId])

        fftSegments = per_segment_extractor(timeSegments, frameWindower, utt.sampleFreq)

        for index, s in enumerate(fftSegments):
            if np.max(s) > max:
                max = np.max(s)
            if np.min(s) < min:
                min = np.min(s)

            segments.append(Segment(utt.name, s, labels[index]))
    return segments, max, min


def get_20log10(X):
    Y = np.log10(X)
    values = np.full(X.shape, 20)
    return np.multiply(Y, values)


def create_dev_segments_from_utterance_for_hdf(utterances):
    max = - math.inf
    min = math.inf
    segments = []
    contextWindower = DataWindower(config_windows["preprocess"], config_contextWindow, config_contextOverlap)
    lossWindower = DataWindower(config_windows["preprocess"], config_lossWindow, config_lossOverlap)
    frameWindower = DataWindower(config_windows["extract"], config_fftWindow, config_fftOverlap)

    lossInfo = {}

    for utt in utterances:

        minContextWindow = config_contextWindow * utt.sampleFreq

        if (len(utt.refTimeSamples) < minContextWindow or len(utt.hypTimeSamples) < minContextWindow):
            continue

        # segment and extract the features and save to hdf fro each utterance
        timeSegments = []
        labels = []
        # print("working on: " + utt.name)
        hypTimeSamplesWithloss = get_hyp_utterance_with_loss_samples_eliminated(utt.hypTimeSamples,
                                                                                utt.startLoss, utt.endLoss)
        hypSigWindowedWithoutLoss = contextWindower.split(utt.hypTimeSamples, utt.sampleFreq)
        hypSigWindowedWithLoss = contextWindower.split(hypTimeSamplesWithloss, utt.sampleFreq)
        refSigWindowed = contextWindower.split(utt.refTimeSamples, utt.sampleFreq)

        for refWindowId, refWindow in enumerate(refSigWindowed):
            if refWindowId > len(hypSigWindowedWithLoss) - 1:
                continue
            if utt.frameLabels[refWindowId][0] == 0:
                hypContextWindow = hypSigWindowedWithoutLoss[refWindowId]
            else:
                hypContextWindow = hypSigWindowedWithLoss[refWindowId]

            lossSigWindows = lossWindower.split(hypContextWindow, utt.sampleFreq)
            for lossId, lossWindow in enumerate(lossSigWindows):
                timeSegments.append([lossWindow, refWindow])
                labels.append(utt.frameLabels[refWindowId][lossId])

        fftSegments = per_segment_extractor(timeSegments, frameWindower, utt.sampleFreq)

        for index, s in enumerate(fftSegments):
            if np.max(s) > max:
                max = np.max(s)
            if np.min(s) < min:
                min = np.min(s)
            segments.append(Segment(utt.name, s, labels[index]))

    for s in segments:
        write_hdf(s, "data/dev/fft/", s.utteranceName, max, min)

    return segments, max, min


def write_hdf(segment, destPath, filename, max, min):
    hdf5File = h5py.File(".".join([destPath, filename, "hdf"]), 'w')
    dset = hdf5File.create_dataset('data', segment.features.shape, data=segment.features)
    dset.attrs["sampling_freq"] = 16000
    dset.attrs["startLoss"] = segment.lossInterval[0]
    dset.attrs["endLoss"] = segment.lossInterval[1]
    dset.attrs["max"] = max
    dset.attrs["min"] = min
    hdf5File.close()



def get_pickle(filename):
    with open(filename, "rb") as f:
        p = pickle.load(f)
    return p

def write_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=2)









