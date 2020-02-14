from enum import Enum
from scipy import signal
import logging
import numpy as np

class Session(object):
    def __init__(self, sessionId ):

        self.sessionId   = sessionId
        self.devices     = []



class Device(object):
    def __init__(self, deviceId, name, sampleFreq, contextWindow, refSegments, lossInterval=None):
        self.deviceId = deviceId
        self.name = name
        self.sampleFreq = sampleFreq
        self.contextWindow = contextWindow
        self.refSegments = refSegments
        self.lossInterval = lossInterval



class Utterance(object):
    def __init__(self, name, sampleFreq, refTimeSamples, hypTimeSamples, startLoss, endLoss, frameLabels):
        self.name = name
        self.sampleFreq = sampleFreq
        self.refTimeSamples = refTimeSamples #it is already normalized
        self.hypTimeSamples  = hypTimeSamples
        self.startLoss = startLoss
        self.endLoss = endLoss
        self.frameLabels = frameLabels



class Segment(object):
    def __init__(self, name, hypFeatures, refFeatures, label):
        self.utteranceName = name
        self.hyp = hypFeatures
        self.ref = refFeatures
        self.label = label

    def update_normalized_data(self, index):
        self.hyp = self.features[:index]
        self.ref = self.features[index:]



class DataWindower(object):
    def __init__(self, windowType, windowSizeSec, overlap):

        self.windowType = windowType
        self.overlap = round(float(overlap / 100), 2)
        self.windowSizeSec = windowSizeSec

    def apply_window(self, windows, window_size):
        """
        :param windows: the signals splitted into time windows
        :param window_size: the number of samples in the window
        :return: the windows weighted by the specified window function
        """
        windower = getattr(signal, self.windowType)
        window = windower(window_size)

        return windows * window

    def split(self, samples, samplingFreq, fft=False):
        """
        :param rec: the recording object holding the signals and all information needed
        :return: the signals split into time windows of the specified size
        """
        windowSize = int(samplingFreq * self.windowSizeSec)
        overlapSize = int(self.overlap * windowSize)
        stride = windowSize - overlapSize

        if stride == 0:
            logging.error("Time windows cannot have an overlap of 100%.")

        # inspired by code of robin tibor schirrmeister
        signalCrops = []
        iStart = 0
        for iStart in range(0, samples.shape[-1] - windowSize + 1, stride):
            signalCrops.append(np.take(samples, range(iStart, iStart + windowSize), axis=-1, ))
        start, end = samples.shape[-1] - windowSize, samples.shape[-1]
        if iStart + windowSize < end and not fft:
            signalCrops.append(np.take(samples, range(start, end), axis=-1, ))

        return self.apply_window(np.array(signalCrops), windowSize)








class Normalizer(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        #self.mean = np.mean(np.mean(datapoints, axis = 1), axis = 0)
        #self.std = np.std(np.std(datapoints, axis = 1), axis = 0)


    def normalize(self, datapoint):
        return (datapoint -self.mean)/self.std


class Distribution(Enum):
    NORMAL  = 1
    UNIFORM = 2


class SignalType(Enum):
    REF  = 1
    HYP = 2


