import numpy as np
from librosa import stft
import pyfftw
pyfftw.interfaces.cache.enable()
pyfftw.interfaces.cache.set_keepalive_time(3600)

from config import *
from IPython import embed
from data_structures import DataWindower, Segment



def get_20log10(X):
    Y = np.log10(X)
    values = np.full(X.shape, 20)
    return np.multiply(Y, values)


def extract_magnitude_spectrum_per_frame(frame, sf):
    #ToDo: for now frequency points, in future filter banks
    """

    :param frame: time values in a frame
    :param sf: sampling frequency
    :return:
    """

    #n = len(frame)
    #k = np.arange(len(frame))
    #T = n / sf
    #frq = k / T

    frq = np.arange(len(frame)) / (len(frame) / sf)
    #index = list(frq).index(config_bandLimit)
    fftValues = pyfftw.interfaces.numpy_fft.fft(frame)
    absFFTValues = abs(fftValues[:config_fqValues])
    #logAbs = get_20log10(absFFTValues[:index])

    return absFFTValues


def per_segment_extractor(segments, frameWindower, sf):
    """

    :param segments: an array of array of segments taken from the utterance. Each array has
                    as the first element the hyp segment and as second element the ref
    :param dataWindower: the object which will split into frames each subsegment
    :return: same datastructure if input but this time with fft values
    """

    hypSequence = []
    refSequence = []
    for s in segments:
        hypFftSegments = []
        refFftSegments = []
        hypFrames = frameWindower.split(s[0],sf)
        for f in hypFrames:
            hypFftSegments.append(extract_magnitude_spectrum_per_frame(f, sf))
        hypSequence.append(hypFftSegments)
        refFrames = frameWindower.split(s[1],sf)
        for f in refFrames:
            refFftSegments.append(extract_magnitude_spectrum_per_frame(f, sf))
        refSequence.append(refFftSegments)



    return [hypSequence, refSequence]


def get_magnitude_sepctrum_from_segments(segments, useLibrosa=False):

    fftSegments = []

    if useLibrosa:
        for s in segments:
            fftS = stft(s, n_fft=config_nFFT, win_length=config_nFFT, hop_length=config_hop)[:-1]
            fftSegments.append(abs(fftS))

    else:
        frameWindower = DataWindower(config_windows["extract"], config_fftWindow, config_fftOverlap)
        for s in segments:
            frameValues = frameWindower.split(s, config_sf)
            fftValues = []
            for f in frameValues:
                fftValues.append(extract_magnitude_spectrum_per_frame(f, config_sf))
            fftSegments.append(abs(fftValues))

    return fftSegments


def get_segment_multidevice(utt, hypContextWindow, refContextWindow, labels, frameWindower=None, isLog = config_isLog):
    # yuo can pass just name and samlfreq so that it is not data structure dependent
    #ref_stft = stft(refContextWindow, n_fft=config_nFFT, hop_length=config_hop)[:-1]
    #hyp_stft = stft(hypContextWindow, n_fft=config_nFFT, hop_length=config_hop)[:-1]

    if frameWindower is not None:
        hypFftValues = []
        refFftValues = []

        hypFrames = frameWindower.split(hypContextWindow, config_sf)
        for f in hypFrames:
            hypFftValues.append(extract_magnitude_spectrum_per_frame(f, config_sf))

        refFrames = frameWindower.split(refContextWindow, config_sf)
        for f in refFrames:
            refFftValues.append(extract_magnitude_spectrum_per_frame(f, config_sf))

        hypFftValues = np.array(hypFftValues)
        refFftValues = np.array(refFftValues)


    else:
        hypFftValues = abs(stft(hypContextWindow, n_fft=config_nFFT, hop_length=config_hop)[:-1].T)
        refFftValues = abs(stft(refContextWindow, n_fft=config_nFFT, hop_length=config_hop)[:-1].T)


    if isLog:
        hyp = get_20log10(hypFftValues)
        ref = get_20log10(refFftValues)
        if is_nan_in_vector(hyp) or is_nan_in_vector(ref):
            print(utt.name)
            return None

    else:
        hyp = hypFftValues
        ref = refFftValues


    return Segment(utt.name, hyp, ref, labels)

def get_segment_multidevice_chime_utt(hypContextWindow, refContextWindow, labels, frameWindower=None, isLog = config_isLog):
    # yuo can pass just name and samlfreq so that it is not data structure dependent
    #ref_stft = stft(refContextWindow, n_fft=config_nFFT, hop_length=config_hop)[:-1]
    #hyp_stft = stft(hypContextWindow, n_fft=config_nFFT, hop_length=config_hop)[:-1]

    if frameWindower is not None:
        hypFftValues = []
        refFftValues = []

        hypFrames = frameWindower.split(hypContextWindow, config_sf)
        for f in hypFrames:
            hypFftValues.append(extract_magnitude_spectrum_per_frame(f, config_sf))

        refFrames = frameWindower.split(refContextWindow, config_sf)
        for f in refFrames:
            refFftValues.append(extract_magnitude_spectrum_per_frame(f, config_sf))

        hypFftValues = np.array(hypFftValues)
        refFftValues = np.array(refFftValues)


    else:
        hypFftValues = abs(stft(hypContextWindow, n_fft=config_nFFT, hop_length=config_hop)[:-1].T)
        refFftValues = abs(stft(refContextWindow, n_fft=config_nFFT, hop_length=config_hop)[:-1].T)


    if isLog:
        hyp = get_20log10(hypFftValues)
        ref = get_20log10(refFftValues)
        if is_nan_in_vector(hyp) or is_nan_in_vector(ref):
            print("none found")
            return None

    else:
        hyp = hypFftValues
        ref = refFftValues


    return Segment("chime", hyp, ref, labels)






def is_nan_in_vector(X):
    zeros = len(np.array(np.where(X == 0)).reshape(-1))
    if zeros > 0:
        print(zeros)
    return zeros > 0


