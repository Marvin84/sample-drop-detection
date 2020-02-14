import h5py
import os
import pickle
import soundfile as sfile
import xml.etree.ElementTree as ET


from config import *
from data_structures import *



def write_hdf_segment_with_loss(segment, label, filename):

    path = "/".join(["data", "dev", "fft", filename])

    hdf5File = h5py.File(".".join([path, "hdf"]), 'w')
    dset = hdf5File.create_dataset('data', np.array(segment).shape, data=np.array(segment))
    dset.attrs["label"] = label
    dset.attrs["sampling_freq"] = config_sf
    #dset.attrs["contextWindowInfo"] = {"size": config_contextWindow, "overlap": config_contextOverlap}
    #dset.attrs["lossWindowInfo"] = {"size": config_lossWindow, "overlap": config_lossOverlap}
    #dset.attrs["lossInfo"] = {"dictMap": config_trainLossInfo, "separation": config_trainFilenameSeparation}
    hdf5File.close()
    print(filename+" written.")


def walk_from_root_and_take_paths(root, fileType="flac"):
    segments = {}
    for subdir, dirs, files in os.walk(root):
        for file in files:
            if file.endswith(fileType):
                fExtension = "".join([".", fileType])
                parts = file.split("-")
                cleanName = "-".join([parts[0], parts[1], parts[2].split("_")[0]])
                p = os.path.join(root, subdir, file)
                segments[cleanName] = p
    return segments




def create_utterance_dict_mappings(refPath, hypPath, lossInfoFilename):
    #ToDo: for now consider one device, extend it for chime5
    """
    :param refPath: root directory for ref utterances
    :param hypPath: root directory for hyp utterances
    :param lossInfo: a dictionary which associates to each segment the start and end time of loss
    :return: nested dictionary of mapped paths
    """
    refPathsDict = walk_from_root_and_take_paths(refPath)
    hypPathsDict = walk_from_root_and_take_paths(hypPath)
    with open(lossInfoFilename, "rb") as bf:
        infoDict = pickle.load(bf)


    newNestedDict = {}
    for k,v in infoDict.items():
        k = k.split("_")[0]
        newNestedDict[k] = {"name":k,
                            "ref": refPathsDict[k],
                            "hyp": {"path": hypPathsDict[k],
                                    "startLoss": int(v[0]),
                                    "endLoss"  : int(v[1])}}
    return newNestedDict







def read_and_normalize_utterance(filePath):
    x, fs = sfile.read(filePath) #use start and stop
    x -= np.mean(x)
    x /= np.max(np.abs(x))
    return x, fs


def read_sound_file(filePath):
    with open(filePath, 'rb') as f:
        signal, samplFreq = sfile.read(f)
    return signal, samplFreq


def read_xml_scene(xmlPath):
    tree = ET.parse(xmlPath)
    root = tree.getroot()
    # sampFreq = int(root.find("General").find("NominalSamplingRate").text.strip())
    # totLength = int(root.find("General").find("TotalLength_Samples").text.strip())
    devices = root.find("SpatialData").find("Room").find("Kinect_features").findall("Kinect_Sampling")

    return devices



def read_utterance_names_with_and_without_loss(fileNames):

    with open(fileNames["with"], 'r') as f:
        lossNames = [l.rstrip('\n') for l in f]
    with open(fileNames["without"], 'r') as f:
        noLossNames = [l.rstrip('\n') for l in f]
    return lossNames, noLossNames



def write_train_segments_as_hdf(segments, path):
    hyp = np.array([s.hyp for s in segments])
    ref = np.array([s.ref for s in segments])
    labels = np.array([s.label for s in segments])

    hdf5File = h5py.File(".".join([path, "hdf"]), 'w')
    #fetaures
    refGroup = hdf5File.create_group("ref_features")
    refGroup.create_dataset("data", ref.shape, data = ref)
    hypGroup = hdf5File.create_group("hyp_features")
    hypGroup.create_dataset("data", hyp.shape, data=hyp)
    #labels
    labelGroup = hdf5File.create_group("labels")
    labelGroup.create_dataset("data", labels.shape, data=labels)

    hdf5File.close()



def write_train_segments_array_as_hdf(segments, path):
    hyp = np.array([s[0] for s in segments])
    ref = np.array([s[1] for s in segments])
    labels = np.array([s[2] for s in segments])

    hdf5File = h5py.File(".".join([path, "hdf"]), 'w')
    #fetaures
    refGroup = hdf5File.create_group("ref_features")
    refGroup.create_dataset("data", ref.shape, data = ref)
    hypGroup = hdf5File.create_group("hyp_features")
    hypGroup.create_dataset("data", hyp.shape, data=hyp)
    #labels
    labelGroup = hdf5File.create_group("labels")
    labelGroup.create_dataset("data", labels.shape, data=labels)

    hdf5File.close()




def read_hdf_segments_for_train(filename):
    with h5py.File(filename, "r") as f:
        ref = f["ref_features"]["data"]
        hyp = f["hyp_features"]["data"]
        labels = f["labels"]["data"]
        Xref = ref[:]
        Xhyp = hyp[:]
        Y = labels[:]
    return Xhyp, Xref, Y


def write_hdf_for_maurizio(device, path):


    ref = np.array(device.refSegments)
    hyp = np.array(device.contextWindow)

    hdf5File = h5py.File(".".join([path, "hdf"]), 'w')
    #fetaures
    refGroup = hdf5File.create_group("ref_features")
    refGroup.create_dataset("data", ref.shape, data = ref)
    hypGroup = hdf5File.create_group("hyp_features")
    hypGroup.create_dataset("data", hyp.shape, data=hyp)


    hdf5File.close()




def read_hdf_maurizio(filename):
    with h5py.File(filename, "r") as f:
        ref = f["ref_features"]["data"]
        hyp = f["hyp_features"]["data"]
        Xref = ref[:]
        Xhyp = hyp[:]
    return Xhyp, Xref


