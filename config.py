import numpy as np


#Overlaps
config_fftOverlap      = 50  #percent
config_contextOverlap  = 80  #percent
config_lossOverlap     = 50  #percent

#Window information
config_fftWindow     = 0.032 #0.128 #0.032 #0.050 #0.064 #seconds  ###########
config_contextWindow = 1.02  #seconds
config_lossWindow    = 0.25    #seconds


#general
config_uttMargin = 2000
config_cxtMargin = 100
config_minMargin = 2000
config_normChimeContext = 0 #6320#16320*20

config_lossGenParams = [600, 150] # Mean, std
config_minLossLen = 100
config_maxLossLen = 2000

#64-< 512,32  32<-256,64
config_sf = 16000 #ToDo: you do not wanna hardcode it, think about a solution
#features
config_isLibrosa = False
if config_isLibrosa:

    config_nFFT = 1024 #int(16000*config_fftWindow) #            ###########
    config_librosaWin = 512
    config_hop = config_librosaWin//2
config_CTX = 16320
config_bandLimit = 1000
config_fqValues = 256 #256 #1024 #512 #256 #513#512 #int(16000*config_fftWindow/2)
config_tmValues = 63 # 63 #15 #31 #63 #40 #31 #int(np.ceil(config_CTX/config_fqValues))


#train
config_epochs = 25
config_batchSize = 50
config_devPercentage = 0.03
config_learningRate = 0.00005
config_featureMapValues = 5
config_isLog = True
config_trainStep = 300
config_isAttention=False
config_isPretrained=True

"""
These are the possible windows we can use.
config_windows = [
    'boxcar',
    'hamming',
    'barthann',
    'bartlett',
    'blackman',
    'blackmanharris',
    'bohman',
    'cosine',
    'flattop',
    'hann',
    'nuttall',
    'parzen',
    'triang'
]
"""
config_windows = {"preprocess": "boxcar", "extract": "hamming"}





#root directory paths to ref and hyp flac files
config_trainRefPath = "/work/asr3/menne/work/jsalt2019/data/20190715_data/train/ref/train-clean-100"
config_trainHypPath = "/work/asr3/menne/work/jsalt2019/data/20190715_data/train/hyp/train-clean-100"
config_trainLossInfo = "/work/asr3/menne/sharedWorkspace/raissi/sample-loss/sample-loss-files/train-files/loss_info_segments_train_new.pickle"
config_trainLabelMappingDict = "/work/asr3/menne/sharedWorkspace/raissi/sample-loss/sample-loss-files/train-files/train_label_per_utterance_balanced.pickle"

config_devRefPath = "/work/asr3/menne/work/jsalt2019/data/20190715_data/dev/ref/dev-clean"
config_devHypPath = "/work/asr3/menne/work/jsalt2019/data/20190715_data/dev/hyp/dev-clean"
config_devLossInfo = "/work/asr3/menne/sharedWorkspace/raissi/sample-loss/sample-loss-files/dev-files/loss_info_segments_dev.pickle"
config_devLabelMappingDict = "/work/asr3/menne/sharedWorkspace/raissi/sample-loss/sample-loss-files/dev-files/dev_label_per_utterance.pickle"

config_chimePath="/work/asr3/menne/sharedWorkspace/raissi/data/chime5"
config_chimeHdf="/work/asr3/menne/sharedWorkspace/raissi/data/chime5/Maurizio-hdf"

#hdf files
#config_hdfTrain = "/work/asr3/menne/sharedWorkspace/raissi/data/contaminated/hdf/train"
#config_hdfDev = "/work/asr3/menne/sharedWorkspace/raissi/data/contaminated/hdf/dev"
config_hdfTrain = "/work/asr3/menne/sharedWorkspace/raissi/data/contaminated/hdf/train"
config_hdfDev = "/work/asr3/menne/sharedWorkspace/raissi/data/contaminated/hdf/dev"

# syntheticData


config_xmlPath = "/work/asr3/menne/sharedWorkspace/raissi/data/xmlSimulatedNew"
config_wavPath = "/work/asr3/menne/sharedWorkspace/raissi/data/raw-low-rev-synthetic/signal/nodr_lo"

config_hdfTrainMulti = "/work/asr3/menne/sharedWorkspace/raissi/data/hdf-simu/train"
config_hdfDevMulti = "/work/asr3/menne/sharedWorkspace/raissi/data/hdf-simu/dev"

config_numberXmls = 882
config_lastIndexXml = 1182
#config_numberXmls = 50
config_maxnumSegMulti = 1
#config_problematicIndices = [42, 52, 63, 67, 72, 79, 132, 134, 148, 178, 182, 189, 217, 241, 266, 295, 315, 318, 320, 324, 334, 350, 362, 382, 384, 422, 445, 459, 462, 473, 487, 491, 500, 551, 558, 574, 581, 591, 592, 595, 601, 633, 636, 643, 645, 649, 666, 669, 670, 673, 697, 722, 785, 796, 854, 858, 861, 868, 879]
#config_problematicIndices = [10, 12, 14, 22, 31, 35, 36, 42, 52, 55, 63, 67, 72, 74, 79, 112, 113, 118, 132, 134, 148, 152, 155, 158, 168, 178, 181, 182, 189, 212, 215, 217, 225, 237, 241, 242, 245, 256, 266, 295, 306, 309, 313, 315, 318, 320, 323, 324, 334, 339, 349, 350, 362, 380, 382, 384, 386, 406, 414, 418, 422, 445, 458, 459, 462, 473, 480, 487, 490, 491, 500, 520, 541, 548, 551, 558, 569, 574, 579, 581, 591, 592, 595, 601, 609, 611, 633, 635, 643, 645, 647, 649, 662, 666, 669, 670, 673, 689, 692, 694, 697, 722, 727, 747, 776, 785, 786, 796, 807, 813, 832, 845, 854, 858, 861, 862, 867, 868, 879]
config_problematicIndices = []
config_selectedChannel = 0
config_deviceIds = [i for i in range(6)]



dname = ("_").join(["isLibrosa", str(config_isLibrosa), "isLog", str(config_isLog), "fft", str(config_fftWindow)])
#scenes or chime
dsceneName = ("_").join(
            ["scenes", "isLibrosa", str(config_isLibrosa), "isLog", str(config_isLog), "fft", str(config_fftWindow)])

#name = dname
#sceneName = dsceneName


name = ("_").join(["lr", str(config_learningRate), "isLog", str(config_isLog), "fft", str(config_fftWindow)])
#scenes or chime
sceneName = ("_").join(
            ["scenes", "lr", str(config_learningRate), "isLog", str(config_isLog), "fft", str(config_fftWindow)])


config_pretrainedModelName = name
config_modelName = sceneName
config_pretrainedmodelPath = ("/").join(["models", config_pretrainedModelName])
config_modelPath = ("/").join(["models", config_modelName])
config_predictionsPath = ("").join(["predictions/", config_modelName, ".pickle"])
config_lossesPath = ("").join(["losses/", config_modelName, ".pickle"])
config_f1ScoresPath = ("").join(["eval-scores/", config_modelName, ".pickle"])
config_errorsPath = ("").join(["errors/", config_modelName, ".pickle"])



