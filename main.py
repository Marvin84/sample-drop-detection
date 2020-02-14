from utils import *
from config import *
from data_io import *
from training import *
from models import *
from online_stats import *
import numpy as np

from utils import *
import pickle
import torch



def main():





    ################
    #CHiME-5
    ################

    indices = [3,8,7]
    deviceObjects = read_chime_segments_no_shift(indices, 2)
    targets, segments = prepare_segments_and_targets_chime(deviceObjects, indices)

    meanName = "stats/chime-mean.pickle"
    stdName = "stats/chime-std.pickle"
    #
    mean = get_pickle(meanName)
    std = get_pickle(stdName)
    targets, segments = prepare_segments_and_targets_chime(deviceObjects, indices, mean=mean, std=std)




    # prePath = "/work/asr3/menne/sharedWorkspace/raissi/data/chime5/Maurizio-hdf/"
    # for device in deviceObjects:
    #     name = ("-").join([str(device.name[3]), str(device.deviceId), str(device.lossInterval[1])])
    #     p = prePath+name
    #     write_hdf_for_maurizio(device, p)

    for i in range(0,20):

        modelPath = (".").join([("_").join([config_modelPath, str(i)]), "pth"])
        #modelPath = (".").join([("_").join(["models/isLibrosa_False_isLog_True_fft_0.032", str(i)]), "pth_att_lr6"])
        #modelPath = "models/lr_5e-05_isLog_True_fft_0.032_"+str(i)+".pth_att_noPre"
        #modelPath = "models/final/isLibrosa_False_isLog_True_fft_0.032_4.pth"
        #modelPath = "models/lr_5e-05_isLog_True_fft_0.032_0.pth"
        #modelPath = (".").join([("_").join(["models/isLibrosa_False_isLog_True_fft_0.032", str(i)]), "pth_att_lr6"])

        #print(modelPath)

        evaluate_model_with_dev_segments_multiple_devices(modelPath, targets, segments)






    ##########################

    #modelPath = (".").join([("_").join([config_modelPath, str(4)]), "pth_att_lr6"])
    #evaluate_model_with_dev_segments_multiple_devices(modelPath, newTargets, devSegments)
    #modelPath = (".").join([("_").join(["models/newContaminated", str(3)]), "pth"])
    #modelPath = "models/lr_5e-05_isLog_True_fft_0.032_0.pth_att_noPre"
    #read_scenes_for_eval(config_numberXmls, config_numberXmls+300, list(range(0,1)))

    #########################
    # creation of the hdfs #ok
    #########################



    #
    # statsComputer = OnlineStatCompute(config_fqValues)
    #
    # trainMapDictAll = create_utterance_dict_mappings(config_trainRefPath, config_trainHypPath, config_trainLossInfo)
    # keys = list(trainMapDictAll.keys())[:]
    # step = 5000
    #
    # for startIndex in range(25000, len(keys), step):
    #    endInd = startIndex + step
    #    if endInd > len(keys):
    #        endInd = len(keys)
    #    subkeys = list(trainMapDictAll.keys())[startIndex:  endInd]
    #    trainMapDict = {k: trainMapDictAll[k] for k in subkeys}
    #    print("starting with the creation of the utterances")
    #    trainUtterances = create_utterance(trainMapDict, config_trainLabelMappingDict, True)
    #    trainSegments = create_segments_from_utterance_for_rnn(trainUtterances, statsComputer=statsComputer)
    #
    #
    #
    #    hdfPath = ("").join([config_hdfTrain, ("_").join(["/train", name, str(int(startIndex/step))])])
    #    write_train_segments_as_hdf(trainSegments, hdfPath)

    #mean = statsComputer.get_mean()
    #std = statsComputer.get_std()
    #with open(("").join(["stats/", name, "_mean.pickle"]), "wb") as fmean:
    #   pickle.dump(mean, fmean, protocol=2 )
    #with open(("").join(["stats/", name, "_std.pickle"]), "wb") as fstd:
    #   pickle.dump(std, fstd, protocol=2)





    
    ##############################
    #pretrain ##ok
    #################################
    
    # meanName = ("").join(["stats/", dname, "_mean.pickle"])
    # stdName = ("").join(["stats/", dname, "_std.pickle"])
    # mean = get_pickle(meanName)
    # std = get_pickle(stdName)



    # statsComputer = OnlineStatCompute(config_fqValues)
    #
    #
    # segments = []
    #
    # for i in range(6):
    #     hdfPath = ("").join([config_hdfTrain, ("_").join(["/train", dname, str(i)]), ".hdf"])
    #     Xhyp, Xref, Y = read_hdf_segments_for_train(hdfPath)
    #     for j in range(len(Xhyp)):
    #         statsComputer.update_stats(Xhyp[j].T)
    #         statsComputer.update_stats(Xref[j].T)
    #         segments.append([Xhyp[j], Xref[j], Y[j]])
    # #
    #
    #
    # mean = statsComputer.get_mean()
    # std = statsComputer.get_std()
    # with open(("").join(["stats/", name, "_mean.pickle"]), "wb") as fmean:
    #    pickle.dump(mean, fmean, protocol=2 )
    # with open(("").join(["stats/", name, "_std.pickle"]), "wb") as fstd:
    #    pickle.dump(std, fstd, protocol=2)
    #
    #
    # for s in segments:
    #     s[0] = (s[0] - mean) / std
    #     s[1] = (s[1] - mean) / std

    # segments = []
    # print(name)
    # for i in range(6):
    #     hdfPath = ("").join([config_hdfTrain, ("_").join(["/train", dname, str(i)]), ".hdf"])
    #     Xhyp, Xref, Y = read_hdf_segments_for_train(hdfPath)
    #     for j in range(len(Xhyp)):
    #         label = Y[j]
    #         normXhyp = (Xhyp[j] - mean)/std
    #         normXref = (Xref[j] - mean) / std
    #         segments.append([normXhyp, normXref, label])
    #
    #
    #
    # train_model_with_segments_with_hdf(segments)


    
  

    ##########################
    # train working pre-trained
    ##########################

    #read_scenes_for_train(0, config_numberXmls)

    #
    # meanName = ("").join(["stats/", dsceneName, "_mean.pickle"])
    # stdName = ("").join(["stats/", dsceneName, "_std.pickle"])
    #
    # mean = get_pickle(meanName)
    # std = get_pickle(stdName)
    # segments = []
    #
    #
    #
    # hdfPath = ("").join([config_hdfTrainMulti, ("_").join(["/train", dsceneName]), ".hdf"])
    # Xhyp, Xref, Y = read_hdf_segments_for_train(hdfPath)
    # for j in range(len(Xhyp)):
    #     label = Y[j]
    #     normXhyp = (Xhyp[j] - mean)/std
    #     normXref = (Xref[j] - mean) / std
    #     if np.sum(np.isnan(normXhyp)) > 0 or np.sum(np.isnan(normXref)):
    #         print(j)
    #         continue
    #     segments.append([normXhyp, normXref, label])




    #modelPath = (".").join([("_").join([config_pretrainedmodelPath, str(21)]), "pth"])

    #modelPath = (".").join([("_").join([config_modelPath, str(11)]), "pth"])
    #print(modelPath)
    #train_model_with_segments_with_hdf_and_pretrained(modelPath,segments, startEpoch=0)
    






    ##########################
    # working with mini-scenes
    ##########################



    # meanName = ("").join(["stats/", sceneName, "_mean.pickle"])
    # stdName = ("").join(["stats/", sceneName, "_std.pickle"])
    # mean = get_pickle(meanName)
    # std = get_pickle(stdName)
    #
    # meanName = ("").join(["stats/devshortmeanAllLog"])
    # stdName = ("").join(["stats/devshortstdAllLog"])
    # mean = get_pickle(meanName)
    # std = get_pickle(stdName)


    # targets = get_pickle("shortscenes_all.pickle")
    # trainSegments, hdfSegments = get_train_scenes_with_targets(0, config_numberXmls, targets)
    # embed()
    #
    # statsComputer = OnlineStatCompute(config_fqValues)
    # hdfPath = ("").join([config_hdfDevMulti, ("").join(["/dev-0.050.hdf"])])
    # # write_train_segments_as_hdf(hdfSegments, hdfPath)
    #segments = []
    #
    #
    #
    # XhypLog = []
    # XrefLog = []
    # #
    #Xhyp, Xref, Y = read_hdf_segments_for_train(hdfPath)
    # #
    # #
    # for j in range(len(Xhyp)):
    #     h = np.log10(Xhyp[j])
    #     r = np.log10(Xref[j])
    #     XhypLog.append(20*np.log10(Xhyp[j]))
    #     XrefLog.append(20*np.log10(Xref[j]))
    #     statsComputer.update_stats(h.T)
    #     statsComputer.update_stats(r.T)
    #
    #
    #     #statsComputer.update_stats(Xhyp[j].T)
    #     #statsComputer.update_stats(Xref[j].T)
    #     mean = statsComputer.get_mean()
    #     std = statsComputer.get_std()
    #     if np.sum(np.isnan(std)) > 0 or np.sum(np.isnan(mean)):
    #         print(j)
    #         embed()
    #
    #
    # mean = statsComputer.get_mean()
    # std = statsComputer.get_std()
    # print("ok")
    # # embed()
    # #
    # #
    # #
    # #
    # for j in range(len(Xhyp)):
    #   normXhyp = (Xhyp[j] - mean)/std
    #   normXref = (Xref[j] - mean) / std
    #   if np.sum(np.isnan(normXhyp)) > 0 or np.sum(np.isnan(normXref)):
    #       print(j)
    #       continue
    #
    #   segments.append([normXhyp, normXref, Y[j]])
    #
    #
    # modelPath = (".").join([("_").join(["models/64LogLibrosa", str(19)]), "pth"])
    # train_model_with_segments_with_hdf_and_pretrained(modelPath, segments)

    #modelPath = (".").join([("_").join([config_pretrainedmodelPath, str(19)]), "pth"])
    #train_model_with_segments_with_hdf_and_pretrained(modelPath,segments)



    # modelPath = (".").join([("_").join(["models/newContaminated_librosa", str(19)]), "pth"])
    # evalNet = RNNSampleLoss(num_inputs=segments[0][0].shape[1])
    # evalNet.load_state_dict(torch.load(modelPath, map_location="cpu"))
    #
    # get_scores_from_eval_rnn(evalNet, segments)





    #########################
    # creation of the hdfs
    #########################

    # trainMapDictAll = create_utterance_dict_mappings(config_trainRefPath, config_trainHypPath, config_trainLossInfo)
    # keys = list(trainMapDictAll.keys())[:5000]
    # step = 1000
    #
    # for startIndex in range(0, len(keys) - step, step):
    #     subkeys = list(trainMapDictAll.keys())[startIndex:  startIndex + step]
    #     trainMapDict = {k: trainMapDictAll[k] for k in subkeys}
    #     print("starting with the creation of the utterances")
    #     trainUtterances = create_utterance(trainMapDict, config_trainLabelMappingDict, True)
    #     trainSegments = create_segments_from_utterance_for_rnn(trainUtterances)



    # trainMapDictAll = create_utterance_dict_mappings(config_trainRefPath, config_trainHypPath, config_trainLossInfo)
    # keys = list(trainMapDictAll.keys())[:50]
    # trainMapDict = {k: trainMapDictAll[k] for k in keys}

    # print("starting with the creation of the utterances")
    # trainUtterances = create_utterance(trainMapDict, config_trainLabelMappingDict)
    # print(len(trainUtterances), "utterances done. Starting with the segments.")
    # trainSegments, max_, min_ = create_both_segments_from_utterance_with_high_time_scale(trainUtterances)
    # print(len(trainSegments), "segments are ready, starting with the training")

    # hdfPath = ("").join([config_trainSetPath, "/train"])
    # write_train_segments_with_hdf(trainSegments, max_, min_, hdfPath)

    ###########################
    # train without hdf
    ###########################

    # net = Net()
    # net.cuda()
    # criterion = nn.BCELoss().cuda()
    # opt = optim.Adam(net.parameters(), lr=config_learningRate)


    # trainMapDictAll = create_utterance_dict_mappings(config_trainRefPath, config_trainHypPath, config_trainLossInfo)
    # keys = list(trainMapDictAll.keys())[:5000]
    # trainMapDict = {k: trainMapDictAll[k] for k in keys}

    # print("starting with the creation of the utterances")
    # trainUtterances = create_utterance(trainMapDict, config_trainLabelMappingDict)
    # print(len(trainUtterances), "utterances done. Starting with the segments.")
    # trainSegments, max_, min_ = create_both_segments_from_utterance_with_high_time_scale(trainUtterances)
    # print(len(trainSegments), "segments are ready, starting with the training")
    # train_model_with_segments(trainSegments, max_, min_, net, criterion, opt)


    ########################
    # trainning with hdfs
    ########################


    # net = Net()   #it is declared in the trainning.py
    # net.cuda()
    # criterion = nn.BCELoss().cuda()
    # opt = optim.Adam(net.parameters(), lr=config_learningRate)
    # train_model_with_segments_with_hdf(net, criterion, opt)


    ###########################
    # test
    ###########################

    # devMapDictAll = create_utterance_dict_mappings(config_devRefPath, config_devHypPath, config_devLossInfo)
    # devUtterances = create_utterance(devMapDictAll, config_devLabelMappingDict)
    # print("preparing the segments for the evaluation")
    # devSegments = create_segments_from_utterance_for_rnn(devUtterances)
    # hdfPath = ("/").join([config_devSetPath, "dev"])
    # write_train_segments_as_hdf(devSegments, hdfPath)


    # print("evaluation started")
    # modelPath = ("").join([config_modelPath, "_9.pth"])
    # evaluate_model_with_dev_segments(modelPath, devSegments, max_, min_)


    ####################################################
    # getting the statistics with the online algorithm
    ####################################################

    # statsComputer = OnlineStatCompute(513)
    #
    #
    # trainMapDictAll = create_utterance_dict_mappings(config_trainRefPath, config_trainHypPath, config_trainLossInfo)
    # keys = list(trainMapDictAll.keys())
    # step = 1000
    #
    # for startIndex in range(0, len(keys), step):
    #     subkeys = list(trainMapDictAll.keys())[startIndex :  startIndex+step]
    #     trainMapDict = {k: trainMapDictAll[k] for k in subkeys}
    #     print("starting with the creation of the utterances")
    #     trainUtterances = create_utterance(trainMapDict, config_trainLabelMappingDict, True)
    #     trainSegments = create_segments_from_utterance_for_rnn(trainUtterances, statsComputer=statsComputer)
    #     hdfPath = ("").join([config_trainSetPath, "/train_", str(int(startIndex/step))])
    #     write_train_segments_as_hdf(trainSegments, hdfPath)
    #
    # mean = statsComputer.get_mean()
    # std = statsComputer.get_std()
    # with open("stats/mean.pickle", "wb") as fmean:
    #     pickle.dump(mean, fmean, protocol=2 )
    # with open("stats/std.pickle", "wb") as fstd:
    #     pickle.dump(std, fstd, protocol=2)




if __name__ == '__main__':
    main()
