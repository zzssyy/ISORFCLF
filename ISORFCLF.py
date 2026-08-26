import data_processing
from sklearn.model_selection import KFold
import numpy as np
import ensemble_learning
import test_scores as score
import random
from sklearn import preprocessing
import pandas as pd
import joblib
from avp_data_util import  toBLOSUM62_NoTarget,get_ee_feats
from diffEntropy import entropy_with_metrics

def ISORFCLF():
    
    tfile = "./datasets/train.fasta"
    
    posi_aas_samples, nega_aas_samples, aas_f_index, aas_f_name = data_processing.get_data(tfile, flag='sorfs')
    posi_aas_samples, nega_aas_samples = data_processing.conn_shuf_split_dataset(posi_aas_samples, nega_aas_samples)
    
    # print("starting aas feature selction...")
    posi_aas_samples, nega_aas_samples, aas_Eachfeature = data_processing.feature_selection(posi_aas_samples, nega_aas_samples, aas_f_index) 
    
    posi_samples, nega_samples = posi_aas_samples, nega_aas_samples
    
    aas_f_nef = data_processing.get_fea_name_dict(aas_f_name, aas_Eachfeature)

    metric_list = []
    n_fold = 5
    cluster_num = 5
    fold_index = 0
    for fold in range(n_fold):
        p_train = [i for i in range(posi_samples.shape[0]) if i%n_fold !=fold]
        p_test = [i for i in range(posi_samples.shape[0]) if i%n_fold ==fold]
        n_train = [i for i in range(nega_samples.shape[0]) if i%n_fold !=fold]
        n_test = [i for i in range(nega_samples.shape[0]) if i%n_fold ==fold]

        posi_train, posi_test = posi_samples[p_train,:], posi_samples[p_test,:]
        nega_train, nega_test = nega_samples[n_train,:], nega_samples[n_test,:]
        # print(posi_train.shape, posi_test.shape)
        data_test = np.concatenate((posi_test[:,:-1], nega_test[:,:-1]), axis=0)
        y_test = np.concatenate((posi_test[:,-1], nega_test[:,-1]), axis=0)
        
        aas_data_test = data_test
                
        y_test_probs = []
        sub_factors = []
        nega_num = len(nega_train)
        cr_nega_train = nega_train
        for count in range(3):
            splited_nega_trains = data_processing.spliting_by_clustering(cr_nega_train, cluster_num, method = 'PEDP')
            nega_train, cr_nega_train, nega_num= data_processing.sampling_from_clusters(splited_nega_trains, posi_train.shape[0], nega_num)
            train_samples = posi_train.tolist() + nega_train
            random.shuffle(train_samples)
            train_samples = np.array(train_samples)
            data_train = train_samples[:,:-1]
            y_train = train_samples[:,-1]
            
            aas_data_train = data_train
                        
            #aas
            aas_y_test_prob, aas_sub_performance = ensemble_learning.ensem_learning_val(aas_data_train, y_train, 0.3, aas_data_test, aas_f_nef, flag='sorfs')
            
            #ensemble
            y_test_prob = aas_y_test_prob
            sub_performance = aas_sub_performance
            
            y_test_probs.append(y_test_prob)
            sub_factors.append(sub_performance)
            tp, fp, tn, fn, BACC, MCC, f1_score, AUC, AUPR = score.calculate_performace(y_test_prob, y_test)
            print('\nSubset'+str(count+1)+'\ntp = \t'+ str(tp) + '\t fp = \t'+ str(fp) + '\t tn = \t'+ str(tn)+ '\t fn = \t'+ str(fn)+'\n  BACC = \t', BACC, '\n  MCC = \t', MCC, '\n  f1_score = \t', f1_score, '\n  AUC = \t', AUC, '\n  AUPR = \t', AUPR)

        sub_weights = ensemble_learning.get_subset_weights(sub_factors)
        print(sub_weights)
        final_prob = 0
        for i in range(len(y_test_probs)):
            final_prob += np.array(y_test_probs[i])*sub_weights[i]

        tp, fp, tn, fn, BACC, MCC, f1_score, AUC, AUPR = score.calculate_performace(final_prob, y_test)
        metric_list.append([tp, fp, tn, fn, BACC, MCC, f1_score, AUC, AUPR])

        print('\n------------------ Fold ', fold_index+1, '----------------------\ntp = \t'+ str(tp) + '\t fp = \t'+ str(fp) + '\t tn = \t'+ str(tn)+ '\t fn = \t'+ str(fn)+'\n  BACC = \t', BACC, '\n  MCC = \t', MCC, '\n  f1_score = \t', f1_score, '\n  AUC = \t', AUC, '\n  AUPR = \t', AUPR)
        fold_index += 1

    ave_tp, ave_fp, ave_tn, ave_fn, ave_BACC, ave_mcc, ave_f1_score, ave_auc, ave_aupr = score.get_average_metrics(metric_list)
    print('\n BACC = \t'+ str(ave_BACC)+ '\n MCC = \t'+str(ave_mcc)+'\n f1_score = \t'+str(ave_f1_score)+'\n AUC = \t'+ str(ave_auc) + '\n AUPR =\t'+str(ave_aupr)+'\n')

ISORFCLF()
