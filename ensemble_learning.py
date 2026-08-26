import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import math
from sklearn.metrics import roc_auc_score, average_precision_score
import ipso
from sklearn.metrics import confusion_matrix
import xgboost as xgb 
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
#np.set_printoptions(suppress=True)


def dynamic_pruning(weights, val_probs, y_val):
    init_weights = weights/weights.sum()
    prob = 0
    for i in range(init_weights.shape[0]):
        prob += val_probs[i] * init_weights[i]
        ori_aupr = average_precision_score(y_val, prob)
    index = np.argsort(weights)
    for prun_num in range(1, weights.shape[0]-1):
        learner_index = sorted(index[prun_num:])
        val_probs_ = np.array(val_probs)[learner_index,:]
        weights_ = weights[learner_index]/weights[learner_index].sum()
        prob_ = 0
        for i in range(weights_.shape[0]):
            prob_ += val_probs_[i] * weights_[i]
            aupr = average_precision_score(y_val, prob_)
        if aupr < ori_aupr:
            return sorted(index[prun_num-1:])
        else:
            ori_aupr = aupr
    return sorted(index[len(weights)-1:])


def get_prob_by_optimization(val_probs, y_val):
    optimasation = ipso.get_optimasation_function(val_probs, y_val)
    lb, ub = [0.1]*len(val_probs), [0.5]*len(val_probs)
    optx, fopt = ipso.pso(optimasation, lb, ub, swarmsize=100, maxiter=100)
    weights = np.array(optx)/sum(optx)
    index =  dynamic_pruning(weights, val_probs, y_val)
    weights = weights[index]/weights[index].sum()
    val_probs = np.array(val_probs)[index,:]
    prob = 0
    for i in range(weights.shape[0]):
        prob += val_probs[i]*weights[i]
    return prob, index, weights


def ensem_learning_val(train_data, train_label, prop, test_data, f_nef, flag='aas'):
    print(f_nef)
    for key,value in f_nef.items():
        if value[0] == value[1]:
            continue
        else:
            if key == 'aac':
                aac_train = train_data[:,value[0]:value[1]]
                aac_test = test_data[:,value[0]:value[1]]
            elif key == 'dpc':
                dpc_train = train_data[:,value[0]:value[1]]
                dpc_test = test_data[:,value[0]:value[1]]
            elif key == 'cks':
                csk_train = train_data[:,value[0]:value[1]]
                csk_test = test_data[:,value[0]:value[1]]
            elif key == 'asdc':
                asdc_train = train_data[:,value[0]:value[1]]
                asdc_test = test_data[:,value[0]:value[1]]
            elif key == 'gga':
                gdc_train = train_data[:,value[0]:value[1]]
                gdc_test = test_data[:,value[0]:value[1]]
            elif key == 'qso':
                qso_train = train_data[:,value[0]:value[1]]
                qso_test = test_data[:,value[0]:value[1]]
            elif key == 'gtp':
                gtp_train = train_data[:,value[0]:value[1]]
                gtp_test = test_data[:,value[0]:value[1]]
            elif key == 'paac':
                paac_train = train_data[:,value[0]:value[1]]
                paac_test = test_data[:,value[0]:value[1]]
            elif key == 'ctd':
                ctd_train = train_data[:,value[0]:value[1]]
                ctd_test = test_data[:,value[0]:value[1]]
            elif key == 'es3mer':
                es3mer_train = train_data[:,value[0]:value[1]]
                es3mer_test = test_data[:,value[0]:value[1]]
            elif key == 'es4mer':
                es4mer_train = train_data[:,value[0]:value[1]]
                es4mer_test = test_data[:,value[0]:value[1]]
            elif key == 'es5mer':
                es5mer_train = train_data[:,value[0]:value[1]]
                es5mer_test = test_data[:,value[0]:value[1]]
            elif key == 'ssm':
                ssm_train = train_data[:,value[0]:value[1]]
                ssm_test = test_data[:,value[0]:value[1]]
            elif key == "rnactd":
                rctd_train = train_data[:,value[0]:value[1]]
                rctd_test = test_data[:,value[0]:value[1]]    
            else:
                print("New features have been introduced, please modify the python code!!!")
                exit(0)

    clf_val_probs = []
    fea_ind_list = []
    fea_weight_list = []

    if flag == 'aas':
        train_datasets = [aac_train,dpc_train,csk_train,asdc_train,gdc_train,qso_train,gtp_train,paac_train,ctd_train]
        test_datasets = [aac_test,dpc_test,csk_test,asdc_test,gdc_test,qso_test,gtp_test,paac_test,ctd_test]
    else:
        train_datasets = [es3mer_train,es4mer_train,es5mer_train]
        test_datasets = [es3mer_test,es4mer_test,es5mer_test]
    
    learners = [
                lgb.LGBMClassifier(n_estimators=200,max_depth=15), 
                DecisionTreeClassifier(random_state=50),
                GaussianNB(),
                RandomForestClassifier(n_estimators=200,max_depth=15),
                xgb.XGBClassifier(random_state=50, eval_metric='rmse')
                ]
    
    val_num = int(train_data.shape[0]*prop)
    for clf in learners:
        fea_val_probs = []
        for data in train_datasets:
            X_train, X_val, y_train, y_val = data[val_num:,:], data[:val_num,:], train_label[val_num:], train_label[:val_num]
            print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)
            
            clf.fit(X_train, y_train)
            val_prob = clf.predict_proba(X_val)[:,-1]
            fea_val_probs.append(val_prob)
        fea_prob, fea_index, fea_weights = get_prob_by_optimization(fea_val_probs, y_val)
        clf_val_probs.append(fea_prob)
        fea_ind_list.append(fea_index)
        fea_weight_list.append(fea_weights)
    
    clf_prob, clf_index, clf_weights = get_prob_by_optimization(clf_val_probs, y_val)
    sub_performance = average_precision_score(y_val, clf_prob)

    test_clf_probs = []
    for i in range(len(learners)):
        clf = learners[i]
        test_fea_probs = []
        for data_index in range(len(test_datasets)):
            train_data = train_datasets[data_index]
            test_data = test_datasets[data_index]
            clf.fit(train_data, train_label)
            test_prob = clf.predict_proba(test_data)[:,-1]
            test_fea_probs.append(test_prob)
        test_fea_probs = np.array(test_fea_probs)[fea_ind_list[i],:]
        test_fea_prob = 0
        for j in range(fea_weight_list[i].shape[0]):
            test_fea_prob += test_fea_probs[j]*fea_weight_list[i][j]
        test_clf_probs.append(test_fea_prob)
    
    test_clf_probs = np.array(test_clf_probs)[clf_index,:]
    sub_prob = 0
    for i in range(clf_weights.shape[0]):
        sub_prob += test_clf_probs[i]*clf_weights[i]

    return sub_prob, sub_performance

def get_subset_weights(factors):
    t_factors = []
    for factor in factors:
        t_factors.append(2.71818**factor)
    return np.array(t_factors)/sum(t_factors)
