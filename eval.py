import numpy as np

def tp_tn_fp_fn(states_pred, states_ground):
    tp = np.sum(np.logical_and(states_pred == 1, states_ground == 1),axis = 0).reshape([1,-1])
    fp = np.sum(np.logical_and(states_pred == 1, states_ground == 0),axis = 0).reshape([1,-1])
    fn = np.sum(np.logical_and(states_pred == 0, states_ground == 1),axis = 0).reshape([1,-1])
    tn = np.sum(np.logical_and(states_pred == 0, states_ground == 0),axis = 0).reshape([1,-1])
    return tp, tn, fp, fn

def recall_precision_accuracy_f1(pred,ground):
    thresh = [100,50,20,200,10]
    sum_samples = pred.shape[0]
    pr = np.array([0 if pred[i][j] < thresh[j] else 1 for i in range(pred.shape[0]) for j in range(pred.shape[1])]).reshape([pred.shape[0],pred.shape[1]])
    gr = np.array([0 if ground[i][j] < thresh[j] else 1 for i in range(ground.shape[0]) for j in range(pred.shape[1])]).reshape([pred.shape[0],pred.shape[1]])
    tp,tn,fp,fn = tp_tn_fp_fn(pr,gr)
    res_recall = recall(tp,fn)
    res_precision = precision(tp,fp)
    res_f1 = f1(res_precision,res_recall)
    res_accuracy = accuracy(tp,tn,sum_samples)
    return (res_recall,res_precision,res_accuracy,res_f1)


def recall(tp,fn):
    return tp / (tp + fn)

def precision(tp,fp):
    return tp / (tp + fp)

def f1(prec,rec):
    return 2 * (prec * rec) / (prec + rec)

def accuracy(tp,tn,samples):
    return (tp + tn) / samples

def mean_absolute_error(pred,ground):
    sum_samples = pred.shape[0]
    total_sum = np.sum(abs(pred - ground),axis = 0)
    return total_sum / sum_samples

def get_sae(pred,ground):
    sample_period = 60
    r = np.sum(ground * sample_period * 1.0 / 3600,axis = 0)
    rhat = np.sum(pred * sample_period * 1.0 / 3600,axis = 0)
    return np.abs(r - rhat) / np.abs(r)
