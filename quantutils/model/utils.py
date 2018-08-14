from __future__ import print_function
import numpy as np
import pandas as pd
import quantutils.dataset.pipeline as ppl

def evaluate(predictions, data_y, threshold=0):
    
    a = np.argmax(predictions,axis=1)
    b = np.argmax(data_y,axis=1) 
    
    a = a[(predictions >= threshold).any(axis=1)]
    b = b[(predictions >= threshold).any(axis=1)]
        
    num = np.float32(np.sum(a == b))
    den = np.float32(b.shape[0])

    print("Won : " + str(num))
    print("Lost : " + str(den - num))
    print("Total : " + str(den))
    print("Diff : " + str(num-(den-num)))
    print("Edge : " + str(100*(num-(den-num))/den) +"%")
    print("IR : " + str(((num-(den-num))/den)*np.sqrt(den)))

    #recall = np.float32(b.shape[0]) / data_y.shape[0] # Number of Days traded
    #F_score = (2.0 * precision * recall) / (precision + recall)
        
    return num / den # precision

## aggregatePredictions:
##
## Options
#1 np.nanmean(results["test_predictions"], axis=0)
#2 np.nanmean(results["test_predictions"]) ## Remove conflicting predictions across markets
#3 (np.sum(np.sign(r-.5), axis=0)/40)+.5 ## Vote rather than average (removes effects of outliers). Voting will ignore the strength of the individual prediction.
#4 (np.sum(np.sign(r-.5))/80)+.5  ## Vote and Remove conflicting predictions across markets
#5 as #1 but don't score any entries that are in conflict
#6 as #3 but don't score any entries that are in conflict

def aggregatePredictions(predictions_list, method='vote_unanimous_all'):
    a = predictions_list[0]
    if (method=='mean_all'): ## Remove conflicting predictions across markets
        for predictions in predictions_list[1:]:
            a = a.add(predictions)            
        a = a / len(predictions_list)
        result = a.mean(axis=1).to_frame()
    if (method=='vote_majority'): ## Vote rather than average (removes effects of outliers) across all data. Voting will ignore the strength of the individual prediction.
        a = np.sign(a-.5)
        for predictions in predictions_list[1:]:
            a = a.add(np.sign(predictions-.5))            
        result = ((a.sum(axis=1)/ (predictions_list[0].shape[1] * len(predictions_list) * 2)) +.5).to_frame()
    if (method=='vote_unanimous_pred'): ## Vote and Remove conflicting predictions across sub-predictions
        a = np.sign(np.sign(a-.5).sum(axis=1))
        for predictions in predictions_list[1:]:
            a = a.add(np.sign(np.sign(predictions-.5).sum(axis=1)))            
        result = (a/ (len(predictions_list) * 2) +.5).to_frame()
    if (method=='vote_unanimous_markets'): ## Vote and Remove conflicting predictions across markets
        a = np.sign(a-.5)
        for predictions in predictions_list[1:]:
            a = a.add(np.sign(predictions-.5))            
        a = ((a.sum(axis=1)/ (predictions_list[0].shape[1] * len(predictions_list) * 2)) +.5)
        result = a[a!=.5].to_frame()
    if (method=='vote_unanimous_all'): ## Vote and Remove conflicting predictions across sub-predictions and markets
        a = np.sign(np.sign(a-.5).sum(axis=1))
        for predictions in predictions_list[1:]:
            a = a.add(np.sign(np.sign(predictions-.5).sum(axis=1)))
        a = (a/((len(predictions_list) * 2)))+.5
        result = a[a!=.5].to_frame()
    return result

# TODO : Rename, these are not signals, they are "wins/losses"
def getSignals(predictions, data_y, threshold):
    signals = np.ones(len(data_y))
    a = np.argmax(predictions,axis=1) 
    b = np.argmax(data_y,axis=1)
    signals[a != b] = -1
    signals[(predictions < threshold).all(axis=1)] = 0
    return signals

def getPredictionSignals(predictions, threshold):
    signals = np.ones(len(predictions))
    a = np.argmax(ppl.onehot(predictions),axis=1)
    signals[(a==1)] = -1 # Set any DOWN signals to -1 (UP signals will pick up the default of 1)
    signals[(predictions < threshold).all(axis=1)] = 0
    return signals


def sample(training_set, method="RANDOM", prop=.9, loo=0, boost = []): 
    if (method == "RANDOM"):
        training_set = training_set.sample(frac=1).reset_index(drop=True)
        idx = np.arange(0,len(training_set)) / float(len(training_set))
        return [training_set[idx<prop], training_set[idx>=prop]]
    elif (method == "LOO"):
        idx = np.array(range(0,len(training_set)))
        return [training_set[idx!=loo], training_set[idx==loo]]
    elif (method == "BOOTSTRAP"):
        idx = np.array(range(0,len(training_set)))
        sample = np.random.choice(idx, len(training_set), replace=True)
        return pd.DataFrame(training_set.values[sample,:]), training_set[~np.in1d(idx, sample)]
    elif (method == "BOOSTING"):
        idx = np.array(range(0,len(training_set)))
        sample = np.random.choice(idx, len(training_set), replace=True, p=boost)
        return pd.DataFrame(training_set.values[sample,:]), training_set[~np.in1d(idx, sample)]

def bootstrapTrain(model, training_set, test_set, lamda, iterations, threshold=0, debug=False):

    metrics = {
        "train_loss":[],
        "train_precision":[],
        "val_loss":[],
        "val_precision":[],
        "test_loss":[],
        "test_precision":[],
        "test_predictions":[],
        "weights":[]
    }
    
    NUM_FEATURES = model.featureCount();
    test_X, test_y = ppl.splitCol(test_set, NUM_FEATURES)

    for i in range(0, iterations):
        
        print(".", end='')

        train_sample, val_sample = sample(training_set, method="BOOTSTRAP", loo=i)

        train_sample_X, train_sample_y = ppl.splitCol(train_sample, NUM_FEATURES)
        val_sample_X, val_sample_y = ppl.splitCol(val_sample, NUM_FEATURES)        

        results = model.train( \
            {'features': train_sample_X, 'labels': train_sample_y, 'lamda': lamda}, \
            {'features': val_sample_X, 'labels': val_sample_y, 'lamda': lamda}, \
            {'features': test_X, 'labels': test_y, 'lamda': lamda}, \
            threshold, 1, debug)

        metrics["train_loss"].append(results["train_loss"]["mean"])
        metrics["train_precision"].append(results["train_precision"]["mean"])
        metrics["val_loss"].append(results["val_loss"]["mean"])
        metrics["val_precision"].append(results["val_precision"]["mean"])
        metrics["test_loss"].append(results["test_loss"]["mean"])
        metrics["test_precision"].append(results["test_precision"]["mean"])
        metrics["test_predictions"].append(results["test_predictions"])
        metrics["weights"].append(results["weights"][0]) # Because we called train() with only 1 iteration 


    results = {
        "train_loss": {"mean":np.nanmean(metrics["train_loss"]), "std":np.nanstd(metrics["train_loss"]), "values":metrics["train_loss"]},
        "train_precision": {"mean":np.nanmean(metrics["train_precision"]), "std":np.nanstd(metrics["train_precision"]), "values":metrics["train_precision"]},
        "val_loss": {"mean":np.nanmean(metrics["val_loss"]), "std":np.nanstd(metrics["val_loss"]), "values":metrics["val_loss"]},
        "val_precision":{"mean":np.nanmean(metrics["val_precision"]), "std":np.nanstd(metrics["val_precision"]), "values":metrics["val_precision"]},
        "test_loss": {"mean":np.nanmean(metrics["test_loss"]), "std":np.nanstd(metrics["test_loss"]), "values":metrics["test_loss"]},
        "test_precision":{"mean":np.nanmean(metrics["test_precision"]), "std":np.nanstd(metrics["test_precision"]), "values":metrics["test_precision"]},
        "test_predictions": metrics["test_predictions"],
        "weights": metrics["weights"],
    }

    if debug:
        print("Iteration : %d Lambda : %.2f, Threshold : %.2f" % (i, lamda, threshold))
        print("Training loss : %.2f+/-%.2f, precision : %.2f+/-%.2f" % 
              (results["train_loss"]["mean"], results["train_loss"]["std"],
               results["train_precision"]["mean"], results["train_precision"]["std"]))
        print("Validation loss : %.2f+/-%.2f, precision : %.2f+/-%.2f" % 
              (results["val_loss"]["mean"], results["val_loss"]["std"],
               results["val_precision"]["mean"], results["val_precision"]["std"]))
        print("Test loss : %.2f+/-%.2f, precision : %.2f+/-%.2f" % 
              (results["test_loss"]["mean"], results["test_loss"]["std"],
               results["test_precision"]["mean"], results["test_precision"]["std"]))

    return results

### 
### BOOSTING
###

def boostingTrain(model, training_set, test_set, lamda, iterations, debug=False):

    metrics = {
        "train_loss":[],
        "train_precision":[],
        "val_loss":[],
        "val_precision":[],
        "test_loss":[],
        "test_precision":[],
        "test_predictions":[],
        "weights":[]
    }
    
    NUM_FEATURES = model.featureCount()
    test_X, test_y = ppl.splitCol(test_set, NUM_FEATURES)
    train_X, train_y = ppl.splitCol(training_set, NUM_FEATURES)
    threshold = 0 # For boosting to work this must be 0
    boost = np.array([1.0/len(training_set)] * len(training_set))

    for i in range(0, iterations):
        
        print(".", end='')

        train_sample, val_sample = sample(training_set, method="BOOSTING", boost=boost)

        train_sample_X, train_sample_y = ppl.splitCol(train_sample, NUM_FEATURES)
        val_sample_X, val_sample_y = ppl.splitCol(val_sample, NUM_FEATURES)        

        results = model.train( \
            {'features': train_sample_X, 'labels': train_sample_y, 'lamda': lamda}, \
            {'features': val_sample_X, 'labels': val_sample_y, 'lamda': lamda}, \
            {'features': test_X, 'labels': test_y, 'lamda': lamda}, \
            threshold, 1, debug)

        #Evaluate the results and calculate the odds of misclassification
        _, _, train_predictions = model.evaluate(model.to_feed_dict({'features':train_X, 'labels':train_y, 'lamda':lamda}), threshold)
        precision = np.argmax(ppl.onehot(train_predictions),axis=1) == np.argmax(ppl.onehot(train_y),axis=1) # TODO : This only works for onehot encoding
        epsilon = sum(boost[~precision]) 
        delta = epsilon / (1.0 - epsilon)
        boost[precision] = boost[precision] * delta
        boost = boost / sum(boost)
                
        metrics["train_loss"].append(results["train_loss"]["mean"])
        metrics["train_precision"].append(results["train_precision"]["mean"])
        metrics["val_loss"].append(results["val_loss"]["mean"])
        metrics["val_precision"].append(results["val_precision"]["mean"])
        metrics["test_loss"].append(results["test_loss"]["mean"])
        metrics["test_precision"].append(results["test_precision"]["mean"])
        metrics["test_predictions"].append(results["test_predictions"])
        metrics["weights"].append(results["weights"][0]) # Because we called train() with only 1 iteration       

    results = {
        "train_loss": {"mean":np.nanmean(metrics["train_loss"]), "std":np.nanstd(metrics["train_loss"]), "values":metrics["train_loss"]},
        "train_precision": {"mean":np.nanmean(metrics["train_precision"]), "std":np.nanstd(metrics["train_precision"]), "values":metrics["train_precision"]},
        "val_loss": {"mean":np.nanmean(metrics["val_loss"]), "std":np.nanstd(metrics["val_loss"]), "values":metrics["val_loss"]},
        "val_precision":{"mean":np.nanmean(metrics["val_precision"]), "std":np.nanstd(metrics["val_precision"]), "values":metrics["val_precision"]},
        "test_loss": {"mean":np.nanmean(metrics["test_loss"]), "std":np.nanstd(metrics["test_loss"]), "values":metrics["test_loss"]},
        "test_precision":{"mean":np.nanmean(metrics["test_precision"]), "std":np.nanstd(metrics["test_precision"]), "values":metrics["test_precision"]},
        "test_predictions": metrics["test_predictions"],
        "weights": metrics["weights"]
    }

    if debug:
        print("Iteration : %d Lambda : %.2f, Threshold : %.2f" % (i, lamda, threshold))
        print("Training loss : %.2f+/-%.2f, precision : %.2f+/-%.2f" % 
              (results["train_loss"]["mean"], results["train_loss"]["std"],
               results["train_precision"]["mean"], results["train_precision"]["std"]))
        print("Validation loss : %.2f+/-%.2f, precision : %.2f+/-%.2f" % 
              (results["val_loss"]["mean"], results["val_loss"]["std"],
               results["val_precision"]["mean"], results["val_precision"]["std"]))
        print("Test loss : %.2f+/-%.2f, precision : %.2f+/-%.2f" % 
              (results["test_loss"]["mean"], results["test_loss"]["std"],
               results["test_precision"]["mean"], results["test_precision"]["std"]))

    return results
