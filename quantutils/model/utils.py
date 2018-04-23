from __future__ import print_function
import numpy as np
import pandas as pd
import quantutils.dataset.pipeline as ppl

def evaluate(predictions, data_y, threshold, majorityVote=False):
    
    a = np.argmax(predictions,axis=1)
    b = np.argmax(data_y,axis=1) 
    
    if (majorityVote):
        msk = a[1::2] == a[::2]
        msk = np.dstack([msk,msk]).flatten()
        a = a[msk]
        b = b[msk]
        predictions = predictions[msk]
    
    a = a[(predictions > threshold).any(axis=1)]
    b = b[(predictions > threshold).any(axis=1)]
        
    precision = np.float32(np.sum(a == b) / np.float32(b.shape[0]))
    recall = np.float32(np.sum(a == b) / np.float32(data_y.shape[0])) # Correct Recall
    recall = np.float32(b.shape[0]) / data_y.shape[0] # Number of Days traded
    F_score = (2.0 * precision * recall) / (precision + recall)
    return precision, recall, F_score

# TODO : Rename, these are not signals, they are "wins/losses"
def getSignals(predictions, data_y, threshold):
    signals = np.ones(len(data_y))
    a = np.argmax(predictions,axis=1) 
    b = np.argmax(data_y,axis=1)
    signals[a != b] = -1
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
        "train_recall":[],
        "train_f":[],
        "val_loss":[],
        "val_precision":[],
        "val_recall":[],
        "val_f":[],
        "test_loss":[],
        "test_precision":[],
        "test_recall":[],
        "test_f":[],
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
        metrics["train_recall"].append(results["train_recall"]["mean"])
        metrics["train_f"].append(results["train_f"]["mean"])
        metrics["val_loss"].append(results["val_loss"]["mean"])
        metrics["val_precision"].append(results["val_precision"]["mean"])
        metrics["val_recall"].append(results["val_recall"]["mean"])
        metrics["val_f"].append(results["val_f"]["mean"])
        metrics["test_loss"].append(results["test_loss"]["mean"])
        metrics["test_precision"].append(results["test_precision"]["mean"])
        metrics["test_recall"].append(results["test_recall"]["mean"])
        metrics["test_f"].append(results["test_f"]["mean"])
        metrics["test_predictions"].append(results["test_predictions"])
        metrics["weights"].append(results["weights"])  


    results = {
        "train_loss": {"mean":np.nanmean(metrics["train_loss"]), "std":np.nanstd(metrics["train_loss"]), "values":metrics["train_loss"]},
        "train_precision": {"mean":np.nanmean(metrics["train_precision"]), "std":np.nanstd(metrics["train_precision"]), "values":metrics["train_precision"]},
        "train_recall": {"mean":np.nanmean(metrics["train_recall"]), "std":np.nanstd(metrics["train_recall"]), "values":metrics["train_recall"]},
        "train_f": {"mean":np.nanmean(metrics["train_f"]), "std":np.nanstd(metrics["train_f"]), "values":metrics["train_f"]},
        "val_loss": {"mean":np.nanmean(metrics["val_loss"]), "std":np.nanstd(metrics["val_loss"]), "values":metrics["val_loss"]},
        "val_precision":{"mean":np.nanmean(metrics["val_precision"]), "std":np.nanstd(metrics["val_precision"]), "values":metrics["val_precision"]},
        "val_recall": {"mean":np.nanmean(metrics["val_recall"]), "std":np.nanstd(metrics["val_recall"]), "values":metrics["val_recall"]},
        "val_f": {"mean":np.nanmean(metrics["val_f"]), "std":np.nanstd(metrics["val_f"]), "values":metrics["val_f"]},
        "test_loss": {"mean":np.nanmean(metrics["test_loss"]), "std":np.nanstd(metrics["test_loss"]), "values":metrics["test_loss"]},
        "test_precision":{"mean":np.nanmean(metrics["test_precision"]), "std":np.nanstd(metrics["test_precision"]), "values":metrics["test_precision"]},
        "test_recall": {"mean":np.nanmean(metrics["test_recall"]), "std":np.nanstd(metrics["test_recall"]), "values":metrics["test_recall"]},
        "test_f": {"mean":np.nanmean(metrics["test_f"]), "std":np.nanstd(metrics["test_f"]), "values":metrics["test_f"]},
        "test_predictions": metrics["test_predictions"],
        "weights": metrics["weights"],
    }

    if debug:
        print("Iteration : %d Lambda : %.2f, Threshold : %.2f" % (i, lamda, threshold))
        print("Training loss : %.2f+/-%.2f, precision : %.2f+/-%.2f, recall : %.2f+/-%.2f, F : %.2f+/-%.2f" % 
              (results["train_loss"]["mean"], results["train_loss"]["std"],
               results["train_precision"]["mean"], results["train_precision"]["std"],
               results["train_recall"]["mean"], results["train_recall"]["std"],
               results["train_f"]["mean"], results["train_f"]["std"]))
        print("Validation loss : %.2f+/-%.2f, precision : %.2f+/-%.2f, recall : %.2f+/-%.2f, F : %.2f+/-%.2f" % 
              (results["val_loss"]["mean"], results["val_loss"]["std"],
               results["val_precision"]["mean"], results["val_precision"]["std"],
               results["val_recall"]["mean"], results["val_recall"]["std"],
               results["val_f"]["mean"], results["val_f"]["std"]))
        print("Test loss : %.2f+/-%.2f, precision : %.2f+/-%.2f, recall : %.2f+/-%.2f, F : %.2f+/-%.2f" % 
              (results["test_loss"]["mean"], results["test_loss"]["std"],
               results["test_precision"]["mean"], results["test_precision"]["std"],
               results["test_recall"]["mean"], results["test_recall"]["std"],
               results["test_f"]["mean"], results["test_f"]["std"]))

    return results

### 
### BOOSTING
###

def boostingTrain(model, training_set, test_set, lamda, iterations, debug=False):

    metrics = {
        "train_loss":[],
        "train_precision":[],
        "train_recall":[],
        "train_f":[],
        "val_loss":[],
        "val_precision":[],
        "val_recall":[],
        "val_f":[],
        "test_loss":[],
        "test_precision":[],
        "test_recall":[],
        "test_f":[],
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
        _, _, _, _, train_predictions = model.predict({'features':train_X, 'labels':train_y, 'lamda':lamda}, threshold)
        precision = np.argmax(train_predictions,axis=1) == np.argmax(train_y,axis=1)
        epsilon = sum(boost[~precision]) 
        delta = epsilon / (1.0 - epsilon)
        boost[precision] = boost[precision] * delta
        boost = boost / sum(boost)
                
        metrics["train_loss"].append(results["train_loss"]["mean"])
        metrics["train_precision"].append(results["train_precision"]["mean"])
        metrics["train_recall"].append(results["train_recall"]["mean"])
        metrics["train_f"].append(results["train_f"]["mean"])
        metrics["val_loss"].append(results["val_loss"]["mean"])
        metrics["val_precision"].append(results["val_precision"]["mean"])
        metrics["val_recall"].append(results["val_recall"]["mean"])
        metrics["val_f"].append(results["val_f"]["mean"])
        metrics["test_loss"].append(results["test_loss"]["mean"])
        metrics["test_precision"].append(results["test_precision"]["mean"])
        metrics["test_recall"].append(results["test_recall"]["mean"])
        metrics["test_f"].append(results["test_f"]["mean"])
        metrics["test_predictions"].append(results["test_predictions"])
        metrics["weights"].append(results["weights"])       

    results = {
        "train_loss": {"mean":np.nanmean(metrics["train_loss"]), "std":np.nanstd(metrics["train_loss"]), "values":metrics["train_loss"]},
        "train_precision": {"mean":np.nanmean(metrics["train_precision"]), "std":np.nanstd(metrics["train_precision"]), "values":metrics["train_precision"]},
        "train_recall": {"mean":np.nanmean(metrics["train_recall"]), "std":np.nanstd(metrics["train_recall"]), "values":metrics["train_recall"]},
        "train_f": {"mean":np.nanmean(metrics["train_f"]), "std":np.nanstd(metrics["train_f"]), "values":metrics["train_f"]},
        "val_loss": {"mean":np.nanmean(metrics["val_loss"]), "std":np.nanstd(metrics["val_loss"]), "values":metrics["val_loss"]},
        "val_precision":{"mean":np.nanmean(metrics["val_precision"]), "std":np.nanstd(metrics["val_precision"]), "values":metrics["val_precision"]},
        "val_recall": {"mean":np.nanmean(metrics["val_recall"]), "std":np.nanstd(metrics["val_recall"]), "values":metrics["val_recall"]},
        "val_f": {"mean":np.nanmean(metrics["val_f"]), "std":np.nanstd(metrics["val_f"]), "values":metrics["val_f"]},
        "test_loss": {"mean":np.nanmean(metrics["test_loss"]), "std":np.nanstd(metrics["test_loss"]), "values":metrics["test_loss"]},
        "test_precision":{"mean":np.nanmean(metrics["test_precision"]), "std":np.nanstd(metrics["test_precision"]), "values":metrics["test_precision"]},
        "test_recall": {"mean":np.nanmean(metrics["test_recall"]), "std":np.nanstd(metrics["test_recall"]), "values":metrics["test_recall"]},
        "test_f": {"mean":np.nanmean(metrics["test_f"]), "std":np.nanstd(metrics["test_f"]), "values":metrics["test_f"]},
        "test_predictions": metrics["test_predictions"],
        "weights": metrics["weights"]
    }

    if debug:
        print("Iteration : %d Lambda : %.2f, Threshold : %.2f" % (i, lamda, threshold))
        print("Training loss : %.2f+/-%.2f, precision : %.2f+/-%.2f, recall : %.2f+/-%.2f, F : %.2f+/-%.2f" % 
              (results["train_loss"]["mean"], results["train_loss"]["std"],
               results["train_precision"]["mean"], results["train_precision"]["std"],
               results["train_recall"]["mean"], results["train_recall"]["std"],
               results["train_f"]["mean"], results["train_f"]["std"]))
        print("Validation loss : %.2f+/-%.2f, precision : %.2f+/-%.2f, recall : %.2f+/-%.2f, F : %.2f+/-%.2f" % 
              (results["val_loss"]["mean"], results["val_loss"]["std"],
               results["val_precision"]["mean"], results["val_precision"]["std"],
               results["val_recall"]["mean"], results["val_recall"]["std"],
               results["val_f"]["mean"], results["val_f"]["std"]))
        print("Test loss : %.2f+/-%.2f, precision : %.2f+/-%.2f, recall : %.2f+/-%.2f, F : %.2f+/-%.2f" % 
              (results["test_loss"]["mean"], results["test_loss"]["std"],
               results["test_precision"]["mean"], results["test_precision"]["std"],
               results["test_recall"]["mean"], results["test_recall"]["std"],
               results["test_f"]["mean"], results["test_f"]["std"]))

    return results
