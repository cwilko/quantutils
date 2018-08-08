from __future__ import print_function
import numpy as np
import tensorflow as tf #v1.3.0 from DSX
import quantutils.model.utils as mlutils

class Model():

    def __init__(self, NUM_FEATURES, NUM_LABELS, CONFIG):
        self.OPT_CNF = CONFIG['optimizer']
        self.NTWK_CNF = CONFIG['network']

        self.createNetwork(NUM_FEATURES, NUM_LABELS)
        
    #### Define the architecture
    def createNetwork(self, NUM_FEATURES, NUM_LABELS):
        
        self.NUM_FEATURES = NUM_FEATURES
        self.NUM_LABELS = NUM_LABELS

        HIDDEN_UNITS = self.NTWK_CNF["hidden_units"]
        # The random seed that defines initialization.
        SEED = self.NTWK_CNF["weights"]["seed"]
        # The stdev of the initialised random weights 
        STDEV = self.NTWK_CNF["weights"]["stdev"]
        # Network bias
        BIAS = self.NTWK_CNF["bias"]

        # Reset graph and seeds
        tf.reset_default_graph()
        np.random.seed(SEED)
        tf.set_random_seed(SEED)

        # The variables below hold all the trainable weights. For each, the
        # parameter defines how the variables will be initialized. 
        self.Theta1 = tf.Variable( tf.truncated_normal([HIDDEN_UNITS, NUM_FEATURES], stddev=STDEV, seed=SEED))
        self.Theta2 = tf.Variable( tf.truncated_normal([NUM_LABELS, HIDDEN_UNITS], stddev=STDEV, seed=SEED))
        self.bias = tf.Variable(tf.constant(BIAS, shape=[NUM_LABELS]))

        # This is where training samples and labels are fed to the graph.
        # These placeholder nodes will be fed a batch of training data at each
        # training step, which we'll write once we define the graph structure.
        self.train_data_node = tf.placeholder(tf.float32, shape=(None, NUM_FEATURES))
        self.train_labels_node = tf.placeholder(tf.float32, shape=(None, NUM_LABELS))
        self.lam = tf.placeholder(tf.float32)

        self.weights1 = tf.placeholder_with_default(self.Theta1, shape=(HIDDEN_UNITS, NUM_FEATURES))
        self.weights2 = tf.placeholder_with_default(self.Theta2, shape=(NUM_LABELS, HIDDEN_UNITS))
        self.weights3 = tf.placeholder_with_default(self.bias, shape=[NUM_LABELS])

        yhat = self.model(self.train_data_node, self.weights1, self.weights2, self.weights3)

        # Change the weights by subtracting derivative with respect to that weight
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.train_labels_node, logits=yhat))
        # Regularization using L2 Loss function 
        regularizer = tf.nn.l2_loss(self.Theta1) + tf.nn.l2_loss(self.Theta2)
        reg = (self.lam / tf.to_float(tf.shape(self.train_labels_node)[0])) * regularizer
        loss_reg = self.loss + reg

        # Optimizer: 

        # Gradient Descent
        self.optimizer = self.createOptimizer(loss_reg)
        #update_weights = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

        # Predictions
        self.prediction = tf.sigmoid(yhat)

    ## The Model definition ##
    def model(self, X, Theta1, Theta2, bias):        
        # Perceptron        
        layer1 = tf.nn.sigmoid(tf.matmul(X, tf.transpose(Theta1)))                            
        output = tf.nn.bias_add(tf.matmul(layer1, tf.transpose(Theta2)),bias)
        return output

    def createOptimizer(self, loss):
        return tf.contrib.opt.ScipyOptimizerInterface(loss, options={'maxiter':self.OPT_CNF['maxIter']})
        
    def minimize(self, feed_dict):        
        #optimizer.minimize(feed_dict=feed_dict, fetches=[loss_reg], loss_callback=loss_callback)
        self.optimizer.minimize(feed_dict=feed_dict)

    def evaluate(self, feed_dict, threshold):
        loss = self.loss.eval(feed_dict)     
        predictions = self.prediction.eval(feed_dict)
        precision, recall, F_score = mlutils.evaluate(predictions, feed_dict[self.train_labels_node], threshold)
        return loss, precision, recall, F_score, predictions

    def featureCount(self):
        return self.NUM_FEATURES

    def getWeights(self):
        return np.concatenate([self.Theta1.eval().flatten(), self.Theta2.eval().flatten(), self.bias.eval().flatten()]).tolist()

    def predict(self, weights, data): 

        HIDDEN_UNITS = self.NTWK_CNF["hidden_units"]
        predictions = np.empty((0,len(data), self.NUM_LABELS))

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            for iteration in weights:
            
                Theta1 = iteration[:(HIDDEN_UNITS*self.NUM_FEATURES)].reshape(HIDDEN_UNITS, self.NUM_FEATURES)      
                Theta2 = iteration[(HIDDEN_UNITS*self.NUM_FEATURES):-self.NUM_LABELS].reshape(self.NUM_LABELS, HIDDEN_UNITS)
                bias = iteration[-self.NUM_LABELS:]
                        
                feed_dict = {
                    self.train_data_node:data,
                    self.weights1:Theta1,
                    self.weights2:Theta2,
                    self.weights3:bias
                }

                predictions = np.concatenate([predictions, [self.prediction.eval(feed_dict)]])

        return predictions

    def to_feed_dict(self, string_dict):
        return { \
                self.train_data_node: string_dict['features'], \
                self.train_labels_node: string_dict['labels'], \
                self.lam: string_dict['lamda'], \
                }

    def train(self, train_dict, val_dict, test_dict, threshold, iterations=50, debug=True):
        
        tf.logging.set_verbosity(tf.logging.ERROR)
        
        SEED = self.NTWK_CNF["weights"]["seed"]

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
        
        for i in range(0,iterations):
            
            for j in range(0, self.OPT_CNF['minimize_iterations']):
                
                # Create a new interactive session that we'll use in
                # subsequent code cells.
                s = tf.InteractiveSession()
                s.as_default()
                
                # Turn on determinism for a set of tf variables/object instances (N.B. also need to set SEED on variables)
                np.random.seed(SEED)
                tf.set_random_seed(SEED)
                            
                # Initialize all the variables we defined above.
                tf.global_variables_initializer().run()
                            
                self.minimize(self.to_feed_dict(train_dict))
                train_loss, train_precision, train_recall, train_f, _ = self.evaluate(self.to_feed_dict(train_dict), threshold)

                if (train_loss < self.OPT_CNF['training_loss_error_case']):
                    print('.', end='')
                    metrics["train_loss"].append(train_loss)
                    metrics["train_precision"].append(train_precision)
                    metrics["train_recall"].append(train_recall)
                    metrics["train_f"].append(train_f)

                    val_loss, val_precision, val_recall, val_f, _= self.evaluate(self.to_feed_dict(val_dict), threshold)

                    metrics["val_loss"].append(val_loss)
                    metrics["val_precision"].append(val_precision)
                    metrics["val_recall"].append(val_recall)
                    metrics["val_f"].append(val_f)
                    
                    test_loss, test_precision, test_recall, test_f, test_predictions = self.evaluate(self.to_feed_dict(test_dict), threshold)

                    metrics["test_loss"].append(test_loss)
                    metrics["test_precision"].append(test_precision)
                    metrics["test_recall"].append(test_recall)
                    metrics["test_f"].append(test_f)
                    metrics["test_predictions"] = test_predictions # return the last set of predictions ( TODO could return the one with the best val score)

                    metrics["weights"].append(self.getWeights())

                    del s
                    break;
                else:
                    del s
            
            if (j >= self.OPT_CNF['minimize_iterations']):
                print("ERROR : Failed to minimise function")
                
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
        
        print(".", end='')
        if debug:
            print("Iterations : %d Lambda : %.2f, Threshold : %.2f" % (iterations, val_dict['lamda'], threshold))
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