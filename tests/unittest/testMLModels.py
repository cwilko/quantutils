import unittest
import numpy as np

import quantutils.dataset.pipeline as ppl
from quantutils.api.marketinsights import MarketInsights
import quantutils.model.utils as mlutils
from quantutils.model.ml import Model

class MLModelTestCase(unittest.TestCase):

	def setUp(self):

		MODEL_ID = "3a491b1a-8af6-416d-aa14-f812cbd660bb"

		MARKET1 = "DOW"
		MARKET2 = "SPY"

		PIPELINE_ID = "marketdirection"

		#
		# Get dataset from MI API #
		#

		print "Loading data..."
		mi = MarketInsights('cred/MIOapi_cred.json')

		self.CONFIG = mi.get_model(MODEL_ID)
		

		mkt1 = mi.get_dataset(MARKET1, PIPELINE_ID)
		mkt2 = mi.get_dataset(MARKET2, PIPELINE_ID)

		# Interleave (part of the "added insight" for this model)
		mkt1, mkt2, isect = ppl.intersect(mkt1,mkt2)
		dataset = ppl.interleave(mkt1,mkt2)

		testSetLength = 430
		self.training_set = dataset[:-(testSetLength)]
		self.test_set = dataset[-(testSetLength):]

	def testFFNN_BootstrapTrain(self):

		NUM_FEATURES = (2 * 4) + 1
		NUM_LABELS = 2
		TRN_CNF = self.CONFIG['training']

		print "Creating model..."
		# Create ML model
		ffnn = Model(NUM_FEATURES, NUM_LABELS, self.CONFIG)

		##
		## BOOTSTRAP TRAINING
		##

		print "Training",
		_, test_y = ppl.splitCol(self.test_set, NUM_FEATURES)
		results = mlutils.bootstrapTrain(ffnn, self.training_set, self.test_set, TRN_CNF['lamda'], TRN_CNF['iterations'], TRN_CNF['threshold'], True)
		predictions =  np.nanmean(results["test_predictions"], axis=0)
		result = mlutils.evaluate(predictions, test_y, .0)

		print "".join(["Received : ", str(result)])
		print "Expected : (0.53023255, 1.0, 0.6930091104902956)"

		self.assertTrue(np.allclose(result, np.array([0.53023255, 1.0, 0.6930091104902956]))) # Local results

	def testFFNN_BoostingTrain(self):

		NUM_FEATURES = (2 * 4) + 1
		NUM_LABELS = 2
		TRN_CNF = self.CONFIG['training']

		print "Creating model..."
		# Create ML model
		ffnn = Model(NUM_FEATURES, NUM_LABELS, self.CONFIG)

		##
		## BOOTSTRAP TRAINING
		##
		print "Training",
		_, test_y = ppl.splitCol(self.test_set, NUM_FEATURES)
		results = mlutils.boostingTrain(ffnn, self.training_set, self.test_set, TRN_CNF['lamda'], TRN_CNF['iterations'], True)
		predictions =  np.nanmean(results["test_predictions"], axis=0)
		result = mlutils.evaluate(predictions, test_y, .0)

		print "".join(["Received : ", str(result)])
		print "Expected : (0.51627904, 1.0, 0.6809815707344107)"

		self.assertTrue(np.allclose(result, np.array([0.51627904, 1.0, 0.6809815707344107]))) # Local results


if __name__ == '__main__':
    unittest.main()