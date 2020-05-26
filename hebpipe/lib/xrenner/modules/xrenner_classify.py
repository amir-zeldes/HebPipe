import numpy as np

"""
modules/xrenner_classify.py

Adapter class to accommodate different types of classifiers. 
  * Supplies a uniform predict_proba() method accessed via .classify_many()
  * Reads pickled Encoders/Binarizers/Scalers and ensures encoding of OOV feature values as _unknown_
  * Compatible with several prominent sklearn estimators (e.g. GradientBoosting, RandomForest, MLP and linear models)

Author: Amir Zeldes
"""

class Classifier:
	def __init__(self, cls, encoder_dict, headers):
		self.cls = cls
		self.encoder_dict = encoder_dict
		self.headers = headers
		t = str(type(cls))
		if "Ridge" in t or "Elastic" in t or "Logistic" in t:
			self.cls_type = "decision"
		elif "RandomForest" in t or "Perceptron" in t or "StochasticGradient" in t or "Boost" in t or \
			"XGB" in t:
			self.cls_type = "tuple"
		else:
			self.cls_type = "predict_proba"


	def classify_many(self, input_list):
		"""
		Classifies a list of xrenner anaphor-antecedent candidate dump features

		:param input_list: list of numpy arrays containing markable pair features
		:return: numpy array of coref probabilities for the input pairs
		"""

		encoded_arrays = []
		for ana, ante, candidate_list, lex in input_list:
			feats = ana.extract_features(lex, ante, candidate_list)
			encoded = []
			for header in self.headers:
				if header in self.encoder_dict:  # Categorical feature
					if self.encoder_dict[header][1] == "binarizer":
						if feats[header] in self.encoder_dict[header][2]:
							cols = self.encoder_dict[header][0].transform([feats[header]])
							encoded += cols.tolist()[0]
						else:  # OOV item
							cols = self.encoder_dict[header][0].transform(["_unknown_"])
							encoded += cols.tolist()[0]
					elif self.encoder_dict[header][1] == "scale":
						encoded.append(self.encoder_dict[header][0].transform(np.array(feats[header])))
					else:  # ordinal feature
						if feats[header] in self.encoder_dict[header][2]:
							encoded.append(self.encoder_dict[header][0].transform(np.array([feats[header]]).reshape(-1,1)))
						else:  # OOV item
							encoded.append(self.encoder_dict[header][0].transform(np.array(["_unknown_"]).reshape(-1,1)))
				else:  # Untransformed numerical feature
					encoded.append(feats[header])
			encoded_array = np.array(encoded).reshape((1, -1))
			encoded_arrays.append(encoded_array)
		return self.predict_proba(np.array(encoded_arrays).reshape(len(input_list),-1))

	def predict_proba(self, feat_matrix):
		if self.cls_type == "predict_proba":
			return self.cls.predict_proba(feat_matrix)
		elif self.cls_type == "decision":  # Returns proba via decision function, get exp
			d = self.cls.decision_function(feat_matrix)#[0]
			probs = np.exp(d) / (1 + np.exp(d))
			return probs
		elif self.cls_type == "tuple":  # Returns tuple with class probabilities for each case
			probas = self.cls.predict_proba(feat_matrix)
			return [tpl[1] for tpl in probas]




