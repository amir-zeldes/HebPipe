#!/usr/bin/python
# -*- coding: utf-8 -*-

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import numpy as np
import sys

PY3 = sys.version_info[0] == 3


def lambda_underscore():  # Module level named lambda-function to make defaultdict picklable
	return "_"


class LetterConfig:
	def __init__(self,letters=None, vowels=None, pos_lookup=None):

		if letters is None:
			self.letters = defaultdict(set)
			self.vowels = set()
			self.pos_lookup = defaultdict(lambda: "_")
		else:
			letter_cats = ["current_letter", "prev_prev_letter", "prev_letter", "next_letter", "next_next_letter", "prev_grp_first", "prev_grp_last", "next_grp_first", "next_grp_last"]
			self.letters = defaultdict(set)
			if "group_in_lex" in letters or "current_letter" in letters:  # Letters dictionary already instantiated - we are loading from disk
				self.letters.update(letters)
			else:
				for cat in letter_cats:
					self.letters[cat] = letters
			self.vowels = vowels
			self.pos_lookup = pos_lookup


class DataFrameSelector(BaseEstimator, TransformerMixin):
	def __init__(self, attribute_names):
		self.attribute_names = attribute_names
	def fit(self, X, y=None):
		return self
	def transform(self, X):
		return X[self.attribute_names].values


class MultiColumnLabelEncoder(LabelEncoder):
	"""
	Wraps sklearn LabelEncoder functionality for use on multiple columns of a
	pandas dataframe.

	"""

	def __init__(self, columns=None):
		self.encoder_dict = {}
		if isinstance(columns, list):
			self.columns = np.array(columns)
		else:
			self.columns = columns

	def fit(self, dframe):
		"""
		Fit label encoder to pandas columns.

		Access individual column classes via indexing `self.all_classes_`

		Access individual column encoders via indexing
		`self.all_encoders_`
		"""
		# if columns are provided, iterate through and get `classes_`
		if self.columns is not None:
			# ndarray to hold LabelEncoder().classes_ for each
			# column; should match the shape of specified `columns`
			self.all_classes_ = np.ndarray(shape=self.columns.shape, dtype=object)
			self.all_encoders_ = np.ndarray(shape=self.columns.shape, dtype=object)
			for idx, column in enumerate(self.columns):
				# fit LabelEncoder to get `classes_` for the column
				le = LabelEncoder()
				le.fit(dframe.loc[:, column].values)
				# append the `classes_` to our ndarray container
				self.all_classes_[idx] = (column, np.array(le.classes_.tolist(), dtype=object))
				# append this column's encoder
				self.all_encoders_[idx] = le
		else:
			# no columns specified; assume all are to be encoded
			self.columns = dframe.iloc[:, :].columns
			self.all_classes_ = np.ndarray(shape=self.columns.shape, dtype=object)
			for idx, column in enumerate(self.columns):
				le = LabelEncoder()
				le.fit(dframe.loc[:, column].values)
				self.all_classes_[idx] = (column, np.array(le.classes_.tolist(), dtype=object))
				self.all_encoders_[idx] = le
		return self

	def fit_transform(self, dframe):
		"""
		Fit label encoder and return encoded labels.

		Access individual column classes via indexing
		`self.all_classes_`

		Access individual column encoders via indexing
		`self.all_encoders_`

		Access individual column encoded labels via indexing
		`self.all_labels_`
		"""
		# if columns are provided, iterate through and get `classes_`
		if self.columns is not None:
			# ndarray to hold LabelEncoder().classes_ for each
			# column; should match the shape of specified `columns`
			self.all_classes_ = np.ndarray(shape=self.columns.shape, dtype=object)
			self.all_encoders_ = np.ndarray(shape=self.columns.shape, dtype=object)
			self.all_labels_ = np.ndarray(shape=self.columns.shape, dtype=object)
			for idx, column in enumerate(self.columns):
				# instantiate LabelEncoder
				le = LabelEncoder()
				# fit and transform labels in the column
				dframe.loc[:, column] = le.fit_transform(dframe.loc[:, column].values)
				self.encoder_dict[column] = le
				# append the `classes_` to our ndarray container
				self.all_classes_[idx] = (column, np.array(le.classes_.tolist(),dtype=object))
				self.all_encoders_[idx] = le
				self.all_labels_[idx] = le
		else:
			# no columns specified; assume all are to be encoded
			self.columns = dframe.iloc[:, :].columns
			self.all_classes_ = np.ndarray(shape=self.columns.shape,
										   dtype=object)
			for idx, column in enumerate(self.columns):
				le = LabelEncoder()
				dframe.loc[:, column] = le.fit_transform(
					dframe.loc[:, column].values)
				self.all_classes_[idx] = (column, np.array(le.classes_.tolist(), dtype=object))
				self.all_encoders_[idx] = le
		return dframe

	def transform(self, dframe):
		"""
		Transform labels to normalized encoding.
		"""
		if self.columns is not None:
			for idx, column in enumerate(self.columns):
				dframe.loc[:, column] = self.all_encoders_[idx].transform(dframe.loc[:, column].values)
		else:
			self.columns = dframe.iloc[:, :].columns
			for idx, column in enumerate(self.columns):
				dframe.loc[:, column] = self.all_encoders_[idx].transform(dframe.loc[:, column].values)
		return dframe

	def inverse_transform(self, dframe):
		"""
		Transform labels back to original encoding.
		"""
		if self.columns is not None:
			for idx, column in enumerate(self.columns):
				dframe.loc[:, column] = self.all_encoders_[idx] \
					.inverse_transform(dframe.loc[:, column].values)
		else:
			self.columns = dframe.iloc[:, :].columns
			for idx, column in enumerate(self.columns):
				dframe.loc[:, column] = self.all_encoders_[idx] \
					.inverse_transform(dframe.loc[:, column].values)
		return dframe


#@profile
def bg2array(bound_group, prev_group="", next_group="", print_headers=False, grp_id=-1, is_test=-1, config=None, train=False, freqs=None):

	output = []

	letters = config.letters
	vowels = config.vowels
	pos_lookup = config.pos_lookup

	group_in_lex = pos_lookup[bound_group] if pos_lookup[bound_group] in letters["group_in_lex"] or train else "_"

	for idx in range(len(bound_group)):

		char_feats = []
		so_far_substr = bound_group[:idx+1]
		remaining_substr = bound_group[idx+1:]
		remaining_substr_mns1 = bound_group[max(0,idx):]
		remaining_substr_mns2 = bound_group[max(0,idx-1):]
		so_far_pos = pos_lookup[so_far_substr] if pos_lookup[so_far_substr] in letters["so_far_pos"] or train else "_"
		remaining_pos = pos_lookup[remaining_substr] if pos_lookup[remaining_substr] in letters["remaining_pos"] or train else "_"
		remaining_pos_mns1 = pos_lookup[remaining_substr_mns1] if pos_lookup[remaining_substr_mns1] in letters["remaining_pos_mns1"] or train else "_"
		remaining_pos_mns2 = pos_lookup[remaining_substr_mns2] if pos_lookup[remaining_substr_mns2] in letters["remaining_pos_mns2"] or train else "_"

		current_letter = bound_group[idx] if bound_group[idx] in letters["current_letter"] or train else "_"
		current_vowel = 1 if bound_group[idx] in vowels else 0
		if idx > 0:
			prev_letter = bound_group[idx-1] if bound_group[idx-1] in letters["prev_letter"] or train else "_"
			prev_vowel = 1 if bound_group[idx-1] in vowels else 0
		else:
			prev_letter = "_"
			prev_vowel = -1
		if idx > 1:
			prev_prev_letter = bound_group[idx-2] if bound_group[idx-2] in letters["prev_prev_letter"] else"_"
			prev_prev_vowel = 1 if bound_group[idx-2] in vowels else 0
		else:
			prev_prev_letter = "_"
			prev_prev_vowel = -1
		if idx < len(bound_group)-1:
			next_letter = bound_group[idx+1] if bound_group[idx+1] in letters["next_letter"] else"_"
			next_vowel = 1 if bound_group[idx+1] in vowels else 0
		else:
			next_letter = "_"
			next_vowel = -1
		if idx < len(bound_group)-2:
			next_next_letter = bound_group[idx+2] if bound_group[idx+2] in letters["next_next_letter"] or train else "_"
			next_next_vowel = 1 if bound_group[idx+2] in vowels else 0
		else:
			next_next_letter = "_"
			next_next_vowel = -1

		char_feats += [idx, group_in_lex, current_letter, prev_prev_letter, prev_letter, next_letter, next_next_letter, len(bound_group)]
		char_feats += [current_vowel, prev_prev_vowel, prev_vowel, next_vowel, next_next_vowel, so_far_pos, remaining_pos, remaining_pos_mns1, remaining_pos_mns2]
		headers = ["idx","group_in_lex","current_letter","prev_prev_letter","prev_letter","next_letter","next_next_letter","len_bound_group"]
		headers += ["current_vowel","prev_prev_vowel","prev_vowel","next_vowel","next_next_vowel", "so_far_pos", "remaining_pos","remaining_pos_mns1","remaining_pos_mns2"]

		# POS lookup features
		all_pos_feats = []
		for chargram_idx, prev_char in enumerate([-4,-3,-2,-1,1,2,3,4]):
			substr = ""
			header_prefix = "_"
			if prev_char < 0:
				header_prefix = "mns" + str(abs(prev_char)) + "_"
				if idx + prev_char >= 0:
					substr = bound_group[idx+prev_char:idx+1]
			elif prev_char > 0:
				header_prefix = "pls" + str(abs(prev_char)) + "_"
				if idx + prev_char <= len(bound_group):
					substr = bound_group[idx:idx+prev_char+1]

			coarse_tag = pos_lookup[substr] if pos_lookup[substr] in letters[header_prefix + "coarse"] or train else "_"

			pos_feats = [coarse_tag]
			if print_headers:
				headers.append(header_prefix + "coarse")
			all_pos_feats += pos_feats #clean_pos_feats

		char_feats += all_pos_feats

		if prev_group != "":
			prev_first = prev_group[0] if prev_group[0] in letters["prev_grp_first"] or train else "_"
			prev_last = prev_group[-1] if prev_group[-1] in letters["prev_grp_last"] or train else "_"
			prev_grp_pos = pos_lookup[prev_group] if pos_lookup[prev_group] in letters["prev_grp_pos"] or train else "_"
			prev_feats = [prev_first,prev_last,len(prev_group),prev_grp_pos]
			char_feats += prev_feats
			if print_headers:
				headers += ["prev_grp_first","prev_grp_last","prev_grp_len","prev_grp_pos"]
		if next_group != "":
			next_first = next_group[0] if next_group[0] in letters["next_grp_first"] or train else "_"
			next_last = next_group[-1] if next_group[-1] in letters["next_grp_last"] or train else "_"
			next_grp_pos = pos_lookup[next_group] if pos_lookup[next_group] in letters["next_grp_pos"] or train else "_"
			next_feats = [next_first,next_last,len(next_group),next_grp_pos]
			char_feats += next_feats
			if print_headers:
				headers += ["next_grp_first","next_grp_last","next_grp_len","next_grp_pos"]

		headers += ["freq_ratio"]
		if freqs is None:
			char_feats += [0.0]
		else:
			f_sofar = freqs[so_far_substr]
			f_remain = freqs[remaining_substr]
			f_whole = freqs[bound_group] + 0.0000000001  # Delta smooth whole
			char_feats += [f_sofar*f_remain/f_whole]

		if grp_id > -1:
			char_feats += [grp_id]
			if print_headers:
				headers += ["grp_id"]
		if is_test > -1:
			char_feats += [is_test]
			if print_headers:
				headers += ["is_test"]

		char_feats += [bound_group,prev_group,next_group]
		headers += ["this_group","prev_group","next_group"]

		output.append(char_feats)


	if print_headers:
		return headers
	else:
		return output


def segs2array(segs):
	output = []
	cursor = 0
	while cursor < len(segs) - 1:  # Word not over yet
		if segs[cursor + 1] == "|":
			output.append(1)  # 1 = positive class
			cursor += 2
		else:
			output.append(0)  # 0 = negative class
			cursor += 1
	output.append(0)

	return output


