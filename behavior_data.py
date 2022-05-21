import numpy as np
import os
import matplotlib.pyplot as plt
from utils.state_data import StateData

# class to manage loading and encoding behavioral data
class BehaviorData:
    
    def __init__(self, minw=2, maxw=8, include_pid=True, include_state=True):
        # minw, maxw: min and max weeks to collect behavior from
        # include_pid: should the participant id be a feature to the model
        # include_state: should the participant state be a feature
        self.minw, self.maxw = minw, maxw
        self.include_pid = include_pid
        self.include_state = include_state
        self.data = self.build()
        
    @property
    def dimensions(self):
        # helper to get the x and y input dimensions
        x, y = self._encode_row(self.data, self.data.index[0])
        return x.shape[0], y.shape[0]
        
    def iterate_training(self):
        # for each subject, for each series, yield the series
        # a series is a slice of their behavior from [x-t:x+t],[y-t:y+t]
        for subj in self.iterate_subjects():
            for ser in self.iterate_subject_series(subj):
                yield self.encode(subj, ser)
        
    def iterate_subjects(self):
        # find the unique participants and yield their subset
        # of data sorted by week
        for pid in self.data["pid"].unique():
            dpid = self.data[self.data["pid"] == pid]
            dpid = dpid.sort_values(by="week")
            yield dpid
            
    def iterate_subject_series(self, subj, window=3):
        # step over the whole behavior of a subject and yield
        # the series of size window
        idx = subj.index
        for i in range(idx.shape[0]-window+1):
            ser_idx = idx[i:i+window]
            yield ser_idx
        
    def encode(self, data, rows):
        # encode the row locations of data
        # data: pd.DataFrame
        # rows: list or pd.Series of indexed to loc
        X, Y = [], []
        for i, row in enumerate(rows):
            x, y = self._encode_row(data, row)
            X.append(x)
            Y.append(y)
        X, Y = np.stack(X), np.stack(Y)
        return X, Y
                
    def _encode_row(self, data, rowloc):
        # here we take a row from the main behavior dataset and 
        # encode all of the features for our model
        # Features:
        #  - participant ID                    (enumeration of participants)
        #  - dynamic state elements            (real values between (1,3)
        #  - action id which prompted question (enumerated 1-5,  binary encoded)
        #  - message ids of action             (enumerated 1-57, binary encoded)
        #  - question ids to predict repsonse  (enumerated 1-32, binary encoded
        def _padded_binary(a, b):
            # helper function to binary encode a and 
            # pad it to be the length of encoded b
            a, b = int(a), int(b)
            l = len(format(b,"b"))
            a = format(a,f"0{l}b")
            return np.array([int(_) for _ in a])
        row = data.loc[rowloc]
        feats_to_enc = np.array(row[["paction_sids", "pmsg_ids", "qids"]].values)
        feats_to_enc = feats_to_enc.tolist()
        if self.include_pid:
            X = np.array([row["pid"]])
        else:
            X = np.array([])
        if self.include_state:
            X = np.append(X, row["state"])
        # max value for each (state elem, message id, question id) for padding
        ls = [5,57,32]
        for j in range(len(feats_to_enc)):
            for k in range(len(feats_to_enc[j])):
                # encode the feature and add it to our feat vector
                bin_feat = _padded_binary(feats_to_enc[j][k],ls[j])
                X = np.append(X, bin_feat)
        # encode the responses
        Y = row["response"]
        Y = [_padded_binary(y,3) for y in Y]
        Y = np.concatenate(Y)
        return X, Y
    
    def build(self):
        # call StateData and build our initial unencoded dataset
        sd = StateData()
        self.data = sd.build(self.minw, self.maxw, encode_ids=True)
        self.data = self.data.sort_values(by="pid")
        return self.data.copy()
