import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
from utils.state_data import StateData

# class to manage loading and encoding behavioral data
class BehaviorData:
    
    def __init__(self, minw=2, maxw=8, include_pid=True, include_state=True, active_samp=.1):
        # minw, maxw: min and max weeks to collect behavior from
        # include_pid: should the participant id be a feature to the model
        # include_state: should the participant state be a feature
        self.minw, self.maxw = minw, maxw
        self.include_pid = include_pid
        self.include_state = include_state
        self.active_samp = active_samp if active_samp is not None else 1
        self.data = self.build()
        
    @property
    def dimensions(self):
        # helper to get the x and y input dimensions
        x, y = self.encode_row(self.data, self.data.index[0])
        return x.shape[0], y.shape[0]
    
    def train_iter(self, n_subj=-1, n_ser=-1):
        for (i_subj, subj) in enumerate(self.iterate_subjects()):
            if n_subj >= 0 and i_subj >= n_subj:
                break
            for (i_ser, ser) in enumerate(self.iterate_subject_series(subj)):
                if n_ser >= 0 and i_ser >= n_ser:
                    break
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
            x, y = self.encode_row(data, row)
            X.append(x)
            Y.append(y)
        X, Y = np.stack(X), np.stack(Y)
        return X, Y
                
    def encode_row(self, data, rowloc):
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
        # responses are the labels
        Y = np.array(row["response"])
        return X, Y
    
    def build(self):
        # call StateData and build our initial unencoded dataset
        sd = StateData()
        d = sd.buildby("pid", minw=self.minw, maxw=self.maxw)
        ids = sd.active_responders(
            self.active_samp, sd.analyze(d))["ids"]
        d = pd.concat([d[i] for i in ids])
        enc = OrdinalEncoder().fit_transform
        d["pid"] = enc(d["pid"].values.reshape(-1,1)).astype(int)
        return d
        
