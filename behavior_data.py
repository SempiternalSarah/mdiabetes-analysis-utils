import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
from utils.state_data import StateData
from torch import save, load

# class to manage loading and encoding behavioral data
class BehaviorData:
    
    def __init__(self, 
                 minw=2, maxw=8, 
                 include_pid=True, include_state=True, 
                 active_samp=.1, 
                 window=3,
                 load=None):
        # minw, maxw: min and max weeks to collect behavior from
        # include_pid: should the participant id be a feature to the model
        # include_state: should the participant state be a feature
        if load is not None:
            self.load(load)
            return
        self.minw, self.maxw = minw, maxw
        self.include_pid = include_pid
        self.include_state = include_state
        self.active_samp = active_samp if active_samp is not None else 1
        self.window = window
        self.data = self.build()
        
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
        
    def iterate_subjects(self, n_subj=None):
        # find the unique participants and yield their subset
        # of data sorted by week
        for i_subj, pid in enumerate(self.data["pid"].unique()):
            if n_subj is not None and i_subj >= n_subj:
                break
            dpid = self.data[self.data["pid"] == pid]
            dpid = dpid.sort_values(by="week")
            yield dpid
            
    def subject_series(self, subj):
        return self.encode(subj, subj.index)
        
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
        def _onehot(a, l):
            vec = np.zeros(l)
            vec[a] = 1
            return vec
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
        Y = np.array([])
        for i,r in enumerate(row["response"]):
            Y = np.append(Y, _onehot(r,4))
        return X, Y
    
    def save(self, p):
        out = {"minw": self.minw, "maxw": self.maxw, "include_pid": self.include_pid,
               "include_state": self.include_state, "active_samp": self.active_samp,
               "window": self.window, "data": self.data}
        save(out, p)
        
    def load(self, p):
        d = load(p)
        self.minw, self.maxw = d["minw"], d["maxw"]
        self.include_pid = d["include_pid"]
        self.include_state = d["include_state"]
        self.active_samp = d["active_samp"]
        self.window = d["window"]
        self.data = d["data"]
        
    @property
    def dimensions(self):
        # helper to get the x and y input dimensions
        x, y = self.encode_row(self.data, self.data.index[0])
        return x.shape[0], y.shape[0]
        
