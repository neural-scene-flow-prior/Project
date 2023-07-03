#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import numpy as np
from torch.utils.data import Dataset

class KITTISceneFlowDataset(Dataset):
    def __init__(self, options, train=False):
        self.options = options
        self.train = train
    
        if self.train:
            self.datapath = sorted(glob.glob(f"{self.options.dataset_path}/*.npz"))[:100]
        else:
            self.datapath = sorted(glob.glob(f"{self.options.dataset_path}/*.npz"))[100:]

        self.cache = {}
        self.cache_size = 30000
        
    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, index):
        fn = self.datapath[index]
        with open(fn, 'rb') as fp:
            data = np.load(fp)
            pc1 = data['pos1'].astype('float32')
            pc2 = data['pos2'].astype('float32')
            flow = data['gt'].astype('float32')

        n1 = pc1.shape[0]
        n2 = pc2.shape[0]
        if not self.options.use_all_points:
            num_points = self.options.num_points

            if n1 >= num_points:
                sample_idx1 = np.random.choice(n1, num_points, replace=False)
            else:
                sample_idx1 = np.concatenate((np.arange(n1), np.random.choice(n1, num_points - n1, replace=True)), axis=-1)

            if n2 >= num_points:
                sample_idx2 = np.random.choice(n2, num_points, replace=False)
            else:
                sample_idx2 = np.concatenate((np.arange(n2), np.random.choice(n2, num_points - n2, replace=True)), axis=-1)
                
            pc1 = pc1[sample_idx1, :].astype('float32')
            pc2 = pc2[sample_idx2, :].astype('float32')
            flow = flow[sample_idx1, :].astype('float32')

        return pc1, pc2, flow





