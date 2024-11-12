import sys
import os
import gc
import re
import numpy as np
import pandas as pd
import torch
import torch.utils.data as du
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

# import psutils

# def check_mem_usage():
#     """
#     Check memory used, in GB
#     """
#     mem_info = psutil.virtual_memory()
#     total_mem = mem_info[0]
#     avail_mem = mem_info[1]
#     mem_used = np.float(total_mem - avail_mem)/np.power(1000, 3)
#     now = datetime.datetime.now()
#     print(now.strftime("%Y-%m-%d %H:%M:%S") + ' mem used: {} GB'.format(mem_used))

class DepMapPairsDataset(Dataset):
    def __init__(self, omics_matrix_files, ppis_file, index_file, crispr_score_file=None):
        """
        Args:
            omics_matrix_file = list of csv files containing omics matrices. expected that row index is samples, column index features.
            ppis_file = csv file containing ppis of genes denoted by columns in crispr_score_file (i.e. the genes that get deleted). expected that rows here are aligned with columns of crispr_score_file
            crispr_score_file = csv file containing crispr scores. expected that row index is samples, column index features.
                                expected that rows be aligned with omics matrix file. expected that columns be aligned with rows of ppis_file
        Notes:
            omics_matrix file rows and ppis_file rows are used in pairwise manner to attempt to predict scores in crispr_score_file.
        """
        self.omics_matrix_files = omics_matrix_files
        self.omics_matrix_feats = []
        self.omics_matrix_idx = []
        omics_idx = 0
        for i in range(len(self.omics_matrix_files)):
            file_i = self.omics_matrix_files[i]
            feats_file_i = re.sub('.csv$', '', file_i) + "_feats.csv"
            feats_i = np.loadtxt(feats_file_i, delimiter=',', dtype='str')
            self.omics_matrix_feats.append(feats_i)
            self.omics_matrix_idx.append(np.arange(omics_idx, omics_idx + len(feats_i)))
            omics_idx += len(feats_i)
        self.ppis_file = ppis_file
        self.ppis_feats_file = re.sub('.csv$', '', ppis_file) + "_columns.csv"
        feats_ppi = np.loadtxt(self.ppis_feats_file, delimiter=',', dtype='str')
        self.ppi_feats = feats_ppi
        self.ppi_matrix_idx = np.arange(omics_idx, omics_idx + len(feats_ppi))
        self.crispr_score_file = crispr_score_file
        self.index = np.loadtxt(index_file, delimiter=',').astype('int32')
        
    def __getitem__(self, idx):
        """
        Get item.
        """
        idx_i, idx_j = self.index[idx]
        X_omics_list = []
        
        for k in range(len(self.omics_matrix_files)):
            file_k = self.omics_matrix_files[k]
            X_k = np.loadtxt(file_k, delimiter=',', skiprows=idx_i, max_rows=1).astype('float32')
            X_omics_list.append(X_k)
            del X_k
        
        X_omics = np.concatenate(X_omics_list)
        X_ppi = np.loadtxt(self.ppis_file, delimiter=',', skiprows=idx_j, max_rows=1).astype('float32')
        X_final = torch.from_numpy(np.concatenate([X_omics, X_ppi]))
        
        if self.crispr_score_file is not None:
            y_crispr = torch.from_numpy(np.loadtxt(self.crispr_score_file, delimiter=',', skiprows=idx_i, max_rows=1, usecols=idx_j).astype('float32'))
            return X_final, y_crispr
        else:
            return X_final

    def __len__(self):
        return(self.index.shape[0])
        

class DepMapPairsEncoder(nn.Module):
    def __init__(self, omics_matrix_idx_list, ppi_matrix_idx, n_hiddens_level0=3, n_hiddens_top=3, contraction_ratio=10):
        
        super(DepMapPairsEncoder, self).__init__()
        self.omics_matrix_idx = omics_matrix_idx_list
        self.ppi_matrix_idx = ppi_matrix_idx
        self.n_hiddens_level0=n_hiddens_level0
        self.n_hiddens_top = n_hiddens_top
        
        top_layer_input_size = 0
        for i in range(len(self.omics_matrix_idx)):
            idx_i = self.omics_matrix_idx[i]
            input_size = len(idx_i)
            for j in range(n_hiddens_level0):
                output_size = max(int(input_size/contraction_ratio), 10)
                self.add_module('FC_omics_{}_layer_{}'.format(str(i), str(j)), nn.Linear(input_size, output_size))
                del input_size
                input_size = output_size
                if j == (n_hiddens_level0 - 1):
                    top_layer_input_size += output_size
                del output_size
        
        
        del input_size
        input_size = len(self.ppi_matrix_idx)
        
        for j in range(n_hiddens_level0):
            output_size = max(int(input_size/contraction_ratio), 10)
            self.add_module('FC_ppi_layer_{}'.format(str(j)), nn.Linear(input_size, output_size))
            del input_size
            input_size = output_size
            if j == (n_hiddens_level0 - 1):
                top_layer_input_size += output_size
            del output_size
        
        del input_size
        input_size = top_layer_input_size
        top_layer_output_size = 0
        
        for j in range(n_hiddens_top):
            output_size = max(int(input_size/contraction_ratio), 10)
            self.add_module('FC_top_layer_{}'.format(str(j)), nn.Linear(input_size, output_size))
            del input_size
            input_size = output_size
            if j == (n_hiddens_top - 1):
                top_layer_output_size += output_size
            del output_size
            
        self.add_module('final_linear', nn.Linear(top_layer_output_size, 1))
        
    def forward(self, x):
        # stores output of individual NNs applied to -omics and PPIs
        intermediate_tensors = []
        # get omics intermediate tensors
        for i in range(len(self.omics_matrix_idx)):
            idx_i = self.omics_matrix_idx[i]
            start_i = int(idx_i[0])
            len_i = len(idx_i)
            x_i = x.narrow(1, start_i, len_i)
            for j in range(self.n_hiddens_level0):
                omics_layer_i_j = 'FC_omics_{}_layer_{}'.format(str(i), str(j))
                x_i = torch.tanh(self._modules[omics_layer_i_j](x_i))
            intermediate_tensors.append(x_i)
        
        # get ppi based tensor
        idx_ppi = self.ppi_matrix_idx
        start_ppi = int(idx_ppi[0])
        len_ppi = len(idx_ppi)
        x_ppi = x.narrow(1, start_ppi, len_ppi)
        
        for j in range(self.n_hiddens_level0):
            ppi_layer_j = 'FC_ppi_layer_{}'.format(str(j))
            x_ppi = torch.tanh(self._modules[ppi_layer_j](x_ppi))
            
        intermediate_tensors.append(x_ppi)
        # setup intermediate input for top set of layers
        x_intermediate = torch.cat(intermediate_tensors, axis=1)
        for j in range(self.n_hiddens_top):
            top_layer_j = 'FC_top_layer_{}'.format(str(j))
            x_intermediate = torch.tanh(self._modules[top_layer_j](x_intermediate))
            
        x_final = self._modules['final_linear'](x_intermediate).flatten()
        return x_final
        
        