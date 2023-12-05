import jax
import numpy as np
from sklearn.metrics import f1_score,jaccard_score
import model
import data
from torch.utils.data import DataLoader
import torch
import jax.numpy as jnp
from datasets.davis import DAVIS2017

class FineTuner():
    def __init__(self,pretrained_model):
        self.pretrained_model = pretrained_model
        self.davis_dataset = DAVIS2017()
        self.davis_loader = DataLoader(self.davis_dataset, batch_size=1)
        self.data_loader = DataLoader(data.PreTrainingDataset(),batch_size =4)
        self.rng = jax.random.PRNGKey(0)
        init_batch_f1, init_batch_f2 = next(iter(self.data_loader))
        self.params = pretrained_model.init(self.rng,init_batch_f1.flatten(0,1).numpy())

    def affinity_matrix(self,f,g):  
        A = f@g
        return A
    
    def compute_mask(F,Y):

        def in_neighborhood(idx_N1,neigborhood,H=14,W=14):
            i_1,j_1 = idx_N1 // H , idx_N1 % H
            for idx_N2 in range(H*W):
                i_2, j_2 = idx_N2 // H, idx_N2 % H
                if i_1 - i_2

        n_labels = Y.shape[0]
        n_pixels = F.shape[1]
        n_context = F.shape[0]
        mask = np.zeros((n_labels,n_pixels,n_pixels,n_context))
        for l in range(n_labels):
            for N_1 in range(n_pixels):
                for N_2 in range(n_pixels):
                    for n in range(n_context):
                        if Y[l,N_1,n] == 1 and N_2 :
                            mask[l,N_1,N_2,n] = 1

        return mask

    def topk(input):
        topk_idxs = np.argsort(input)
        topk_vals = input[topk_idxs]
        return topk_vals, topk_idxs
        
    def label_propagation(self,f,g,y,k,tau=1):
        A = self.affinity_matrix(f,g)
        M = self.compute_mask(y,f,g)
        z = np.zeros((M.shape[0],M.shape[1]))
        for l in range(M.shape[0]):
            A_l = M[l]*A
            for j in range(M.shape[1]):
                topk_vals, topk_idxs = self.topk(A_l[j],k)
                topk_labels = y[l,topk_idxs]
                topk_vals = jax.nn.softmax(topk_vals/tau,axis=1)
                z[l,j] = topk_vals.dot(topk_labels)/M.shape[3]
        
        return z 



    def DAVIS_2017(self,top_k=7, queue_length=20,neighborhood_size=20):

        f1_scores = []
        jaccard_scores = []
        
        for i, [frames, annot] in enumerate(self.davis_dataset):
            F = [self.pretrained_model.apply(self.params,frames[0])]
            Y = [annot[0]]
            for j in range(1,len(frames)+1):
                g = self.pretrained_model.apply(self.params,frames[j])
                if j < queue_length:
                    f = np.concatenate(F,axis =0)
                    y = np.concatenate(Y,axis =0)
                elif j>=queue_length:
                    F.pop(0)
                    f = np.concatenate(F,axis =0)
                    y = np.concatenate(Y[-20:],axis =0)
                z = self.label_propagation(self,f,g,y,top_k,tau=1)
                Y.append(z)
                F.append(g)

            f1_scores.append(f1_score(annot,Y))
            jaccard_scores.append(jaccard_score(annot,Y))
                    
        return f1_scores, jaccard_scores
    
    def JHMDB(self,top_k=10, queue_length=20,neighborhood_size=8):
        
        for i, data in enumerate(self.jhmdb_dataloader):
            return None
    
    def VIP(self,top_k=7, queue_length=20,neighborhood_size=20):

        for i, data in enumerate(self.vip_dataloader):
            return None

def main():
    fine_tuner_model = model.FineTuneSiamMAE()
    fine_tuner = FineTuner(fine_tuner_model)
    fine_tuner.DAVIS_2017()

if __name__ == '__main__':
    main()