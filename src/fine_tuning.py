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
        self.data_loader = DataLoader(data.PreTrainingDataset(),batch_size =4)
        self.rng = jax.random.PRNGKey(0)
        init_batch_f1, init_batch_f2 = next(iter(self.data_loader))
        self.params = pretrained_model.init(self.rng,init_batch_f1.flatten(0,1).numpy())

    def affinity_matrix(self,embed):   
        A = np.zeros((embed.shape[0]-1,embed.shape[1]-1,embed.shape[1]-1))
        for i in range(embed.shape[0]-1):
            A[i] = np.exp(embed[i][:-1].dot(np.transpose(embed[i+1][:-1])))
            A[i] = A[i]/np.sum(A[i],axis=0)
        print(A.shape)
        return A
        
    def label_propagation(m):
                
        for i in range(m):
            return None
        
    def test_dataset(self,top_k=7, queue_length=20,neighborhood_size=20):

        for i, [f1s,f2s] in enumerate(self.data_loader):
            embed = self.pretrained_model.apply(self.params,f1s.flatten(0,1).numpy())
            for i in range(embed.shape[0]-1):
                embed_1 = embed[i]
                embed_2 = embed[i+1]
                aff_matrix = self.affinity_matrix(embed_1,embed_2)
                

            return None

    def DAVIS_2017(self,top_k=7, queue_length=20,neighborhood_size=20):
        
        for i, [frames, annot] in enumerate(self.davis_dataset):
            print(frames.shape)
            print(annot.shape)
            embed = self.pretrained_model.apply(self.params,frames.numpy())
            A = self.affinity_matrix(embed)
            A_v, A_idxs = torch.topk(torch.from_numpy(A).flatten(1),k=top_k,dim =1)
            print(A_idxs.shape)
            print(A_v.shape) 
            return None

        return None
    
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