import jax
import numpy as np
from sklearn.metrics import f1_score,jaccard_score
import model
import data
from torch.utils.data import DataLoader
import torch

class FineTuner():
    def __init__(self,pretrained_model):
        self.pretrained_model = pretrained_model
        self.data_loader = DataLoader(data.PreTrainingDataset(),batch_size =4)
        self.rng = jax.random.PRNGKey(0)
        init_batch_f1, init_batch_f2 = next(iter(self.data_loader))
        print(init_batch_f1.shape)
        print(init_batch_f1.flatten(0,1).shape)
        self.params = pretrained_model.init(self.rng,init_batch_f1.flatten(0,1).numpy())

    def label_propagation(m):
                
        for i in range(m):
            return None
        
    def test_dataset(self,top_k=7, queue_length=20,neighborhood_size=20):
        for i, [f1s,f2s] in enumerate(self.data_loader):
            print(f1s.shape)

    def DAVIS_2017(self,top_k=7, queue_length=20,neighborhood_size=20):
        
        for i, data in enumerate(self.davis_dataloader):
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
    fine_tuner.test_dataset()

if __name__ == '__main__':
    main()