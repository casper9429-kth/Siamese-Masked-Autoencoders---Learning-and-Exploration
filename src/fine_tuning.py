import numpy as np
from sklearn.metrics import f1_score,jaccard_score
import model

class FineTuner():
    def __init__(self,pretrained_model):
        self.pretrained_model = pretrained_model
        self.davis_dataloader = 
        self.jhmdb_dataloader =
        self.vip_dataloader = 


    def label_propagation():
        
        for i in range(m)



    def DAVIS_2017(self,top_k=7, queue_length=20,neighborhood_size=20):
        
        for i, data in enumerate(self.davis_dataloader):
            embed = apply_model(data)
            pred = label_propagation(embed,f1)
            

        return None
    
    def JHMDB(self,top_k=10, queue_length=20,neighborhood_size=8):
        
        for i, data in enumerate(self.jhmdb_dataloader):
            return None
    
    def VIP(self,top_k=7, queue_length=20,neighborhood_size=20):

        for i, data in enumerate(self.vip_dataloader):
            return None

def main():
    model = model.SiamMAE()
    



if __name__ == '__main__':
    main()