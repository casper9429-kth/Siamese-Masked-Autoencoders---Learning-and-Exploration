import jax 
from model import SiamMAE
from data import PreTrainingDataset
from torch.utils.data import DataLoader

def test():
    model = SiamMAE()
    rng = jax.random.PRNGKey(42)
    dataset = PreTrainingDataset()
    dataloader = DataLoader(dataset,batch_size =4,shuffle=False)
    f1s , f2s = next(iter(dataloader))
    f1s = f1s.reshape((f1s.shape[0]*f1s.shape[1],f1s.shape[2],f1s.shape[3],f1s.shape[4]))
    f2s = f2s.reshape((f2s.shape[0]*f2s.shape[1],f2s.shape[2],f2s.shape[3],f2s.shape[4]))
    params = model.init(rng,f1s,f2s)

test()