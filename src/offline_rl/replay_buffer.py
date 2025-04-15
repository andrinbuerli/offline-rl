import torch
from tensordict import TensorDict
from torch.utils.data import DataLoader
from torchrl.data.replay_buffers import LazyTensorStorage, ReplayBuffer
from tqdm.auto import tqdm


def collate_minari_databatch(batch):
    result = {
        #"id": torch.Tensor([x.id for x in batch]),
        "action": torch.stack([torch.as_tensor(x.actions) for x in batch]).float(),
    }
    
    observations = torch.stack([TensorDict(x.observations) for x in batch])
    observation = observations["observation"].float()
    desired_goal = observations["desired_goal"].float()
    if "wall_info" in observations:
        wall_info = observations["wall_info"].float()
    else:
        wall_info = None
    
    result["observation"] = observation[:, :-1] # remove the last step
    result["desired_goal"] = desired_goal[:, :-1] # remove the last step
    if wall_info is not None:
        result["wall_info"] = wall_info[:, :-1] # remove the last step
        
    # build "next" dict with reward, done, observation
    result["next"] = {}
    
    result["next"]["reward"] = torch.stack([torch.as_tensor(x.rewards) for x in batch]).unsqueeze(-1).float()
    # done as in "terminated" or "truncated"
    result["next"]["done"] = (
        torch.stack([torch.as_tensor(x.terminations) for x in batch]) 
        | torch.stack([torch.as_tensor(x.truncations) for x in batch])
    ).unsqueeze(-1)
    result["next"]["observation"] = observation[:, 1:] # remove the first step
    result["next"]["desired_goal"] = desired_goal[:, 1:] # remove the first step
    if wall_info is not None:
        result["next"]["wall_info"] = wall_info[:, 1:] # remove the first step
    
    # flatten dim 0/1 (batch, time)
    for key in result:
        if key == "next":
            for subkey in result[key]:
                result[key][subkey] = result[key][subkey].reshape(-1, *result[key][subkey].shape[2:])
        else:
            result[key] = result[key].reshape(-1, *result[key].shape[2:])
            
    batch_size = observation.shape[0] * (observation.shape[1] - 1)
    return TensorDict(result, batch_size=batch_size)
    

class LocalMinariReplayBuffer(ReplayBuffer):
    def __init__(self, dataset, max_size=1000000, load_bsize=32):
        super().__init__(storage=LazyTensorStorage(max_size=max_size))
        if not isinstance(dataset, list):
            dataset = [dataset]
            
        for ds in dataset:
            self.load_into_buffer(ds, load_bsize)

    def load_into_buffer(self, dataset, load_bsize):
        self.dataset = dataset
        dataloader = DataLoader(
            dataset,
            batch_size=load_bsize,
            shuffle=False,
            collate_fn=collate_minari_databatch,
        )
        
        for batch in tqdm(dataloader, desc="Loading batched dataset into buffer", unit="batch"):
            self.extend(batch)