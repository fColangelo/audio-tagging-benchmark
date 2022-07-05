from torchmetrics import Metric
import torch
from sklearn.metrics import label_ranking_average_precision_score
import numpy as np


class Lwlrap(Metric):
    
    def __init__(self, 
                 num_classes: int,
                 label_weigths: torch.Tensor=None, 
                 dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        
        self.num_classes     = num_classes
        
        if label_weigths is not None:
            self.class_weigths = label_weigths
        else:
            self.class_weigths = torch.ones(self.num_classes)
            
        self.add_state("cw_cum_precision",\
                       default=torch.zeros(self.num_classes),\
                       dist_reduce_fx="sum")
        self.add_state("cw_count",\
                       default=torch.zeros(self.num_classes),\
                       dist_reduce_fx="sum")
    
    def sample_class_precisions(self, scores, truth):
        """Calculate precisions for each true class for a single sample.

        Args:
        scores: np.array of (num_classes,) giving the individual classifier scores.
        truth: np.array of (num_classes,) bools indicating which classes are true.

        Returns:
        pos_class_indices: np.array of indices of the true classes for this sample.
        pos_class_precisions: np.array of precisions corresponding to each of those
          classes.
        """
        num_classes = scores.shape[0]
        pos_class_indices = np.flatnonzero(truth > 0)
        # Only calculate precisions if there are some true classes.
        if not len(pos_class_indices):
            return pos_class_indices, np.zeros(0)
        # Retrieval list of classes for this sample. 
        retrieved_classes = np.argsort(scores)[::-1]
        # class_rankings[top_scoring_class_index] == 0 etc.
        class_rankings = np.zeros(num_classes, dtype=np.int)
        class_rankings[retrieved_classes] = range(num_classes)
        # Which of these is a true label?
        retrieved_class_true = np.zeros(num_classes, dtype=np.bool)
        retrieved_class_true[class_rankings[pos_class_indices]] = True
        # Num hits for every truncated retrieval list.
        retrieved_cumulative_hits = np.cumsum(retrieved_class_true)
        # Precision of retrieval list truncated at each hit, in order of pos_labels.
        precision_at_hits = (
          retrieved_cumulative_hits[class_rankings[pos_class_indices]] / 
          (1 + class_rankings[pos_class_indices].astype(np.float)))
        return pos_class_indices, precision_at_hits

    def update(self,  GTs: torch.Tensor, preds: torch.Tensor,):
        # Called at each batch
        for idx in range(GTs.shape[0]):
            class_idx, p     = self.sample_class_precisions(GTs[idx], preds[idx])
            for idx_2 in range(class_idx.shape[0]):
                index = class_idx[idx_2]
                self.cw_cum_precision[index] += p[index] 
                self.cw_count[index]         += 1 
            
    def compute(self):
        stable_count = torch.max( torch.stack((self.cw_count,\
                        torch.ones(self.num_classes)), dim=1), dim=1)[0]
        return torch.mul(self.class_weigths,\
                         torch.divide(self.cw_cum_precision, stable_count))
