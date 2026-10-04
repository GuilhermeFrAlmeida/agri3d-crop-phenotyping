import numpy as np

class SegmentationEvaluator:
    """Dataset-wide confusion matrix; absent-class IoU is None and excluded from mIoU."""
    def __init__(self, num_classes=2):
        if num_classes < 2:
            raise ValueError("At least two classes")
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, preds, targets, valid_mask=None):
        preds, targets = np.asarray(preds), np.asarray(targets)
        if preds.shape != targets.shape or preds.size == 0:
            raise ValueError("Expected matching nonempty predictions and targets")
        if preds.dtype.kind not in 'iu' or targets.dtype.kind not in 'iu':
            raise ValueError("Class IDs must be integers")
        mask = np.ones(targets.shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask)
        if mask.shape != targets.shape or mask.dtype != bool:
            raise ValueError("Invalid validity mask")
        p, t = preds[mask], targets[mask]
        if np.any((p<0)|(p>=self.num_classes)) or np.any((t<0)|(t>=self.num_classes)):
            raise ValueError("Class ID out of range")
        self.confusion += np.bincount(t*self.num_classes+p,
            minlength=self.num_classes**2).reshape(self.confusion.shape)

    def compute(self):
        c = self.confusion
        if c.sum() == 0:
            raise ValueError("No evaluated points")
        tp = np.diag(c); true = c.sum(axis=1); pred = c.sum(axis=0)
        result = {'accuracy': float(tp.sum()/c.sum()), 'evaluated_points': int(c.sum())}
        ious = []
        for i in range(self.num_classes):
            union = true[i]+pred[i]-tp[i]
            iou = float(tp[i]/union) if union else None
            result[f'class_{i}_iou'] = iou
            if iou is not None: ious.append(iou)
            result[f'class_{i}_precision'] = float(tp[i]/pred[i]) if pred[i] else None
            result[f'class_{i}_recall'] = float(tp[i]/true[i]) if true[i] else None
            result[f'class_{i}_f1'] = float(2*tp[i]/(true[i]+pred[i])) if true[i]+pred[i] else None
        result['mIoU'] = float(np.mean(ious))
        return result

    def calculate_metrics(self, preds, targets, valid_mask=None):
        evaluator = SegmentationEvaluator(self.num_classes)
        evaluator.update(preds, targets, valid_mask)
        return evaluator.compute()
