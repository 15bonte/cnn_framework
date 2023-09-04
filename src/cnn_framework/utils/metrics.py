from abc import abstractmethod
from matplotlib import pyplot as plt
import numpy as np
import torch
from torchmetrics import AUROC, ROC, MeanSquaredError, PearsonCorrCoef
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.classification import JaccardIndex
from torchmetrics.classification import Accuracy
import skdim
import sklearn.metrics as sklearn_metrics


class AbstractMetric:
    def __init__(self, device="cpu", num_classes=None):
        self.device = device
        self.num_classes = num_classes

    @abstractmethod
    def update(self, predictions, targets, adds=None):
        pass

    @abstractmethod
    def get_score(self):
        return 0, None

    @abstractmethod
    def reset(self):
        pass


class DetectionMeanAveragePrecision(AbstractMetric):
    def __init__(self, *args):
        super().__init__(*args)
        self.metric = MeanAveragePrecision()
        self.name = "Detection mAP"

    def update(self, predictions, targets, _=None):
        self.metric.update(predictions, targets)

    def get_score(self):
        return self.metric.compute().item(), None

    def reset(self):
        self.metric = MeanAveragePrecision()


class PCC(AbstractMetric):
    def __init__(self, *args):
        super().__init__(*args)
        self.metric = PearsonCorrCoef().to(self.device)
        self.name = "PCC"

    def update(self, predictions, targets, _=None):
        self.metric.update(torch.flatten(predictions), torch.flatten(targets).float())

    def get_score(self):
        return self.metric.compute().item(), None

    def reset(self):
        self.metric = PearsonCorrCoef()


class IoU(AbstractMetric):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = "IoU"
        self.metric = JaccardIndex(num_classes=2)

    def update(self, predictions, targets, _=None):
        self.metric.update(predictions.detach().cpu(), targets.int().detach().cpu())

    def get_score(self):
        return self.metric.compute().item(), None

    def reset(self):
        self.metric = JaccardIndex(num_classes=2)


class ClassificationAccuracy(AbstractMetric):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = "ClassificationAccuracy"
        self.true, self.pred = torch.empty(0).to(self.device), torch.empty(0).to(self.device)
        self.metric = Accuracy(task="multiclass", num_classes=self.num_classes).to(self.device)

    def update(self, predictions, targets, _=None):
        # From vector to classification
        predictions_argmax = torch.argmax(predictions, dim=1)
        targets_argmax = torch.argmax(targets, dim=1)
        # Update metric
        self.metric.update(
            predictions_argmax,
            targets_argmax,
        )
        # Update current values
        self.true = torch.cat((self.true, targets_argmax))
        self.pred = torch.cat((self.pred, predictions_argmax))

    def get_score(self):
        return self.metric.compute().item(), (self.true, self.pred)

    def reset(self):
        self.metric = Accuracy(task="multiclass", num_classes=self.num_classes).to(self.device)
        self.true, self.pred = torch.empty(0).to(self.device), torch.empty(0).to(self.device)


class DummyMetric(AbstractMetric):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = "DummyMetric"

    def update(self, _, __, ___=None):
        pass

    def get_score(self):
        return 0, None

    def reset(self):
        pass


class IntrinsicDimensionMetric(AbstractMetric):
    """Estimate local intrinsic dimension (dimension in k-nearest-neighborhoods around each point)"""

    def __init__(self, *args):
        super().__init__(*args)
        self.name = "IntrinsicDimensionMetric"
        self.local_dimensions = []

    def update(self, predictions, __, ___=None):
        lpca = skdim.id.lPCA().fit_pw(
            predictions.detach().cpu().numpy(), n_neighbors=predictions.shape[0] - 1, n_jobs=1
        )
        normalized_local_dimensions = lpca.dimension_pw_ / predictions.shape[1]
        self.local_dimensions = np.concatenate(
            (self.local_dimensions, normalized_local_dimensions), axis=None
        )

    def get_score(self):
        return np.mean(self.local_dimensions), None

    def reset(self):
        self.local_dimensions = []


class PositivePairMatchingMetric(AbstractMetric):
    """Computes the accuracy over the k top predictions for the specified values of k"""

    def __init__(self, *args):
        super().__init__(*args)
        self.name = "PositivePairMatchingMetric"
        self.local_matchings = []

    def update(self, predictions, targets, ___=None):
        topk = (1,)

        with torch.no_grad():
            maxk = topk[0]
            batch_size = targets.size(0)

            _, pred = predictions.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(targets.view(1, -1).expand_as(pred))

            correct_k = correct[:maxk].reshape(-1).float().sum(0, keepdim=True)
            matching = correct_k.div_(batch_size)[0].item()
            self.local_matchings.append(matching)

    def get_score(self):
        return np.mean(self.local_matchings), None

    def reset(self):
        self.local_matchings = []


class AUROCMetric(AbstractMetric):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = "Weighted AUROC"
        self.true, self.pred = torch.empty(0).to(self.device), torch.empty(0).to(self.device)
        self.metric, self.alternative_metric = self.initialize_metric()

    def initialize_metric(self):
        return (
            AUROC(task="multiclass", num_classes=self.num_classes, average="weighted").to(
                self.device
            ),
            ROC(task="multiclass", num_classes=self.num_classes, average="weighted").to(
                self.device
            ),
        )

    def update(self, predictions, targets, _=None):
        # From vector to classification
        predictions_argmax = torch.argmax(predictions, dim=1)
        targets_argmax = torch.argmax(targets, dim=1)
        # Update metric
        self.metric.update(
            predictions,
            targets_argmax,
        )
        self.alternative_metric.update(
            predictions,
            targets_argmax,
        )
        # Update current values
        self.true = torch.cat((self.true, targets_argmax))
        self.pred = torch.cat((self.pred, predictions_argmax))

    def get_score(self):
        # Plot ROC curves
        fpr, tpr, thresholds = self.alternative_metric.compute()
        for class_id in range(self.num_classes):
            local_fpr = fpr[class_id].detach().cpu().numpy()
            local_tpr = tpr[class_id].detach().cpu().numpy()
            local_thresholds = thresholds[class_id].detach().cpu().numpy()
            # Get optimal threshold
            # Calculate the G-mean
            gmean = np.sqrt(local_tpr * (1 - local_fpr))
            # Find the optimal threshold
            index = np.argmax(gmean)
            threshold_opt = round(local_thresholds[index], ndigits=4)
            print("Threshold:", threshold_opt)
            # Plot ROC curve
            roc_auc = sklearn_metrics.auc(local_fpr, local_tpr)
            plt.title("Receiver Operating Characteristic")
            plt.plot(
                local_fpr,
                local_tpr,
                label="AUC = %0.2f" % roc_auc,
            )
            plt.title(
                f"ROC curve for class {class_id}",
            )
            plt.legend(loc="lower right")
            plt.plot([0, 1], [0, 1], "r--")
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel("True Positive Rate")
            plt.xlabel("False Positive Rate")
            plt.show()
        # Return AUC
        return self.metric.compute().item(), (self.true, self.pred)

    def reset(self):
        self.metric, self.alternative_metric = self.initialize_metric()
        self.true, self.pred = torch.empty(0).to(self.device), torch.empty(0).to(self.device)


class MeanSquaredErrorMetric(AbstractMetric):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = "MeanSquaredError"
        self.metric = MeanSquaredError(squared=True).to(self.device)

    def update(self, predictions, targets, _=None):
        # Update metric
        self.metric.update(
            predictions,
            targets,
        )

    def get_score(self):
        return -self.metric.compute().item(), None

    def reset(self):
        self.metric = MeanSquaredError(squared=True).to(self.device)


class MeanErrorMetric(AbstractMetric):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = "MeanError"
        self.metric = MeanSquaredError(squared=False).to(self.device)

    def update(self, predictions, targets, _=None):
        # Update metric
        self.metric.update(
            predictions,
            targets,
        )

    def get_score(self):
        return -self.metric.compute().item(), None

    def reset(self):
        self.metric = MeanSquaredError(squared=False).to(self.device)
