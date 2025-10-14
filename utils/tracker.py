from utils.metrics_printer import print_metrics
import matplotlib.pyplot as plt
from datetime import datetime


class ExperimentTracker:
    def __init__(
            self, 
            scalar_names, 
            metrics_names
    ):
        assert isinstance(scalar_names, (list, tuple)), f"scalar_names must be either list or tupple, got {type(scalar_names)}"
        assert isinstance(metrics_names, (list, tuple)), f"metrics_names must be either list or tupple, got {type(metrics_names)}"

        self.scalars = dict()
        for scalar_name in scalar_names:
            self.scalars[scalar_name] = []
        self.best_metrics = dict()
        for metrics_name in metrics_names:
            print(metrics_name)
            self.best_metrics[metrics_name] = 0.0
        
    def log_scalar(
            self, 
            name:str, 
            value
    ):
        self.scalars[name].append(value)

    def update_metric(
            self,
            name:str,
            value
    ):
        result = False
        if value > self.best_metrics[name]:
            self.best_metrics[name] = value
            result = True

        return result
    
    def print_best_metrics(self):
        print_metrics(self.best_metrics)
    
    def plot_figure(
            self, 
            scalar_names:list,
            filename,
            title=None,
            xlabel=None,
            ylabel=None
    ):
        for scalar_name in scalar_names:
            plt.plot(
                self.scalars[scalar_name], 
                label=scalar_name
            )

        plt.grid(True)
        if len(scalar_names) > 1:
            plt.legend()
        if title is not None:
            plt.title(title)
        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def log_training_start(self):
        self.train_start_time = datetime.now()

    def log_training_end(self):
        self.train_end_time = datetime.now()
        self.train_duration = self.train_end_time-self.train_start_time
        print(f"Total training time: {self.train_duration}")