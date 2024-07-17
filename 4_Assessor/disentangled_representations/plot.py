import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

import sys
from labels import Labels


def get_dictfile(file: str, dirpath: str = './'):
    """Load a dictionary from file."""
    with open(dirpath+file, 'rb') as f:
        return pickle.load(f)


class ResultPlots():
    def __init__(self, model_id, parent_dir: str = './'):

        # Get training stats
        self.results = get_dictfile(
            f'{model_id}-training_stats.pkl',
            dirpath=parent_dir+'models/'
        )

        # Get best metrics from validation models
        self.validation_metrics = get_dictfile(
            file=f'{model_id}-metrics.pkl',
            dirpath=parent_dir+'validation_models/'
        )

        # Load task_label_mapping
        self.dataset_name = self.results.get('dataset_name')
        self.task_label_mapping = Labels(dataset_name=self.dataset_name).task_mapping

    def _idx_to_label(self, idx: int) -> str:
        """Map task index to label string."""
        return list(self.task_label_mapping.keys())[idx]

    def learning_curves(self, results: dict = None):
        """Plot the learning curves (loss and accuracy on train and validation sample)."""

        if results is None:
            results = self.results
        
        fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(6,8))
        fig.subplots_adjust(hspace=0.3)

        abline_kwargs = dict(color='grey', lw=0.5, linestyle='dashed')
        marker_kwargs = dict(zorder=10, s=20, color='red', marker='x', label='Best model')
        
        # Same plot for loss and accuracy
        for i, metric in enumerate(['loss', 'accuracy']):

            train = results[metric]['train']
            valid = results[metric]['validation']

            axs[i].plot(train, label='Training set')
            axs[i].plot(valid, label='Validation set')
            axs[i].axvline(results[metric]['best_epoch'], **abline_kwargs)
            axs[i].axhline(results[metric]['best'], **abline_kwargs)
            axs[i].scatter(results[metric]['best_epoch'], results[metric]['best'], **marker_kwargs)
            axs[i].set_title(metric.title())
            
        axs[1].set_xlabel('Epoch')

        axs[0].legend(loc='upper left', bbox_to_anchor=(1.05, 1.04))

        return fig

    def learning_curves_val_model(
            self,
            model_id: int,
            feat_task: str,
            target_task: str,
            hyperparams_str: str = 'max_epochs=1000-lr=0.0001-seed=4243',
        ):
        """Wrapper for `plot_learning_curves()` function for validation models."""

        # Load training stats
        d = get_dictfile(
            file=f'model_id={model_id}-feat_task={feat_task}-target_task={target_task}-{hyperparams_str}-training_stats.pkl',
            dirpath='validation_models/'
        )

        # Plot stats
        self.learning_curves(results=d)

    def drilldown(self, metric: str, sample: str):
        """Plot the drilldown of a metric (accuracy, loss prediction, loss dcor)."""

        if metric.startswith('loss'):
            metric, losstype = metric.split()

        # Data has structure (epoch, [losstype], task1, task2) 
        data = np.stack(self.results[f'{metric}_drilldown'][sample])

        if metric.startswith('loss'):
            losstype_idx = ['dcor', 'prediction'].index(losstype)
            data = data[:, losstype_idx, :, :]
        
        n_tasks = data.shape[-1]
        
        # Plot
        fig, axs = plt.subplots(nrows=n_tasks, ncols=1, sharex=True, figsize=(6,8))

        color_lines = [Line2D([0], [0], color='C'+str(i), lw=2) for i in range(n_tasks)]
        color_labels = [self._idx_to_label(i) for i in range(n_tasks)]
        style_lines = [
            Line2D([0], [0], color='black', lw=1, linestyle='-'),
            Line2D([0], [0], color='black', lw=1, linestyle='--')
        ]
        style_labels = ['Primary', 'Secondary']

        title = {
            'accuracy': 'Accuracy',
            'dcor': 'Distance correlation',
            'prediction': 'Cross-entropy loss'
        }

        for task1 in range(n_tasks):
            for task2 in range(n_tasks):
                axs[task1].plot(
                    data[:,task1,task2],
                    label=self._idx_to_label(task2),
                    linestyle='--' if task1 != task2 else '-',)
                axs[task1].set_title(
                    f'Branch: {self._idx_to_label(task1)}',
                    color=f'C{task1}',
                    fontweight='normal',
                )
                axs[task1].set_ylabel(
                    title[metric if metric == 'accuracy' else losstype]
                )
                axs[task1].set_ylim([
                    0,
                    (1 if metric == 'accuracy' else data.max()*1.07)
                ])

            # Add the legends (first axis only)
            if task1 == 0:
                legend1 = axs[task1].legend(color_lines, color_labels, loc='upper left', bbox_to_anchor=(1.05, 1.04))
                legend1.set_title('Task')
                legend1._legend_box.align = 'left'
                
                legend2 = axs[task1].legend(style_lines, style_labels, loc='upper left', bbox_to_anchor=(1.05, 0.5))
                legend2.set_title('Task type')
                legend2._legend_box.align = 'left'

                # Add the first legend manually to the current Axes.
                axs[task1].add_artist(legend1)

        axs[-1].set_xlabel('Epoch')

        return fig
    
    def _get_best_metrics_df(self) -> pd.DataFrame:
        """Convert best validation metrics to pandas DataFrame."""

        # Convert to pandas DataFrame
        # and sort such that colors of target and feature tasks match
        best_metrics_df = (
            pd.DataFrame.from_dict(
                self.validation_metrics['values'],
                orient='index'
            )
            .reset_index()
            .rename(columns={
                'level_0': 'Feature task', 
                'level_1': 'Target task'
            })
        )
        # Convert indices to strings
        for col in ['Feature task', 'Target task']:
            best_metrics_df[col] = best_metrics_df[col].apply(self._idx_to_label)

        return best_metrics_df

    def best_validation_metrics(self) -> None:
        """
        Create a grouped bar plot that displays the best metrics from each
        validation model.
        """

        metric = self.validation_metrics['metric'].title()
        best_metrics_df = (
            self._get_best_metrics_df()
            .rename(columns={'best': metric})
        )

        # Grouped barplot for best metrics
        ax = sns.barplot(
            data=best_metrics_df,
            x='Feature task', y=metric, hue='Target task'
        )

        # Annotate values above the bars
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height+0.01),
                        ha='center', va='bottom', xytext=(0, 0),
                        textcoords='offset points', fontsize=7, rotation=90)
            
        # Add a vertical line at the baseline value for each bar
        for i, p in enumerate(ax.patches):
            ax.hlines(
                y=best_metrics_df.baseline.iloc[i],
                xmin=p.get_x(), xmax=p.get_x() + p.get_width(),
                color='lightgrey', linewidth=1
            )            

        # Grouped barplot for baseline f1 scores
        sns.barplot(
            data=best_metrics_df,
            x='Feature task', y='baseline', hue='Target task',
            palette='pastel',  # Use the lighter color palette
            ax=ax
        )           

        # Only keep the first half of the handles and labels for legend
        handles, labels = ax.get_legend_handles_labels()
        half_length = len(handles) // 2
        handles, labels = handles[:half_length], labels[:half_length]
        ax.legend(handles, labels, title='Target task')

        # Add summary metrics as text
        summary_metrics_str = (
            'Avg. improvement over baseline:'
            ' factor={baseline_improvement_factor:.2f}'
            ', diff={baseline_improvement_diff:.2%}'
            '\n'
            'Avg. feature usability: factor={feature_usability_factor:.2f}'
            '\n'
            'Harmonic mean: {harmonic_mean:.2f}'
        ).format(**self.validation_metrics['summary'])
        ax.text(
            0.5, 0.97, summary_metrics_str,
            ha='center', va='top', transform=ax.transAxes
        )

        # Move legend
        sns.move_legend(ax, 'upper left', bbox_to_anchor=(1.05, 1.02))

        # Change tick colors
        for i, tick_label in enumerate(ax.xaxis.get_ticklabels()):
            tick_label.set_color(sns.color_palette()[i])

        # Add title
        plt.title(f'Validation model performance ({metric})')

        # Set y-axis title
        plt.ylabel(metric)

        # Set y-axis limits
        plt.ylim(0, 1)

        return plt.gcf()
    
if __name__ == '__main__':

    ### Plot validation performance for different dcor loss factors ###
    import json
    from matplotlib.backends.backend_pdf import PdfPages
    from tqdm.auto import tqdm

    # Load model ids and hyperparams
    with open('model_ids.json') as f:
        model_ids_mapping = json.load(f)
    start_id = 121  # Adjust
    model_ids_mapping = {k: v for k, v in model_ids_mapping.items() if int(k) >= start_id}

    # Safe to multi-page pdf
    with PdfPages('../sharing/validation_model_performance.pdf') as pdf:
        for model_id, hyperparams in tqdm(model_ids_mapping.items()):
            plt.figure(figsize=(10, 6))
            result_plots = ResultPlots(model_id=model_id)
            result_plots.best_validation_metrics()
            plt.title(f'Validation model performance\nmodel {model_id} - dcor_loss_factor={hyperparams["dcor_loss_factor"]}')
            pdf.savefig(bbox_inches='tight')
            plt.clf()
            plt.close()
