
import os
import numpy as np
from zipfile import ZipFile

from .rotations import qad as _qad, qcip as _qcip
from .imu import relative_orientations
from .utils import apply_procrustes_alignment

# TODO: add temporal consistency metric (to penalize high movement jitter)


def rmse(pred, target, weights=1.0, reduce=True):
    """
    Computes the RMSE between pred and target.

    Args:
        pred(np.ndarray): array of predicted data.
        target(np.ndarray): array of target data.
        weights(list[float], float): weights for each value.
        reduce(bool): if reduction method should be applied.

    Returns:
        (np.ndarray): rmse between pred and target

    """
    dist = weights * np.sqrt(((pred - target) ** 2).mean(axis=-1))
    return dist.mean() if reduce else dist


def mse(pred, target, weights=1.0, reduce=True):
    """
    Computes the MSE between pred and target.

    Args:
        pred(np.ndarray): array of predicted data.
        target(np.ndarray): array of target data.
        weights(list[float], float): weights for each value.
        reduce(bool): if reduction method should be applied.

    Returns:
        (np.ndarray): mse between pred and target

    """
    dist = weights * ((pred - target) ** 2).mean(axis=-1)
    return dist.mean() if reduce else dist


def mae(pred, target, weights=1.0, reduce=True):
    """
    Computes the MAE between pred and target.

    Args:
        pred(np.ndarray): array of predicted data.
        target(np.ndarray): array of target data.
        weights(list[float], float): weights for each value.
        reduce(bool): if reduction method should be applied.

    Returns:
        (np.ndarray): mae between pred and target

    """
    dist = weights * np.abs(pred - target).mean(axis=-1)
    return dist.mean() if reduce else dist


def l2norm(pred, target, weights=1.0, reduce=True):
    """
    Computes the L2Norm between pred and target.

    Args:
        pred(np.ndarray): array of predicted data.
        target(np.ndarray): array of target data.
        weights(list[float], float): weights for each value.
        reduce(bool): if reduction method should be applied.

    Returns:
        (np.ndarray): mae between pred and target

    """
    dist = weights * np.linalg.norm(pred - target, axis=-1)
    return dist.mean() if reduce else dist


def qad(pred, target, weights=1.0, reduce=True):
    """
    Computes the QAD(quaternion angle distance) between pred and target.

    Args:
        pred(np.ndarray): array of predicted data.
        target(np.ndarray): array of target data.
        weights(list[float], float): weights for each value.
        reduce(bool): if reduction method should be applied.

    Returns:
        (np.ndarray): qad between pred and target

    """
    dist = weights * _qad(pred, target, keepdims=False)
    return dist.mean() if reduce else dist


def pcp(pred, target, dist_func, threshold, reduce=True):
    """
    Computes the percentage of correct predictions (PCP) between pred
    and target.

    Args:
        pred(np.ndarray): array of predicted data.
        target(np.ndarray): array of target data.
        dist_func(function): function to use to compute distance.
        threshold(float): threshold to consider the detection correct.
        reduce(bool): if reduction method should be applied.

    Returns:
        (np.ndarray): pcp between pred and target

    """
    dists = dist_func(pred, target)
    pck = dists < threshold
    return pck.mean() if reduce else pck


def auc(pred, target, dist_func, pck_range, n_values=40):
    """
    Computes percentage of correct (metric) for multiple threshold
    values.

    Args:
        pred(np.ndarray): array of predicted data.
        target(np.ndarray): array of target data.
        dist_func(function): function to use to compute distance.
        pck_range(list[float]): range of data to evaluate.
        n_values(int): number of threshold values to evaluate.

    Returns:
        (np.ndarray): auc between pred and target

    """
    thresholds = np.linspace(1e-12 + pck_range[0], pck_range[1], n_values, dtype=np.float32)
    return [pcp(pred, target, dist_func, threshold, reduce=True)
            for i, threshold in enumerate(thresholds)]


###################################################################################################
###################################################################################################

class Metric(object):
    """
    General interface object to handle data computed using some metric.

    Args:
        name(str): name of the metric.
        mtype(str): type of metric. Available types are ("raw",
            "time", "dist", "pcp", "auc").
        units(str): unit quantity associated with the metric.
        description(str): description of the metric.
        mrange(tuple[float]): metric upper and lower range or None if
            unbounded.
        colnames(list[str]): name of each if metric is computed for
            multiple entries.
        colweights(list[float]): weighting to apply for each entry when
            averaging values along all entries.
        err_thresh(float): desired error threshold for this metric.

    """

    def __init__(self, name, mtype, units="units", description=None, mrange=None,
                 colnames=None, colweights=None, err_thresh=None):
        self.name = name
        self.mtype = mtype
        self.units = units
        self.description = description if description is not None else name
        self.mrange = mrange
        self.colnames = colnames
        self.colweights = colweights if colweights is not None else 1.0
        self.err_thresh = err_thresh
        assert mtype in ("raw", "time", "dist", "pcp", "auc")
        assert (mrange is None or len(mrange) == 2)
        # store data
        self._data = []

    def __len__(self):
        return len(self._data)

    def reset(self):
        """
        Resets computed metrics buffer.
        """
        self._data = []

    @property
    def data(self):
        """
        Returns computed metric data as a np.array.
        """
        return np.array(self._data)

    def update(self, *args):
        """
        Computes and stores metric data from new sample of data. This
        function should be implemented by child classes.
        """
        raise NotImplementedError

###################################################################################################
# raw data
class RawData(Metric):
    """
    Stores raw data, but uses same interface as the Metric class in
    order to access MetricsAnalyser functionality (namely exporting
    data + time plots).
    """
    def __init__(self, name, **kwargs):
        super(RawData, self).__init__(name=name, mtype="raw", **kwargs)

    def update(self, raw_data):
        self._data.append(raw_data)


###################################################################################################
class TimeMetric(Metric):
    """
    Stores metrics related with time (e.g. inference time, etc...)
    """
    def __init__(self, name, units="s", **kwargs):
        super(TimeMetric, self).__init__(name=name, mtype="time", units=units, **kwargs)

    def update(self, t):
        self._data.append(t)


###################################################################################################
# distance metrics
class DistMetric(Metric):
    """
    Stores a distance metric data between predicted and target inputs.
    """
    def __init__(self, name, dist, **kwargs):
        assert dist in ["rmse", "mse", "mae", "l2norm", "qad"]
        self._dist_func = dict(rmse=rmse, mse=mse, mae=mae, l2norm=l2norm, qad=qad)[dist]
        super(DistMetric, self).__init__(name=name, mtype="dist", **kwargs)

    def update(self, pred, target):
        self._data.append(self._dist_func(pred, target, weights=self.colweights, reduce=False))


class RelDistMetric(Metric):
    """
    Stores a distance metric data between predicted and target inputs
    relative to one of the inputs (ex. root_joint).
    """
    def __init__(self, name, dist, rel_idx=0, **kwargs):
        assert dist in ["rmse", "mse", "mae", "l2norm"]
        self._dist_func = dict(rmse=rmse, mse=mse, mae=mae, l2norm=l2norm)[dist]
        self.rel_idx = rel_idx
        super(RelDistMetric, self).__init__(name=name, mtype="dist", **kwargs)

    def update(self, pred, target):
        pred = pred - pred[self.rel_idx]
        target = target - target[self.rel_idx]
        self._data.append(self._dist_func(pred, target, weights=self.colweights, reduce=False))


class ProcrustesDistMetric(Metric):
    """
    Stores distance metric data between predicted and target inputs
    after Procrustes alignment.
    """
    def __init__(self, name, dist, **kwargs):
        assert dist in ["rmse", "mse", "mae", "l2norm"]
        self._dist_func = dict(rmse=rmse, mse=mse, mae=mae, l2norm=l2norm)[dist]
        super(ProcrustesDistMetric, self).__init__(name=name, mtype="dist", **kwargs)

    def update(self, pred, target):
        pred = apply_procrustes_alignment(pred, target)
        self._data.append(self._dist_func(pred, target, weights=self.colweights, reduce=False))


class QADistMetric(Metric):
    """
    Stores quaternion angle distance (QAD) error between predicted
    and target inputs.
    """
    def __init__(self, name, show_degrees=False, **kwargs):
        units = "deg" if show_degrees else "rads"
        mrange = (0., np.pi)
        if show_degrees and "err_thresh" in kwargs:
            kwargs["err_thresh"] = np.rad2deg(kwargs["err_thresh"])
            mrange = (0., 180.)

        self._show_degrees = show_degrees
        super(QADistMetric, self).__init__(name=name, mtype="dist", units=units,
                                           mrange=mrange, **kwargs)

    def update(self, pred, target):
        assert (np.all(((pred >= -1.0) & (pred <= 1.0)))
                and np.all(((target >= -1.0) & (target <= 1.0)))), \
            "QADistMetric inputs need to be normalized quaternions!"
        if self._show_degrees:
            self._data.append(np.rad2deg(qad(pred, target, reduce=False)))
        else:
            self._data.append(qad(pred, target, reduce=False))


class RelQADistMetric(Metric):
    """
    Stores quaternion angle distance (QAD) error between predicted
    and target inputs, with normalized orientations relative to one of
    the inputs (ex. root_joint).
    """
    def __init__(self, name, rel_idx=0, show_degrees=False, **kwargs):
        units = "deg" if show_degrees else "rads"
        mrange = (0., np.pi)
        if show_degrees and "err_thresh" in kwargs:
            kwargs["err_thresh"] = np.rad2deg(kwargs["err_thresh"])
            mrange = (0., 180.)

        self.rel_idx = rel_idx
        self._show_degrees = show_degrees
        super(RelQADistMetric, self).__init__(name=name, mtype="dist", units=units,
                                              mrange=mrange, **kwargs)

    def update(self, pred, target):
        assert (np.all(((pred >= -1.0) & (pred <= 1.0)))
                and np.all(((target >= -1.0) & (target <= 1.0)))), \
            "QADistMetric inputs need to be normalized quaternions!"

        pred = relative_orientations(pred, ref_quat=pred[..., self.rel_idx, :])
        target = relative_orientations(target, ref_quat=target[..., self.rel_idx, :])
        if self._show_degrees:
            self._data.append(np.rad2deg(qad(pred, target, reduce=False)))
        else:
            self._data.append(qad(pred, target, reduce=False))


###################################################################################################
# percentage of correct predictions metrics
class PCPMetric(Metric):
    """
    Percentage of Correct Predictions (PCP). Stores percentage of
    samples error, between predicted and target inputs, bellow the
    defined threshold.
    """
    def __init__(self, name, dist, threshold, units="%", **kwargs):
        assert dist in ["rmse", "mse", "mae", "l2norm", "qad"]
        self._dist_func = dict(rmse=rmse, mse=mse, mae=mae, l2norm=l2norm, qad=qad)[dist]
        super(PCPMetric, self).__init__(name=name, mtype="pcp", mrange=(0.0, 100.0),
                                        units=units, err_thresh=threshold, **kwargs)

    def update(self, pred, target):
        self._data.append(pcp(pred, target, dist_func=self._dist_func,
                              threshold=self.err_thresh, reduce=False))


class AUCMetric(Metric):
    """
    Stores the percentage of correct predictions for multiple
    threshold points.
    """
    def __init__(self, name, dist, pcp_thresh_range, n_thresholds=40, units="%", **kwargs):
        super(AUCMetric, self).__init__(name=name, mtype="auc", mrange=(0.0, 100.0),
                                        units=units, colnames=None, **kwargs)
        assert dist in ["rmse", "mse", "mae", "l2norm", "qad"]
        assert len(pcp_thresh_range) == 2
        self.pck_thresh_range = pcp_thresh_range
        self.n_thresholds = n_thresholds
        self._dist_func = dict(rmse=rmse, mse=mse, mae=mae, l2norm=l2norm, qad=qad)[dist]

    def update(self, pred, target):
        self._data.append(auc(pred, target, dist_func=self._dist_func,
                              pck_range=self.pck_thresh_range,
                              n_values=self.n_thresholds))

###################################################################################################
###################################################################################################


class MetricsAnalyser(object):
    """
    Computes, stores and analyses relevant metrics for the evaluation
    of pose estimation algorithms.

    Args:
        metrics(dict[str, Metric]): dictionary with metrics to compute.
        exp_name(str): name of the experiment. Useful for logging.

    TODO:
        - supports options to plot/export raw data
    """

    def __init__(self, metrics, exp_name="MetricsAnalyser"):
        self.exp_name = exp_name
        self._metrics = metrics
        self._n = 0

    def reset(self):
        self._n = 0
        self._time = []
        for m in self._metrics.values():
            m.reset()

    def update(self, data_dict):
        """
        Updates metrics with new samples of data.

        Args:
            data_dict (dict[str, any]): dictionary containing data for
                each of the initialized metrics. Keys need be the same
                as the names for each metric given on init, values are
                data to be computed.

        """
        # update number of samples + inference time
        self._n += 1
        # update metrics
        for k, v in data_dict.items():
            self._metrics[k].update(*v)

    def get_metrics(self):
        """
        Get values computed for each metric.

        Returns:
            (dict[str, np.ndarray]): dict with computed metric
                values for each metric.

        """
        d = dict()
        for m in self._metrics.values():
            d[m.name] = m.data
        return d

    def get_avg_metrics(self):
        """
        Get average values (and confidence interval) obtained for
        each of the computed metrics.

        Returns:
            (dict[str, list[float, float]]): dict with average (and ci)
                values for each metric.

        """
        d = dict()
        for m in self._metrics.values():
            vals = m.data
            mean = np.nanmean(vals)
            ci = np.nanstd(vals)  # np.mean((st.sem(vals) * st.t.ppf((1 + 0.95) / 2, len(vals) - 1)))
            d[m.name] = [mean, ci]
        return d

    def print_avg_metrics(self, save_path=None, print_metrics=True,
                          extra_info=""):
        """
        Prints averages of all the computed metrics.

        Args:
            save_path(str, None): path to save the plots, or None if
                saving is not wanted.
            print_metrics(bool): if metrics should be printed.
            extra_info(str): extra information to add to the
                terminal log.

        """
        s = f"Experiment:  {self.exp_name}"
        s += f"\nAveraged metrics results for ({self._n}) samples:"
        for m in self._metrics.values():
            vals = m.data
            mean = np.round(np.nanmean(vals), 3)
            ci = np.round(np.nanstd(vals), 3)  # np.mean((st.sem(vals) * st.t.ppf((1 + 0.95) / 2, len(vals) - 1)))
            s += f"\n  {(m.name + ':').ljust(25)} {str(mean)} ± {str(ci)} {m.units}"

        if extra_info:
            s += "\n\nExtraInfo:\n" + extra_info

        if print_metrics:
            print("##########################################")
            print(s)
            print("##########################################")

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w") as f:
                f.write(s)
        return s

    def plot_over_time(self, save_path=None, show_plots=True, remove_outliers=None):
        """
        Plots the metrics over time.

        Args:
            save_path(str, None): path to save the plots, or None if
                saving is not wanted.
            show_plots (bool): if plots should be visualized.
            remove_outliers(bool): if extreme outliers should be
                removed (q > percentile ± 3.0 * IQR).

        """

        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set()

        valid_metrics = [m for m in self._metrics.values()
                         if (m.mtype in ("raw", "time", "dist")) and (len(m) > 0)]

        if not valid_metrics:
            print("No metrics available for temporal error plots!")
            return

        fig, ax = plt.subplots(len(valid_metrics), 1,
                               figsize=(35, 10 * len(valid_metrics)))
        fig.suptitle(self.exp_name, fontsize=54)
        fig.subplots_adjust(left=0.05, right=0.975, bottom=0.05, top=0.925, hspace=0.3)

        if not hasattr(ax, "__iter__"): ax = [ax]
        for i, m in enumerate(valid_metrics):
            # create pandas dataframe to handle data for seaborn
            data_table = pd.DataFrame(data=m.data, columns=m.colnames)

            # remove extreme outliers(3 * IQR)
            if remove_outliers:
                Q1 = data_table.quantile(0.25)
                Q3 = data_table.quantile(0.75)
                IQR = Q3 - Q1
                data_table = data_table[~((data_table < (Q1 - 3.0 * IQR))
                                          | (data_table > (Q3 + 3.0 * IQR))
                                          ).any(axis=1)]

            if data_table.shape[1] > 1: # if multiple columns of data
                # plot metric for each column over time
                sns.lineplot(data=data_table, ax=ax[i], dashes=False,
                             palette=sns.color_palette("nipy_spectral", data_table.shape[1]))

                # plot mean metric over time
                sns.lineplot(data=data_table.mean(axis=1), ax=ax[i],
                             color="dimgray", linewidth=6, dashes=False)
            else:
                sns.lineplot(data=data_table, ax=ax[i])

            # plots metadata
            ax[i].set_title(str(m.name), fontsize=44)
            ax[i].set_xlabel("Step", fontsize=30)
            ax[i].set_ylabel(m.units, fontsize=30)
            ax[i].tick_params(axis="both", which="major", labelsize=22)
            ax[i].legend(prop={"size": 20})
            if m.mrange is not None:
                ax[i].set_ylim(m.mrange[0], m.mrange[1])

            # add error threshold line
            if m.err_thresh is not None:
                ax[i].axhline(y=m.err_thresh, color="k", linestyle="dashed", lw=3)

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path)
        if show_plots:
            plt.show(block=False)

    def plot_boxplot(self, save_path=None, show_plots=True, remove_outliers=True):
        """
        Plots boxplots to analyse error for each column of data.

        Args:
            save_path(str, None): path to save the plots, or None if
                saving is not wanted.
            show_plots (bool): if plots should be visualized.
            remove_outliers(bool): if extreme outliers should be
                removed (q > percentile ± 3.0 * IQR).

        """
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set()

        # select valid metrics to show in boxplots (rmse)
        valid_metrics = [m for m in self._metrics.values()
                         if (m.mtype == "dist") and (len(m) > 0)]

        if not valid_metrics:
            print("No rmse metrics available for boxplots!")
            return

        fig, ax = plt.subplots(len(valid_metrics), 1,
                               figsize=(23, 15 * len(valid_metrics)))
        fig.suptitle(self.exp_name, fontsize=54)
        fig.subplots_adjust(left=0.075, right=0.95, bottom=0.15, top=0.92, hspace=0.35)

        if not hasattr(ax, "__iter__"): ax = [ax]   # if only 1 plot
        for i, m in enumerate(valid_metrics):
            # create pandas dataframe to handle data for seaborn
            data_table = pd.DataFrame(data=m.data,
                                      columns=m.colnames)

            # remove extreme outliers(3.0 * IQR)
            if remove_outliers:
                Q1 = data_table.quantile(0.25)
                Q3 = data_table.quantile(0.75)
                IQR = Q3 - Q1
                data_table = data_table[~((data_table < (Q1 - 3.0 * IQR))
                                          | (data_table > (Q3 + 3.0 * IQR))
                                          ).any(axis=1)]

            # add boxplots
            sns.boxplot(data=data_table, ax=ax[i],
                        palette=sns.color_palette("nipy_spectral", data_table.shape[1]))

            # plots metadata
            ax[i].set_title(str(m.name), fontsize=46)
            ax[i].set_xlabel(m.description, fontsize=30)
            ax[i].set_ylabel(m.units, fontsize=30)
            ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=75)
            ax[i].tick_params(axis="both", which="major", labelsize=22)
            if m.mrange is not None:
                ax[i].set_ylim(m.mrange[0], m.mrange[1])

            # add error threshold line
            if m.err_thresh is not None:
                ax[i].axhline(y=m.err_thresh, color="k", linestyle="dashed", lw=3)

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path)
        if show_plots:
            plt.show(block=False)

    def plot_auc(self, save_path=None, show_plots=True, remove_outliers=True):
        """
        Plots auc plots to analyse PCK across different thresholds.

        Args:
            save_path(str, None): path to save the plots, or None if
                saving is not wanted.
            show_plots (bool): if plots should be visualized.
            remove_outliers(bool): if extreme outliers should be
                removed (q > percentile ± 3.0 * IQR).

        """
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set()

        valid_metrics = [m for m in self._metrics.values()
                         if (m.mtype == "auc") and (len(m) > 0)]

        if not valid_metrics:
            print("No AUC metrics available for AUC plots!")
            return

        fig, ax = plt.subplots(len(valid_metrics), 1,
                               figsize=(20, 10 * len(valid_metrics)))
        fig.suptitle(self.exp_name, fontsize=44)
        fig.subplots_adjust(left=0.05, right=0.975, bottom=0.075, top=0.92, hspace=0.3)

        if not hasattr(ax, "__iter__"): ax = [ax]
        for i, m in enumerate(valid_metrics):
            # create pandas dataframe to handle data for seaborn
            m_vals = m.data * 100.0
            thresholds = np.linspace(m.pck_thresh_range[0], m.pck_thresh_range[1],
                                     m_vals.shape[1], dtype=np.float32)
            data_table = pd.DataFrame(data=np.mean(m_vals, axis=0),
                                      index=list(map(str, np.round(thresholds, 4))))

            # remove extreme outliers(3.0 * IQR)
            if remove_outliers:
                Q1 = data_table.quantile(0.25)
                Q3 = data_table.quantile(0.75)
                IQR = Q3 - Q1
                data_table = data_table[~((data_table < (Q1 - 3.0 * IQR))
                                          | (data_table > (Q3 + 3.0 * IQR))
                                          ).any(axis=1)]

            sns.lineplot(data=data_table, ax=ax[i], legend=None, markers=True)

            # plots metadata
            ax[i].set_title(str(m.name).upper(), fontsize=28)
            ax[i].set_xlabel(f"Error Thresholds ({m.description})", fontsize=22)
            ax[i].set_ylabel(m.units, fontsize=22)
            ax[i].set_xticklabels(list(map(str, np.round(thresholds, 3))), rotation=45)
            if m.mrange is not None:
                ax[i].set_ylim(m.mrange[0], m.mrange[1])

            # add error threshold line
            if m.err_thresh is not None:
                ax[i].axvline(x=(m.err_thresh * (len(ax[i].get_xticklabels())
                                                 / (m.pck_thresh_range[1]))) - 0.5,
                              color="k", linestyle="dashed", lw=2)

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path)
        if show_plots:
            plt.show(block=False)

    def export_data(self, save_path):
        """
        Exports all computed metrics to csv and zips them.

        Args:
            save_path(str): file path to save the plots.

        """
        import pandas as pd

        if not self._metrics:
            print("No metrics available to export!")
            return

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_path_name = os.path.dirname(save_path)
        save_file_name = os.path.basename(save_path)

        # export computed metrics to csv
        file_names = []
        for m in self._metrics.values():
            # create dataframe
            m_vals = m.data
            if len(m_vals) > 0:
                data = pd.DataFrame(data=m_vals, columns=m.colnames)

                # export metrics
                fname = save_path_name + f"/{m.name}.csv"
                file_names.append(fname)
                data.to_csv(fname, float_format="%.5f")

        # zip all data
        with ZipFile(save_path_name + "/" + save_file_name.replace(".zip", "") + ".zip", "w") as z:
            for m in file_names:
                z.write(m, os.path.basename(m))
                os.remove(m)

    def log_all(self, save_path=None, show_plots=True, print_metrics=True,
                remove_outliers=True, extra_info="", export=False):
        """
        Utility function to log all options available.

        Args:
            save_path(str, None): path to save the plots, or None if
                saving is not wanted.
            show_plots (bool): if plots should be visualized.
            print_metrics(bool): if metrics should be printed.
            remove_outliers(bool): if extreme outliers should be
                removed (q > percentile ± 3.0 * IQR).
            extra_info(str): extra information to add to the terminal
                output.
            export(bool): if computed metrics should be exported to csv.

        """
        if export:
            self.export_data(save_path + "/metric_data.zip" if save_path is not None else "./")

        self.print_avg_metrics(save_path + "/metrics.txt" if save_path is not None else None,
                               print_metrics=print_metrics, extra_info=extra_info)

        self.plot_boxplot(save_path + "/boxplots.png" if save_path is not None else None,
                          show_plots=show_plots, remove_outliers=remove_outliers)

        self.plot_auc(save_path + "/auc_plots.png" if save_path is not None else None,
                      show_plots=show_plots, remove_outliers=remove_outliers)

        self.plot_over_time(save_path + "/error_plot.png" if save_path is not None else None,
                            show_plots=show_plots, remove_outliers=remove_outliers)
