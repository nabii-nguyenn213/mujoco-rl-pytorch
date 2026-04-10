from pathlib import Path
from math import ceil
import pandas as pd
import matplotlib.pyplot as plt
import argparse

class Plotter:
    def __init__(
        self,
        csv_dir,
        save_dir=None,
        dpi=300,
        default_figsize=(4.8, 3.2),
        style="paper",
        save_formats=("png", "pdf"),
        # save_formats=("pdf"),
    ):
        self.csv_dir = Path(csv_dir)
        self.save_dir = Path(save_dir) if save_dir is not None else Path("results/plots/figures")
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.dpi = dpi
        self.default_figsize = default_figsize
        self.style = style
        self.save_formats = save_formats

        self.metric_labels = {
            "episode_return": "Episode Return",
            "avg_return": "Average Return",
            "best_return": "Best Return",
            "episode_length": "Episode Length",
            "length": "Episode Length",
            "actor_loss": "Actor Loss",
            "critic_loss": "Critic Loss",
            "entropy": "Entropy",
            "alpha": "Temperature",
            "q1": "Q1",
            "q2": "Q2",
            "log_pi": "Log π",
        }

        self.metric_groups = {
            "returns": ["episode_return", "avg_return", "best_return"],
            "losses": ["actor_loss", "critic_loss"],
            "lengths": ["episode_length", "length"],
        }

        self._configure_matplotlib()

    def _configure_matplotlib(self):
        plt.rcParams.update({
            "font.family": "serif",
            # "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif", "STIXGeneral"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.usetex": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "lines.linewidth": 1.8,
            "figure.figsize": self.default_figsize,
        })

    def _extract_metric_name(self, csv_path):
        csv_path = Path(csv_path)
        stem = csv_path.stem
        if "tag-" in stem:
            return stem.split("tag-")[-1]
        return stem

    def _prettify_metric_name(self, metric):
        return self.metric_labels.get(metric, metric.replace("_", " ").title())

    def _load_csv(self, csv_path):
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        required_cols = {"Step", "Value"}
        if not required_cols.issubset(df.columns):
            raise ValueError(
                f"{csv_path} must contain columns {required_cols}, got {list(df.columns)}"
            )

        df = df.sort_values("Step").reset_index(drop=True)
        return df

    def _smooth(self, y, window):
        if window <= 1:
            return pd.Series(y)
        return pd.Series(y).rolling(window=window, min_periods=1).mean()

    def _resolve_csv_path(self, csv_file):
        csv_path = Path(csv_file)
        if not csv_path.is_absolute() and csv_path.parent == Path("."):
            csv_path = self.csv_dir / csv_path
        return csv_path

    def _get_x_scale(self, step_series):
        max_step = float(pd.Series(step_series).max())
        if max_step >= 1e6:
            return 1e6, r"Step ($\times 10^6$)"
        elif max_step >= 1e5:
            return 1e5, r"Step ($\times 10^5$)"
        elif max_step >= 1e4:
            return 1e4, r"Step ($\times 10^4$)"
        elif max_step >= 1e3:
            return 1e3, r"Step ($\times 10^3$)"
        else:
            return 1.0, "Step"

    def _infer_default_smooth_window(self, metric):
        metric = metric.lower()
        if "loss" in metric:
            return 50
        if "return" in metric:
            return 20
        if "length" in metric:
            return 20
        return 10

    def _save_figure(self, fig, stem_name):
        saved_paths = []
        for ext in self.save_formats:
            out_path = self.save_dir / f"{stem_name}.{ext}"
            if ext.lower() == "png":
                fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
            else:
                fig.savefig(out_path, bbox_inches="tight")
            saved_paths.append(out_path)
        return saved_paths

    def available_metrics(self):
        csv_files = sorted(self.csv_dir.glob("*.csv"))
        return sorted([self._extract_metric_name(f) for f in csv_files])

    def get_metric_file_map(self):
        return {self._extract_metric_name(f): f for f in self.csv_dir.glob("*.csv")}

    def plot_metric(
        self,
        csv_file,
        smooth_window=None,
        show_raw=True,
        raw_alpha=0.25,
        figsize=None,
        title=None,
        xlabel=None,
        ylabel=None,
        save_name=None,
        close=True,
    ):
        csv_path = self._resolve_csv_path(csv_file)
        df = self._load_csv(csv_path)
        metric = self._extract_metric_name(csv_path)
        if smooth_window is None:
            smooth_window = self._infer_default_smooth_window(metric)
        scale, auto_xlabel = self._get_x_scale(df["Step"])
        x = df["Step"] / scale
        y = df["Value"]
        y_smooth = self._smooth(y, smooth_window)
        fig, ax = plt.subplots(figsize=figsize or self.default_figsize)
        if show_raw and smooth_window > 1:
            ax.plot(x, y, alpha=raw_alpha, linewidth=1.0, label="Raw")
        smooth_label = "Value" if smooth_window <= 1 else f"Smoothed ({smooth_window})"
        ax.plot(x, y_smooth, label=smooth_label)
        ax.set_xlabel(xlabel or auto_xlabel)
        ax.set_ylabel(ylabel or self._prettify_metric_name(metric))
        ax.set_title(title or self._prettify_metric_name(metric))
        ax.legend(frameon=False)
        ax.margins(x=0.02)
        fig.tight_layout()
        stem_name = save_name or metric
        saved_paths = self._save_figure(fig, stem_name)
        for p in saved_paths:
            print(f"Saved: {p}")
        if close:
            plt.close(fig)
        return fig, ax

    def plot_selected(
        self,
        metrics,
        smooth_window=None,
        show_raw=True,
        raw_alpha=0.25,
        figsize=None,
    ):
        csv_map = self.get_metric_file_map()
        for metric in metrics:
            if metric not in csv_map:
                print(f"[Warning] Metric '{metric}' not found.")
                continue
            self.plot_metric(
                csv_file=csv_map[metric],
                smooth_window=smooth_window,
                show_raw=show_raw,
                raw_alpha=raw_alpha,
                figsize=figsize,
            )

    def plot_all(
        self,
        smooth_window=None,
        show_raw=True,
        raw_alpha=0.25,
        figsize=None,
    ):
        csv_files = sorted(self.csv_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.csv_dir}")
        for f in csv_files:
            try:
                self.plot_metric(
                    csv_file=f,
                    smooth_window=smooth_window,
                    show_raw=show_raw,
                    raw_alpha=raw_alpha,
                    figsize=figsize,
                )
            except Exception as e:
                print(f"Skipping {f.name}: {e}")

    def plot_dashboard(
        self,
        metrics=None,
        smooth_window=None,
        show_raw=False,
        raw_alpha=0.20,
        ncols=2,
        figsize_per_panel=(4.8, 3.0),
        save_name="dashboard",
        close=True,
    ):
        csv_map = self.get_metric_file_map()
        if metrics is None:
            preferred = [
                "episode_return",
                "avg_return",
                "best_return",
                "episode_length",
                "actor_loss",
                "critic_loss",
            ]
            metrics = [m for m in preferred if m in csv_map]
            if not metrics:
                metrics = list(csv_map.keys())
        valid_metrics = [m for m in metrics if m in csv_map]
        if not valid_metrics:
            raise ValueError("No valid metrics found for dashboard.")
        n = len(valid_metrics)
        nrows = ceil(n / ncols)
        fig_width = figsize_per_panel[0] * ncols
        fig_height = figsize_per_panel[1] * nrows
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height))
        if nrows == 1 and ncols == 1:
            axes = [axes]
        else:
            axes = list(pd.Series(axes.flatten()))
        for ax, metric in zip(axes, valid_metrics):
            csv_path = csv_map[metric]
            df = self._load_csv(csv_path)
            local_smooth = smooth_window
            if local_smooth is None:
                local_smooth = self._infer_default_smooth_window(metric)
            scale, auto_xlabel = self._get_x_scale(df["Step"])
            x = df["Step"] / scale
            y = df["Value"]
            y_smooth = self._smooth(y, local_smooth)
            if show_raw and local_smooth > 1:
                ax.plot(x, y, alpha=raw_alpha, linewidth=1.0, label="Raw")
            label = "Value" if local_smooth <= 1 else f"Smoothed ({local_smooth})"
            ax.plot(x, y_smooth, label=label)
            ax.set_title(self._prettify_metric_name(metric))
            ax.set_xlabel(auto_xlabel)
            ax.set_ylabel(self._prettify_metric_name(metric))
            ax.margins(x=0.02)
            if show_raw or local_smooth > 1:
                ax.legend(frameon=False)
        for ax in axes[len(valid_metrics):]:
            ax.axis("off")
        fig.tight_layout()
        saved_paths = self._save_figure(fig, save_name)
        for p in saved_paths:
            print(f"Saved: {p}")
        if close:
            plt.close(fig)
        return fig, axes

    def compare_runs(
        self,
        run_csv_files,
        labels=None,
        metric_name=None,
        smooth_window=None,
        figsize=None,
        xlabel=None,
        ylabel=None,
        title=None,
        save_name=None,
        close=True,
    ):
        if not run_csv_files:
            raise ValueError("run_csv_files must not be empty.")
        resolved_paths = [self._resolve_csv_path(f) for f in run_csv_files]
        if labels is None:
            labels = [Path(f).stem for f in resolved_paths]
        if len(labels) != len(resolved_paths):
            raise ValueError("labels and run_csv_files must have the same length.")
        if metric_name is None:
            metric_name = self._extract_metric_name(resolved_paths[0])
        if smooth_window is None:
            smooth_window = self._infer_default_smooth_window(metric_name)
        first_df = self._load_csv(resolved_paths[0])
        scale, auto_xlabel = self._get_x_scale(first_df["Step"])
        fig, ax = plt.subplots(figsize=figsize or self.default_figsize)
        for path, label in zip(resolved_paths, labels):
            df = self._load_csv(path)
            x = df["Step"] / scale
            y = self._smooth(df["Value"], smooth_window)
            ax.plot(x, y, label=label)
        ax.set_xlabel(xlabel or auto_xlabel)
        ax.set_ylabel(ylabel or self._prettify_metric_name(metric_name))
        ax.set_title(title or f"{self._prettify_metric_name(metric_name)} Comparison")
        ax.legend(frameon=False)
        ax.margins(x=0.02)
        fig.tight_layout()
        stem_name = save_name or f"compare_{metric_name}"
        saved_paths = self._save_figure(fig, stem_name)
        for p in saved_paths:
            print(f"Saved: {p}")
        if close:
            plt.close(fig)
        return fig, ax

    def plot_episode_return_all(self, smooth_window=20, legends=None):
        csv_files = sorted(self.csv_dir.glob("*tag-episode_return*.csv"))

        fig, ax = plt.subplots(figsize=self.default_figsize)

        for i, csv_file in enumerate(csv_files):
            df = self._load_csv(csv_file)

            scale, auto_xlabel = self._get_x_scale(df["Step"])
            x = df["Step"] / scale
            y = self._smooth(df["Value"], smooth_window)

            # choose legend name
            if legends is not None:
                label = legends[i]
            else:
                label = csv_file.stem

            ax.plot(x, y, label=label)

        ax.set_xlabel(auto_xlabel)
        ax.set_ylabel("Episode Return")
        ax.set_title("Episode Return Comparison")
        ax.legend(frameon=False)

        fig.tight_layout()
        self._save_figure(fig, "episode_return_compare")
        plt.close(fig)

    def compare_multiple_runs_from_dirs(
        self,
        run_dirs,
        metric,
        labels=None,
        smooth_window=None,
        figsize=None,
        save_name=None,
        close=True,
    ):
        csv_paths = []
        for run_dir in run_dirs:
            run_dir = Path(run_dir)
            candidates = list(run_dir.glob(f"*tag-{metric}.csv"))
            if not candidates:
                raise FileNotFoundError(f"Cannot find metric '{metric}' in {run_dir}")
            csv_paths.append(candidates[0])
        return self.compare_runs(
            run_csv_files=csv_paths,
            labels=labels,
            metric_name=metric,
            smooth_window=smooth_window,
            figsize=figsize,
            save_name=save_name or f"compare_{metric}",
            close=close,
        )

    def plot_group(
        self,
        group_name,
        smooth_window=None,
        show_raw=False,
        raw_alpha=0.20,
        ncols=2,
        figsize_per_panel=(4.8, 3.0),
        close=True,
    ):
        if group_name not in self.metric_groups:
            raise ValueError(
                f"Unknown group '{group_name}'. Available groups: {list(self.metric_groups.keys())}"
            )
        return self.plot_dashboard(
            metrics=self.metric_groups[group_name],
            smooth_window=smooth_window,
            show_raw=show_raw,
            raw_alpha=raw_alpha,
            ncols=ncols,
            figsize_per_panel=figsize_per_panel,
            save_name=group_name,
            close=close,
        )

def get_parser(): 
    args = argparse.ArgumentParser()
    args.add_argument("--csv", type=str, default="results/plots/Ant-v5")
    args.add_argument("--save", type=str, default="assets/plots/SAC/")
    return args.parse_args()

if __name__ == "__main__":
    args = get_parser()
    plotter = Plotter(
        csv_dir=args.csv, 
        save_dir=args.save, 
        dpi=300,
    )

    # plotter.plot_selected([
    #     "episode_return",
    #     "avg_return",
    #     "best_return",
    #     "episode_length",
    #     "actor_loss",
    #     "critic_loss",
    # ])
    #
    # plotter.plot_dashboard()

    # plotter.plot_group("returns")

    # plotter.plot_metric("run-.-tag-episode_return_rank_0.csv", save_name="Hopper_SAC")
    # plotter.plot_metric("run-.-tag-episode_return_rank_1.csv", save_name="Hopper_SAC")
    # plotter.plot_metric("run-.-tag-episode_return_rank_2.csv", save_name="Hopper_SAC")
    # plotter.plot_metric("run-.-tag-episode_return_rank_3.csv", save_name="Hopper_SAC")

    plotter.plot_episode_return_all(smooth_window=20, legends=["rank_0",
                                                              "rank_1", 
                                                              "rank_2", 
                                                              "rank_3" 
                                                              ]) 
