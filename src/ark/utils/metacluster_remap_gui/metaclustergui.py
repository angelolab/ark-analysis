import warnings

import ipywidgets as widgets
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from IPython.display import display
from scipy.cluster.hierarchy import dendrogram
from scipy.stats import zscore

from .colormap_helper import distinct_cmap
from .metaclusterdata import MetaClusterData
from .throttle import throttle
from .zscore_norm import ZScoreNormalize

# Third party ipympl causing this in it's backend_agg startup code
warnings.filterwarnings("ignore", message="nbagg.transparent is deprecated")

DEBUG_VIEW = widgets.Output(layout={'border': '1px solid black'})
DEFAULT_HEATMAP = sns.diverging_palette(240, 10, n=3, as_cmap=True)


class MetaClusterGui():
    """Coordinate and present the metacluster Graphical User Interface

    Attributes:
        mcd (MetaClusterData)):
            State of the actual clusters at any point in time
        selected_clusters (set[int]):
            Currently selected clusters

    Args:
        data (MetaClusterData)):
            An initialized MetaClusterData instance
        heatmapcolors (matplotlib.colors.ColorMap)):
            If you wish to change the default heatmap colors
        width (float):
            Adjust the actual width to accomodate monitor size, resolution, zoom, etc
        debug (bool):
            Enable debug mode for the GUI. This enables a special logging window where
            output from callbacks can be printed.
        enable_throttle (bool):
            Control whether or not to throttle GUI callbacks. Disabling might be
            helpful for debugging certain race conditions.

    """
    def __init__(self, metaclusterdata, heatmapcolors=DEFAULT_HEATMAP,
                 width=17.0, debug=False, enable_throttle=True):
        self.width: float = width
        self.heatmapcolors: str = heatmapcolors
        self.mcd: MetaClusterData = metaclusterdata
        self.selected_clusters = set()

        self.make_widgets()
        self.make_gui()

        self._heatmaps_stale = True
        self.update_gui()

        display(self.gui)

        if debug:
            self.enable_debug_mode()

        if enable_throttle:
            throttler = throttle(.3)
            self.update_gui = throttler(self.update_gui)

    def make_gui(self):
        """Create and configure all of the plots which make up the GUI

        Below is a map of the physical subplot layout of
        the Axes within the Figure.

        The abbreviation is used both for the axes
            e.g. self.ax_c

        as well as the plotted items.
            e.g. self.im_c, self.rects_cp

        Map of matplotlib Figure::

            |   |    Cluster     | Meta |
            ----------------------------
            |   |    cp          |  cb  | counts of pixels, color bar
            | cd|    c           |  m   | heatmap itself
            |   |    cs          |  ms  | selection markers
            |   |    cl          |  ml  | metacluster color labels

        """
        width_ratios = [
            int(self.mcd.cluster_count / 7),
            self.mcd.cluster_count,
            self.mcd.metacluster_count * 2,
        ]
        marker_ratio = max(self.mcd.marker_count / 20, 1)
        height_ratios = [
            6 * marker_ratio, self.mcd.marker_count * marker_ratio, marker_ratio, marker_ratio
        ]

        subplots = plt.subplots(
            4, 3,
            gridspec_kw={
                'width_ratios': width_ratios,
                'height_ratios': height_ratios},
            figsize=(self.width, 6 * marker_ratio),
        )
        with self.plot_output:
            plt.show()

        (self.fig, (
            (self.ax_01, self.ax_cp, self.ax_cb),
            (self.ax_cd, self.ax_c, self.ax_m),
            (self.ax_02, self.ax_cs, self.ax_ms),
            (self.ax_03, self.ax_cl, self.ax_ml))) = subplots

        self.fig.canvas.toolbar_visible = False
        self.fig.canvas.header_visible = False
        self.fig.canvas.footer_visible = False

        self.fig.canvas.mpl_connect('pick_event', self.onpick)

        # heatmaps
        self.normalizer = ZScoreNormalize(-1, 0, 1)

        def _heatmap(ax, column_count):
            data = np.zeros((self.mcd.marker_count, column_count))
            return ax.imshow(
                data,
                norm=self.normalizer,
                cmap=self.heatmapcolors,
                aspect='auto',
                picker=True,
            )
        self.im_c = _heatmap(self.ax_c, self.mcd.cluster_count)
        self.im_m = _heatmap(self.ax_m, self.mcd.metacluster_count)

        self.ax_c.yaxis.set_tick_params(which='major', labelleft=False)
        self.ax_c.set_yticks(np.arange(self.mcd.marker_count) + 0.5)
        self.ax_c.set_xticks(np.arange(self.mcd.cluster_count) + 0.5)
        self.ax_m.set_xticks(np.arange(self.mcd.metacluster_count) + 0.5)
        self.ax_c.xaxis.set_tick_params(which='both', bottom=False, labelbottom=False)
        self.ax_m.xaxis.set_tick_params(which='both', bottom=False, labelbottom=False)
        self.ax_m.yaxis.set_tick_params(which='both', left=False, labelleft=False)
        self.ax_m.yaxis.set_tick_params(which='both', right=True, labelright=True, labelsize=7)
        self.ax_m.set_yticks(np.arange(self.mcd.marker_count) + 0.5)

        # xaxis metacluster color labels
        self.ax_cl.xaxis.set_tick_params(which='both', bottom=False, labelbottom=False)
        self.ax_ml.xaxis.set_tick_params(which='both', bottom=True, labelbottom=True)
        self.ax_cl.yaxis.set_tick_params(which='both', left=False, labelleft=True)
        self.ax_cl.set_yticks([0.5])
        self.ax_cl.set_yticklabels(["Metacluster"])
        self.ax_ml.yaxis.set_tick_params(which='both', left=False, labelleft=False)

        def _color_labels(ax, column_count):
            data = np.zeros((1, column_count))
            return ax.imshow(data, aspect='auto', picker=True, vmin=1, vmax=self.mcd.cluster_count)

        self.im_cl = _color_labels(self.ax_cl, self.mcd.cluster_count)
        self.im_ml = _color_labels(self.ax_ml, self.mcd.metacluster_count)

        # xaxis cluster selection labels
        self.ax_cs.xaxis.set_tick_params(which='both', bottom=False, labelbottom=False)
        self.ax_cs.yaxis.set_tick_params(which='both', left=False, labelleft=True)
        self.ax_cs.set_yticks([0.5])
        self.ax_cs.set_yticklabels(["Selected"])

        self.im_cs = self.ax_cs.imshow(
            np.zeros((1, self.mcd.marker_count)),
            cmap='Blues',
            aspect='auto',
            picker=True,
            vmin=-0.3,
            vmax=1,
        )

        # xaxis pixelcount graphs
        self.ax_cp.xaxis.set_tick_params(which='both', bottom=False, labelbottom=False)
        self.ax_cp.yaxis.set_tick_params(which='both', left=False, labelleft=False)
        self.ax_cp.set_ylabel("Count (k)", rotation=90)
        self.ax_cp.set_xlim(0, self.mcd.cluster_count)
        self.rects_cp = self.ax_cp.bar(
            np.arange(self.mcd.cluster_count) + 0.5,
            np.zeros(self.mcd.cluster_count))
        self.labels_cp = []
        label_alignment_fudge = 0.08
        for x in np.arange(self.mcd.cluster_count) + 0.5 + label_alignment_fudge:
            label = self.ax_cp.text(
                x=x, y=0, s="-", va='bottom',
                ha='center', rotation=90, color='black', fontsize=8)
            self.labels_cp.append(label)

        # colorbar
        self.cb = plt.colorbar(self.im_c, ax=self.ax_cb, orientation='horizontal',
                               fraction=.75, shrink=.95, aspect=15)
        self.cb.ax.xaxis.set_tick_params(which='both', labelsize=7, labelrotation=90)

        # dendrogram
        self.ddg = dendrogram(
            self.mcd.linkage_matrix,
            ax=self.ax_cd,
            orientation='left',
            labels=self.mcd.fixed_width_marker_names,
            leaf_font_size=8,
            )
        self.mcd.set_marker_order(self.ddg['leaves'][::-1])
        self.ax_m.set_yticklabels(self.mcd.marker_names[::-1])

        self.ax_cd.figure.frameon = False
        self.ax_cd.spines["top"].set_visible(False)
        self.ax_cd.spines["left"].set_visible(False)
        self.ax_cd.spines["right"].set_visible(False)
        self.ax_cd.spines["bottom"].set_visible(False)
        self.ax_cd.xaxis.set_tick_params(which='both', bottom=False, labelbottom=False)
        self.ax_cd.yaxis.set_tick_params(which='both', pad=-2)
        self.ax_cd.tick_params(axis="y", direction="in")
        self.move_dendro_labels(self.ax_cd)

        self.ax_01.axis('off')
        self.ax_02.axis('off')
        self.ax_03.axis('off')
        self.ax_cb.axis('off')
        self.ax_ms.axis('off')

        # space for longer labels hack
        self.ax_ml.set_xticks([0.5])
        self.ax_ml.set_xticklabels(["SpaceHolder--"], rotation=90, fontsize=8)

        # Tighten layout based on display
        self.fig.tight_layout()
        plt.subplots_adjust(hspace=.0)  # make color labels touch heatmap
        plt.subplots_adjust(wspace=.02)

    def make_widgets(self):
        """Create the physical ipywidgets that display below the GUI plot."""

        # zscore adjuster
        self.zscore_clamp_slider = widgets.FloatSlider(
            value=3,
            min=1,
            max=10.0,
            step=0.5,
            description='Max Zscore:',
            disabled=False,
            continuous_update=True,
            orientation='horizontal',
            readout=True,
            readout_format='.1f',
            tooltip='Clamp/Clip zscore to a certain max value.',
        )
        self.zscore_clamp_slider.observe(self.update_zscore)

        # clear_selection button
        self.clear_selection_button = widgets.Button(
            description='Clear Selection',
            disabled=False,
            button_style='warning',
            tooltip='Clear currently selected clusters',
            icon='ban',
        )
        self.clear_selection_button.on_click(self.clear_selection)

        # new metacluster button
        self.new_metacluster_button = widgets.Button(
            description='New metacluster',
            disabled=False,
            button_style='success',
            tooltip='Create new metacluster from current selection',
            icon='plus',
        )
        self.new_metacluster_button.on_click(self.new_metacluster)

        # metacluster metadata
        self.current_metacluster = widgets.Dropdown(
            value=self.mcd.metaclusters.index[0],
            options=list(zip(self.mcd.metacluster_displaynames, self.mcd.metaclusters.index)),
            description='MetaCluster:',
        )
        self.current_metacluster.observe(
            self.update_current_metacluster_handler, type="change", names="value"
        )

        self.current_metacluster_displayname = widgets.Text(
            value=self.mcd.get_metacluster_displayname(self.current_metacluster.value),
            placeholder='Metacluster Displayname',
            description='Edit Name:',
            disabled=False,
        )
        self.current_metacluster_displayname.observe(
            self.update_current_metacluster_displayname,
            type="change",
            names="value"
        )

        # group widgets to look nice
        self.metacluster_info = widgets.VBox([
            self.current_metacluster,
            self.current_metacluster_displayname
        ])
        self.tools = widgets.HBox([
            self.zscore_clamp_slider,
            self.clear_selection_button,
            self.new_metacluster_button,
        ])
        self.toolbar = widgets.HBox([
            self.tools,
            self.metacluster_info
        ])

        self.toolbar.layout.justify_content = 'center'
        self.plot_output = widgets.Output()
        self.gui = widgets.VBox([self.plot_output, self.toolbar])

    def move_dendro_labels(self, ax, dendrosplit_ratio=1.8):
        """Overlay axis labels directly onto a scipy dendrogram

        Final image will use the ratio 1:dendrosplit_ratio
        for tree_region:labels_region

        Args:
            ax (matplotlib.axes.Axes):
                The axis containing the existing scipy dendrogram
            dendrosplit_ratio (float):
                How big to make the the labels compared to the tree
        """
        def add_room_for_labels():
            ax.set_axisbelow(False)
            xlim = ax.get_xlim()
            ax.set_xlim((xlim[0], -(xlim[0] * dendrosplit_ratio)))

        def stretch_dendro_leaves():
            for c in ax.collections:
                for path in c.get_paths():
                    for v in path.vertices:
                        if v[0] == 0:
                            v[0] = ax.get_xlim()[1]

        def get_ax_width_points(ax):
            bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
            return bbox.width * 72  # points = 1/72 in

        def move_ax_labels():
            dr = dendrosplit_ratio
            width = get_ax_width_points(ax)
            dedent = -(width * dr / (1 + dr))
            ax.yaxis.set_tick_params(which='both', pad=dedent)

        def restyle_ax_labels():
            for lb in ax.get_yticklabels():
                lb.set_path_effects([
                    path_effects.Stroke(linewidth=4, foreground='white'),
                    path_effects.Normal(),
                    ])
                lb.set_family('monospace')
                lb.set_zorder(4)

        add_room_for_labels()
        stretch_dendro_leaves()
        move_ax_labels()
        restyle_ax_labels()

    @property
    def selection_mask(self):
        """2D boolean mask of shape (1,cluster_count) of currently selected clusters"""
        def is_selected(cluster):
            if cluster in self.selected_clusters:
                return 1
            else:
                return 0
        return [[is_selected(c) for c in self.mcd.clusters.index]]

    def update_gui(self):
        """Update and redraw any updated GUI elements"""
        self.im_cs.set_data(self.selection_mask)
        self.im_cs.set_extent((0, self.mcd.cluster_count, 0, 1))

        if not self._heatmaps_stale:
            print("skipping other repaints")
            self.fig.canvas.draw()
            return

        def _preplot(df):
            return df.apply(zscore).clip(upper=self.zscore_clamp_slider.value).T

        self.normalizer.calibrate(_preplot(self.mcd.clusters).values)

        # clusters heatmap
        self.im_c.set_data(_preplot(self.mcd.clusters))
        self.im_c.set_extent((0, self.mcd.cluster_count, 0, self.mcd.marker_count))
        self.im_c.set_clim(self.normalizer.vmin, self.normalizer.vmax)

        # metaclusters heatmap
        self.im_m.set_data(_preplot(self.mcd.metaclusters))
        self.im_m.set_extent((0, self.mcd.metacluster_count, 0, self.mcd.marker_count))
        self.im_m.set_clim(self.normalizer.vmin, self.normalizer.vmax)

        # retrieve the current value of the zscore sliders
        zscore_cap = self.zscore_clamp_slider.value

        # due to delays, a zscore_cap modulo of 1 also needs to be considered here
        # due to floating point error, allclose must be used
        if np.allclose(zscore_cap % 1, 0) or np.allclose(zscore_cap % 1, 1):
            new_ticks = np.arange(-zscore_cap, zscore_cap + 1)
        else:
            # fractional intervals are always in increments of 1/2
            new_ticks = np.arange(-zscore_cap + 0.5, zscore_cap - 0.5 + 1)
            new_ticks = np.insert(new_ticks, 0, -zscore_cap)
            new_ticks = np.append(new_ticks, zscore_cap)

        self.cb.ax.set_xticks(new_ticks)

        # xaxis metacluster color labels
        assert len(self.mcd.metaclusters.index) <= self.mcd.cluster_count, \
            "Can't support num metaclusters > cluster count"
        mc_cmap = distinct_cmap(self.mcd.cluster_count)  # metaclusters < clusters
        self.im_cl.set_data([self.mcd.clusters_with_metaclusters['metacluster']])
        self.im_cl.set_extent((0, self.mcd.cluster_count, 0, 1))
        self.im_cl.set_cmap(mc_cmap)
        self.ax_ml.set_xticks(np.arange(self.mcd.metacluster_count) + 0.5)
        self.ax_ml.set_xticklabels(self.mcd.metacluster_displaynames, rotation=90, fontsize=7)
        self.im_ml.set_data([self.mcd.metaclusters.index])
        self.im_ml.set_extent((0, self.mcd.metacluster_count, 0, 1))
        self.im_ml.set_cmap(mc_cmap)

        # xaxis pixelcount graphs
        ax_cp_ymax = max(self.mcd.cluster_pixelcounts['count']) * 1.65
        self.ax_cp.set_ylim(0, ax_cp_ymax)
        sorted_pixel_counts = self.mcd.clusters.join(self.mcd.cluster_pixelcounts)['count']
        for rect, h in zip(self.rects_cp, sorted_pixel_counts):
            rect.set_height(h)
        for label, y in zip(self.labels_cp, sorted_pixel_counts):
            text = str(y)
            label_y_spacing = ax_cp_ymax * 0.05
            label.set_y(y + label_y_spacing)
            label.set_text(text)

        self.fig.canvas.draw()
        self._heatmaps_stale = False

    def enable_debug_mode(self):
        """Display the debug output widget as part of the GUI

        This is used to route logging, output, and tracebacks that happen
        in any of the event handler callbacks.
        """
        self.fig.canvas.footer_visible = True
        DEBUG_VIEW.clear_output()
        DEBUG_VIEW.append_stdout("Debug mode started\n")
        display(DEBUG_VIEW)

    def remap_current_selection(self, metacluster):
        """Instruct the MetaClusterData to remap the selected clusters

        All selected clusters will be remapped to the metacluster id which is passed

        Args:
            metacluster (int):
                metacluster id to map the current selection to
        """
        for cluster in self.selected_clusters:
            print('remapping', cluster, metacluster)
            self.mcd.remap(cluster, metacluster)
        self._heatmaps_stale = True
        self.mcd.save_output_mapping()

    @DEBUG_VIEW.capture(clear_output=False)
    def update_zscore(self, e):
        self._heatmaps_stale = True
        self.update_gui()

    @DEBUG_VIEW.capture(clear_output=False)
    def clear_selection(self, e):
        self.selected_clusters.clear()
        self.update_gui()

    @DEBUG_VIEW.capture(clear_output=False)
    def new_metacluster(self, e):
        metacluster = self.mcd.new_metacluster()
        self.remap_current_selection(metacluster)
        self.update_current_metacluster(metacluster)
        self.update_gui()

    def update_current_metacluster_handler(self, t):
        return self.update_current_metacluster(t.new)

    @DEBUG_VIEW.capture(clear_output=False)
    def update_current_metacluster(self, metacluster):
        self.current_metacluster.options = \
            list(zip(self.mcd.metacluster_displaynames, self.mcd.metaclusters.index))
        self.current_metacluster.value = metacluster
        self.current_metacluster_displayname.value = \
            self.mcd.get_metacluster_displayname(metacluster)

    @DEBUG_VIEW.capture(clear_output=False)
    def update_current_metacluster_displayname(self, t):
        self.mcd.change_displayname(self.current_metacluster.value, t.new)
        old_current_metacluster = self.current_metacluster.value

        self.current_metacluster.unobserve(
            self.update_current_metacluster_handler, type="change", names="value"
        )

        self.current_metacluster.options = \
            list(zip(self.mcd.metacluster_displaynames, self.mcd.metaclusters.index))
        self.current_metacluster.value = old_current_metacluster

        self.current_metacluster.observe(
            self.update_current_metacluster_handler, type="change", names="value")

        self._heatmaps_stale = True
        self.update_gui()

    @DEBUG_VIEW.capture(clear_output=False)
    def onpick(self, e):
        self.e = e
        if e.mouseevent.name != 'button_press_event':
            return
        if e.mouseevent.button == 1:
            self.onpick_select(e)
        elif e.mouseevent.button == 3:
            self.onpick_remap(e)
        self.update_gui()

    def onpick_select(self, e):
        """Handle or route for handling all clicks to any matplotlib plots."""
        selected_ix = int(e.mouseevent.xdata)
        if e.artist in [self.im_c, self.im_cs]:
            selected_cluster = self.mcd.clusters.index[selected_ix]
            # Toggle selection
            if selected_cluster in self.selected_clusters:
                self.selected_clusters.remove(selected_cluster)
            else:
                self.selected_clusters.add(selected_cluster)
        elif e.artist in [self.im_m, self.im_ml]:
            self.select_metacluster(self.mcd.metaclusters.index[selected_ix])
        elif e.artist in [self.im_cl]:
            selected_cluster = self.mcd.clusters_with_metaclusters.index[selected_ix]
            metacluster = self.mcd.which_metacluster(cluster=selected_cluster)
            self.select_metacluster(metacluster)

    def select_metacluster(self, metacluster):
        self.update_current_metacluster(metacluster)
        clusters = self.mcd.cluster_in_metacluster(metacluster)
        # Toggle entire metacluster
        if all(c in self.selected_clusters for c in clusters):
            # remove whole metacluster
            self.selected_clusters.difference_update(clusters)
        else:
            # select whole metacluster
            self.selected_clusters.update(clusters)

    def onpick_remap(self, e):
        selected_ix = int(e.mouseevent.xdata)
        metacluster = None
        if e.artist in [self.im_c, self.im_cs]:
            selected_cluster = self.mcd.clusters.index[selected_ix]
            metacluster = self.mcd.which_metacluster(cluster=selected_cluster)
        elif e.artist in [self.im_m, self.im_ml]:
            metacluster = self.mcd.metaclusters.index[selected_ix]
        elif e.artist in [self.im_cl]:
            selected_cluster = self.mcd.clusters_with_metaclusters.index[selected_ix]
            metacluster = self.mcd.which_metacluster(cluster=selected_cluster)

        self.update_current_metacluster(metacluster)
        self.remap_current_selection(metacluster)
