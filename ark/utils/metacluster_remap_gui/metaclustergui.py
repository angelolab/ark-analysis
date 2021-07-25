import warnings

import numpy as np
import ipywidgets as widgets
import matplotlib.pyplot as plt
from scipy.stats import zscore
from IPython.display import display

from .colormap_helper import distinct_cmap

# Third party ipympl causing this in it's backend_agg startup code
warnings.filterwarnings("ignore", message="nbagg.transparent is deprecated")

DEBUG_VIEW = widgets.Output(layout={'border': '1px solid black'})

class MetaClusterGui():
    def __init__(self, metaclusterdata, heatmapcolors='seismic', width=17, debug=False):
        self.width = width
        self.heatmapcolors = heatmapcolors
        self.mcd = metaclusterdata
        self.selected_clusters = set()
        self.make_gui()
        if debug:
            self.enable_debug_mode()

    @property
    def max_zscore(self):
        return self.zscore_clamp_slider.value

    @property
    def cmap(self):
        return distinct_cmap(self.mcd.metacluster_count)

    def preplot(self, df):
        return df.apply(zscore).clip(upper=self.max_zscore).T

    def make_gui(self):
        #  |    Cluster     | Meta |
        #  -------------------------
        #  |    cp          |  mp  | counts of pixels
        #  |    c           |  m   | heatmap itself
        #  |    cs          |  ms  | selection markers
        #  |    cl          |  ml  | metacluster color labels
        subplots = plt.subplots(4,2,
            gridspec_kw={
                'width_ratios': [self.mcd.cluster_count, self.mcd.metacluster_count], # cluster plot bigger than metacluster plot
                'height_ratios': [5, self.mcd.marker_count, 1, 1]},
            figsize=(self.width, 5),
            )

        (self.fig, (
            (self.ax_cp, self.ax_mp),
            (self.ax_c, self.ax_m),
            (self.ax_cs, self.ax_ms),
            (self.ax_cl, self.ax_ml))) = subplots

        self.fig.canvas.toolbar_visible = False
        self.fig.canvas.header_visible = False
        self.fig.canvas.footer_visible = False

        self.fig.canvas.mpl_connect('pick_event', self.onpick)

        # heatmap axis
        self.ax_c.yaxis.set_tick_params(which='major', labelsize=8)
        self.ax_c.set_yticks(np.arange(self.mcd.marker_count)+0.5)
        self.ax_c.set_yticklabels(self.mcd.marker_names)
        self.ax_c.set_xticks(np.arange(self.mcd.cluster_count)+0.5)
        self.ax_m.set_xticks(np.arange(self.mcd.metacluster_count)+0.5)
        self.ax_c.xaxis.set_tick_params(which='both', bottom=False, labelbottom=False)
        self.ax_m.xaxis.set_tick_params(which='both', bottom=False, labelbottom=False)

        # heatmaps
        self.im_c = self.ax_c.imshow(np.zeros((self.mcd.marker_count, self.mcd.cluster_count)), cmap=self.heatmapcolors, aspect='auto', picker=True)
        self.im_m = self.ax_m.imshow(np.zeros((self.mcd.marker_count, self.mcd.metacluster_count)), cmap=self.heatmapcolors, aspect='auto', picker=True)
        self.ax_m.yaxis.set_tick_params(which='both', left=True, labelleft=False)

        # xaxis metacluster color labels
        self.ax_cl.xaxis.set_tick_params(which='both', bottom=False, labelbottom=False)
        self.ax_ml.xaxis.set_tick_params(which='both', bottom=False, labelbottom=False)
        self.ax_cl.yaxis.set_tick_params(which='both', left=False, labelleft=True)
        self.ax_cl.set_yticks([0.5])
        self.ax_cl.set_yticklabels(["Metacluster"])
        self.ax_ml.yaxis.set_tick_params(which='both', left=False, labelleft=False)

        self.im_cl = self.ax_cl.imshow(np.zeros((1, self.mcd.cluster_count)), aspect='auto', picker=True)
        self.im_ml = self.ax_ml.imshow(np.zeros((1, self.mcd.metacluster_count)), aspect='auto', picker=True)

        # xaxis cluster selection labels
        self.ax_cs.xaxis.set_tick_params(which='both', bottom=False, labelbottom=False)
        self.ax_ms.xaxis.set_tick_params(which='both', bottom=False, labelbottom=False)
        self.ax_cs.yaxis.set_tick_params(which='both', left=False, labelleft=True)
        self.ax_cs.set_yticks([0.5])
        self.ax_cs.set_yticklabels(["Selected"])
        self.ax_ms.yaxis.set_tick_params(which='both', left=False, labelleft=False)
        self.im_cs = self.ax_cs.imshow(np.zeros((1, self.mcd.marker_count)), cmap='Blues', aspect='auto', picker=True, vmin=-0.3, vmax=1)
        self.im_ms = self.ax_ms.imshow(np.zeros((1, self.mcd.marker_count)), cmap='Blues', aspect='auto', picker=True, vmin=-0.3, vmax=1)

        # xaxis pixelcount graphs
        self.ax_cp.xaxis.set_tick_params(which='both', bottom=False, labelbottom=False)
        self.ax_mp.xaxis.set_tick_params(which='both', bottom=False, labelbottom=False)
        self.ax_cp.yaxis.set_tick_params(which='both', left=False, labelleft=False)
        self.ax_mp.yaxis.set_tick_params(which='both', left=False, labelleft=False)
        self.ax_cp.set_ylabel("Pixels (k)", rotation=90)
        self.ax_cp.set_xlim(0, self.mcd.cluster_count)
        self.rects_cp = self.ax_cp.bar(np.arange(self.mcd.cluster_count)+0.5, np.zeros(self.mcd.cluster_count))
        self.labels_cp = []
        label_alignment_fudge = 0.08
        for x in np.arange(self.mcd.cluster_count)+0.5+label_alignment_fudge:
            label = self.ax_cp.text(x=x, y=0, s="-", va='bottom', ha='center', rotation=90, color='black', fontsize=8)
            self.labels_cp.append(label)


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
        display(self.zscore_clamp_slider)

        # clear_selection button
        self.clear_selection_button = widgets.Button(
            description='Clear Selection',
            disabled=False,
            button_style='warning',
            tooltip='Clear currently selected clusters',
            icon='ban',
            )
        self.clear_selection_button.on_click(self.clear_selection)
        display(self.clear_selection_button)

        # new metacluster button
        self.new_metacluster_button = widgets.Button(
            description='New metacluster',
            disabled=False,
            button_style='success',
            tooltip='Create new metacluster from current selection',
            icon='plus',
            )
        self.new_metacluster_button.on_click(self.new_metacluster)
        display(self.new_metacluster_button)


        self._heatmaps_stale = True

        # initilize data, etc
        self.update_gui()
        # Tighten layout based on display
        self.fig.tight_layout()
        plt.subplots_adjust(hspace = .0) # make color labels touch heatmap

    def update_gui(self):
        # xaxis cluster selection labels
        selection_mask = [[1 if c in self.selected_clusters else 0 for c in self.mcd.clusters.index]]
        self.im_cs.set_data(selection_mask)
        self.im_cs.set_extent((0, self.mcd.cluster_count, 0, 1))

        if not self._heatmaps_stale:
            print("skipping other repaints")
            return

        # clusters heatmap
        self.im_c.set_data(self.preplot(self.mcd.clusters))
        self.im_c.set_extent((0, self.mcd.cluster_count, 0, self.mcd.marker_count))
        self.im_c.set_clim(0, self.max_zscore)

        # metaclusters heatmap
        self.im_m.set_data(self.preplot(self.mcd.metaclusters))
        self.im_m.set_extent((0, self.mcd.metacluster_count, 0, self.mcd.marker_count))
        self.im_m.set_clim(0, self.max_zscore)

        # xaxis metacluster color labels
        metacluster_iloc = {mc:i+1 for (mc,i) in zip(self.mcd.metaclusters.index, range(self.mcd.metacluster_count))}
        self.im_cl.set_data([[metacluster_iloc[mc] for mc in self.mcd.clusters_with_metaclusters['hCluster_cap']]])
        self.im_cl.set_extent((0, self.mcd.metacluster_count, 0, 1))
        self.im_cl.set_cmap(self.cmap)
        self.im_cl.set_clim(1, self.mcd.metacluster_count)

        self.im_ml.set_data([[metacluster_iloc[mc] for mc in self.mcd.metaclusters.index]])
        self.im_ml.set_extent((0, self.mcd.metacluster_count, 0, 1))
        self.im_ml.set_cmap(self.cmap)
        self.im_ml.set_clim(1, self.mcd.metacluster_count)

        # xaxis pixelcount graphs
        ax_cp_ymax = max(self.mcd.cluster_pixelcounts['count'])*1.5
        self.ax_cp.set_ylim(0, ax_cp_ymax)
        sorted_pixel_counts = self.mcd.clusters.join(self.mcd.cluster_pixelcounts)['count']
        for rect, h in zip(self.rects_cp, sorted_pixel_counts):
            rect.set_height(h)
        for label, y in zip(self.labels_cp, sorted_pixel_counts):
            text = "{:0.0f}".format(y / 1000)
            label_y_spacing = ax_cp_ymax * 0.05
            label.set_y(y + label_y_spacing)
            label.set_text(text)


        self._heatmaps_stale = False

    def enable_debug_mode(self):
        self.fig.canvas.footer_visible = True
        DEBUG_VIEW.clear_output()
        DEBUG_VIEW.append_stdout("Debug mode started\n")
        display(DEBUG_VIEW)

    def remap_current_selection(self, metacluster):
        for cluster in self.selected_clusters:
            print('remapping', cluster, metacluster)
            self.mcd.remap(cluster, metacluster)
        self._heatmaps_stale = True

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
        metacluster = max(self.mcd.mapping['hCluster_cap']) + 1
        print(metacluster)
        self.remap_current_selection(metacluster)
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
        selected_ix = int(e.mouseevent.xdata)
        if e.artist in [self.im_c, self.im_cs]:
            selected_cluster = self.mcd.clusters.index[selected_ix]
            # Toggle selection
            if selected_cluster in self.selected_clusters:
                self.selected_clusters.remove(selected_cluster)
            else:
                self.selected_clusters.add(selected_cluster)
        elif e.artist in [self.im_m, self.im_ml, self.im_ms]:
            self.select_metacluster(self.mcd.metaclusters.index[selected_ix])
        elif e.artist in [self.im_cl]:
            selected_cluster = self.mcd.clusters_with_metaclusters.index[selected_ix]
            self.select_metacluster(self.mcd.which_metacluster(cluster=selected_cluster))

    def select_metacluster(self, metacluster):
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
        elif e.artist in [self.im_m, self.im_ml, self.im_ms]:
            metacluster = self.mcd.metaclusters.index[selected_ix]
        elif e.artist in [self.im_cl]:
            selected_cluster = self.mcd.clusters_with_metaclusters.index[selected_ix]
            metacluster = self.mcd.which_metacluster(cluster=selected_cluster)
        self.remap_current_selection(metacluster)
