import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from alpineer import misc_utils


def plot_dim_reduced_data(component_one, component_two, fig_id, hue, cell_data,
                          title, title_fontsize=24, palette="Spectral", alpha=0.3,
                          legend_type="full", bbox_to_anchor=(1.05, 1), legend_loc=2,
                          legend_borderaxespad=0., dpi=None, save_dir=None, save_file=None):
    """Helper function to visualize_dimensionality_reduction

    Args:
        component_one (pandas.Series):
            the data corresponding to the first component
        component_two (pandas.Series):
            the data corresponding to the second component
        fig_id (int):
            the figure identifier for the visualization
        hue (pandas.Series):
            define the hue for each data point
        cell_data (pandas.DataFrame):
            Dataframe containing columns for dimensionality reduction and category
        title (str):
            the title we wish to set for the graph
        title_fontsize (int):
            the fontsize of the title we want
        palette (str):
            the color palette we wish to visualize with
        alpha (float):
            a value to define the opacity of the points visualized
        legend_type (str):
            what type of legend we wish to specify
        bbox_to_anchor (tuple):
            the bounding box of the legend
        legend_loc (str):
            an string describing where we want the legend located
        legend_borderaxespad (float):
            the pad between the axes and legend border
        dpi (float):
            The resolution of the image to save, ignored if save_dir is None
        save_dir (str):
            Directory to save plots, default is None
        save_file (str):
            If save_dir specified, specify a file name you wish to save to.
            Ignored if save_dir is None
    """

    plt.figure(fig_id)
    sns.scatterplot(x=component_one, y=component_two, hue=hue, palette=palette,
                    data=cell_data, legend=legend_type, alpha=alpha)

    plt.legend(bbox_to_anchor=bbox_to_anchor, loc=legend_loc, borderaxespad=legend_borderaxespad)
    plt.title(title, fontsize=title_fontsize)

    if save_dir is not None:
        misc_utils.save_figure(save_dir, save_file, dpi=dpi)


def visualize_dimensionality_reduction(cell_data, columns, category, color_map="Spectral",
                                       algorithm="UMAP", dpi=None, save_dir=None):
    """Plots the dimensionality reduction of specified population columns

    Args:
        cell_data (pandas.DataFrame):
            Dataframe containing columns for dimensionality reduction and category
        columns (list):
            List of column names that are included for dimensionality reduction
        category (str):
            Name of column in dataframe containing population or patient data
        color_map (str):
            Name of MatPlotLib ColorMap used
        algorithm (str):
            Name of dimensionality reduction algorithm, must be UMAP, PCA, or tSNE
        dpi (float):
            The resolution of the image to save, ignored if save_dir is None
        save_dir (str):
            Directory to save plots, default is None
    """

    cell_data = cell_data.dropna()
    dim_reduction_algos = ["UMAP", "PCA", "tSNE"]

    misc_utils.verify_in_list(algorithm=algorithm,
                              dimensionality_reduction_algorithms=dim_reduction_algos)

    graph_title = "%s projection of data" % algorithm

    if algorithm == "UMAP":
        reducer = umap.UMAP()

        column_data = cell_data[columns].values
        scaled_column_data = StandardScaler().fit_transform(column_data)
        embedding = reducer.fit_transform(scaled_column_data)

        plot_dim_reduced_data(embedding[:, 0], embedding[:, 1], fig_id=1,
                              hue=cell_data[category], cell_data=cell_data, title=graph_title,
                              dpi=dpi, save_dir=save_dir, save_file="UMAPVisualization.png")

    elif algorithm == "PCA":
        pca = PCA()
        pca_result = pca.fit_transform(cell_data[columns].values)

        plot_dim_reduced_data(pca_result[:, 0], pca_result[:, 1], fig_id=2,
                              hue=cell_data[category], cell_data=cell_data, title=graph_title,
                              dpi=dpi, save_dir=save_dir, save_file="PCAVisualization.png")

    elif algorithm == "tSNE":
        tsne = TSNE()
        tsne_results = tsne.fit_transform(cell_data[columns].values)

        plot_dim_reduced_data(tsne_results[:, 0], tsne_results[:, 1], fig_id=3,
                              hue=cell_data[category], cell_data=cell_data, title=graph_title,
                              dpi=dpi, save_dir=save_dir, save_file="tSNEVisualization.png")
