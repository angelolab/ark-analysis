import subprocess
import tempfile
import os


def _exec_notebook(nb_filename):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        '..', '..', 'templates', nb_filename)
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as fout:
        args = ["jupyter", "nbconvert", "--to", "notebook", "--execute",
                "--ExecutePreprocessor.timeout=1000",
                "--output", fout.name, path]
        subprocess.check_call(args)


def test_preprocessing():
    _exec_notebook('Deepcell_Preprocessing.ipynb')


def test_postprocessing():
    _exec_notebook('Deepcell_Postprocessing.ipynb')


def test_example_spatial_analysis():
    _exec_notebook('example_spatial_analysis_script.ipynb')
