import subprocess
import tempfile


def _exec_notebook(path):
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as fout:
        args = ["jupyter", "nbconvert", "--to", "notebook", "--execute",
                "--ExecutePreprocessor.timeout=1000",
                "--output", fout.name, path]
        subprocess.check_call(args)


def test_preprocessing():
    _exec_notebook('scripts/Deepcell_Preprocessing.ipynb')


def test_postprocessing():
    _exec_notebook('scripts/Deepcell_Postprocessing.ipynb')


