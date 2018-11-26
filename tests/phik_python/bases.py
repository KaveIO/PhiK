import os
import unittest
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors.execute import CellExecutionError


class NotebookTest(unittest.TestCase):
    """Unit test notebook"""

    def run_notebook(self, notebook):
        """ Test notebook """

        # load notebook
        with open(notebook) as f:
            nb = nbformat.read(f, as_version=4)

        # execute notebook
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        try:
            ep.preprocess(nb, {})
            status = True
        except CellExecutionError:
            # store if failed
            status = False
            executed_notebook = os.getcwd() + '/' + notebook.split('/')[-1]
            with open(executed_notebook, mode='wt') as f:
                nbformat.write(nb, f)

        # check status
        self.assertTrue(status, 'Notebook execution failed (%s)' % notebook)
