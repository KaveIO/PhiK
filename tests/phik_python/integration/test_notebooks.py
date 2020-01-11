from phik_python.bases import NotebookTest
from phik import resources
import pytest


@pytest.mark.filterwarnings("ignore:Session._key_changed is deprecated")
class PipelineNotebookTest(NotebookTest):
    """Unit test notebook"""

    def test_basic_tutorial(self):
        self.run_notebook( resources.notebook('phik_tutorial_basic.ipynb') )

    def test_advanced_tutorial(self):
        self.run_notebook( resources.notebook('phik_tutorial_advanced.ipynb') )
