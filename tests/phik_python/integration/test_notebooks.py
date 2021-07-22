import sys
import asyncio
from phik_python.bases import NotebookTest
from phik import resources
import pytest

# See https://bugs.python.org/issue37373 :(
if (
    sys.version_info[0] == 3
    and sys.version_info[1] >= 8
    and sys.platform.startswith("win")
):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


@pytest.mark.filterwarnings("ignore:Session._key_changed is deprecated")
class PipelineNotebookTest(NotebookTest):
    """Unit test notebook"""

    def test_basic_tutorial(self):
        self.run_notebook(resources.notebook("phik_tutorial_basic.ipynb"))

    def test_advanced_tutorial(self):
        self.run_notebook(resources.notebook("phik_tutorial_advanced.ipynb"))
