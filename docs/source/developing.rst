===========================
Developing and Contributing
===========================


Working on the package
----------------------
You have some cool feature and/or algorithm you want to add to the package. How do you go about it?

First clone the package.

.. code-block:: bash

  git clone https://github.com/KaveIO/PhiK.git

then

.. code-block:: bash

  pip install -e PhiK/

this will install ``PhiK`` in editable mode, which will allow you to edit the code and run it as
you would with a normal installation of the ``PhiK`` correlation analyzer package.

To make sure that everything works try executing the tests, e.g.

.. code-block:: bash

  cd PhiK/
  phik_trial .

or 

.. code-block:: bash

  cd PhiK/
  python setup.py test

That's it.


Contributing
------------

When contributing to this repository, please first discuss the change you wish to make via issue, email, or any
other method with the owners of this repository before making a change. You can find the contact information on the
`index <index.html>`_ page.

Note that when contributing that all tests should succeed.


Tips and Tricks
---------------

- Enable auto reload in ``jupyter``:

.. code-block:: python

  %load_ext autoreload

this will reload modules before executing any user code.
