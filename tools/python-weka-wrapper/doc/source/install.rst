Installation
============

The following sections should help you install the library on your machine.

Some of the instructions are based on the *CellProfiler*
`instructions <https://github.com/CellProfiler/python-javabridge/blob/master/docs/installation.rst>`_, the guys
behind the *javabridge* library.

However, if you should encounter problems or if you would like to submit improvements
on the instructions below, please use the following mailing list:

https://groups.google.com/forum/#!forum/python-weka-wrapper


Videos
------

You can find videos on `YouTube <https://www.youtube.com/channel/UCYkzno8dbnAasWakSXVsuPA>`_ that show the installation process:

* `Ubuntu 14.04 (64-bit) <https://www.youtube.com/watch?v=8d0PVBlttM4>`_
* `Windows 7 (32-bit) <https://www.youtube.com/watch?v=KdDt9rT5wTo>`_
* `Windows 8.1 (64-bit) <https://www.youtube.com/watch?v=PeUfDVOA_1Y>`_
* `Mac OSX 10.9.5 (Mavericks) <https://www.youtube.com/watch?v=CORXWYam36E>`_


Prerequisites for all plaforms
------------------------------

You need an `Oracle JDK (1.6+) <http://www.oracle.com/technetwork/java/javase/downloads/>`_
installed and the `JAVA_HOME` `environment variable <http://docs.oracle.com/cd/E19182-01/820-7851/inst_cli_jdk_javahome_t/index.html>`_
pointing to the installation directory in order to use *python-weka-wrapper* library.

*Why not OpenJDK?* Weka is developed and tested with Oracle's JDK/JRE. There is
no guarantee that it will work with OpenJDK.


Ubuntu
------

First, you need to be able to compile C/C++ code and Python modules:

.. code-block:: bash

   $ sudo apt-get install build-essential python-dev

Now, you can install the various packages that we require for installing
`python-weka-wrapper`:

.. code-block:: bash

   $ sudo apt-get install python-pip python-numpy

The following packages are optional, but necessary if you also want some
graphical output:

.. code-block:: bash

   $ sudo apt-get install python-imaging python-matplotlib python-pygraphviz

Install OpenJDK as well, in order to get all the header files that *javabridge*
compiles against (but don't use it for starting up JVMs):

.. code-block:: bash

   $ sudo apt-get install default-jdk

Finally, you can use `pip` to install the Python packages that are not
available in the repositories:

.. code-block:: bash

   $ sudo pip install javabridge
   $ sudo pip install python-weka-wrapper


Debian
------

First, become the superuser:

.. code-block:: bash

   $ su

You need to be able to compile C/C++ code and Python modules:

.. code-block:: bash

   $ apt-get install build-essential python-dev

Now, you can install the various packages that we require for installing
`python-weka-wrapper`:

.. code-block:: bash

   $ apt-get install python-pip python-numpy

The following packages are optional, but necessary if you also want some
graphical output:

.. code-block:: bash

   $ apt-get install python-imaging python-matplotlib python-pygraphviz

Download an Oracle JDK and un-tar it in `/opt` (e.g., `/opt/jdk1.7.0_75/`).
Export the Java home directory as follows (required for the `javabridge`
installation):

.. code-block:: bash

   $ export JAVA_HOME=/opt/jdk1.7.0_75/

Finally, you can use `pip` to install the Python packages that are not
available in the repositories:

.. code-block:: bash

   $ pip install javabridge
   $ pip install python-weka-wrapper

Please note, when using `python-weka-wrapper` as a *normal* user, don't forget
to export the `JAVA_HOME` environment variable as described above (e.g., add it
to your `.profile`).


Other Linux distributions
-------------------------

See `these <http://docs.python-guide.org/en/latest/starting/install/linux/>`_
general instructions for installing Python on Linux. You need to be able to
compile C/C++ code and Python modules (i.e., Python header files are required).
By installing OpenJDK, you should be able to compile *javabridge* against its
header files (for JNI access).

Then you need to install the following Python packages, preferably through your
package manager (e.g., `yum`).  Please note that on a *headless* machine, you
can omit the packages marked as *optional*, as they are only required for
graphical output and plots:

* pip
* numpy
* PIL (optional)
* matplotlib (optional)
* pygraphviz (optional)

Once these libraries are installed, you can use `pip` to install the remaining
Python packages:

.. code-block:: bash

   $ sudo pip install javabridge
   $ sudo pip install python-weka-wrapper


Mac OSX
-------

Please follow `these <http://docs.python-guide.org/en/latest/starting/install/osx/>`_
general instructions for installing Python.

In order to compile C/C++ code, you need to install *Xcode* through Apple's App
Store. Once installed you can install the *XCode command-line tools* by issuing
the command `xcode-select --install` in a terminal.

Also, install *graphviz* using homebrew (`brew install pkg-config` and 
`brew install graphviz`) for visualizing trees and graphs.

You need to install the following Python packages:

* numpy
* pillow
* matplotlib
* pygraphviz

Once these libraries are installed, you can use `pip` to install the remaining
Python packages:

.. code-block:: bash

   $ pip install javabridge
   $ pip install python-weka-wrapper


Windows
-------

**Please note:** You need to make sure that the *bitness* of your environment
is consistent.  I.e., if you install a 32-bit version of Python, you need to
install a 32-bit JDK and 32-bit numpy (or all of them are 64-bit).

Perform the following steps:

* install `Python <http://www.python.org/downloads>`_, make sure you check `Add python.exe to path` during the installation
* add the Python scripts directory to your `PATH` environment variable, e.g., `C:\\Python27\\Scripts`
* install `pip` with these steps:

 * download from `here <https://bootstrap.pypa.io/get-pip.py>`_
 * install using `python get-pip.py`

* install `numpy 1.9.x <http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy>`_
* install `.Net 4.0 <http://go.microsoft.com/fwlink/?LinkID=187668>`_ (if not already installed)
* install `Windows SDK 7.1 <http://www.microsoft.com/download/details.aspx?id=8279>`_

* open Windows SDK command prompt (**not** the regular command prompt!) and install `javabridge` and `python-weka-wrapper`

  .. code-block:: bat

     set MSSdk=1
     set DISTUTILS_USE_SDK=1
     pip install javabridge
     pip install python-weka-wrapper

Now you can run `python-weka-wrapper` using the regular command-prompt as well.


From source
-----------

You can either download a source archive or clone the github repository
(`git clone https://github.com/fracpete/python-weka-wrapper.git`). Once you
have done this, you can install the library using the following command:

.. code-block:: bash

   $ python setup.py install

Check out the section on *virtualenv* as well, if you would rather install it
in a *disposable* location.
