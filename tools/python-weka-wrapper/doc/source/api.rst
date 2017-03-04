API
===

The following sections explain in more detail of how to use *python-weka-wrapper* from Python using the API.

A lot more examples you will find in the (aptly named) `examples` module.


Java Virtual Machine
--------------------

In order to use the library, you need to manage the Java Virtual Machine (JVM).

For starting up the library, use the following code:

.. code-block:: python

   >>> import weka.core.jvm as jvm
   >>> jvm.start()

If you want to use the classpath environment variable and all currently installed Weka packages,
use the following call:

.. code-block:: python

   >>> jvm.start(system_cp=True, packages=True)

In case your Weka home directory is not located in `wekafiles` in your user's home directory,
then you have two options for specifying the alternative location: use the `WEKA_HOME` environment
variable or the `packages` parameter, supplying a directory. The latter is shown below:

.. code-block:: python

   >>> jvm.start(packages="/my/packages/are/somwhere/else")

Most of the times, you will want to increase the maximum heap size available to the JVM.
The following example reserves 512 MB:

.. code-block:: python

   >>> jvm.start(max_heap_size="512m")

And, finally, in order to stop the JVM again, use the following call:

.. code-block:: python

   >>> jvm.stop()


Data generators
---------------

Artifical data can be generated using one of Weka's data generators, e.g., the
`Agrawal` classification generator:

.. code-block:: python

   >>> from weka.datagenerators import DataGenerator
   >>> generator = DataGenerator(classname="weka.datagenerators.classifiers.classification.Agrawal", options=["-B", "-P", "0.05"])
   >>> DataGenerator.make_data(generator, ["-o", "/some/where/outputfile.arff"])

Or using the low-level API (outputting data to stdout):

.. code-block:: python

   >>> generator = DataGenerator(classname="weka.datagenerators.classifiers.classification.Agrawal", options=["-n", "10", "-r", "agrawal"])
   >>> generator.dataset_format = generator.define_data_format()
   >>> print(generator.dataset_format)
   >>> if generator.single_mode_flag:
   >>>     for i in xrange(generator.num_examples_act):
   >>>         print(generator.generate_example())
   >>> else:
   >>>     print(generator.generate_examples())


Loaders and Savers
------------------

You can load and save datasets of various data formats using the `Loader` and `Saver` classes.

The following example loads an ARFF file and saves it as CSV:

.. code-block:: python

   >>> from weka.core.converters import Loader, Saver
   >>> loader = Loader(classname="weka.core.converters.ArffLoader")
   >>> data = loader.load_file("/some/where/iris.arff")
   >>> print(data)
   >>> saver = Saver(classname="weka.core.converters.CSVSaver")
   >>> saver.save_file(data, "/some/where/iris.csv")


Filters
-------

The `Filter` class from the `weka.filters` module allows you to filter datasets, e.g.,
removing the last attribute using the `Remove` filter:

.. code-block:: python

   >>> from weka.filters import Filter
   >>> data = ...                       # previously loaded data
   >>> remove = Filter(classname="weka.filters.unsupervised.attribute.Remove", options=["-R", "last"])
   >>> remove.inputformat(data)     # let the filter know about the type of data to filter
   >>> filtered = remove.filter(data)   # filter the data
   >>> print(filtered)                  # output the filtered data

Classifiers
-----------

Here is an example on how to cross-validate a `J48` classifier (with confidence factor 0.3)
on a dataset and output the summary and some specific statistics:

.. code-block:: python

   >>> from weka.classifiers import Classifier, Evaluation
   >>> from weka.core.classes import Random
   >>> data = ...             # previously loaded data
   >>> data.class_is_last()   # set class attribute
   >>> classifier = Classifier(classname="weka.classifiers.trees.J48", options=["-C", "0.3"])
   >>> evaluation = Evaluation(data)                     # initialize with priors
   >>> evaluation.crossvalidate_model(classifier, iris_data, 10, Random(42))  # 10-fold CV
   >>> print(evaluation.summary())
   >>> print("pctCorrect: " + str(evaluation.percent_correct))
   >>> print("incorrect: " + str(evaluation.incorrect))

Here we train a classifier and output predictions:

.. code-block:: python

   >>> from weka.classifiers import Classifier
   >>> data = ...             # previously loaded data
   >>> data.class_is_last()   # set class attribute
   >>> cls = Classifier(classname="weka.classifiers.trees.J48", options=["-C", "0.3"])
   >>> cls.build_classifier(data)
   >>> for index, inst in enumerate(data):
   >>>     pred = cls.classify_instance(inst)
   >>>     dist = cls.distribution_for_instance(inst)
   >>>     print(str(index+1) + ": label index=" + str(pred) + ", class distribution=" + str(dist))

Clusterers
----------

In the following an example on how to build a `SimpleKMeans` (with 3 clusters)
using a previously loaded dataset without a class attribute:

.. code-block:: python

   >>> from weka.clusterers import Clusterer
   >>> data = ... # previously loaded dataset
   >>> clusterer = Clusterer(classname="weka.clusterers.SimpleKMeans", options=["-N", "3"])
   >>> clusterer.build_clusterer(data)
   >>> print(clusterer)

Once a clusterer is built, it can be used to cluster Instance objects:

.. code-block:: python

   >>> for inst in data:
   >>>     cl = clusterer.cluster_instance(inst)  # 0-based cluster index
   >>>     dist = clusterer.distribution_for_instance(inst)   # cluster membership distribution
   >>>     print("cluster=" + str(cl) + ", distribution=" + str(dist))


Attribute selection
-------------------

You can perform attribute selection using `BestFirst` as search algorithm and
`CfsSubsetEval` as evaluator as follows:

.. code-block:: python

   >>> from weka.attribute_selection import ASSearch, ASEvaluation, AttributeSelection
   >>> data = ...   # previously loaded dataset
   >>> search = ASSearch(classname="weka.attributeSelection.BestFirst", options=["-D", "1", "-N", "5"])
   >>> evaluator = ASEvaluation(classname="weka.attributeSelection.CfsSubsetEval", options=["-P", "1", "-E", "1"])
   >>> attsel = AttributeSelection()
   >>> attsel.search(search)
   >>> attsel.evaluator(evaluator)
   >>> attsel.select_attributes(data)
   >>> print("# attributes: " + str(attsel.number_attributes_selected))
   >>> print("attributes: " + str(attsel.selected_attributes))
   >>> print("result string:\n" + attsel.results_string)


Associators
-----------

Associators, like `Apriori`, can be built and output like this:

.. code-block:: python

   >>> from weka.associations import Associator
   >>> data = ...   # previously loaded dataset
   >>> associator = Associator(classname="weka.associations.Apriori", options=["-N", "9", "-I"])
   >>> associator.build_associations(data)
   >>> print(associator)


Serialization
-------------

You can easily serialize and de-serialize as well.

Here we just save a trained classifier to a file, load it again from disk and output the model:

.. code-block:: python

   >>> import weka.core.serialization as serialization
   >>> from weka.classifiers import Classifier
   >>> classifier = ...  # previously built classifier
   >>> serialization.write("/some/where/out.model", classifier)
   >>> ...
   >>> classifier2 = Classifier(jobject=serialization.read("/some/where/out.model"))
   >>> print(classifier2)

Weka usually saves the header of the dataset that was used for training as well (e.g., in order to determine
whether test data is compatible). This is done as follows:

.. code-block:: python

   >>> import weka.core.serialization as serialization
   >>> from weka.classifiers import Classifier
   >>> from weka.core.dataset import Instances
   >>> classifier = ...  # previously built Classifier
   >>> data = ... # previously loaded/generated Instances
   >>> serialization.write_all("/some/where/out.model", [classifier, Instances.template_instances(data)])
   >>> ...
   >>> objects = serialization.read_all("/some/where/out.model")
   >>> classifier2 = Classifier(jobject=objects[0])
   >>> data2 = Instances(jobject=objects[1])
   >>> print(classifier2)
   >>> print(data2)


Experiments
-----------

Experiments, like they are run in Weka's Experimenter, can be configured and executed as well.

Here is an example for performing a cross-validated classification experiment:

.. code-block:: python

   >>> from weka.experiments import SimpleCrossValidationExperiment, SimpleRandomSplitExperiment, Tester, ResultMatrix
   >>> from weka.classifiers import Classifier
   >>> import weka.core.converters as converters
   >>> # configure experiment
   >>> datasets = ["iris.arff", "anneal.arff"]
   >>> classifiers = [Classifier(classname="weka.classifiers.rules.ZeroR"), Classifier(classname="weka.classifiers.trees.J48")]
   >>> outfile = "results-cv.arff"   # store results for later analysis
   >>> exp = SimpleCrossValidationExperiment(
   >>>     classification=True,
   >>>     runs=10,
   >>>     folds=10,
   >>>     datasets=datasets,
   >>>     classifiers=classifiers,
   >>>     result=outfile)
   >>> exp.setup()
   >>> exp.run()
   >>> # evaluate previous run
   >>> loader = converters.loader_for_file(outfile)
   >>> data   = loader.load_file(outfile)
   >>> matrix = ResultMatrix(classname="weka.experiment.ResultMatrixPlainText")
   >>> tester = Tester(classname="weka.experiment.PairedCorrectedTTester")
   >>> tester.resultmatrix = matrix
   >>> comparison_col = data.attribute_by_name("Percent_correct").index
   >>> tester.instances = data
   >>> print(tester.header(comparison_col))
   >>> print(tester.multi_resultset_full(0, comparison_col))

And a setup for performing regression experiments on random splits on the datasets:

.. code-block:: python

   >>> from weka.experiments import SimpleCrossValidationExperiment, SimpleRandomSplitExperiment, Tester, ResultMatrix
   >>> from weka.classifiers import Classifier
   >>> import weka.core.converters as converters
   >>> # configure experiment
   >>> datasets = ["bolts.arff", "bodyfat.arff"]
   >>> classifiers = [Classifier(classname="weka.classifiers.rules.ZeroR"), Classifier(classname="weka.classifiers.functions.LinearRegression")]
   >>> outfile = "results-rs.arff"   # store results for later analysis
   >>> exp = SimpleRandomSplitExperiment(
   >>>     classification=False,
   >>>     runs=10,
   >>>     percentage=66.6,
   >>>     preserve_order=False,
   >>>     datasets=datasets,
   >>>     classifiers=classifiers,
   >>>     result=outfile)
   >>> exp.setup()
   >>> exp.run()
   >>> # evaluate previous run
   >>> loader = converters.loader_for_file(outfile)
   >>> data   = loader.load_file(outfile)
   >>> matrix = ResultMatrix(classname="weka.experiment.ResultMatrixPlainText")
   >>> tester = Tester(classname="weka.experiment.PairedCorrectedTTester")
   >>> tester.resultmatrix = matrix
   >>> comparison_col = data.attribute_by_name("Correlation_coefficient").index
   >>> tester.instances = data
   >>> print(tester.header(comparison_col))
   >>> print(tester.multi_resultset_full(0, comparison_col))


Packages
--------

Packages can be listed, installed and uninstalled using the `weka.core.packages` module:

.. code-block:: python

   # refresh package cache
   import weka.core.packages as packages
   packages.refresh_cache()

   # list all packages (name and URL)
   items = packages.all_packages()
   for item in items:
       print(item.name + " " + item.url)

   # install CLOPE package
   packages.install_package("CLOPE")
   items = packages.installed_packages()
   for item in items:
       print(item.name + " " + item.url)

   # uninstall CLOPE package
   packages.uninstall_package("CLOPE")
   items = packages.installed_packages()
   for item in items:
       print(item.name + " " + item.url)
