import inspect
import pkgutil
import importlib
from typing import Set, List

import merlin
from merlin.plots._base import AbstractPlot
from merlin.plots._base import PlotMetadata


def get_available_plots() -> Set:
    """ Get all plots defined within any submodule of merlin.plots

    Returns: a set of references to the plots
    """
    plotSet = set()
    for importer, modname, ispkg in pkgutil.iter_modules(merlin.plots.__path__):
        currentModule = importlib.import_module(
            merlin.plots.__name__ + '.' + modname)
        for name, obj in inspect.getmembers(currentModule):
            if inspect.isclass(obj)\
                    and issubclass(obj, AbstractPlot)\
                    and obj != AbstractPlot:
                plotSet.add(obj)
    return plotSet


class PlotEngine:

    def __init__(self, plotTask, taskDict):
        """ Create a new plot engine.

        Args:
            plotTask: the analysis task to save the plots and plot
                metadata into
            taskDict: a dictionary containing references to the analysis
                tasks to use for plotting results.
        """
        self.taskDict = taskDict
        availablePlots = [x(plotTask) for x in get_available_plots()]
        self.plotList = [x for x in availablePlots if x.is_relevant(taskDict)]

        requiredMetadata = \
            {m for p in self.plotList for m in p.get_required_metadata()}
        self.metadataDict = {x.metadata_name(): x(plotTask, taskDict)
                             for x in requiredMetadata}

    def get_plots(self) -> List[AbstractPlot]:
        """ Get a list of the plots that this plot engine will generate.

        Returns: A list of the plot objects that will be generated by this
            plot engine.
        """
        return self.plotList

    def take_step(self) -> bool:
        """ Generate metadata and plots from newly available analysis results.

        Returns: True if all plots have been generated and otherwise false.
        """

        incompletePlots = [p for p in self.plotList if not p.is_complete()]
        if len(incompletePlots) == 0:
            return True

        for m in self.metadataDict.values():
            m.update()

        completeTasks = [k for k, v in self.taskDict.items() if v.is_complete()]
        completeMetadata = [k for k, v in self.metadataDict.items()
                            if v.is_complete()]
        readyPlots = [p for p in incompletePlots
                      if p.is_ready(completeTasks, completeMetadata)]
        for p in readyPlots:
            p.plot(self.taskDict, self.metadataDict)

        return len([p for p in self.plotList if not p.is_complete()]) == 0