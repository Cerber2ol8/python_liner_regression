import os
import six
class PlotData(object):
    def __init__(self):
        self.step = []
        self.value = []

    def append(self, step, value):
        self.step.append(step)
        self.value.append(value)

    def reset(self):
        self.step = []
        self.value = []


class Ploter(object):
    """
        Plot input data in a 2D graph
        
        Args:
            title: assign the title of input data.
            step: x_axis of the data.
            value: y_axis of the data.
    """

    def __init__(self, *args):
        self.__args__ = args
        self.__plot_data__ = {}
        for title in args:
            self.__plot_data__[title] = PlotData()
        # demo in notebooks will use Ploter to plot figure, but when we convert
        # the ipydb to py file for testing, the import of matplotlib will make the
        # script crash. So we can use `export DISABLE_PLOT=True` to disable import
        # these libs
        self.__disable_plot__ = os.environ.get("DISABLE_PLOT")
        if not self.__plot_is_disabled__():
            import matplotlib.pyplot as plt
            self.plt = plt

    def __plot_is_disabled__(self):
        return self.__disable_plot__ == "True"

    def append(self, title, step, value):
        """
        Feed data

        Args:
                title: assign the group data to this subtitle.
                step: the x_axis of data.
                value: the y_axis of data.
            
            Examples:
                .. code-block:: python
                plot_curve = Ploter("Curve 1","Curve 2")
                plot_curve.append(title="Curve 1",step=1,value=1)
        """
        assert isinstance(title, six.string_types)
        assert title in self.__plot_data__
        data = self.__plot_data__[title]
        assert isinstance(data, PlotData)
        data.append(step, value)

    def plot(self, path=None):
        """
            Plot data in a 2D graph

            Args:
                path: store the figure to this file path. Defaul None. 
              
            Examples:
                .. code-block:: python
                plot_curve = Ploter()
                plot_cure.plot()
        """
        if self.__plot_is_disabled__():
            return

        titles = []
        for title in self.__args__:
            data = self.__plot_data__[title]
            assert isinstance(data, PlotData)
            if len(data.step) > 0:
                titles.append(title)
                self.plt.plot(data.step, data.value)
        self.plt.legend(titles, loc='upper left')

    def reset(self):
        for key in self.__plot_data__:
            data = self.__plot_data__[key]
            assert isinstance(data, PlotData)
            data.reset()
