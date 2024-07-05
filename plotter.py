import os, math
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

""" Displays the graph of the training and validations runs on the screen. """
class Plotter():  
    def __init__(self):
        self.save_path = './plotter/'
          
    """ Plots a graph with the give points list and shows it on screen.
    @train_list       list of train values to plot.
    @validation_list  list of validation values to plot.
    @x_axis_min       minimum value on the x-axis.
    @x_axis_max       maximum value on the x-axis. 
    @y_axis_min       minimum value on the y-axis.
    @y_axis_max       maximum value on the x-axis.
    @y_label          labels the y-axis.
    @x_label          labels the x-axis.
    @title            title of the graph.
    @legend           descirbes the shown graph.
    @optimizer_name   Name of the selected optimizer.
    @graph_name       specifies the name of the graph."""
    def plot_loss(self, train_list, validation_list, x_axis_min, 
                x_axis_max, y_axis_min, y_axis_max, y_label, x_label, 
                title, save_file, graph_name, batch_size):

        for entry in train_list:
            plt.plot(entry) 
        for entry in validation_list:
            plt.plot(entry)
        plt.axis([int(x_axis_min), int(x_axis_max), y_axis_min, y_axis_max])
        plt.legend(['SGD train loss', 'Adam train loss', 'LAMB train loss', 
                    'SGD Val loss', 'Adam Val loss', 'LAMB Val loss'])
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.title(title)
        try:
            os.mkdir(self.save_path)
        except FileExistsError:
            pass
        plt.savefig(f'{self.save_path}{save_file}_{graph_name}_{batch_size}.png')
        plt.show()
    
    """ Plots a graph with the give points list and shows it on screen.
    @value_list       list of points to plot.
    @x_axis_min       minimum value on the x-axis.
    @x_axis_max       maximum value on the x-axis. 
    @y_axis_min       minimum value on the y-axis.
    @y_axis_max       maximum value on the x-axis.
    @y_label          labels the y-axis.
    @x_label          labels the x-axis.
    @title            title of the graph.
    @legend           descirbes the shown graph.
    @optimizer_name   Name of the selected optimizer.
    @graph_name       specifies the name of the graph. """
    def plot_accuracy(self, acc_list, x_axis_min, x_axis_max, 
                    y_axis_min, y_axis_max, y_label, x_label, 
                    title, save_file, graph_name, batch_size):
        for entry in acc_list:
            plt.plot(entry, '-o') 
        plt.axis([x_axis_min, x_axis_max, y_axis_min, y_axis_max])
        plt.legend(['SGD', 'Adam', 'LAMB'])
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.xticks([])
        plt.title(title)
        try:
            os.mkdir(self.save_path)
        except FileExistsError:
            pass
        plt.savefig(f'{self.save_path}{save_file}_{graph_name}_{batch_size}.png')
        plt.show()
