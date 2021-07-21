import sys, os, pathlib, torch, time, math
import torch_optimizer as optim
import warnings

from torch.serialization import SourceChangeWarning
from service import Service
from plotter import Plotter
from storage import Storage
from calculator import Calculator

# Contains all menu relevant functions.
class Menu():
    def __init__(self):       
        self.service = Service()
        self.epochs = 100
        self.learning_rate = 0.001
        self.plotter = Plotter()
        self.storage = Storage()
        self.calculator = Calculator()

    # Shows a menu to choose from 3 optimizers and their benchmark graphs, or to exit the application. 
    # @min      smallest number in the menu -> 1.
    # @max      largest number in the menu -> 5.
    def show_menu(self, min, max):
        while True:
            print('Menu')
            print('----')
            print('Choose an optimizer to train the convolutional network or if a trained model already exists') 
            print('check the performances of the different optimizers under point 4.\n')
            print('1. SGD')
            print('2. Adam')
            print('3. LAMB')
            print('4. Benchmark overview')
            print('---------------------\n')
            print('5. Exit\n')
            selection = input('Input: ')
            if not selection.isnumeric() or int(selection) > max or int(selection) < min :
                print('Input not in range or not a number.')
                time.sleep(3)
                continue 
            self.selected_option(selection)

    # Calls all the neccessary functions to train, validate, save the completed training run.
    # @optimizer_name    Name of the choosen optimizer.
    # @learning_rate     User input vslue which defines the step size in the data record 
    # @epochs            User input which defines the cycles over the data  
    def validate_and_safe(self, optimizer_name, learning_rate, epochs):
        self.service.validate_cifar(self.epochs)
        self.storage.save_csv_file(self.service.optimizer_name, 'training', self.service.training_loss)
        self.storage.save_csv_file(self.service.optimizer_name, 'validation', self.service.validation_loss)
        self.storage.save_csv_file(self.service.optimizer_name, 'accuracy', self.service.validation_accuracy)
        print('Data is saved.\n')

    # Calls functions to show the selected optimizer benchmark.
    # @optimizer_name    sequence of all optimizers.
    def show_benchmark(self, *optimizer_name):
        train_loss_avg_list = []
        val_loss_avg_list = []
        accuracy_avg_list = [] 
        for entry in range(len(optimizer_name)):
            # Load data
            train_loss_list, number_of_lists = self.storage.load_loss_csv(optimizer_name[entry], 'training') 
            val_loss_list, number_of_lists = self.storage.load_loss_csv(optimizer_name[entry], 'validation')
            accuracy_list = self.storage.load_accuracy_csv(optimizer_name[entry], 'accuracy') 
            #Calculate averages
            train_loss_average = self.calculator.calc_loss_average(train_loss_list, number_of_lists)
            val_loss_average = self.calculator.calc_loss_average(val_loss_list, number_of_lists)
            accuracy_average = self.calculator.calc_accuracy_average(accuracy_list, number_of_lists)
            # Add averages to lists
            train_loss_avg_list.append(train_loss_average)
            val_loss_avg_list.append(val_loss_average)
            accuracy_avg_list.append(accuracy_average)
        self.plotter.plot_loss(train_loss_avg_list, val_loss_avg_list, 0, self.epochs, 0, 1, 'Loss', 'Epochs / runs', 'Training', 'Loss', 'Benchmark_overview', 'training')
        self.plotter.plot_accuracy(accuracy_avg_list, None, None, 0, 100, 'Percent', None, 'Validation', 'Accuracy','Benchmark_overview', 'accuracy')

    # Executes the functions based on the menu selectection.
    # @selection      selected number    
    def selected_option(self, selection): 
        if selection == '1':
            print('You selected the SGD optimizer. Parameters are set to 10 epochs and to 0.001 as learning rate.')
            self.service.training(torch.optim.SGD(self.service.model.parameters(), lr=self.learning_rate), 'SGD', self.learning_rate, self.epochs)
            self.validate_and_safe('SGD', self.learning_rate, self.epochs)
        elif selection == '2':
            print('You selected the Adam optimizer. Parameters are set to 10 epochs and to 0.001 as learning rate.') 
            self.service.training(torch.optim.Adam(self.service.model.parameters(), lr=self.learning_rate), 'Adam', self.learning_rate, self.epochs)
            self.validate_and_safe('Adam', self.learning_rate, self.epochs)
        elif selection == '3':
            print('You selected the LAMB optimizer. Parameters are set to 10 epochs and to 0.001 as learning rate.') 
            self.service.training(optim.Lamb(self.service.model.parameters(), lr=self.learning_rate), 'LAMB', self.learning_rate, self.epochs)
            self.validate_and_safe('LAMB', self.learning_rate, self.epochs) 
        elif selection == '4':
            self.show_benchmark('SGD', 'Adam', 'LAMB')
        elif selection == '5':
            print('Bye, bye')
            sys.exit()