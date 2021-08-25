import sys, os, pathlib, torch, time, math
import torch_optimizer as optim
#import warnings
#from torch.serialization import SourceChangeWarning
from service import Service
from plotter import Plotter
from storage import Storage
from calculator import Calculator

# Contains all menu relevant functions.
class Menu():
    def __init__(self):       
        self.service = Service()
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
            print('check the performances of the different optimizers under point 4 (batch size = 64)')
            print('and 5 (batch size = 512).\n')
            print('1. SGD')
            print('2. Adam')
            print('3. LAMB')
            print('4. 1. Test series benchmarks (batch = 64)')
            print('5. 2. Test series benchmarks (batch = 512)')
            print('----------------------------\n')
            print('6. Exit\n')
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
    def validate_and_save(self, optimizer_name, learning_rate, batch_size):
        self.service.validate_cifar(self.service.epochs)
        self.storage.save_csv_file(self.service.optimizer_name, 'training', self.service.training_loss, self.service.batch_size)
        self.storage.save_csv_file(self.service.optimizer_name, 'validation', self.service.validation_loss, self.service.batch_size)
        self.storage.save_csv_file(self.service.optimizer_name, 'accuracy', self.service.validation_accuracy, self.service.batch_size)
        print('Data is saved.\n')

    # Calls functions to show the selected optimizer benchmark.
    # @optimizer_name    sequence of all optimizers.
    # @batch_size        
    def show_benchmark(self, batch_size, *optimizer_name):
        train_loss_avg_list = []
        val_loss_avg_list = []
        accuracy_avg_list = [] 
        y_axis_range = [] 
        for entry in range(len(optimizer_name)):
            # Load data
            train_loss_list, number_of_lists = self.storage.load_loss_csv(optimizer_name[entry], 'training', batch_size) 
            val_loss_list, number_of_lists = self.storage.load_loss_csv(optimizer_name[entry], 'validation', batch_size)
            accuracy_list = self.storage.load_accuracy_csv(optimizer_name[entry], 'accuracy', batch_size) 
            #Calculate averages
            train_loss_average = self.calculator.calc_loss_average(train_loss_list, number_of_lists)
            val_loss_average = self.calculator.calc_loss_average(val_loss_list, number_of_lists)
            accuracy_average = self.calculator.calc_accuracy_average(accuracy_list, number_of_lists)
            # Add averages to lists
            train_loss_avg_list.append(train_loss_average)
            val_loss_avg_list.append(val_loss_average)
            accuracy_avg_list.append(accuracy_average)
            y_axis_range.append(val_loss_average[-1])
            y_axis_range.sort()
        self.plotter.plot_loss(train_loss_avg_list, val_loss_avg_list, 0, self.service.epochs + 2, 0, y_axis_range[-1] + 0.02, 'Loss', 'Epochs', 'Training', 'Loss', 'Benchmark_overview', 'training', self.service.epochs)
        self.plotter.plot_accuracy(accuracy_avg_list, None, None, 0, 100, 'Percent', 'Run', 'Validation', 'Accuracy','Benchmark_overview', 'accuracy', self.service.epochs)

    # Executes the functions based on the menu selectection.
    # @selection      selected number    
    def selected_option(self, selection): 
        if selection == '1':
            self.service.training(torch.optim.SGD(self.service.model.parameters(), lr=self.service.learning_rate), 'SGD', self.service.learning_rate, self.service.epochs)
            self.validate_and_save('SGD', self.service.learning_rate, self.service.batch_size)
        elif selection == '2':
            self.service.training(torch.optim.Adam(self.service.model.parameters(), lr=self.service.learning_rate), 'Adam', self.service.learning_rate, self.service.epochs)
            self.validate_and_save('Adam', self.service.learning_rate, self.service.batch_size)
        elif selection == '3':
            self.service.training(optim.Lamb(self.service.model.parameters(), lr=self.service.learning_rate), 'LAMB', self.service.learning_rate, self.service.epochs)
            self.validate_and_save('LAMB', self.service.learning_rate, self.service.batch_size) 
        elif selection == '4':
            self.show_benchmark(64, 'SGD', 'Adam', 'LAMB')
        elif selection == '5':
            self.show_benchmark(512, 'SGD', 'Adam', 'LAMB')
        elif selection == '6':
            print('Bye, bye')
            sys.exit()
