import csv, pathlib, os

class Storage():

    def __init__(self):
        self.save_path = './csv/'

    # Saves the values from the trainig runs and the calculated validation accuracy to a csv file.
    # @optimizer_name      Name of the selected optimizer. Necessaray to load the correct file.   
    # @kind                specifies the save file string either to training or validation.
    # @data_to_save        what kind of data to be saved: training_loss or validation accuracy
    def save_csv_file(self, optimizer_name, kind, data_to_save):
        try:
            os.mkdir(self.save_path)
        except FileExistsError:
            pass
        with open(f'{self.save_path}{optimizer_name}_{kind}.csv', 'a', newline='') as csv_validation:
            writer = csv.writer(csv_validation, delimiter=',')
            writer.writerow(data_to_save)
            data_to_save.clear()

    # Loads the saved training data csv file.
    # @optimizer_name      Name of the selected optimizer. Necessaray to load the correct file.
    # @kind                specifies the save file string either to training or validation.                
    # @return              a list for each row in the csv file, number of lists
    def load_loss_csv(self, optimizer_name, kind): 
        list_in_list = []    
  
        with open(f'{self.save_path}{optimizer_name}_{kind}.csv', newline='') as csv_loss:
            loss = csv.reader(csv_loss, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)          
            number_of_lists = 0
            data_list = []
            for row in loss:
                data_list.append(row)
                number_of_lists += 1
        list_in_list.append(data_list) 
      #  print(*list_in_list) 
        return list_in_list, number_of_lists

    # Loads the saved validation data csv file.
    # @optimizer_name      Name of the selected optimizer. Necessaray to load the correct file.
    # @kind                specifies the save file string either to training or validation.
    # @return              a list of all measured accuracys.
    def load_accuracy_csv(self, optimizer_name, kind):
        validation_list = []
        try:
            with open(f'{self.save_path}{optimizer_name}_{kind}.csv', newline='') as csv_validation:
                validation = csv.reader(csv_validation, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
                for row in validation:
                    validation_list.append(row)
        except FileExistsError:
            print('File not found.')            
        return validation_list