import math

class Calculator():

    # Calculates the average of all training runs of a specific optimizer.
    # @value_list       list of all values to be calculated.    
    def calc_loss_average(self, loss_list, number_of_lists):
        average_list = []
        for j in range(len(loss_list[0][0])):
            element = 0.0           
            for i in range(len(loss_list[0])):
                element += float(loss_list[0][i][j])                 
            average_list.append(element / float(number_of_lists)) 
        return average_list
 
    # Calculates the average of all saved training cycles.
    # @accuracy_list     list of all saved accuracies. 
    # @return            average of all training cycles.
    def calc_accuracy_average(self, accuracy_list, number_of_lists):
        sum = 0.0   
        for entry in range(len(accuracy_list)):
            sum += accuracy_list[entry][0]      
        return sum / float(number_of_lists)