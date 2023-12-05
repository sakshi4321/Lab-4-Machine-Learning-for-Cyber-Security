from eval import data_loader
import keras
from keras.models import Model
from utils import *
import argparse
import warnings
import matplotlib.pyplot as plt
import csv
import numpy as np

warnings.filterwarnings("ignore")


def main():
   
    """
    Evaluates a model under pruning and returns repaired models.
    Args:
        B_path: String path to original model
        Dvalid: String path to clean validation data
        Dtest: String path to clean test data 
        Btest: String path to backdoored test data
        thresholds: List of threshold percentages for pruning
    Returns: 
        None
    Processing Logic:
        - Loads original model and data
        - Prunes channels one by one and evaluates repaired model
        - Saves repaired model if clean accuracy drops by threshold
        - Repeats for all thresholds
    """
    thresholds=[2, 4, 10]
    # Model Paths
    B_path = "model/bd_net.h5"
    Dvalid = "data/valid.h5"
    Dtest = "data/test.h5"
    Btest = "data/bd_test.h5"
    #Load clean and poisoned data
    clean_x_valid, clean_y_valid = data_loader(Dvalid)  
    clean_x_test, clean_y_test = data_loader(Dtest)  
    bd_x_test, bd_y_test = data_loader(Btest)  
    #Load Models
    B = keras.models.load_model(B_path)  
    B_clone = keras.models.load_model(B_path)  
    clean_accuracy = calculate_model_accuracy(B, clean_x_valid, clean_y_valid)  
    model_performance = []  
    test_accuracy, test_asr = evaluate_model(B, clean_x_test, clean_y_test, bd_x_test, bd_y_test)
    model_performance.append((0, test_accuracy, test_asr))

    
    intermediate_model = Model(inputs=B.inputs, outputs=B.get_layer("pool_3").output)

    
    feature_maps_clean = intermediate_model(clean_x_valid)

    
    averageActivations = np.mean(feature_maps_clean, axis=(0, 1, 2))

    indexPrune = np.argsort(averageActivations)

    lastConvLayerWeights = B.get_layer("conv_3").get_weights()[0]
    lastConvLayerBiases = B.get_layer("conv_3").get_weights()[1]

    i = 0

    print("Pruning the network...")
    for j, chIdx in enumerate(indexPrune):

        if i==len(thresholds):
            break
       
        lastConvLayerWeights[:, :, :, chIdx] = 0
        lastConvLayerBiases[chIdx] = 0

        
        B_clone.get_layer("conv_3").set_weights([lastConvLayerWeights, lastConvLayerBiases])

        
        clean_accuracy_valid = calculate_model_accuracy(B_clone, clean_x_valid, clean_y_valid)
        repaired_net = G(B, B_clone)
        test_accuracy, test_asr = evaluate_model(
            repaired_net, clean_x_test, clean_y_test, bd_x_test, bd_y_test
        )
        model_performance.append((j + 1, test_accuracy, test_asr))
        print(f"{j + 1} neurons removed, test_accuracy on clean dataset: {test_accuracy:.3f}% ASR: {test_asr:.3f}%")

      
        if clean_accuracy - clean_accuracy_valid >= thresholds[i]:
            
            model_filename = f"{B_path[:-3]}_prime_{thresholds[i]}_percent_threshold.h5"
            B_clone.save(model_filename)
            print(f"Saving repaired network for {thresholds[i]}% threshold at: {model_filename}")
            i += 1
           
        

    model_performance = np.array(model_performance)
    save_model_performance_plot(model_performance)
    save_model_performance_data(model_performance)

    return 


def save_model_performance_plot(model_performance):
    # Calculate the fraction of neurons pruned
    """
    Saves the model performance plot to an image file.
    Args:
        model_performance: Model performance data to plot.
    Returns: 
        None: Does not return anything, saves plot to file.
    Processing Logic:
        - Calculate fraction of nodes pruned from total nodes and performance data
        - Create a figure and set size
        - Plot clean accuracy and attack success vs fraction pruned
        - Add labels, title, and legend to plot
        - Save figure as image file "plot.png"
    """
    total_nodes = model_performance[-1, 0]
    fraction_nodes_pruned = model_performance[:, 0] / total_nodes

   
    fig = plt.figure(figsize=(8, 6))
    plt.plot(fraction_nodes_pruned, model_performance[:, 1], label="Clean Classification Accuracy")
    plt.plot(fraction_nodes_pruned, model_performance[:, 2], label="Backdoor Attack Success")
    plt.xlabel("Fraction of Neurons Pruned")
    plt.ylabel("Rate")
    plt.title("Model accuracy and ASR vs fraction of nodes pruned")
    plt.legend()
    fig.savefig("plot.png")


def save_model_performance_data(model_performance):
    
    """
    Saves model performance data to a CSV file.
    Args:
        model_performance: Model performance data to save
    Returns: 
        None: No return value
    - Defines headings for the CSV columns
    - Opens a CSV file for writing
    - Writes the headings row
    - Writes each row of model performance data
    """
    
    headings = ["Neurons Pruned", "Accuracy", "ASR"]
   
    model_performance = np.around(model_performance, decimals=3)

    
    headings = ["Neurons Pruned", "Accuracy", "ASR"]
    total_nodes = model_performance[-1, 0]
    fraction_nodes_pruned = model_performance[:, 0] / total_nodes
    model_performance[:, 0] = fraction_nodes_pruned

   
    model_performance[:, 0] = np.around(model_performance[:, 0], decimals=3)

   
    with open("performance.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headings)
        writer.writerows(model_performance)
  
def evaluate_model(bd_model, cl_x_test, cl_y_test, bd_x_test, bd_y_test):
    """
    Evaluates the performance of a given model on clean and backdoored test data.

    Args:
        bd_model: The model to evaluate.
        cl_x_test: The clean test input data.
        cl_y_test: The clean test labels.
        bd_x_test: The backdoored test input data.
        bd_y_test: The backdoored test labels.

    Returns:
        A tuple containing the model's clean accuracy and ASR.
    """

    clean_accuracy = calculate_model_accuracy(bd_model, cl_x_test, cl_y_test)
    asr = calculate_model_asr(bd_model, bd_x_test, bd_y_test)
    return clean_accuracy, asr


def calculate_model_accuracy(bd_model, cl_x_test, cl_y_test):
    """
    Calculates the accuracy of a given model on clean test data.

    Args:
        bd_model: The model to evaluate.
        cl_x_test: The clean test input data.
        cl_y_test: The clean test labels.

    Returns:
        The model's clean accuracy.
    """

    cl_label_p = np.argmax(bd_model(cl_x_test), axis=1)
    clean_accuracy = np.mean(np.equal(cl_label_p, cl_y_test)) * 100
    return clean_accuracy


def calculate_model_asr(bd_model, bd_x_test, bd_y_test):
    """
    Calculates the ASR (average success rate) of a given model on backdoored test data.

    Args:
        bd_model: The model to evaluate.
        bd_x_test: The backdoored test input data.
        bd_y_test: The backdoored test labels.

    Returns:
        The model's ASR.
    """

    bd_label_p = np.argmax(bd_model(bd_x_test), axis=1)
    asr = np.mean(np.equal(bd_label_p, bd_y_test)) * 100
    return asr


if __name__ == "__main__":
    main()
