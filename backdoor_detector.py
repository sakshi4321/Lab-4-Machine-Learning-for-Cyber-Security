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
    for j, idx in enumerate(indexPrune):

        if i==len(thresholds)-2:
            break
       
        lastConvLayerWeights[:, :, :, idx] = 0
        lastConvLayerBiases[idx] = 0

        
        B_clone.get_layer("conv_3").set_weights([lastConvLayerWeights, lastConvLayerBiases])

        
        clean_accuracy_valid = calculate_model_accuracy(B_clone, clean_x_valid, clean_y_valid)
        repaired_net = G(B, B_clone)
        test_accuracy, test_asr = evaluate_model(repaired_net, clean_x_test, clean_y_test, bd_x_test, bd_y_test)
        model_performance.append((j + 1, test_accuracy, test_asr))
        print(f"{j + 1} neurons were removed, test_accuracy on clean dataset: {test_accuracy:.3f}%, ASR: {test_asr:.3f}%")

        #save model if threshhold is reached
        if clean_accuracy - clean_accuracy_valid >= thresholds[i]:
            
            model_filename = f"{B_path[:-3]}_prime_{thresholds[i]}_percent_threshold.h5"
            B_clone.save(model_filename)
            print(f"Saving repaired network for {thresholds[i]}% threshold at: {model_filename}")
            i += 1
           
        

    model_performance = np.array(model_performance)
    save_model_performance(model_performance)
   

    return 


def save_model_performance(model_performance):
  
    
    # Calculate fraction of nodes pruned
    total_nodes = model_performance[-1, 0]
    fraction_nodes_pruned = model_performance[:, 0] / total_nodes

    # Create plot
    fig = plt.figure(figsize=(8, 6))
    plt.plot(fraction_nodes_pruned, model_performance[:, 1], label="Clean Classification Accuracy")
    plt.plot(fraction_nodes_pruned, model_performance[:, 2], label="Backdoor Attack Success")
    plt.xlabel("Fraction of Neurons Pruned")
    plt.ylabel("Rate")
    plt.title("Model accuracy and ASR vs fraction of nodes pruned")
    plt.legend()

     # Save figure
    fig.savefig("plot.png")

    #save to csv for table
    headings = ["Neurons Pruned", "Accuracy", "ASR"]
   
   #just get values till 3 decimal places
    model_performance = np.around(model_performance, decimals=3)  
    total_nodes = model_performance[-1, 0]
    fraction_nodes_pruned = model_performance[:, 0] / total_nodes
    model_performance[:, 0] = fraction_nodes_pruned

   
    model_performance[:, 0] = np.around(model_performance[:, 0], decimals=3)

   
    with open("performance.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headings)
        writer.writerows(model_performance)



    

    
  
def evaluate_model(bd_model, clean_x_test, clean_y_test, bd_x_test, bd_y_test):
   

    """
    Evaluates a model on clean and backdoored test data
    Args:
        bd_model: Backdoored model to evaluate
        clean_x_test: Clean test features 
        clean_y_test: Clean test labels
        bd_x_test: Backdoored test features
        bd_y_test: Backdoored test labels
    Returns: 
        clean_accuracy, asr: Clean accuracy and attack success rate
    - Calculates clean accuracy on clean test data
    - Calculates attack success rate on backdoored test data 
    - Returns both metrics
    """
    clean_accuracy = calculate_model_accuracy(bd_model, clean_x_test, clean_y_test)
    asr = calculate_model_asr(bd_model, bd_x_test, bd_y_test)
    return clean_accuracy, asr


def calculate_model_accuracy(bd_model, clean_x_test, clean_y_test):
    
    """
    Calculate accuracy of a model on clean test data
    Args:
        bd_model: Trained model
        clean_x_test: Clean test input data 
        clean_y_test: Clean test label data
    Returns:
        clean_accuracy: Accuracy percentage of model on clean test data
    Processing Logic:
        - Predict labels for clean test data using the model
        - Compare predicted labels with actual clean test labels
        - Calculate percentage of predictions that matched actual labels
        - Return the accuracy percentage
    
    """

    # Predict labels for clean test data
    predicted_labels = np.argmax(bd_model(clean_x_test), axis=1)
    
    # Compare predicted labels with actual labels 
    matches = np.equal(predicted_labels, clean_y_test)
    
    # Calculate accuracy percentage
    accuracy = np.mean(matches) * 100
    
    return accuracy


def calculate_model_asr(bd_model, bd_x_test, bd_y_test):
    
    """Calculates the accuracy score of a model on test data
    Args:
        bd_model: The trained model
        bd_x_test: The test data inputs
        bd_y_test: The test data true labels
    Returns: 
        asr: The accuracy score of the model on test data as a percentage
    Processing Logic:
        - Predict labels for test data using the model
        - Compare predicted labels to true labels
        - Calculate percentage of predictions that matched true labels
        - Return the accuracy score as a percentage
    
    """
    
    # Predict labels for test data
    predicted_labels = np.argmax(bd_model(bd_x_test), axis=1)
    
    # Compare predicted labels to true labels
    matches = np.equal(predicted_labels, bd_y_test)
    
    # Calculate percentage of predictions that matched true labels
    accuracy_score = np.mean(matches) * 100
    
    return accuracy_score



if __name__ == "__main__":
    main()
