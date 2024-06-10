
# ----------------------
# Imports
# ----------------------
from Levenshtein import distance

# ----------------------
# Function Definitions
# ----------------------

def parse_data(file_path):
    # print(f"Opening file: {file_path}")
    # Open the file in read mode
    with open(file_path, 'r') as file:
        # Read the entire file content
        data = file.read()

    # Split the data into elements
    elements = data.split()

    # print(f"Finished parsing file: {file_path}")
    return elements

def calculate_recall_per_element(ground_truth_file, predicted_flow_file):
    # Read the ground truth and predicted flow txt files
    ground_truth_data = parse_data(ground_truth_file)
    predicted_flow_data = parse_data(predicted_flow_file)

    # Get unique elements in the ground truth data
    unique_elements = set(ground_truth_data)

    # Initialize a dictionary to store recall values for each unique element
    recall_values = {}

    # Iterate over the unique elements
    for element in unique_elements:
        # Count the total occurrences of the element in the ground truth data
        total_occurrences = ground_truth_data.count(element)

        # Count the correctly predicted occurrences of the element in the predicted flow data
        correctly_predicted_occurrences = predicted_flow_data.count(element)

        # Calculate the recall for the element
        recall = correctly_predicted_occurrences / total_occurrences

        # Store the recall value in the dictionary
        recall_values[element] = recall

    print("Finished recall calculation.")
    return recall_values



def calculate_miss_detection_rate(ground_truth_file, predicted_flow_file):
    # Read the ground truth and predicted flow txt files
    ground_truth_data = parse_data(ground_truth_file)
    predicted_flow_data = parse_data(predicted_flow_file)

    # Get unique elements in the ground truth data
    unique_elements = set(ground_truth_data)

    # Initialize a dictionary to store miss detection rate values for each unique element
    miss_detection_rate_values = {}

    # Iterate over the unique elements
    for element in unique_elements:
        # Count the total occurrences of the element in the ground truth data
        total_occurrences = ground_truth_data.count(element)

        # Count the correctly predicted occurrences of the element in the predicted flow data
        correctly_predicted_occurrences = predicted_flow_data.count(element)

        # Calculate the number of false negatives
        false_negatives = total_occurrences - correctly_predicted_occurrences

        # Calculate the miss detection rate for the element
        miss_detection_rate = false_negatives / total_occurrences

        # Store the miss detection rate value in the dictionary
        miss_detection_rate_values[element] = miss_detection_rate

    print("Finished miss detection rate calculation.")
    return miss_detection_rate_values


def calculate_false_detection_rate(ground_truth_file, predicted_flow_file):
    # Read the ground truth and predicted flow txt files
    ground_truth_data = parse_data(ground_truth_file)
    predicted_flow_data = parse_data(predicted_flow_file)

    # Get unique elements in the predicted flow data
    unique_elements = set(predicted_flow_data)

    # Initialize a dictionary to store false detection rate values for each unique element
    false_detection_rate_values = {}

    # Iterate over the unique elements
    for element in unique_elements:
        # Count the total occurrences of the element in the predicted flow data
        total_occurrences = predicted_flow_data.count(element)

        # Count the correctly predicted occurrences of the element in the ground truth data
        correctly_predicted_occurrences = ground_truth_data.count(element)

        # Calculate the number of false positives
        false_positives = total_occurrences - correctly_predicted_occurrences

        # Calculate the false detection rate for the element
        false_detection_rate = false_positives / total_occurrences

        # Store the false detection rate value in the dictionary
        false_detection_rate_values[element] = false_detection_rate

    print("Finished false detection rate calculation.")
    return false_detection_rate_values


def calculate_frame_wise_accuracy(ground_truth_file, predicted_flow_file):
    # Read the ground truth and predicted flow txt files
    ground_truth_data = parse_data(ground_truth_file)
    predicted_flow_data = parse_data(predicted_flow_file)

    # If the ground truth and predicted flow data have different lengths
    if len(ground_truth_data) != len(predicted_flow_data):
        # Determine which list is longer
        longer_list = ground_truth_data if len(ground_truth_data) > len(predicted_flow_data) else predicted_flow_data

        # Remove elements from the end of the longer list until they have the same length
        while len(ground_truth_data) != len(predicted_flow_data):
            longer_list.pop()

    # Initialize a counter for the number of correctly predicted frames
    correctly_predicted_frames = 0

    # Iterate over the frames
    for ground_truth_frame, predicted_frame in zip(ground_truth_data, predicted_flow_data):
        # Check if the predicted action matches the ground truth action
        if ground_truth_frame == predicted_frame:
            correctly_predicted_frames += 1

    # Calculate the frame-wise accuracy
    frame_wise_accuracy = correctly_predicted_frames / len(ground_truth_data)

    print("Finished frame-wise accuracy calculation.")
    return frame_wise_accuracy

def calculate_frame_wise_accuracy_for_each_element(ground_truth_file, predicted_flow_file):
    # Read the ground truth and predicted flow txt files
    ground_truth_data = parse_data(ground_truth_file)
    predicted_flow_data = parse_data(predicted_flow_file)

    # If the ground truth and predicted flow data have different lengths
    if len(ground_truth_data) != len(predicted_flow_data):
        # Determine which list is longer
        longer_list = ground_truth_data if len(ground_truth_data) > len(predicted_flow_data) else predicted_flow_data

        # Remove elements from the end of the longer list until they have the same length
        while len(ground_truth_data) != len(predicted_flow_data):
            longer_list.pop()

    # Get unique elements in the ground truth and predicted flow data
    unique_elements = set(ground_truth_data + predicted_flow_data)

    # Initialize a dictionary to store frame-wise accuracy values for each unique element
    frame_wise_accuracy_values = {}

    # Iterate over the unique elements
    for element in unique_elements:
        # Initialize a counter for the number of correctly predicted frames for the current element
        correctly_predicted_frames = 0

        # Get the total occurrences of the current element in the ground truth data
        total_occurrences = ground_truth_data.count(element)

        # Ensure that the total occurrences is not zero to avoid division by zero error
        if total_occurrences != 0:
            # Iterate over the frames
            for ground_truth_frame, predicted_frame in zip(ground_truth_data, predicted_flow_data):
                # Check if the current element is present in the frame and if the predicted action matches the ground truth action
                if ground_truth_frame == predicted_frame == element:
                    correctly_predicted_frames += 1

            # Calculate the frame-wise accuracy for the current element
            frame_wise_accuracy = correctly_predicted_frames / total_occurrences

            # Store the frame-wise accuracy value in the dictionary
            frame_wise_accuracy_values[element] = frame_wise_accuracy
        else:
            # If total occurrences is zero, set the frame-wise accuracy to a default value (e.g., 0 or NaN)
            frame_wise_accuracy_values[element] = 0

    print("Finished frame-wise accuracy calculation for each element.")
    return frame_wise_accuracy_values



from Levenshtein import distance

def calculate_segmental_edit_distance(ground_truth_file, predicted_flow_file):
    # Read the ground truth and predicted flow txt files
    ground_truth_data = parse_data(ground_truth_file)
    predicted_flow_data = parse_data(predicted_flow_file)

    # Calculate the segmental edit distance
    segmental_edit_distance = distance(''.join(ground_truth_data), ''.join(predicted_flow_data))

    print("Finished segmental edit distance calculation.")
    return segmental_edit_distance


def calculate_segmental_f1_score(ground_truth_file, predicted_flow_file, thresholds=[10, 25, 50]):
    # Read the ground truth and predicted flow txt files
    ground_truth_data = parse_data(ground_truth_file)
    predicted_flow_data = parse_data(predicted_flow_file)

    # Initialize a dictionary to store the F1 scores for each threshold
    f1_scores = {}

    # Iterate over the thresholds
    for threshold in thresholds:
        # Calculate the number of true positives, false positives, and false negatives
        TP = sum([1 for gt, pred in zip(ground_truth_data, predicted_flow_data) if gt == pred and gt != ' '])
        FP = sum([1 for gt, pred in zip(ground_truth_data, predicted_flow_data) if gt != pred and pred != ' '])
        FN = sum([1 for gt, pred in zip(ground_truth_data, predicted_flow_data) if gt != pred and gt != ' '])

        # Calculate the precision and recall
        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0

        # Calculate the F1 score
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        # Store the F1 score in the dictionary
        f1_scores[f'F1@{threshold}'] = f1_score

    print("Finished segmental F1 score calculation.")
    return f1_scores

def calculate_mean_over_frames(ground_truth_file, predicted_flow_file):
    # Read the ground truth and predicted flow txt files
    ground_truth_data = parse_data(ground_truth_file)
    predicted_flow_data = parse_data(predicted_flow_file)

    # Calculate the metric for each frame
    frame_values = [1 if gt == pred else 0 for gt, pred in zip(ground_truth_data, predicted_flow_data)]

    # Calculate the mean over frames
    mean_over_frames = sum(frame_values) / len(frame_values)

    print("Finished mean calculation over frames.")
    return mean_over_frames



# --------------------------------
# Function Calls
# --------------------------------

from prettytable import PrettyTable

predicted_flow_file = 'SR2_LR0.0001_MR0.3/result/snatch/split_1/Snatch_1_flow'
ground_truth_file = 'snatch/groundTruth/Snatch_1_flow.txt'

# Initialize the table
table = PrettyTable()
table.field_names = ["Metric", "Element", "Value"]

# recall_values = calculate_recall_per_element(ground_truth_file, predicted_flow_file)
# for element, recall in recall_values.items():
#     table.add_row(["Recall", element, recall])

# miss_detection_rate_values = calculate_miss_detection_rate(ground_truth_file, predicted_flow_file)
# for element, miss_detection_rate in miss_detection_rate_values.items():
#     table.add_row(["Miss Detection Rate", element, miss_detection_rate])

# false_detection_rate_values = calculate_false_detection_rate(ground_truth_file, predicted_flow_file)
# for element, false_detection_rate in false_detection_rate_values.items():
#     table.add_row(["False Detection Rate", element, false_detection_rate])

# frame_wise_accuracy = calculate_frame_wise_accuracy(ground_truth_file, predicted_flow_file)
# table.add_row(["Frame-wise Accuracy", "Overall", frame_wise_accuracy])

# frame_wise_accuracy_values = calculate_frame_wise_accuracy_for_each_element(ground_truth_file, predicted_flow_file)
# for element, frame_wise_accuracy in frame_wise_accuracy_values.items():
#     table.add_row(["Frame-wise Accuracy", element, frame_wise_accuracy])

# segmental_edit_distance = calculate_segmental_edit_distance(ground_truth_file, predicted_flow_file)
# table.add_row(["Segmental Edit Distance", "Overall", segmental_edit_distance])

# f1_scores = calculate_segmental_f1_score(ground_truth_file, predicted_flow_file)
# for threshold, f1_score in f1_scores.items():
#     table.add_row(["F1 Score", threshold, f1_score])

# mean_over_frames = calculate_mean_over_frames(ground_truth_file, predicted_flow_file)
# table.add_row(["Mean Over Frames", "Overall", mean_over_frames])

# # Save the table to a file
# with open('[SR2LR0.00001Mr0.3]Snatch_1_flow.txt', 'w') as file:
#     file.write(str(table))

# # Print a success message
# print("Table saved to output.txt")







# import os
import pandas as pd

# # Define the directory path
# dir_path = 'SR2_LR0.0001_MR0.3/result/snatch/split_1/'

# # Get a list of all files in the directory
# all_files = os.listdir(dir_path)

# # Filter the list to only include .txt files
# txt_files = [file for file in all_files if file.endswith('.txt')]

# # Initialize an empty DataFrame
# df = pd.DataFrame()

# # Loop over the list of .txt files
# for txt_file in txt_files:
#     # Read the .txt file into a DataFrame
#     file_df = pd.read_csv(os.path.join(dir_path, txt_file), sep="\t", header=None)
    
#     # Append the file DataFrame to the main DataFrame
#     df = df.append(file_df, ignore_index=True)

# # Return the main DataFrame
# df.columns = ['metrics']
# df

# Initialize the dictionary
metrics = {}

recall_values = calculate_recall_per_element(ground_truth_file, predicted_flow_file)
for element, recall in recall_values.items():
    metrics[f"Recall_{element}"] = recall

miss_detection_rate_values = calculate_miss_detection_rate(ground_truth_file, predicted_flow_file)
for element, miss_detection_rate in miss_detection_rate_values.items():
    metrics[f"Miss Detection Rate_{element}"] = miss_detection_rate

false_detection_rate_values = calculate_false_detection_rate(ground_truth_file, predicted_flow_file)
for element, false_detection_rate in false_detection_rate_values.items():
    metrics[f"False Detection Rate_{element}"] = false_detection_rate

frame_wise_accuracy = calculate_frame_wise_accuracy(ground_truth_file, predicted_flow_file)
metrics["Frame-wise Accuracy_Overall"] = frame_wise_accuracy

frame_wise_accuracy_values = calculate_frame_wise_accuracy_for_each_element(ground_truth_file, predicted_flow_file)
for element, frame_wise_accuracy in frame_wise_accuracy_values.items():
    metrics[f"Frame-wise Accuracy_{element}"] = frame_wise_accuracy

segmental_edit_distance = calculate_segmental_edit_distance(ground_truth_file, predicted_flow_file)
metrics["Segmental Edit Distance_Overall"] = segmental_edit_distance

f1_scores = calculate_segmental_f1_score(ground_truth_file, predicted_flow_file)
for threshold, f1_score in f1_scores.items():
    metrics[f"F1 Score_{threshold}"] = f1_score

mean_over_frames = calculate_mean_over_frames(ground_truth_file, predicted_flow_file)
metrics["Mean Over Frames_Overall"] = mean_over_frames

# Convert the dictionary to a DataFrame
df = pd.DataFrame(metrics, index=[0])

# Save the DataFrame to a file
df.to_csv('[SR2LR0.00001Mr0.3]Snatch_1_flow.csv', index=False)

# Print a success message
print("Table saved to output.txt")

df.to_csv('test1.csv', index=False)



