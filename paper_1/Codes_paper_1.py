# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import math
from scipy.stats import binom
import time



for outer in range(35):
    for middle in range(30):
        for inner in range(10):
            print("Inner loop:", inner, "Middle loop", middle, inner * 10 + middle)
            time.sleep(0.5)

partial_estimation = pd.read_csv("/content/Partial_table_estimate.csv")
partial_learning = pd.read_csv("/content/Partial_table_learning.csv")
partial_transfer = pd.read_csv("/content/Partial_table_transfer.csv")
complete_estimation = pd.read_csv("/content/Complete_table_estimate.csv")
complete_learning = pd.read_csv("/content/Complete_table_learning.csv")
complete_transfer = pd.read_csv("/content/Complete_table_transfer.csv")

count_A1 = 0
partial_learning.loc[:, 'chosen'] == 3
if partial_learning.loc[inner, 'chosen'] == 1:
    count_A1 += 1

partial_learning

def calculate_rates(data_frame):
    (Option1, Option3) = (1, 3)
    (Option2, Option4) = (2, 4)
    rates_summary = []
    counts_Option1 = 0
    counts_Option2 = 0
    counts_Option4 = 0
    counts_Option3 = 0
    for outer in range(35):
        for middle in range(30):
            for inner in range(10):
                index = inner + 10 * middle + 300 * outer
                chosen_option = data_frame.loc[index, 'chosen']
                if chosen_option == Option1:
                    counts_Option1 += 1
                elif chosen_option == Option2:
                    counts_Option2 += 1
                elif chosen_option == Option4:
                    counts_Option4 += 1
                elif chosen_option == Option3:
                    counts_Option3 += 1
            total_counts = counts_Option1 + counts_Option2 + counts_Option3 + counts_Option4
            average_rate_Option1 = counts_Option1 / 10
            average_rate_Option2 = counts_Option2 / 10
            average_rate_Option4 = counts_Option4 / 10
            average_rate_Option3 = counts_Option3 / 10
            rates_per_trial = {
                "ID": middle % 10,
                "rate1": average_rate_Option1,
                "rate2": average_rate_Option2,
                "rate3": average_rate_Option3,
                "rate4": average_rate_Option4
            }
            rates_summary.append(rates_per_trial)
            counts_Option1 = 0
            counts_Option2 = 0
            counts_Option4 = 0
            counts_Option3 = 0
    return pd.DataFrame(rates_summary)

result_df = calculate_rates(partial_learning)
print(result_df.head(50))

def compute_rates_per_trial(dataset):
    (Opt1, Opt3) = (1, 3)
    (Opt2, Opt4) = (2, 4)

    rates_list = []
    total_patients = 35
    total_trials = int(dataset.shape[0] / total_patients)

    print(total_trials)
    for patient_idx in range(total_patients):
        event_count = 0
        for trial_block in range(int(total_trials / 10)):
            Opt1_count, Opt2_count, Opt3_count, Opt4_count = 0, 0, 0, 0
            for trial in range(10):
                chosen_value = dataset.loc[patient_idx * total_trials + trial_block * 10 + trial, 'chosen']
                if chosen_value in (Opt1, Opt2, Opt3, Opt4):
                    event_count += 1
                    if chosen_value == Opt1:
                        Opt1_count += 1
                    elif chosen_value == Opt2:
                        Opt2_count += 1
                    elif chosen_value == Opt3:
                        Opt3_count += 1
                    elif chosen_value == Opt4:
                        Opt4_count += 1
            rates_per_trial = {
                "ID": patient_idx,
                "Trial Block": trial_block,
                "Count": event_count,
                "Opt1 Rate": Opt1_count / 10,
                "Opt2 Rate": Opt2_count / 10,
                "Opt3 Rate": Opt3_count / 10,
                "Opt4 Rate": Opt4_count / 10
            }
            rates_list.append(rates_per_trial)

    return pd.DataFrame(rates_list)

learning_rates_partial = compute_rates_per_trial(partial_learning)

transfer_rates_partial = compute_rates_per_trial(partial_transfer)

learning_rates_complete = compute_rates_per_trial(complete_learning)

transfer_rates_complete = compute_rates_per_trial(complete_transfer)

print("Partial DataFrame Columns:", learning_rates_partial.columns)
print("Complete DataFrame Columns:", learning_rates_complete.columns)

filtered_learning_rates_partial = learning_rates_partial[
    ~( (learning_rates_partial['Opt1 Rate'] == 0) & (learning_rates_partial['Opt2 Rate'] == 0) )
]

filtered_learning_rates_complete = learning_rates_complete[
    ~( (learning_rates_complete['Opt1 Rate'] == 0) & (learning_rates_complete['Opt2 Rate'] == 0) )
]

learning_rates_partial.sort_values("Trial Block", inplace=True)
data_to_plot_partial = pd.DataFrame([])
for i in range(30):
    data = learning_rates_partial[learning_rates_partial['Trial Block'] == i]
    data = data[['Opt1 Rate', 'Opt2 Rate', 'Opt3 Rate', 'Opt4 Rate']].mean(axis=0)
    data_to_plot_partial = pd.concat([data_to_plot_partial, data.to_frame().T], ignore_index=True)
data_to_plot_partial.index = range(30)
data_to_plot_partial.index.name = 'Trial Block'
print(data_to_plot_partial.head())

learning_rates_partial.sort_values("Trial Block", inplace=True)
aggregated_means_partial = pd.DataFrame()
for trial_index in range(30):
    trial_data = learning_rates_partial[learning_rates_partial['Trial Block'] == trial_index]
    mean_values = trial_data[['Opt1 Rate', 'Opt2 Rate', 'Opt3 Rate', 'Opt4 Rate']].mean().to_frame().T
    aggregated_means_partial = pd.concat([aggregated_means_partial, mean_values], ignore_index=True)
aggregated_means_partial.index = pd.RangeIndex(start=0, stop=30, step=1)
aggregated_means_partial.index.name = 'Trial Block'
print(aggregated_means_partial.head())

fig, ax = plt.subplots()
ax.plot(data_to_plot_partial.index, data_to_plot_partial['Opt1 Rate'], label='Opt1 Rate', color='blue', marker='o', linestyle='-')
ax.plot(data_to_plot_partial.index, data_to_plot_partial['Opt3 Rate'], label='Opt3 Rate', color='green', marker='x', linestyle='-')
ax.legend()
ax.set_title('Rate Comparisons Over Trials')
ax.set_xlabel('Trial Index')
ax.set_ylabel('Rates')
plt.show()

fig, ax = plt.subplots()
ax.plot(data_to_plot_partial.index, data_to_plot_partial['Opt2 Rate'], label='Opt2 Rate', color='red', marker='^', linestyle='--')
ax.plot(data_to_plot_partial.index, data_to_plot_partial['Opt4 Rate'], label='Opt4 Rate', color='purple', marker='s', linestyle='--')
ax.legend()
ax.set_title('Comparison of Opt2 and Opt4 Rates Across Trials')
ax.set_xlabel('Trial Number')
ax.set_ylabel('Rate Value')
plt.show()

try:
    data_to_plot_complete = learning_rates_complete.groupby('Trial Block').mean()
    print("DataFrame 'data_to_plot_complete' is ready for use.")
except NameError:
    print("DataFrame 'learning_rates_complete' is not defined. Please check data loading and processing steps.")

fig, ax = plt.subplots()
ax.plot(data_to_plot_complete.index, data_to_plot_complete['Opt1 Rate'], label='Opt1 Rate', color='navy', marker='o', linestyle='-')
ax.plot(data_to_plot_complete.index, data_to_plot_complete['Opt3 Rate'], label='Opt3 Rate', color='darkred', marker='x', linestyle='-')
ax.legend()
ax.set_title('Rate Comparison of Opt1 and Opt3 Across Complete Data Trials')
ax.set_xlabel('Trial Index')
ax.set_ylabel('Rate')
plt.show()

fig, ax = plt.subplots()
ax.plot(data_to_plot_complete.index, data_to_plot_complete['Opt2 Rate'], label='Opt2 Rate', color='teal', marker='^', linestyle='--')
ax.plot(data_to_plot_complete.index, data_to_plot_complete['Opt4 Rate'], label='Opt4 Rate', color='magenta', marker='s', linestyle=':')
ax.legend()
ax.set_title('Comparison of Opt2 and Opt4 Rates Over Complete Data Trials')
ax.set_xlabel('Trial Index')
ax.set_ylabel('Rate')
plt.show()

y_Opt1_mean_partial = data_to_plot_partial['Opt1 Rate'].mean()
y_Opt2_mean_partial = data_to_plot_partial['Opt2 Rate'].mean()
y_Opt3_mean_partial = data_to_plot_partial['Opt3 Rate'].mean()
y_Opt4_mean_partial = data_to_plot_partial['Opt4 Rate'].mean()

y_Opt1_mean_complete = data_to_plot_complete['Opt1 Rate'].mean()
y_Opt2_mean_complete = data_to_plot_complete['Opt2 Rate'].mean()
y_Opt3_mean_complete = data_to_plot_complete['Opt3 Rate'].mean()
y_Opt4_mean_complete = data_to_plot_complete['Opt4 Rate'].mean()

labels = ['Opt1', 'Opt2', 'Opt3', 'Opt4']
x_positions = [0, 1, 2, 3]
heights = [y_Opt1_mean_partial, y_Opt2_mean_partial, y_Opt3_mean_partial, y_Opt4_mean_partial]
plt.bar(x_positions, heights, color=['blue', 'green', 'red', 'purple'])
plt.xticks(x_positions, labels)
plt.title('Average Rates Comparison for Partial Data')
plt.xlabel('Rate Types')
plt.ylabel('Mean Rate Values')
plt.show()

labels = ['Opt1', 'Opt2', 'Opt3', 'Opt4']
x_positions = [0, 1, 2, 3]
heights = [y_Opt1_mean_complete, y_Opt2_mean_complete, y_Opt3_mean_complete, y_Opt4_mean_complete]
plt.bar(x_positions, heights, color=['cyan', 'magenta', 'yellow', 'grey'])
plt.xticks(x_positions, labels)
plt.title('Average Rates Comparison for Complete Data')
plt.xlabel('Types of Rates')
plt.ylabel('Average Rate Value')
plt.show()

learning_rates2_partial = learning_rates_partial.drop(columns=['ID', 'Count', 'Trial Block'])
learning_rates2_complete = learning_rates_complete.drop(columns=['ID', 'Count', 'Trial Block'])

ax = sns.barplot(data=learning_rates2_partial, ci="sd", palette=["green", "lime", "darkgreen", "limegreen"])
ax.set_title('Bar Plot of Learning Rates with Standard Deviation as Confidence Interval')
ax.set_xlabel('Rate Categories')
ax.set_ylabel('Values')
plt.show()

ax = sns.barplot(data=learning_rates2_complete, ci="sd", palette=["green", "lime", "darkgreen", "limegreen"])
ax.set_title('Complete Learning Rates with Standard Deviation as Confidence Interval')
ax.set_xlabel('Rate Types')
ax.set_ylabel('Average Rate Values')
plt.show()

ax = sns.barplot(data=learning_rates2_partial, palette=["green", "lime", "darkgreen", "limegreen"])
ax.set_title('Partial Learning Rates Overview')
ax.set_xlabel('Rate Types')
ax.set_ylabel('Average Rate Values')
plt.show()

ax = sns.barplot(data=learning_rates2_complete, palette=["green", "lime", "darkgreen", "limegreen"])
ax.set_title('Complete Learning Rates Overview')
ax.set_xlabel('Rate Categories')
ax.set_ylabel('Average Rate Values')
plt.show()

print(learning_rates2_complete.head())

ax.set_xticklabels(['Opt1', 'Opt2', 'Opt3', 'Opt4'])

y_A1_mean_partial = data_to_plot_partial['Opt1 Rate'].mean()
y_A2_mean_partial = data_to_plot_partial['Opt2 Rate'].mean()
y_B_mean_partial = data_to_plot_partial['Opt3 Rate'].mean()
y_C_mean_partial = data_to_plot_partial['Opt4 Rate'].mean()

labels = ['A1', 'A2', 'B', 'C']
x_positions = [0, 1, 2, 3]
heights = [y_A1_mean_partial, y_A2_mean_partial, y_B_mean_partial, y_C_mean_partial]
plt.bar(x_positions, heights, color=['blue', 'green', 'red', 'purple'])
plt.xticks(x_positions, labels)
plt.title('Average Rates Comparison for Partial Data')
plt.xlabel('Types of Rates')
plt.ylabel('Average Rate Value')
plt.show()

y_A1_mean_partial = data_to_plot_partial['Opt1 Rate'].mean()
y_A2_mean_partial = data_to_plot_partial['Opt2 Rate'].mean()
y_B_mean_partial = data_to_plot_partial['Opt3 Rate'].mean()
y_C_mean_partial = data_to_plot_partial['Opt4 Rate'].mean()
labels = ['A1', 'A2', 'B', 'C']
x_positions = [0, 1, 2, 3]
heights = [y_A1_mean_partial, y_A2_mean_partial, y_B_mean_partial, y_C_mean_partial]
plt.bar(x_positions, heights, color=['blue', 'green', 'red', 'purple'])
plt.xticks(x_positions, labels)
plt.title('Average Rates Comparison for Partial Data')
plt.xlabel('Types of Rates')
plt.ylabel('Average Rate Value')
plt.show()

y_A1_partial = data_to_plot_partial['Opt1 Rate']
y_A2_partial = data_to_plot_partial['Opt2 Rate']
y_B_partial = data_to_plot_partial['Opt3 Rate']
y_C_partial = data_to_plot_partial['Opt4 Rate']
y_A1_complete = data_to_plot_complete['Opt1 Rate']
y_A2_complete = data_to_plot_complete['Opt2 Rate']
y_B_complete = data_to_plot_complete['Opt3 Rate']
y_C_complete = data_to_plot_complete['Opt4 Rate']

y_A1_mean_partial = data_to_plot_partial['Opt1 Rate'].mean()
y_A2_mean_partial = data_to_plot_partial['Opt2 Rate'].mean()
y_B_mean_partial = data_to_plot_partial['Opt3 Rate'].mean()
y_C_mean_partial = data_to_plot_partial['Opt4 Rate'].mean()
y_A1_mean_complete = data_to_plot_complete['Opt1 Rate'].mean()
y_A2_mean_complete = data_to_plot_complete['Opt2 Rate'].mean()
y_B_mean_complete = data_to_plot_complete['Opt3 Rate'].mean()
y_C_mean_complete = data_to_plot_complete['Opt4 Rate'].mean()

plt.figure(figsize=(10, 5))
plt.bar([0, 1, 2, 3], height=[y_A1_mean_partial, y_A2_mean_partial, y_B_mean_partial, y_C_mean_partial], color=['blue', 'green', 'red', 'purple'])
plt.xticks([0, 1, 2, 3], ['Opt1', 'Opt2', 'Opt3', 'Opt4'])
plt.title('Average Rates Comparison for Partial Data')
plt.xlabel('Types of Rates')
plt.ylabel('Average Rate Value')
plt.show()

plt.figure(figsize=(10, 5))
plt.bar([0, 1, 2, 3], height=[y_A1_mean_complete, y_A2_mean_complete, y_B_mean_complete, y_C_mean_complete], color=['cyan', 'magenta', 'yellow', 'grey'])
plt.xticks([0, 1, 2, 3], ['Opt1', 'Opt2', 'Opt3', 'Opt4'])
plt.title('Average Rates Comparison for Complete Data')
plt.xlabel('Types of Rates')
plt.ylabel('Average Rate Value')
plt.show()

y_A1_partial = data_to_plot_partial['Opt1 Rate']
y_A2_partial = data_to_plot_partial['Opt2 Rate']
y_B_partial = data_to_plot_partial['Opt3 Rate']
y_C_partial = data_to_plot_partial['Opt4 Rate']
y_A1_complete = data_to_plot_complete['Opt1 Rate']
y_A2_complete = data_to_plot_complete['Opt2 Rate']
y_B_complete = data_to_plot_complete['Opt3 Rate']
y_C_complete = data_to_plot_complete['Opt4 Rate']

plt.figure(figsize=(10, 5))
plt.bar([0, 1, 2, 3], height=[y_A1_mean_partial, y_A2_mean_partial, y_B_mean_partial, y_C_mean_partial], color=['blue', 'green', 'red', 'purple'])
plt.xticks([0, 1, 2, 3], ['Opt1', 'Opt2', 'Opt3', 'Opt4'])
plt.title('Average Rates Comparison for Partial Data')
plt.xlabel('Types of Rates')
plt.ylabel('Average Rate Value')
plt.show()

plt.figure(figsize=(10, 5))
plt.bar([0, 1, 2, 3], height=[y_A1_mean_complete, y_A2_mean_complete, y_B_mean_complete, y_C_mean_complete], color=['cyan', 'magenta', 'yellow', 'grey'])
plt.xticks([0, 1, 2, 3], ['Opt1', 'Opt2', 'Opt3', 'Opt4'])
plt.title('Average Rates Comparison for Complete Data')
plt.xlabel('Types of Rates')
plt.ylabel('Average Rate Value')
plt.show()

y_A1_mean_partial = data_to_plot_partial['Opt1 Rate'].mean()
y_A2_mean_partial = data_to_plot_partial['Opt2 Rate'].mean()
y_B_mean_partial = data_to_plot_partial['Opt3 Rate'].mean()
y_C_mean_partial = data_to_plot_partial['Opt4 Rate'].mean()

y_A1_mean_complete = data_to_plot_complete['Opt1 Rate'].mean()
y_A2_mean_complete = data_to_plot_complete['Opt2 Rate'].mean()
y_B_mean_complete = data_to_plot_complete['Opt3 Rate'].mean()
y_C_mean_complete = data_to_plot_complete['Opt4 Rate'].mean()
y_A1_partial = data_to_plot_partial['Opt1 Rate']
y_A2_partial = data_to_plot_partial['Opt2 Rate']
y_B_partial = data_to_plot_partial['Opt3 Rate']
y_C_partial = data_to_plot_partial['Opt4 Rate']

y_A1_complete = data_to_plot_complete['Opt1 Rate']
y_A2_complete = data_to_plot_complete['Opt2 Rate']
y_B_complete = data_to_plot_complete['Opt3 Rate']
y_C_complete = data_to_plot_complete['Opt4 Rate']

plt.figure(figsize=(10, 5))
plt.bar([0, 1, 2, 3], height=[y_A1_mean_partial, y_A2_mean_partial, y_B_mean_partial, y_C_mean_partial], color=['blue', 'green', 'red', 'purple'])
plt.xticks([0, 1, 2, 3], ['Opt1', 'Opt2', 'Opt3', 'Opt4'])
plt.title('Average Rates Comparison for Partial Data')
plt.xlabel('Types of Rates')
plt.ylabel('Average Rate Value')
plt.show()

plt.figure(figsize=(10, 5))
plt.bar([0, 1, 2, 3], height=[y_A1_mean_complete, y_A2_mean_complete, y_B_mean_complete, y_C_mean_complete], color=['cyan', 'magenta', 'yellow', 'grey'])
plt.xticks([0, 1, 2, 3], ['Opt1', 'Opt2', 'Opt3', 'Opt4'])
plt.title('Average Rates Comparison for Complete Data')
plt.xlabel('Types of Rates')
plt.ylabel('Average Rate Value')
plt.show()

y_A1_partial = data_to_plot_partial['Opt1 Rate']
y_A2_partial = data_to_plot_partial['Opt2 Rate']

y_A1_complete = data_to_plot_complete['Opt1 Rate']
y_A2_complete = data_to_plot_complete['Opt2 Rate']

t_stat_partial, p_val_partial = ttest_ind(y_A2_partial, y_A1_partial)
t_stat_complete, p_val_complete = ttest_ind(y_A2_complete, y_A1_complete)

p_val_difference = p_val_partial - p_val_complete
print("Difference in p-values: ", p_val_difference)

n_trials_partial = int(y_A1_mean_partial)
p_success_partial = y_A2_mean_partial

mean, var, skew, kurt = binom.stats(n_trials_partial, p_success_partial, moments='mvsk')
print("Mean: ", mean)
print("Variance: ", var)
print("Skewness: ", skew)
print("Kurtosis: ", kurt)

n_trials = y_A1_partial.mean()
p_success = y_A2_partial.mean()

binom_mean = n_trials * p_success
print("Binomial Mean: ", binom_mean)

n_trials_partial = y_A1_partial.mean()
p_success_partial = y_A2_partial.mean()
bio_par = n_trials_partial * p_success_partial

n_trials_complete = y_A1_complete.mean()
p_success_complete = y_A2_complete.mean()
bio_comp = n_trials_complete * p_success_complete

bio_diff = bio_comp - bio_par
print("Difference in Binomial Means: ", bio_diff)

var_A1 = np.array(y_A1_partial).var()
var_A2 = np.array(y_A2_partial).var()
var_C = np.array(y_C_partial).var()
var_B = np.array(y_B_partial).var()

print("Variance A1:", var_A1)
print("Variance A2:", var_A2)
print("Variance B:", var_B)
print("Variance C:", var_C)

t_stat_partial, p_val_partial = ttest_ind(y_A1_partial, y_A2_partial)
print("T-statistic (partial):", t_stat_partial)
print("P-value (partial):", p_val_partial)

t_stat_complete, p_val_complete = ttest_ind(y_A1_complete, y_A2_complete)
print("T-statistic (complete):", t_stat_complete)
print("P-value (complete):", p_val_complete)

def calculate_confidence_interval(data1, data2, alpha=0.05):
    mean1, mean2 = np.mean(data1), np.mean(data2)
    var1, var2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
    n1, n2 = len(data1), len(data2)
    se = np.sqrt(var1/n1 + var2/n2)
    t_critical = stats.t.ppf(1 - alpha/2, df=n1 + n2 - 2)
    diff_mean = mean1 - mean2
    ci_low = diff_mean - t_critical * se
    ci_high = diff_mean + t_critical * se

    return (ci_low, ci_high)

ci_partial = calculate_confidence_interval(y_A1_partial, y_A2_partial)
print("95% Confidence interval for partial data:", ci_partial)

ci_complete = calculate_confidence_interval(y_A1_complete, y_A2_complete)
print("95% Confidence interval for complete data:", ci_complete)
