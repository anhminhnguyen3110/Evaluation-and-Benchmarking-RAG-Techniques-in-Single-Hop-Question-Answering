import glob
import json
from matplotlib.ticker import PercentFormatter
import pandas as pd
import matplotlib.pyplot as plt
import os

pd.set_option('display.max_rows', None)  # Set to None to print all rows
pd.set_option('display.max_columns', None)  # Set to None to print all columns
pd.set_option('display.width', None)  # Make sure the width is wide enough
pd.set_option('display.max_colwidth', None)  # Remove truncation for column content

# Create the result_visualiser directory if it doesn't exist
output_dir = 'result_visualiser'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Initialize an empty list to hold the data
data_list = []

# Use glob to get all JSON files in the result folder
json_files = glob.glob('result/*.json')

# Iterate over the files
for file in json_files:
    # Get the evaluator name from the filename
    evaluator_name = os.path.splitext(os.path.basename(file))[0].replace('evaluator_', '').replace('_results', '')
    with open(file, 'r') as f:
        data = json.load(f)
        # data is a dictionary with datasets as keys
        for dataset, dataset_data in data.items():
            # Start with overall metrics
            overall_metrics = {}
            overall_metrics['evaluator'] = evaluator_name
            overall_metrics['dataset'] = dataset
            overall_metrics['metric_scope'] = 'full'
            # From 'squad_v2'
            if 'squad_v2' in dataset_data:
                squad_v2 = dataset_data['squad_v2']
                overall_metrics['exact_match'] = squad_v2.get('exact_match')
            # From 'scorer_result'
            if 'scorer_result' in dataset_data:
                scorer_result = dataset_data['scorer_result']
                for key, value in scorer_result.items():
                    overall_metrics[key] = value
            # From 'ai_evaluator'
            if 'ai_evaluator' in dataset_data:
                ai_evaluator = dataset_data['ai_evaluator']
                # ai_evaluator is a list containing lists
                # Get the overall ragas metrics
                try:
                    overall_ragas = ai_evaluator[0][1]['ragas']
                    for key, value in overall_ragas.items():
                        overall_metrics[key] = value
                except (IndexError, KeyError):
                    pass  # Handle missing data
            data_list.append(overall_metrics)
            # Now extract metrics per range from 'scorer_e_result'
            if 'scorer_e_result' in dataset_data:
                scorer_e_result = dataset_data['scorer_e_result']
                for range_key, range_metrics in scorer_e_result.items():
                    range_data = {}
                    range_data['evaluator'] = evaluator_name
                    range_data['dataset'] = dataset
                    range_data['metric_scope'] = range_key
                    for key, value in range_metrics.items():
                        range_data[key] = value
                    data_list.append(range_data)
            # Now extract per-range metrics from 'ai_evaluator'
            if 'ai_evaluator' in dataset_data:
                ai_evaluator = dataset_data['ai_evaluator']
                # The per-range data is in ai_evaluator[0][0], which is a dict with ranges
                try:
                    per_range_data = ai_evaluator[0][0]
                    for range_key, range_value in per_range_data.items():
                        ragas_metrics = range_value.get('ragas', {})
                        range_data = {}
                        range_data['evaluator'] = evaluator_name
                        range_data['dataset'] = dataset
                        range_data['metric_scope'] = range_key
                        for key, value in ragas_metrics.items():
                            range_data[key] = value
                        data_list.append(range_data)
                except (IndexError, KeyError):
                    pass  # Handle missing data

# Now, create a DataFrame from the data_list
df = pd.DataFrame(data_list)

# Since we may have multiple entries for the same evaluator, dataset, and metric_scope,
# we can group them and take the first non-null value for each metric
df = df.groupby(['evaluator', 'dataset', 'metric_scope']).first().reset_index()

# Ensure that all combinations of evaluators, datasets, and metric_scopes are represented
evaluators = df['evaluator'].unique()
datasets = df['dataset'].unique()
metric_scopes = df['metric_scope'].unique()

# Create a MultiIndex to ensure all combinations are present
index = pd.MultiIndex.from_product([evaluators, datasets, metric_scopes], names=['evaluator', 'dataset', 'metric_scope'])

# Reindex the DataFrame to include all combinations, filling missing values with NaN
df = df.set_index(['evaluator', 'dataset', 'metric_scope']).reindex(index).reset_index()

# List of metrics
metrics = ['f1', 'exact_match', 'qa_f1_score', 'qa_precision', 'qa_recall', 'rouge_score',
           'answer_correctness', 'answer_similarity', 'answer_relevancy', 'faithfulness',
           'context_precision', 'context_recall']

# For each metric and each range, create a table and bar chart
for metric in metrics:
    for scope in metric_scopes:
        # Create a subset for this metric and scope
        subset = df[df['metric_scope'] == scope]
        if metric in subset.columns:
            pivot_table = subset.pivot_table(index='dataset', columns='evaluator', values=metric, aggfunc='first')
            
            if not pivot_table.empty:
                pivot_table = pivot_table.fillna(0)
                print(f"\nMetric: {metric} - Range: {scope}")
                print(pivot_table)

                if 'full' in scope:
                    title = f"{metric.capitalize()}"
                else :
                    title = f"{metric.capitalize()} ({scope})"
                ax = pivot_table.plot(kind='bar', width=0.9, title=title, figsize=(14, 7))
                

                plt.xlabel("")
                plt.ylabel("")
                plt.xticks(rotation=0)  # Set x-axis labels to horizontal
                plt.tight_layout()

                for container in ax.containers:
                    ax.bar_label(container, fmt='%.1f%%', fontsize=7.3, padding=3)

                max_value = pivot_table.max().max()  
                ymax = min(max_value + 10, 100)  # Adding 10% but not more than 100%
                ax.set_ylim(0, ymax)  # Set the y-limit to the calculated max

                ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
                ax.set_yticks(range(0, int(ymax) + 1, 10))  # Y-axis ticks at every 10%

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)

                ax.grid(True, which='major', axis='y', linestyle='-', linewidth=0.7, alpha=0.5)

                image_filename = f"{metric}_range_{scope}.png".replace('/', '_')
                ax.legend(loc='best', fontsize='small', ncol=2)
                if 'full' in scope:
                    full_output_dir = 'result_visualiser/full'
                    if not os.path.exists(full_output_dir):
                        os.makedirs(full_output_dir)
                    plt.savefig(os.path.join(full_output_dir, image_filename), bbox_inches='tight', pad_inches=0.1)
                else:
                    plt.savefig(os.path.join(output_dir, image_filename), bbox_inches='tight', pad_inches=0.1)
                plt.close()
