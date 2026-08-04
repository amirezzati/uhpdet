import os
import json
import itertools
import argparse
import pandas as pd

# metrics
def agreement_w_question(question_pred, group_preds, type='a'):
    N = len(group_preds)
    if type == 'a': # affirmative group
        score = sum(1 for pred in group_preds if pred == question_pred) / N
    else: # type == 'c' contradictory group
        score = sum(1 for pred in group_preds if pred != question_pred) / N
    return score

def pairwise_agreement(group1_preds, group2_preds=None, opposite=0):
    # PA: pairwise agreement
    agree_pairs = 0
    total_pairs = 0

    if group2_preds: # two groups are give
        if opposite:
            group2_preds = [int(not x) for x in group2_preds]
        
        for (i, y_i), (j, y_j) in itertools.combinations(enumerate(group1_preds + group2_preds), 2):
            total_pairs += 1
            if y_i == y_j:
                agree_pairs += 1
    else: # one group is given
        for (i, y_i), (j, y_j) in itertools.combinations(enumerate(group1_preds), 2):
            total_pairs += 1
            if y_i == y_j:
                agree_pairs += 1
    
    pa = agree_pairs / total_pairs if total_pairs > 0 else 1.0
    return pa


# within groups
def compute_within_group_metrics(sample, groups):
    question_pred = sample['results']['baseline_original']

    sample['features'] = {}
    for key in groups.keys():
        if 'A' in key: # affirmatives
            sample['features'][f'QA_{key}'] = agreement_w_question(question_pred, groups[key], type='a')
        elif 'C' in key: # contradictories
            sample['features'][f'QA_{key}'] = agreement_w_question(question_pred, groups[key], type='c')

        sample['features'][f'PA_{key}'] = pairwise_agreement(groups[key])


# between groups
def compute_between_group_metrics(sample, groups):
    COMBINATIONS = ['ITA_ITC', 'ITA_FSA', 'ITA_FSC', 'ITC_FSA', 'ITC_FSC', 'FSA_FSC']

    print(groups)

    for comb in COMBINATIONS:
        g1, g2 = comb.split('_')

        opposite = 0
        if (g1[-1] == 'C' and g2[-1] == 'A') or (g1[-1] == 'A' and g2[-1] == 'C'):
            opposite = 1

        y_vars_g1 = groups[g1]
        y_vars_g2 = groups[g2]

        pa = pairwise_agreement(y_vars_g1, y_vars_g2, opposite=opposite)
        sample['features'][f'PA_{comb}'] = pa




def build_features(inference_data):
    list_of_samples = []
    for sample in inference_data:
        if 'results' not in sample.keys():
            print(sample['id'])
            continue
        if 'image_perturbations' not in sample['results'].keys():
            print(sample['id'])
            continue
        if 'text_perturbations' not in sample['results'].keys():
            print(sample['id'])
            continue
        if 'affirmative' not in sample['results']['image_perturbations'].keys() or 'contradictory' not in sample['results']['image_perturbations'].keys():
            print(sample['id'])
            continue

        ITA = sample['results']['image_perturbations']['affirmative']
        ITC = sample['results']['image_perturbations']['contradictory']

        affirmative_org = ITA.pop('original', None)
        contradictory_org = ITC.pop('original', None)

        FSA = sample['results']['text_perturbations']['affirmative']
        FSC = sample['results']['text_perturbations']['contradictory']

        groups = {
            'ITA': [affirmative_org] + list(ITA.values()),
            'ITC': [contradictory_org] + list(ITC.values()),
            'FSA': [affirmative_org] + list(FSA.values()),
            'FSC': [contradictory_org] + list(FSC.values()),
        }

        sample_features = {
            'id': sample['id'],
            'type': sample['type'],
            'truth': sample['truth'],
        }

        # compute hallucination label
        pred = sample['results']['baseline_original']
        if sample_features['truth'] == 'yes':
            hal = 1 if pred == 0 else 0
        else: # truth = no
            hal = 1 if pred == 1 else 0
        sample_features['hal_label'] = hal

        # compute features for the sample
        compute_within_group_metrics(sample, groups)
        compute_between_group_metrics(sample, groups)

        for key in sample['features'].keys():
            sample_features[key] = sample['features'][key]

        list_of_samples.append(sample_features)

    return pd.DataFrame(list_of_samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute within/between-group consistency features from an inference.py results file."
    )
    parser.add_argument("--input-file", required=True,
                        help="Path to a results_full_<benchmark>_<model>.json file produced by inference.py.")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for the resulting data_features.csv "
                             "(default: ../Results/Features/<model>/<benchmark>/).")
    args = parser.parse_args()

    with open(args.input_file, 'r') as f:
        inference_data = json.load(f)

    file_name = os.path.splitext(os.path.basename(args.input_file))[0]
    benchmark_name, model_name = file_name.removeprefix('results_full_').rsplit('_', 1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(script_dir), 'Results', 'Features', model_name, benchmark_name
    )
    os.makedirs(output_dir, exist_ok=True)
    feature_file_path = os.path.join(output_dir, 'data_features.csv')

    df = build_features(inference_data)
    print(df.head())

    df.to_csv(feature_file_path, index=False)
    print(f"Saved features for model={model_name}, benchmark={benchmark_name} to {feature_file_path}")

