from SelectiveClassifier_IFAC import IFAC
from copy import deepcopy
from rule_helper_functions import get_instances_covered_by_rule_base_and_consequence, instance_is_from_reference_group
import pandas as pd
import numpy as np
from SituationTesting import SituationTesting
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

def generate_calibration_curve(y_true, y_pred_probs_pos, title_to_save, path_to_save):
    true_pos, pred_pos = calibration_curve(y_true, y_pred_probs_pos, n_bins=10)

    plt.plot(pred_pos,
             true_pos,
             marker='o',
             linewidth=1,
             label='Black Box Classifier')

    # Plot the Perfectly Calibrated by Adding the 45-degree line to the plot
    plt.plot([0, 1],
             [0, 1],
             linestyle='--',
             label='Perfectly Calibrated')

    # Set the title and axis labels for the plot
    plt.title(title_to_save)
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Probability')

    # Add a legend to the plot
    plt.legend(loc='best')

    complete_path_to_save_on = path_to_save + "\\" + title_to_save + ".png"
    plt.savefig(complete_path_to_save_on)
    plt.show()

    return

def put_reject_rules_dict_in_one_list_format(reject_rules_dict):
    all_reject_rules_list = []

    for pd_itemset, corresponding_reject_rules in reject_rules_dict.items():
        all_reject_rules_list.extend(corresponding_reject_rules)

    return all_reject_rules_list

def extract_instances_that_fall_under_reject_rules(reject_rules_list, data, predictions_for_data, prediction_probabilities_for_data):
    positive_label = data.desirable_label
    relevant_data = deepcopy(data.descriptive_data)
    relevant_data = relevant_data.drop(columns=[data.decision_attribute])
    relevant_data[data.decision_attribute] = predictions_for_data

    relevant_data['prediction probability'] = prediction_probabilities_for_data

    data_covered_by_all_rules = pd.DataFrame([])

    for reject_rule in reject_rules_list:
        data_covered_by_rule = get_instances_covered_by_rule_base_and_consequence(reject_rule.rule_base, reject_rule.rule_consequence, relevant_data)
        data_covered_by_rule['relevant_rule'] = reject_rule
        data_covered_by_rule['threat'] = 'Favouritism' if reject_rule.get_rule_consequence_label() == positive_label else 'Discrimination'
        data_covered_by_all_rules = pd.concat([data_covered_by_all_rules, data_covered_by_rule], axis=0)
        #to avoid that we add rules that are covered by two different rules twice
        index_relevance_boolean_indicators = relevant_data.index.isin(data_covered_by_rule.index)
        relevant_data = relevant_data[~index_relevance_boolean_indicators]
    return data_covered_by_all_rules


def extract_instances_with_high_disc_score(situation_tester, validation_data_covered_by_rules, validation_data, sensitive_attributes, reference_group_dict):
    validation_indices_covered_by_rules = validation_data_covered_by_rules.index

    numerical_validation_data = validation_data.numerical_data.loc[:,
                                validation_data.numerical_data.columns != validation_data.decision_attribute]
    relevant_num_validation_data = numerical_validation_data.loc[validation_indices_covered_by_rules]

    high_disc_scores_indices = []

    for index, instance in relevant_num_validation_data.iterrows():
        check_for_favouritism = validation_data_covered_by_rules.loc[index]['threat'] == 'Favouritism'
        sens_attributes_instance = instance[sensitive_attributes].to_dict()
        numpy_instance = instance.to_numpy()
        instance = numpy_instance.reshape(1, -1)
        if check_for_favouritism:
            sit_info = situation_tester.instance_is_favoured_based_on_sit_test(instance, sens_attributes_instance)
            disc_info = "Favoured" if sit_info.favoured else "Neutral"
        else:
            sit_info = situation_tester.instance_is_discriminated_based_on_sit_test(instance, sens_attributes_instance)
            disc_info = "Discriminated" if sit_info.discriminated else "Neutral"
        if disc_info != "Neutral":
            high_disc_scores_indices.append(index)
    print("HIGH DISC SCORE INDICES")
    print(high_disc_scores_indices)
    return high_disc_scores_indices

#Meaning of cut_off_probability: if an instance falls under a discriminatory rule and has a high disc score ->
#Reject from making a prediciton if prob is BIGGER than cut_off_value (unfair but certain)
#Else (if prob is SMALLER than cut_off_value) than Intervent (unfair and uncertain)
def decide_on_probability_threshold_unfair_but_certain(relevant_data, n_instances_to_reject):
    prediction_probs_of_data = relevant_data['prediction probability']
    ordered_prediction_probs = prediction_probs_of_data.sort_values(ascending=False)

    if (n_instances_to_reject > len(relevant_data)):
        cut_off_probability = 0.5
        number_of_rejected_instances = len(relevant_data)

    else:
        cut_off_probability = ordered_prediction_probs.iloc[n_instances_to_reject]
        number_of_rejected_instances = n_instances_to_reject

    return cut_off_probability, number_of_rejected_instances


#Meaning of cut_off_probability: if an instance doesn't fall under any of the discrimination rules OR doesn't have a high
#disc score, then we are only going to reject that instance if it's prediction_probability is SMALLER than the cut_off_probability
def decide_on_probability_threshold_fair_but_uncertain(relevant_data, n_instances_to_reject):
    prediction_probs_of_data = relevant_data['prediction probability']
    ordered_prediction_probs = prediction_probs_of_data.sort_values(ascending=True)

    if (n_instances_to_reject > len(relevant_data)):
        cut_off_probability = 0.5

    else:
        cut_off_probability = ordered_prediction_probs.iloc[n_instances_to_reject]

    return cut_off_probability


def divide_data_into_unfair_and_fair_parts(validation_data, validation_predictions, validation_prediction_probabilities, train_data, pd_itemsets, disc_class_rules_connected_to_pd_itemsets, reference_group_dict, sensitive_attributes, sit_test_k, sit_test_t):
    decision_attribute = validation_data.decision_attribute
    positive_label = validation_data.desirable_label
    negative_label = validation_data.undesirable_label

    situation_tester = SituationTesting(train_data, reference_group_dict, sensitive_attributes, sit_test_k, sit_test_t)
    fairness_rejector = IFAC(disc_class_rules_connected_to_pd_itemsets, pd_itemsets, sensitive_attributes,
                             decision_attribute, negative_label, positive_label, reference_group_dict, cut_off_probability_fair_uncertain=0, cut_off_probability_unfair_certain=0, situation_tester= situation_tester)

    reject_rules_per_protected_itemset = fairness_rejector.reject_rules_per_prot_itemset
    all_reject_rules_in_one_list = put_reject_rules_dict_in_one_list_format(reject_rules_per_protected_itemset)
    data_covered_by_rules = extract_instances_that_fall_under_reject_rules(all_reject_rules_in_one_list,
                                                                           validation_data, validation_predictions,
                                                                           validation_prediction_probabilities)
    indices_of_rule_covered_instances_with_high_disc_score = extract_instances_with_high_disc_score(situation_tester,
                                                                                                    data_covered_by_rules,
                                                                                                    validation_data,
                                                                                                    sensitive_attributes,
                                                                                                    reference_group_dict)
    complete_validation_data = deepcopy(validation_data.descriptive_data)
    complete_validation_data['prediction probability'] = validation_prediction_probabilities

    unfair_data_fraction = complete_validation_data.loc[
        indices_of_rule_covered_instances_with_high_disc_score]
    fair_data_fraction = complete_validation_data.loc[
        ~complete_validation_data.index.isin(indices_of_rule_covered_instances_with_high_disc_score)]

    return unfair_data_fraction, fair_data_fraction


def decide_on_probability_thresholds_fair_and_unfair_data(coverage, fairness_weight, validation_data, validation_predictions, validation_prediction_probabilities_df, train_data, disc_class_rules_connected_to_pd_itemsets, pd_itemsets, sensitive_attributes, reference_group_dict, path, sit_test_k = 10, sit_test_t = 0.3):
    n_of_instances_to_reject = int(len(validation_data.descriptive_data) * (1-coverage))

    n_of_unfair_instances_to_reject = int(n_of_instances_to_reject * fairness_weight)

    highest_pred_probabilities = validation_prediction_probabilities_df.max(axis='columns')
    pos_class_pred_probabilities = validation_prediction_probabilities_df[validation_data.desirable_label]
    data_with_fairness_concerns, data_without_fairness_concerns = divide_data_into_unfair_and_fair_parts(validation_data, validation_predictions, highest_pred_probabilities, train_data, pd_itemsets, disc_class_rules_connected_to_pd_itemsets, reference_group_dict, sensitive_attributes, sit_test_k, sit_test_t)

    val_data_true_labels = validation_data.binary_labels
    pos_class_pred_probabilities_unfair_data = pos_class_pred_probabilities.loc[data_with_fairness_concerns.index]
    true_labels_unfair_data = val_data_true_labels.loc[data_with_fairness_concerns.index]
    pos_class_pred_probabilities_fair_data = pos_class_pred_probabilities.loc[data_without_fairness_concerns.index]
    true_label_fair_data = val_data_true_labels.loc[data_without_fairness_concerns.index]

    # generate_calibration_curve(true_labels_unfair_data, pos_class_pred_probabilities_unfair_data, "Calibration Curve Unfair Portion of Data", path)
    # generate_calibration_curve(true_label_fair_data, pos_class_pred_probabilities_fair_data, "Calibration Curve Fair Portion of Data", path)

    probability_threshold_rejection_unfair_data, number_of_unfair_data_rejected = decide_on_probability_threshold_unfair_but_certain(data_with_fairness_concerns, n_of_unfair_instances_to_reject)
    n_of_fair_instances_to_reject = n_of_instances_to_reject - number_of_unfair_data_rejected
    probability_threshold_rejection_fair_data = decide_on_probability_threshold_fair_but_uncertain(data_without_fairness_concerns, n_of_fair_instances_to_reject)
    print("PROBABILITY THRESHOLD UNFAIR CERTAIN: " + str(probability_threshold_rejection_unfair_data))
    print("PROBABILITY THRESHOLD FAIR UNCERTAIN: " + str(probability_threshold_rejection_fair_data))

    return probability_threshold_rejection_unfair_data, probability_threshold_rejection_fair_data