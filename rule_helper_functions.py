from copy import deepcopy
import itertools
import pandas as pd
from math import sqrt
import scipy.stats
from PD_itemset import PD_itemset
from Rule import Rule



def give_quick_sets_of_rules_for_recidivism_testing_purposes(pd_itemsets):
    disc_class_rules_connected_to_pd_itemsets = dict()
    for pd_itemset in pd_itemsets:
        disc_class_rules_connected_to_pd_itemsets[pd_itemset] = []

    native_american_and_male = PD_itemset({'sex': 'M', 'race': 'American Indian or Alaskan Native'})
    rule_1 = Rule({"race": "American Indian or Alaskan Native", "sex" : "M", "offense": "Operating While Intoxicated"}, {"recidivism" : "yes"}, support = 0.133, confidence = 0.857, lift = 1.151, slift = 0.758, elift = 0.734)
    rule_2 = Rule({"race": "American Indian or Alaskan Native", "sex" : "M", "case_type": "Criminal Traffic"}, {"recidivism" : "yes"}, support = 0.256, confidence = 0.857, lift = 1.151, slift = 0.551, elift = 0.734)

    hispanic = PD_itemset({'race': 'Hispanic'})
    rule_3 = Rule({"prior_criminal_traffic" : "None", "offense": "OAR/OAS", "age": "18-29", "race": "Hispanic",  "case_type": "Criminal Traffic"}, {"recidivism" : "yes"}, support = 0.017, confidence = 1.00, lift = 1.151, slift = 0.647, elift = 0.734)

    disc_class_rules_connected_to_pd_itemsets[native_american_and_male] = [rule_1, rule_2]
    disc_class_rules_connected_to_pd_itemsets[hispanic] = [rule_3]

    return disc_class_rules_connected_to_pd_itemsets



def give_quick_sets_of_rules_for_income_testing_purposes(pd_itemsets):
    disc_class_rules_connected_to_pd_itemsets = dict()
    for pd_itemset in pd_itemsets:
        disc_class_rules_connected_to_pd_itemsets[pd_itemset] = []

    female_and_white = PD_itemset({"sex" : "Female", "race" : "White alone"})
    rule_1 = Rule({"sex" : "Female", "race" : "White alone", "marital status" : "Married"}, {"income" : "low"}, support=0.02, confidence=0.9, lift=1.0, slift=0.51, elift=0.4, slift_p_value=0.00)
    rule_2 = Rule({"sex" : "Female",  "race" : "White alone", "workinghours" : "More than 50"}, {"income" : "low"}, support=0.02, confidence=0.9, lift=1.0, slift=0.61, elift=0.4, slift_p_value=0.00)

    male_and_white = PD_itemset({"sex" : "Male", "race" : "White alone"})
    rule_3 = Rule({"sex" : "Male", "race" : "White alone", "workinghours" : "More than 50"}, {"income" : "high"}, support=0.02, confidence=0.9, lift=1.0, slift=0.51, elift=0.4, slift_p_value=0.00)
    #
    male_and_other = PD_itemset({"sex" : "Male", "race" : "Other"})
    rule_4 = Rule({"sex" : "Male", "race" : "Other", "workinghours" : "More than 50"}, {"income" : "high"}, support=0.02, confidence=0.9, lift=1.0, slift=0.51, elift=0.4, slift_p_value=0.00)
    rule_5 = Rule({"sex" : "Male", "race" : "Other", "marital status" : "Seperated"}, {"income" : "low"}, support=0.02, confidence=0.9, lift=1.0, slift=0.51, elift=0.4, slift_p_value=0.00)
    #
    male = PD_itemset({"sex" : "Male"})
    rule_6 = Rule({"sex" : "Male", "workinghours" : "More than 50"},  {"income" : "high"}, support=0.02, confidence=0.9, lift=1.0, slift=0.51, elift=0.4, slift_p_value=0.00)

    disc_class_rules_connected_to_pd_itemsets[female_and_white] = [rule_1, rule_2]
    disc_class_rules_connected_to_pd_itemsets[male_and_white] = [rule_3]
    disc_class_rules_connected_to_pd_itemsets[male_and_other] = [rule_4, rule_5]
    disc_class_rules_connected_to_pd_itemsets[male] = [rule_6]

    return disc_class_rules_connected_to_pd_itemsets


def give_quick_set_of_rules_german_credit(pd_itemsets):
    disc_class_rules_connected_to_pd_itemsets = dict()
    for pd_itemset in pd_itemsets:
        disc_class_rules_connected_to_pd_itemsets[pd_itemset] = []

    female = PD_itemset({"Sex": "Female"})
    rule_1 = Rule({"Sex" : "Female", "Account Balance" : "No Account",  "Credit Amount Range" : "(231.826, 3884.8]"}, {"Creditability": "Not Credible"},
                  support=0.02, confidence=0.9, lift=1.0, slift=0.51, elift=0.4, slift_p_value=0.00)
    rule_2 = Rule({"Sex" : "Female", "Account Balance" : "No Account",  "Duration of Credit (month)" : "1-2 years"}, {"Creditability": "Not Credible"},
                  support=0.02, confidence=0.9, lift=1.0, slift=0.51, elift=0.4, slift_p_value=0.00)

    disc_class_rules_connected_to_pd_itemsets[female] = [rule_1, rule_2]
    return disc_class_rules_connected_to_pd_itemsets


def instance_is_from_reference_group(instance, sensitive_attributes, reference_group_dict):
    sens_attribute_values_of_instance = instance[sensitive_attributes].to_dict()
    if len(sens_attribute_values_of_instance) != len(reference_group_dict):
        return False

    for key, value in sens_attribute_values_of_instance.items():
        if sens_attribute_values_of_instance[key] != reference_group_dict[key]:
            return False
    return True

def get_instances_covered_by_rule_base_and_consequence(rule_base, rule_consequence, data):
    relevant_data = data
    for key in rule_base.keys():
        relevant_data = relevant_data[relevant_data[key] == rule_base[key]]

    for key in rule_consequence.keys():
        relevant_data = relevant_data[relevant_data[key] == rule_consequence[key]]

    return relevant_data


def get_number_of_instances_covered_by_ruleBase_and_by_completeRule(rule_base, rule_consequence, data):
    relevant_data = data
    for key in rule_base.keys():
        relevant_data = relevant_data[relevant_data[key] == rule_base[key]]

    number_of_instances_covered_by_rule_base = len(relevant_data)

    for key in rule_consequence.keys():
        relevant_data = relevant_data[relevant_data[key] == rule_consequence[key]]

    number_of_instances_covered_by_rule_base_and_consequence = len(relevant_data)

    return number_of_instances_covered_by_rule_base, number_of_instances_covered_by_rule_base_and_consequence


def get_instances_covered_by_rule_base(rule_base, data):
    relevant_data = data
    for key in rule_base.keys():
        relevant_data = relevant_data[relevant_data[key] == rule_base[key]]
    return relevant_data

def get_support_of_rule_base(rule_base, data):
    number_instances_covered_by_rule_base = len(get_instances_covered_by_rule_base(rule_base, data))
    return number_instances_covered_by_rule_base/len(data)

def calculate_support_and_confidence_of_rule(rule, data):
    n_instances_covered_by_rule_base, n_instances_covered_by_base_and_cons = get_number_of_instances_covered_by_ruleBase_and_by_completeRule(
        rule.rule_base, rule.rule_consequence, data)
    support = n_instances_covered_by_base_and_cons/len(data)
    if (n_instances_covered_by_rule_base != 0):
        confidence = n_instances_covered_by_base_and_cons/n_instances_covered_by_rule_base
        return support, confidence
    else:
        return support, -999


def calculate_support_and_confidence_of_rule(rule_base, rule_consequence, data):
    n_instances_covered_by_rule_base, n_instances_covered_by_base_and_cons = get_number_of_instances_covered_by_ruleBase_and_by_completeRule(
        rule_base, rule_consequence, data)
    support = n_instances_covered_by_base_and_cons / len(data)
    if (n_instances_covered_by_rule_base != 0):
        confidence = n_instances_covered_by_base_and_cons/n_instances_covered_by_rule_base
        return support, confidence
    else:
        return support, -999


def calculate_support_and_confidence_of_rule_with_negation_part(rule_base, negation_part_rule_base, rule_consequence, data):
    n_instances_covered_by_rule_base_with_negation, n_instances_covered_by_rule_base_with_negation_and_consequence = get_number_of_instances_covered_by_ruleBaseWithNegation_and_completeRule(rule_base, negation_part_rule_base, rule_consequence, data)
    support = n_instances_covered_by_rule_base_with_negation_and_consequence/len(data)

    if (n_instances_covered_by_rule_base_with_negation != 0):
        confidence = n_instances_covered_by_rule_base_with_negation_and_consequence / n_instances_covered_by_rule_base_with_negation
        return support, confidence
    else:
        return support, -999

def get_number_of_instances_covered_by_ruleBaseWithNegation_and_completeRule(rule_base, negation_part_rule_base, rule_consequence, data):
    instances_covered_by_rule_base_with_negation = get_instances_covered_by_rule_with_negation(rule_base,
                                                                                               negation_part_rule_base,
                                                                                               data)
    n_instances_covered_by_rule_base_with_negation = len(instances_covered_by_rule_base_with_negation)

    instances_covered_by_rule_base_with_negation_and_consequence = instances_covered_by_rule_base_with_negation

    for key in rule_consequence.keys():
        instances_covered_by_rule_base_with_negation_and_consequence = \
        instances_covered_by_rule_base_with_negation_and_consequence[
            instances_covered_by_rule_base_with_negation_and_consequence[key] == rule_consequence[key]]

    n_instances_covered_by_rule_base_with_negation_and_consequence = len(
        instances_covered_by_rule_base_with_negation_and_consequence)

    return n_instances_covered_by_rule_base_with_negation, n_instances_covered_by_rule_base_with_negation_and_consequence


def at_least_one_instance_is_classified_correctly(rule_consequence, data_covered_by_rule_base, decision_attribute):
    rule_consequence_value = rule_consequence[decision_attribute]
    rightly_classified_instances = data_covered_by_rule_base[data_covered_by_rule_base[decision_attribute] == rule_consequence_value]
    return len(rightly_classified_instances) >= 1

def number_of_right_and_misclassified_instances_by_rule(rule_consequence, data_covered_by_rule_base, decision_attribute):
    rule_consequence_value = rule_consequence[decision_attribute]
    number_of_rightly_classified_instances = len(data_covered_by_rule_base[
        data_covered_by_rule_base[decision_attribute] == rule_consequence_value])
    number_of_misclassified_instances = len(data_covered_by_rule_base) - number_of_rightly_classified_instances
    return number_of_rightly_classified_instances, number_of_misclassified_instances



#who are neither female nor black.... maybe find sens. attrib. combination with highest difference in conf.??
#so try out all different combinations, see which one is worst?
#could say same for non-binary sensitive attribute -> why only compare black to non-black, if also e.g. hispanic or asian
#people are discriminated
def get_instances_covered_by_rule_with_negation(rule_base, negation_part, data):
    non_relevant_data = data

    for key in negation_part.keys():
        non_relevant_data = non_relevant_data[non_relevant_data[key] == negation_part[key]]

    index_relevance_boolean_indicators = data.index.isin(non_relevant_data.index)
    relevant_data = data[~index_relevance_boolean_indicators]

    for key in rule_base.keys():
        relevant_data = relevant_data[relevant_data[key] == rule_base[key]]

    return relevant_data



def calculate_significance_of_slift(number_instances_covered_by_org_rule_base, number_instances_covered_by_ref_rule_base, number_instances_covered_by_complete_org_rule, number_instances_covered_by_complete_ref_rule, total_number_instances):
    confidence_org_rule = number_instances_covered_by_complete_org_rule / number_instances_covered_by_org_rule_base
    confidence_reference_pd_rule = number_instances_covered_by_complete_ref_rule / number_instances_covered_by_ref_rule_base

    slift_d= confidence_org_rule - confidence_reference_pd_rule
    if (slift_d == 0):
        p_value = 1.0
        return p_value

    if (confidence_reference_pd_rule == 0):
        p_value = 0.0
        return p_value

    total_proportion_both_groups_with_rule_consequence = (number_instances_covered_by_complete_org_rule + number_instances_covered_by_complete_ref_rule) / (number_instances_covered_by_org_rule_base + number_instances_covered_by_ref_rule_base)

    Z = (confidence_org_rule-confidence_reference_pd_rule) / sqrt((total_proportion_both_groups_with_rule_consequence * (1-total_proportion_both_groups_with_rule_consequence) *  ((1/number_instances_covered_by_complete_org_rule) + (1/number_instances_covered_by_complete_ref_rule))))

    p_value = scipy.stats.norm.sf(abs(Z))*2
    # print("number of people covered by org rule: " + str(number_instances_covered_by_complete_org_rule))
    # print("number of people covered by ref rule: " + str(number_instances_covered_by_complete_ref_rule))
    # print("Z is: " + str(Z))
    # print("p_value is: " + str(p_value))
    return p_value



#rule come in this format {'rule_base': {'sex': 'Male'}, 'rule_consequence': {'income': '<=50K'}, 'support': 0.46460489542704464, 'confidence': 0.6942634235888022, 'lift': 0.9144786138946193}
def calculate_slift_and_elift(rule, dataset, protected_itemset, reference_group):
    data = dataset.descriptive_data

    pd_itemset_dict_notation = protected_itemset.dict_notation
    pd_itemset_frozenset_notation = protected_itemset.frozenset_notation

    #check if class rule contains protected itemset. If not than the DCI score will be 0
    if pd_itemset_frozenset_notation==frozenset():
        return 0, 0, 0

    n_instances_covered_by_org_rule_base, n_instances_covered_by_complete_org_rule = get_number_of_instances_covered_by_ruleBase_and_by_completeRule(rule.rule_base, rule.rule_consequence, data)
    support_original_rule, confidence_of_original_rule = calculate_support_and_confidence_of_rule(rule.rule_base, rule.rule_consequence, data)

    rule_base_without_protected_itemset = deepcopy(rule.rule_base)
    for key in pd_itemset_dict_notation.keys():
        rule_base_without_protected_itemset.pop(key, None)

    # this calculations are needed to compute the elift (comparing confidence of rule with prot itemset, with the same rule
    # not containing it)
    n_instances_covered_by_rule_base_without_prot_itemset, n_instances_covered_by_complete_rule_without_prot_itemset = get_number_of_instances_covered_by_ruleBase_and_by_completeRule(rule_base_without_protected_itemset, rule.rule_consequence, data)
    support_rule_without_prot_itemset, confidence_rule_without_prot_itemset = calculate_support_and_confidence_of_rule(rule_base_without_protected_itemset,
                                                                                    rule.rule_consequence, data)
    elift_d = confidence_of_original_rule - confidence_rule_without_prot_itemset

    # this calculations are needed to compute the slift (where we compare the rule containing the prot itemset, with the rule
    # where prot itemset is substituted by reference group)
    # reference_group_base = deepcopy(reference_group)
    # reference_group_base.update(rule_base_without_protected_itemset)
    # n_instances_covered_by_rule_base_with_ref_prot_itemset, n_instances_covered_by_complete_ref_rule = get_number_of_instances_covered_by_ruleBase_and_by_completeRule(reference_group_base, rule.rule_consequence, data)
    # support_comparison_pd, confidence_comparison_pd = calculate_support_and_confidence_of_rule(reference_group_base, rule.rule_consequence, data)

    n_instances_covered_by_rule_base_with_neg_prot_itemset, n_instances_covered_by_complete_neg_rule = get_number_of_instances_covered_by_ruleBaseWithNegation_and_completeRule(rule_base_without_protected_itemset, protected_itemset.dict_notation, rule.rule_consequence, data)
    support_comparison_pd, confidence_comparison_pd = calculate_support_and_confidence_of_rule_with_negation_part(rule_base_without_protected_itemset, protected_itemset.dict_notation, rule.rule_consequence, data)

    if confidence_comparison_pd != -999:
        slift_d = confidence_of_original_rule - confidence_comparison_pd
        p_value_slift_d = calculate_significance_of_slift(n_instances_covered_by_org_rule_base,
                                                  n_instances_covered_by_rule_base_with_neg_prot_itemset,
                                                  n_instances_covered_by_complete_org_rule,
                                                  n_instances_covered_by_complete_neg_rule, len(data))
    else:
        p_value_slift_d = -999
        slift = -999
        slift_d = -999

    return slift_d, elift_d, p_value_slift_d


def dict_to_string_format(rule_dict):
    dict_to_string = "{"
    counter = 1

    for key, value in rule_dict.items():
        dict_to_string += "\"" + key + "\""
        dict_to_string += " : "
        dict_to_string += "\"" + value + "\""
        if counter != len(rule_dict):
            dict_to_string += ", "
        counter += 1
    dict_to_string += "}"
    return dict_to_string



#fairness_rules is a dict in the form of {PD_itemset : list}
def fairness_rules_to_dataframe(fairness_rules):
    dataframe_with_rules = pd.DataFrame([], columns=['sensitive_group', 'rule_base', 'rule_consequence', 'support', 'confidence', 'elift'])
    for pd_itemset in fairness_rules.keys():
        if pd_itemset.dict_notation != {}:
            for fairness_rule in fairness_rules[pd_itemset]:
                sensitive_group_string = dict_to_string_format(pd_itemset.dict_notation)
                rule_base_string = dict_to_string_format(fairness_rule.rule_base)
                rule_consequence_string = dict_to_string_format(fairness_rule.rule_consequence)
                row_entry = {'sensitive_group': sensitive_group_string, 'rule_base': rule_base_string,
                             'rule_consequence': rule_consequence_string, 'support': fairness_rule.support,
                             'confidence': fairness_rule.confidence, 'elift': fairness_rule.elift,
                             'slift': fairness_rule.slift, 'slift_p_value': fairness_rule.slift_p_value}
                dataframe_with_rules = dataframe_with_rules.append(row_entry, ignore_index=True)
    return dataframe_with_rules

def convert_to_apriori_format(X):
    list_of_dicts_format = X.to_dict('records')
    list_of_lists = []
    for dictionary in list_of_dicts_format:
        one_entry = set()
        for key, value in dictionary.items():
            one_entry.add(key + " : " + str(value))
        list_of_lists.append(one_entry)
    return list_of_lists
