from apyori import apriori
from copy import deepcopy
from Rule import *
from rule_helper_functions import calculate_slift_and_elift, get_instances_covered_by_rule_base, convert_to_apriori_format


def extract_class_association_rules_in_predictions(train_data_dataset, predictions, class_items, protected_itemsets, reference_group, sensitive_attributes, min_support=0.05, max_length=4):
    train_data = train_data_dataset.descriptive_data
    decision_label = train_data_dataset.decision_attribute

    train_data_without_ground_truth_but_with_predictions = deepcopy(train_data)
    train_data_without_ground_truth_but_with_predictions = train_data_without_ground_truth_but_with_predictions.drop(columns=[decision_label])
    train_data_without_ground_truth_but_with_predictions[decision_label] = predictions

    new_train_data_dataset = deepcopy(train_data_dataset)
    new_train_data_dataset.descriptive_data = train_data_without_ground_truth_but_with_predictions
    #class_rules = apply_apriori_algorithm_and_extract_class_rules(new_train_data_dataset, class_items, protected_itemsets, reference_group)
    class_rules = apply_apriori_to_find_unfair_patterns_per_pd_itemset(new_train_data_dataset, class_items, protected_itemsets, reference_group, sensitive_attributes, min_support=min_support, max_length=max_length)

    return class_rules


def apply_apriori_to_find_unfair_patterns_per_pd_itemset(train_data_dataset, class_items, protected_itemsets, reference_group, sensitive_attributes, min_support=0.05, max_length=4):
    train_data = train_data_dataset.descriptive_data
    class_rules_per_protected_itemset = {}

    for prot_itemset in protected_itemsets:
        prot_itemset_dict = prot_itemset.dict_notation
        if prot_itemset_dict != {}:
            print(prot_itemset_dict)
            data_belonging_to_prot_itemset = get_instances_covered_by_rule_base(prot_itemset_dict, train_data)
            data_belonging_to_prot_itemset = data_belonging_to_prot_itemset.drop(columns=sensitive_attributes)

            data_apriori_format = convert_to_apriori_format(data_belonging_to_prot_itemset)
            associations = apriori(transactions=data_apriori_format, min_support=min_support,
                                   min_confidence=0.85, min_lift=1.0, min_length=2,
                                   max_length=max_length)

            list_of_associations = list(associations)
            class_rules_for_prot_itemset = extract_class_rules_for_one_protected_itemset(prot_itemset, list_of_associations, class_items,
                                              reference_group, train_data_dataset)
            class_rules_per_protected_itemset[prot_itemset] = class_rules_for_prot_itemset

    return class_rules_per_protected_itemset

def extract_class_rules_for_one_protected_itemset(relevant_pd_itemset, all_rules, class_items, reference_group, data):
    discriminatory_rules = []
    for rule in all_rules:
        if rule.items.isdisjoint(class_items):
            continue
        #covering the case that we're dealing with a rule that contains decision attribute
        #make sure that we save the ordering of the rule, where the consequence ONLY consists of the decision attribute
        for ordering in rule.ordered_statistics:
            rule_base = ordering.items_base
            rule_consequence = ordering.items_add
            if (not rule_consequence.isdisjoint(class_items)) & (len(rule_consequence) == 1):
                rule_base_with_prot_itemset = rule_base.union(relevant_pd_itemset.frozenset_notation)
                myRule = initialize_rule(rule_base_with_prot_itemset, rule_consequence, support=rule.support,
                                         confidence=ordering.confidence, lift=ordering.lift)

                slift, elift, slift_p_value = calculate_slift_and_elift(myRule, data, relevant_pd_itemset, reference_group)
                myRule.set_slift(slift)
                myRule.set_elift(elift)
                myRule.set_slift_p_value(slift_p_value)
                discriminatory_rules.append(myRule)

    return discriminatory_rules