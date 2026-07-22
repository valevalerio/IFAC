from scipy.spatial.distance import pdist, squareform, cdist
import pandas as pd
from copy import deepcopy


#HAVE TO MAYBE RESTRICT THE NEAREST NEIGHBOUR SEARCH, TO ONLY LOOK AT INSTANCES THAT FALL WITHIN DECISION RULE

class SituationTesting():

    def __init__(self, train_data, reference_group_dict, sensitive_attributes, k, threshold):
        self.train_data = train_data
        self.reference_group_dict_list = reference_group_dict
        self.sensitive_attributes = sensitive_attributes
        self.k = k
        self.threshold = threshold


    def extract_data_falling_within_rule_base(self, rule_base_dict):
        #NEED TO FIRST GET RID OF POSSIBLE SENSITVIE ATTRIBUTES IN RULE BASE
        rule_base_without_protected_itemset = deepcopy(rule_base_dict)
        for key in self.sensitive_attributes:
            rule_base_without_protected_itemset.pop(key, None)

        data_falling_within_rule_base = self.train_data.descriptive_data

        for key, value in rule_base_without_protected_itemset.items():
            data_falling_within_rule_base = data_falling_within_rule_base[data_falling_within_rule_base[key] == value]

        falling_within_rule_base_indices = data_falling_within_rule_base.index
        numerical_format_of_falling_within_rule_base_data = self.train_data.numerical_data.loc[falling_within_rule_base_indices]
        return numerical_format_of_falling_within_rule_base_data


    def get_data_from_fixed_ref_group(self, relevant_data):
        reference_group_data = deepcopy(relevant_data)
        list_of_reference_group_data = []

        for reference_group in self.reference_group_dict_list:
            for key, value in reference_group.items():
                reference_group_data = reference_group_data[reference_group_data[key] == value]
            list_of_reference_group_data.append(reference_group_data)
            reference_group_data = deepcopy(relevant_data)

        complete_reference_group_data = pd.concat(list_of_reference_group_data)
        return complete_reference_group_data

    def divide_by_protected_membership_according_to_instances_sens_characteristics(self, sens_attribute_values_of_test_instance, relevant_data):
        relevant_data = relevant_data.loc[:, relevant_data.columns != self.train_data.decision_attribute]

        reference_group_data = self.get_data_from_fixed_ref_group(relevant_data)

        same_category_group_data = deepcopy(relevant_data)
        for key, value in sens_attribute_values_of_test_instance.items():
            same_category_group_data = same_category_group_data[same_category_group_data[key] == value]

        return reference_group_data, same_category_group_data


    def find_nearest_protected_and_unprotected_neighbours(self, test_instance, reference_group_data, non_reference_group_data):
        distance_vector_prot = cdist(test_instance, non_reference_group_data, metric = self.train_data.distance_function)
        distance_vector_prot = pd.Series(distance_vector_prot[0], index = non_reference_group_data.index)
        sorted_distances_prot = distance_vector_prot.argsort()
        sorted_indices_protected = distance_vector_prot.iloc[sorted_distances_prot].index
        closest_indices_protected = sorted_indices_protected[:self.k]
        closest_instances_protected = self.train_data.descriptive_data.loc[closest_indices_protected]

        distance_vector_unprot = cdist(test_instance, reference_group_data, metric = self.train_data.distance_function)
        distance_vector_unprot = pd.Series(distance_vector_unprot[0], index = reference_group_data.index)
        sorted_distances_unprot = distance_vector_unprot.argsort()
        sorted_indices_unprotected = distance_vector_unprot.iloc[sorted_distances_unprot].index
        closest_indices_unprotected = sorted_indices_unprotected[:self.k]
        closest_instances_unprotected = self.train_data.descriptive_data.loc[closest_indices_unprotected]

        return closest_instances_protected, closest_instances_unprotected


    def calc_pos_decision_ratio(self, instances):
        decision_attribute = self.train_data.decision_attribute
        negative_label = self.train_data.undesirable_label

        instances_with_pos_label = instances[instances[decision_attribute] != negative_label]

        if len(instances) > 0:
            return len(instances_with_pos_label)/len(instances)

        else:
            return 0

    def instance_is_discriminated_based_on_sit_test(self, test_instance, sens_attributes_instance, rule_to_search_within=None):
        if rule_to_search_within != None:
            relevant_data_to_run_sit_testing = self.extract_data_falling_within_rule_base(rule_to_search_within)
        else:
            relevant_data_to_run_sit_testing = self.train_data.numerical_data

        reference_group_data, non_reference_group_data = self.divide_by_protected_membership_according_to_instances_sens_characteristics(
            sens_attributes_instance, relevant_data_to_run_sit_testing)

        nearest_prot_neighbours, nearest_unprot_neighbours = self.find_nearest_protected_and_unprotected_neighbours(test_instance, reference_group_data, non_reference_group_data)
        pos_decision_ratio_prot = self.calc_pos_decision_ratio(nearest_prot_neighbours)
        pos_decision_ratio_unprot = self.calc_pos_decision_ratio(nearest_unprot_neighbours)

        disc_score = pos_decision_ratio_unprot - pos_decision_ratio_prot

        sit_test_info = SituationTestingInfo(disc_score, self.k, self.threshold, nearest_prot_neighbours, nearest_unprot_neighbours,
                                             sensitive_attributes=self.sensitive_attributes, decision_attribute=self.train_data.decision_attribute,
                                             desirable_label=self.train_data.desirable_label)
        return sit_test_info

    def instance_is_favoured_based_on_sit_test(self, test_instance, sens_attributes_instance, rule_to_search_within=None):
        if rule_to_search_within != None:
            relevant_data_to_run_sit_testing = self.extract_data_falling_within_rule_base(rule_to_search_within)
        else:
            relevant_data_to_run_sit_testing = self.train_data.numerical_data

        reference_group_data, non_reference_group_data = self.divide_by_protected_membership_according_to_instances_sens_characteristics(sens_attributes_instance, relevant_data_to_run_sit_testing)

        nearest_prot_neighbours, nearest_unprot_neighbours = self.find_nearest_protected_and_unprotected_neighbours(
            test_instance, reference_group_data, non_reference_group_data)
        pos_decision_ratio_prot = self.calc_pos_decision_ratio(nearest_prot_neighbours)
        pos_decision_ratio_unprot = self.calc_pos_decision_ratio(nearest_unprot_neighbours)

        favo_score = pos_decision_ratio_unprot - pos_decision_ratio_prot

        sit_test_info = SituationTestingInfo(favo_score, self.k, self.threshold, nearest_prot_neighbours,
                                             nearest_unprot_neighbours, from_favoured_group=True,
                                             sensitive_attributes=self.sensitive_attributes, decision_attribute=self.train_data.decision_attribute,
                                             desirable_label=self.train_data.desirable_label)
        return sit_test_info



class SituationTestingInfo:

    def __init__(self, discrimination_score, k, threshold, closest_prot_neighbours, closest_unprot_neighbours, from_favoured_group = False,
                 sensitive_attributes = None, decision_attribute = None, desirable_label = None):
        self.disc_score = discrimination_score
        self.k = k
        self.threshold = threshold
        self.sensitive_attributes = sensitive_attributes
        self.decision_attribute = decision_attribute
        self.desirable_label = desirable_label

        if from_favoured_group:
            self.discriminated = False
            self.favoured = self.disc_score >= self.threshold
        else:
            self.favoured = False
            self.discriminated = self.disc_score >= self.threshold

        self.closest_prot_neighbours = closest_prot_neighbours
        self.closest_unprot_neighbours = closest_unprot_neighbours

    def __str__(self):
        str_repr = f"Disc Score: {self.disc_score:.2f}"
        str_repr += "\nClosest neighbours from reference group:\n"
        str_repr += self.closest_unprot_neighbours.to_string()
        str_repr += "\nClosest neighbours from other protected groups:\n"
        str_repr += self.closest_prot_neighbours.to_string()
        return str_repr

