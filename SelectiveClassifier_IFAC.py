from PD_itemset import PD_itemset
from itertools import combinations
from SituationTesting import SituationTesting
from Rule import rule1_is_subset_of_rule2
from Rejects import FairnessRejectWithIntervention, FairnessRejectWithoutIntervention, ProbabilisticReject


class IFAC:

    def __init__(self, class_rules_per_prot_itemset, pd_itemsets, sensitive_attributes, decision_attribute, negative_label, positive_label, reference_group_dict, cut_off_probability_unfair_certain, cut_off_probability_fair_uncertain, situation_tester=None, min_slift=None):
        self.class_rules_per_prot_itemset = class_rules_per_prot_itemset
        self.pd_itemsets = pd_itemsets
        self.sensitive_attributes = sensitive_attributes
        self.decision_attribute = decision_attribute
        self.negative_label = negative_label
        self.positive_label = positive_label
        self.reference_group_dict_list = reference_group_dict
        self.minimum_slift = min_slift if min_slift is not None else 0.3
        self.max_pvalue_slift = 0.01 #0.01

        self.reject_rules_per_prot_itemset = self.construct_protected_itemset_and_their_reject_rules_dict()
        #self.sens_attribute_combinations = [('race', 'sex')]
        self.sens_attribute_combinations = self.make_sens_attribute_combinations()
        self.situation_test = situation_tester
        self.cut_off_probability_unfair_certain = cut_off_probability_unfair_certain
        self.cut_off_probability_fair_uncertain = cut_off_probability_fair_uncertain


    def get_reject_rules_per_prot_itemset(self):
        return self.reject_rules_per_prot_itemset

    def remove_rules_that_are_subsets_from_other_rules(self, list_of_rules):
        result = []
        for index1, rule1 in enumerate(list_of_rules):
            is_subset_of_any = False
            for index2, rule2 in enumerate(list_of_rules):
                if index1 != index2 and rule1_is_subset_of_rule2(rule1, rule2):
                    is_subset_of_any = True
                    break
            if not is_subset_of_any:
                result.append(rule1)
        return result

    def construct_protected_itemset_and_their_reject_rules_dict(self):
        reject_rules_per_prot_itemset = {}
        for pd_itemset in self.pd_itemsets:
            if pd_itemset.dict_notation != {}:
                reject_rules_per_prot_itemset[pd_itemset] = []

        reject_rules_per_prot_itemset = {}
        #could do a more advanced thing here, also taking confidence and everything into account
        for pd_itemset, rules in self.class_rules_per_prot_itemset.items():
            if pd_itemset.dict_notation != {}:
                significant_rules_with_high_slift = [rule for rule in rules if (rule.slift_p_value < self.max_pvalue_slift) & (((rule.confidence - rule.slift) < 0.5) | (abs(rule.slift) >= self.minimum_slift))]
                rules_with_high_slift_no_subrules = self.remove_rules_that_are_subsets_from_other_rules(significant_rules_with_high_slift)
                reject_rules_per_prot_itemset[pd_itemset] = rules_with_high_slift_no_subrules
                #this part is written to only have 'favouritism' rules for reference group
                if pd_itemset.dict_notation not in self.reference_group_dict_list:
                    rules_with_high_slift_no_subrules_no_favouritism_Rules = [rule for rule in rules_with_high_slift_no_subrules if (rule.rule_consequence[self.decision_attribute] != self.positive_label)]
                    reject_rules_per_prot_itemset[pd_itemset] = rules_with_high_slift_no_subrules_no_favouritism_Rules

        for pd_itemset, reject_rules in reject_rules_per_prot_itemset.items():
            print(pd_itemset)
            for reject_rule in reject_rules:
                print(reject_rule)
        return reject_rules_per_prot_itemset


    def make_sens_attribute_combinations(self):
        sens_attribute_combinations = []

        for n in range(1, len(self.sensitive_attributes) + 1):
            sens_attribute_combinations += list(combinations(self.sensitive_attributes, n))
        return sens_attribute_combinations

    def check_if_instance_without_fairness_threat_should_be_rejected_based_on_probability(self, descriptive_format_instance, prediction_for_instance, prediction_probability_for_instance):
        if prediction_probability_for_instance < self.cut_off_probability_fair_uncertain:
            opposite_prediction = self.positive_label if prediction_for_instance == self.negative_label else self.negative_label
            reject = ProbabilisticReject(descriptive_format_instance, prediction_for_instance,  prediction_probability_for_instance, opposite_prediction)
            return True, reject
        else:
            return False, None

    def check_if_instance_should_be_rejected_based_on_rules_prediction_probability_and_situation_testing(self, descriptive_format_instance, numeric_format_instance, prediction_for_instance, prediction_probability_for_instance):
        instance_falls_under_reject_rules, rule_to_reject_upon = self.check_if_instance_should_be_rejected_based_on_any_of_sens_char_reject_rules(descriptive_format_instance, prediction_for_instance)
        sens_attributes_of_test_instance = descriptive_format_instance[self.sensitive_attributes].to_dict()

        if not instance_falls_under_reject_rules:
            return self.check_if_instance_without_fairness_threat_should_be_rejected_based_on_probability(
                descriptive_format_instance, prediction_for_instance, prediction_probability_for_instance)

        sit_test_info = self.check_if_instance_should_be_rejected_based_on_situation_testing(numeric_format_instance, sens_attributes_of_test_instance,
                                                                                             rule_to_reject_upon)
        if not (sit_test_info.discriminated or sit_test_info.favoured):
            return self.check_if_instance_without_fairness_threat_should_be_rejected_based_on_probability(descriptive_format_instance, prediction_for_instance, prediction_probability_for_instance)

        if prediction_probability_for_instance > self.cut_off_probability_unfair_certain:
            opposite_prediction = self.positive_label if prediction_for_instance == self.negative_label else self.negative_label
            reject = FairnessRejectWithoutIntervention(descriptive_format_instance, prediction_for_instance, prediction_probability_for_instance, rule_to_reject_upon, opposite_prediction, sit_test_summary=sit_test_info)
            return True, reject
        else:
            alternative_prediction = self.positive_label if prediction_for_instance == self.negative_label else self.negative_label
            reject = FairnessRejectWithIntervention(descriptive_format_instance, prediction_for_instance, prediction_probability_for_instance, rule_to_reject_upon, alternative_prediction)
            return True, reject


    def check_if_instance_should_be_rejected_based_on_rules_and_prediction_probability(self, descriptive_format_instance, numeric_format_instance, prediction_for_instance, prediction_probability_for_instance):
        instance_falls_under_reject_rules, rule_to_reject_upon = self.check_if_instance_should_be_rejected_based_on_any_of_sens_char_reject_rules(descriptive_format_instance, prediction_for_instance)

        if instance_falls_under_reject_rules:
            if prediction_probability_for_instance > self.cut_off_probability:
                reject = FairnessRejectWithoutIntervention(descriptive_format_instance, prediction_for_instance, prediction_probability_for_instance, rule_to_reject_upon)
                return True, reject
            else:
                alternative_prediction = self.positive_label if prediction_for_instance == self.negative_label else self.negative_label
                reject = FairnessRejectWithIntervention(descriptive_format_instance, prediction_for_instance, prediction_probability_for_instance, rule_to_reject_upon, alternative_prediction)
                return True, reject
        return False, None

    def check_if_instance_should_be_rejected_based_on_situation_testing(self, numerical_format_test_instance, sens_attributes_of_test_instance, rule_to_reject_upon):
        rule_consequence_outcome = rule_to_reject_upon.rule_consequence[self.decision_attribute]
        if (rule_consequence_outcome == self.positive_label):
            return self.situation_test.instance_is_favoured_based_on_sit_test(numerical_format_test_instance, sens_attributes_of_test_instance, rule_to_reject_upon.rule_base)
        else:
            return self.situation_test.instance_is_discriminated_based_on_sit_test(numerical_format_test_instance, rule_to_reject_upon.rule_base)

    def check_if_instance_should_be_rejected_based_on_any_of_sens_char_reject_rules(self, descriptive_format_test_instance,  prediction_for_instance):
        for sens_attribute_combination in self.sens_attribute_combinations:
            sens_attribute_combination_as_list = list(sens_attribute_combination)
            sens_attribute_values_of_instance = PD_itemset(descriptive_format_test_instance[sens_attribute_combination_as_list].to_dict())
            rule_to_reject_upon = self.check_which_of_reject_rules_in_dict_applies(
                sens_attribute_values_of_instance, descriptive_format_test_instance, prediction_for_instance)
            if rule_to_reject_upon is not None:
                return (True, rule_to_reject_upon)
        return (False, None)


    #this function will return None if none of the reject rules applies
    def check_which_of_reject_rules_in_dict_applies(self, prot_itemset, instance, prediction_for_instance):
        corresponding_reject_rules = self.reject_rules_per_prot_itemset.get(prot_itemset)

        if corresponding_reject_rules is None or len(corresponding_reject_rules) == 0:
            return None

        for rule in corresponding_reject_rules:
            if self.check_if_reject_rule_applies(rule, instance, prediction_for_instance):
                return rule

        return None

    def check_if_reject_rule_applies(self, one_rule, instance, prediction_for_instance):
        instance_as_dict = instance.to_dict()
        rule_base = one_rule.rule_base
        for key, value in rule_base.items():
            if rule_base[key] != instance_as_dict[key]:
                return False

        rule_consequence = one_rule.rule_consequence[self.decision_attribute]
        return rule_consequence == prediction_for_instance


class ProbabilisticRejector:
    def __init__(self, threshold, sensitive_attributes, decision_attribute, negative_label, positive_label, reference_group_dict):
        self.threshold = threshold

        self.sensitive_attributes = sensitive_attributes
        self.decision_attribute = decision_attribute
        self.negative_label = negative_label
        self.positive_label = positive_label

        self.reference_group_dict = reference_group_dict

    def check_if_instance_should_be_rejected_based_on_prob_and_situation_testing(self):
        return

    def check_if_instance_should_be_rejected_based_on_prob(self, test_instance_descriptive_format, prediction_probability_dict):
        label_with_higest_prob = max(prediction_probability_dict, key= lambda x: prediction_probability_dict[x])
        highest_prob = prediction_probability_dict[label_with_higest_prob]
        if highest_prob <= self.threshold:
            sens_attributes_of_instance = test_instance_descriptive_format[self.sensitive_attributes].to_dict()
            if sens_attributes_of_instance == self.reference_group_dict:
                reject_info = ProbabilisticReject(test_instance_descriptive_format, prediction_without_reject=label_with_higest_prob, alternative_prediction=self.negative_label)
            else:
                reject_info = ProbabilisticReject(test_instance_descriptive_format, prediction_without_reject=label_with_higest_prob, alternative_prediction=self.positive_label)
            return True, reject_info

        return False, None
