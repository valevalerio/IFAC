class Rule:

    #rule_base and rule_consequence are both dictionaries
    def __init__(self, rule_base, rule_consequence, support, confidence, lift, slift=0, elift=0, slift_p_value=0):
        self.rule_base = rule_base
        self.rule_consequence = rule_consequence
        self.support = support
        self.confidence = confidence
        self.lift = lift
        self.slift = slift
        self.elift = elift
        self.slift_p_value = slift_p_value

    def set_slift(self, slift):
        self.slift = slift

    def set_elift(self, elift):
        self.elift = elift

    def set_slift_p_value(self, p_value):
        self.slift_p_value = p_value

    def get_rule_consequence_label(self):
        name_of_consequence_key = list(self.rule_consequence.keys())[0]
        return self.rule_consequence[name_of_consequence_key]

    def __str__(self):
        output_string = "("
        counter = 1
        for rule_key in self.rule_base.keys():
            output_string += rule_key + " = " + str(self.rule_base[rule_key])
            if counter != len(self.rule_base):
                output_string += " AND "
            counter += 1
        output_string += ") -> "

        counter = 1
        for rule_key in self.rule_consequence.keys():
            output_string += "(" + rule_key + " = " + str(self.rule_consequence[rule_key])
            if counter != len(self.rule_consequence):
                output_string += " AND "
            counter += 1
        output_string += ")"

        output_string += f", Support: {self.support:.3f}, Confidence: {self.confidence:.3f}, Lift: {self.lift:.3f}, SLift: {self.slift:.3f}, Elift: {self.elift:.3f}"
        return output_string

    def __repr__(self):
        output_string = "Rule: ("
        counter = 1
        for rule_key in self.rule_base.keys():
            output_string += str(self.rule_base[rule_key])
            if counter != len(self.rule_base):
                output_string += " AND "
            counter += 1
        output_string += ") -> "

        counter = 1
        for rule_key in self.rule_consequence.keys():
            output_string += str(self.rule_consequence[rule_key])
            if counter != len(self.rule_consequence):
                output_string += " AND "
            counter += 1

        output_string += f" Confidence: {self.confidence}, Support: {self.support}, Slift: {self.slift}, p-value: {self.slift_p_value}"
        return output_string


def convert_frozenset_rule_format_to_dict_format(frozentset_rule_representation):
    rule_as_dict = {}
    #rule is a frozenset, where each item follows the following format 'key : value'
    #each itemstring needs to be added to the dictionary as one key, value pair
    for rule_item in frozentset_rule_representation:
        #rule_item is string
        splitted_rule = rule_item.split(" : ")
        key_of_rule = splitted_rule[0]
        value_of_rule = splitted_rule[1]
        if value_of_rule.isdigit():
            rule_as_dict[key_of_rule] = int(value_of_rule)
        else:
            rule_as_dict[key_of_rule] = value_of_rule
    return rule_as_dict


def initialize_rule(rule_base_frozenset, rule_consequence_frozenset, support, confidence, lift):
    rule_base_dict = convert_frozenset_rule_format_to_dict_format(rule_base_frozenset)
    rule_consequence_dict = convert_frozenset_rule_format_to_dict_format(rule_consequence_frozenset)
    rule = Rule(rule_base_dict, rule_consequence_dict, support, confidence, lift)
    return rule


def rule1_is_subset_of_rule2(rule1, rule2):
    if rule1.rule_consequence != rule2.rule_consequence:
        return False

    return set(rule2.rule_base.items()).issubset(set(rule1.rule_base.items()))
