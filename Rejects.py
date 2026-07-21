import numpy as np
class Reject:
    def __init__(self, rejected_instance, reject_threat, prediction_without_reject, prediction_probability, alternative_prediction):
        self.rejected_instance = rejected_instance
        self.reject_threat = reject_threat
        self.prediction_without_reject = prediction_without_reject
        self.prediction_probability = prediction_probability
        self.alternative_prediction = alternative_prediction

    def __str__(self):
        reject_str_pres = "\n______________________________\n"
        reject_str_pres += self.reject_threat + "-based Reject for this instance\n"
        reject_str_pres += str(self.rejected_instance.to_dict())
        reject_str_pres += "\nPrediction that would have been made: " + str(self.prediction_without_reject)
        reject_str_pres += "\nPrediction Probability: " + str(np.round(self.prediction_probability,2))
        return reject_str_pres
class SchreuderReject(Reject):
    def __init__(self, rejected_instance, prediction_without_reject, prediction_probability):
        Reject.__init__(self, rejected_instance, "Schreuder", prediction_without_reject,  prediction_probability, alternative_prediction=None)

    def __str__(self):
        str_pres = Reject.__str__(self)
        str_pres += "\nDecision will be deferred to human"
        return str_pres

class FairnessReject(Reject):
    def __init__(self, rejected_instance, prediction_without_reject, prediction_probability, rule_reject_is_based_upon, sit_test_summary, alternative_prediction):
        Reject.__init__(self, rejected_instance, "Fairness", prediction_without_reject, prediction_probability, alternative_prediction)
        self.sit_test_summary = sit_test_summary
        self.rule_reject_is_based_upon = rule_reject_is_based_upon


    def __str__(self):
        str_pres = Reject.__str__(self)
        str_pres += "\nRejection Based on this rule\n"
        str_pres += str(self.rule_reject_is_based_upon)

        if self.sit_test_summary != None:
            str_pres += "\nSituation Testing Score: " + str(self.sit_test_summary)
        return str_pres

    def silent_str(self):
        """
        This new method is used to print the reject information without the prediction that would have been made and its probability, which is useful for test with human-in-the-loop, where the prediction is not known to the human (the black-box classifier has abstained from making a prediction, and the human is asked to make a decision instead).
        The silent_str method returns a string representation of the reject information without revealing the prediction that would have been made and its probability, it reveals the rejected instance, it says the black-box classifier has abstained from making a prediction, it indicates the sensitive attributes of the rejected instance. and then the 
        """
        reject_str_pres = "\n______________________________\n"
        reject_str_pres += self.reject_threat + "-based Reject for this instance\n"
        reject_str_pres += str(self.rejected_instance.to_dict())
        # reject_str_pres += "\nPrediction that would have been made: " + str(self.prediction_without_reject)
        # reject_str_pres += "\nPrediction Probability: " + str(np.round(self.prediction_probability,2))
        return reject_str_pres

class FairnessRejectWithoutIntervention(Reject):
    def __init__(self, rejected_instance, prediction_without_reject, prediction_probability, rule_reject_is_based_upon, opposite_prediction, sit_test_summary=None):
        FairnessReject.__init__(self, rejected_instance, prediction_without_reject, prediction_probability, rule_reject_is_based_upon, sit_test_summary, alternative_prediction=None)
        self.opposite_prediction = opposite_prediction

    def __str__(self):
        str_pres = FairnessReject.__str__(self)
        str_pres += "\nDecision will be deferred to human"
        return str_pres

class FairnessRejectWithIntervention(FairnessReject):
    def __init__(self, rejected_instance, prediction_without_reject, prediction_probability, rule_reject_is_based_upon, alternative_prediction, sit_test_summary=None):
        FairnessReject.__init__(self, rejected_instance, prediction_without_reject, prediction_probability, rule_reject_is_based_upon, sit_test_summary, alternative_prediction)

    def __str__(self):
        str_pres = FairnessReject.__str__(self)
        str_pres += "\nIntervention!"
        return str_pres

class ProbabilisticReject(Reject):

    def __init__(self, rejected_instance, prediction_without_reject,  prediction_probability, opposite_prediction):
        Reject.__init__(self, rejected_instance, "Uncertain Probability", prediction_without_reject,  prediction_probability, alternative_prediction=None)
        self.opposite_prediction = opposite_prediction

    def __str__(self):
        str_pres = Reject.__str__(self)
        str_pres += "\nDecision will be deferred to human"
        return str_pres