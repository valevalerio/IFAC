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
        reject_str_pres += self.reject_threat + "-based Abstention for following instance\n"
        reject_str_pres += str(self.rejected_instance.to_dict())
        reject_str_pres += "\nPrediction that would have been made: " + str(self.prediction_without_reject)
        reject_str_pres += "\nPrediction Probability: " + str(np.round(self.prediction_probability,2))
        return reject_str_pres

    def to_structured_dict(self):
        """Numeric/categorical fields for CSV export, as a flat dict - the structured
        counterpart to __str__()/silent_str(). Base case has nothing beyond what's already
        a plain column elsewhere (bb_prediction, reject_type already tell you the flipped
        label and whether it's the WithIntervention case); subclasses with rule/
        situation-testing data override this to add those fields."""
        return {}
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
        For human-in-the-loop review: same as __str__ but withholds the black-box's own
        prediction/probability, and replaces the raw situation-testing score with a plain-language
        breakdown of the situation-testing neighbours, so the human isn't anchored on the model's
        prediction or an opaque disc/favo score when making their own call.
        """
        reject_str_pres = "\n______________________________\n"
        reject_str_pres += "The AI has abstained from making a prediction for possible fairness concerns for the following instance:\n"
        
        reject_str_pres += str(
                    self.rejected_instance.drop('decision').to_dict()
                    )

        if self.sit_test_summary is None:
            return reject_str_pres

        sensitive_attributes = self.sit_test_summary.sensitive_attributes
        own_category = ", ".join(f"{attr}={self.rejected_instance[attr]}" for attr in sensitive_attributes)
        reject_str_pres += f"\nThe sensitive attributes of this applicant are: {own_category}"

        reference_neighbours = self.sit_test_summary.closest_unprot_neighbours
        same_category_neighbours = self.sit_test_summary.closest_prot_neighbours
        reference_category = (
            ", ".join(f"{attr}={reference_neighbours.iloc[0][attr]}" for attr in sensitive_attributes)
            if len(reference_neighbours) > 0 else "the reference group"
        )

        similarity_basis = ", ".join(
            f"{key}={value}" for key, value in self.rule_reject_is_based_upon.rule_base.items()
            if key not in sensitive_attributes
        )
        rule_confidence = self.rule_reject_is_based_upon.confidence
        rule_lift = self.rule_reject_is_based_upon.lift
        reject_str_pres += (
            f"\nThe discriminated subgroup for the category {own_category} is described as: {similarity_basis} "
            # confidence is already a 0-1 fraction; :.1% multiplies by 100 itself, so a
            # `rule_confidence*100` here would double-scale it (0.877 -> "8770.0%").
            f"(rule confidence: {rule_confidence:.1%}, lift: {rule_lift:.2f})"
        )

        reject_str_pres += self._sit_test_paragraph(reference_category, reference_neighbours)
        reject_str_pres += self._sit_test_paragraph(own_category, same_category_neighbours)
        return reject_str_pres

    def _sit_test_paragraph(self, category_label, neighbours):
        decision_attribute = self.sit_test_summary.decision_attribute
        desirable_label = self.sit_test_summary.desirable_label

        n_granted = int((neighbours[decision_attribute] == desirable_label).sum())
        n_total = len(neighbours)

        return f"\nSimilar instances with {category_label} that had loan granted {n_granted}/{n_total}."

    def to_structured_dict(self):
        """The rule this reject is based on, as plain values instead of buried in
        silent_str()'s/__str__()'s formatted paragraph - support/confidence/lift/slift plus
        which conditions (rule_base) actually fired, so "does this applicant match the rule
        that triggered the reject" is a column lookup instead of a text-parsing job."""
        out = super().to_structured_dict()
        out.update(
            rule_conditions=dict(self.rule_reject_is_based_upon.rule_base),
            rule_support=self.rule_reject_is_based_upon.support,
            rule_confidence=self.rule_reject_is_based_upon.confidence,
            rule_lift=self.rule_reject_is_based_upon.lift,
            rule_slift=self.rule_reject_is_based_upon.slift,
        )
        return out

class FairnessRejectWithoutIntervention(FairnessReject):
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