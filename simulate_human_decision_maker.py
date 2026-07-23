import json
import os
import sys
from copy import deepcopy

from SelectiveClassifier_IFAC import FairnessRejectWithoutIntervention, ProbabilisticReject

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "llm_eval"))
from utils import is_valid_result  # noqa: E402


def simulate_perfectly_accurate_human_decision_maker():
    return


def _applicant_view_text(prediction):
    """Same anchoring-free view llm_eval/utils.py::applicant_view_text builds from the
    flattened CSV, but read directly off the live Reject object: FairnessReject already
    withholds the black box's own prediction/rule via silent_str() (see IFAC/Rejects.py);
    other reject types (e.g. ProbabilisticReject) don't define silent_str(), so fall back
    to the instance's own features with the ground-truth `decision` dropped."""
    if hasattr(prediction, "silent_str"):
        return prediction.silent_str()
    instance = prediction.rejected_instance
    if "decision" in instance.index:
        instance = instance.drop("decision")
    return "Applicant Profile:\n" + json.dumps(instance.to_dict(), indent=2)


def simulate_llm_human_decision_maker(predictions_with_fairness_intervention, human_deferred_predictions, llm_officer):
    """Defers each rejected instance to the LLM-simulated human officer (see
    llm_eval/pi_rpc_officer.py::PiRpcLoanOfficer), using its would_grant_loan answer as the
    substituted decision. Falls back to the black box's own un-rejected prediction when the
    officer fails to produce a valid decision (mirrors is_valid_result's -1-sentinel
    handling in llm_eval/run_llm_officer_eval.py)."""
    complete_predictions = deepcopy(predictions_with_fairness_intervention)
    for index, prediction in human_deferred_predictions.iteritems():
        applicant_text = _applicant_view_text(prediction)
        result = llm_officer.evaluate_application(applicant_text)
        valid, _ = is_valid_result(result)
        if valid:
            complete_predictions[index] = "approved" if result["would_grant_loan"] == "Y" else "denied"
        else:
            complete_predictions[index] = prediction.prediction_without_reject
    complete_predictions.sort_index(inplace=True)
    return complete_predictions


#This human will always make the opposite prediction from what the black box would have made
def simulate_black_box_hater_human_decision_maker(predictions_with_fairness_intervention, human_deferred_predictions):
    complete_predictions = deepcopy(predictions_with_fairness_intervention)
    for index, prediction in human_deferred_predictions.iteritems():
        opposite_prediction_to_original_one = prediction.opposite_prediction
        complete_predictions[index] = opposite_prediction_to_original_one
    complete_predictions.sort_index(inplace=True)
    print(complete_predictions)
    return complete_predictions


#This human will always make the same prediction that the black box would have made
def simulate_black_box_follower_human_decision_maker(predictions_with_fairness_intervention, human_deferred_predictions):
    complete_predictions = deepcopy(predictions_with_fairness_intervention)
    for index, prediction in human_deferred_predictions.iteritems():
        original_black_box_prediction = prediction.prediction_without_reject
        complete_predictions[index] = original_black_box_prediction
    complete_predictions.sort_index(inplace=True)
    return complete_predictions


def simulate_perfectly_accurate_human_decision_maker(predictions_with_fairness_intervention, human_deferred_predictions, ground_truth):
    complete_predictions = deepcopy(predictions_with_fairness_intervention)
    for index, prediction in human_deferred_predictions.iteritems():
        ground_truth_label = ground_truth[index]
        complete_predictions[index] = ground_truth_label
    complete_predictions.sort_index(inplace=True)
    return complete_predictions


def simulate_human_decision_maker_following_uncertainty_reject_advice_opposing_fairness_reject_advice(predictions_with_fairness_intervention, human_deferred_predictions):
    complete_predictions = deepcopy(predictions_with_fairness_intervention)
    for index, prediction in human_deferred_predictions.iteritems():
        if isinstance(prediction, ProbabilisticReject):
            opposite_black_box_prediction = prediction.opposite_prediction
            complete_predictions[index] = opposite_black_box_prediction
        else:
            original_black_box_prediction = prediction.prediction_without_reject
            complete_predictions[index] = original_black_box_prediction
    return complete_predictions


def simulate_human_decision_maker_following_fairness_reject_advice_opposing_uncertainty_reject_advice(predictions_with_fairness_intervention, human_deferred_predictions):
    complete_predictions = deepcopy(predictions_with_fairness_intervention)
    for index, prediction in human_deferred_predictions.iteritems():
        if isinstance(prediction, FairnessRejectWithoutIntervention):
            opposite_black_box_prediction = prediction.opposite_prediction
            complete_predictions[index] = opposite_black_box_prediction
        else:
            original_black_box_prediction = prediction.prediction_without_reject
            complete_predictions[index] = original_black_box_prediction
    return complete_predictions
