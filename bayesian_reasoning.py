"""
CATEGORY 9(a): BAYESIAN REASONING
Use case: Simple medical diagnosis using Naive Bayes.

This program estimates the probability of Flu, Cold, and Allergy
from observed symptoms. The probabilities are illustrative examples
for an academic demonstration, not real medical diagnostic values.

Run:
    python bayesian_reasoning.py
"""

CLASSES = {
    "Flu": {
        "prior": 0.30,
        "symptoms": {
            "fever": 0.90,
            "cough": 0.80,
            "sneezing": 0.20,
        },
    },
    "Cold": {
        "prior": 0.40,
        "symptoms": {
            "fever": 0.30,
            "cough": 0.70,
            "sneezing": 0.80,
        },
    },
    "Allergy": {
        "prior": 0.30,
        "symptoms": {
            "fever": 0.05,
            "cough": 0.30,
            "sneezing": 0.90,
        },
    },
}


def diagnose(observed_symptoms):
    """
    Calculate posterior probabilities using Naive Bayes.

    observed_symptoms is a dictionary such as:
        {"fever": True, "cough": True, "sneezing": False}
    """
    scores = {}

    for disease, data in CLASSES.items():
        probability = data["prior"]

        for symptom, present in observed_symptoms.items():
            p_present = data["symptoms"][symptom]
            likelihood = p_present if present else (1 - p_present)
            probability *= likelihood

        scores[disease] = probability

    total = sum(scores.values())
    if total == 0:
        raise ValueError("Evidence probabilities sum to zero.")

    posterior = {
        disease: (score / total) * 100
        for disease, score in scores.items()
    }

    prediction = max(posterior, key=posterior.get)
    return prediction, posterior


def run_test_case(case_name, symptoms):
    print(f"\n{case_name}")
    print("Observed symptoms:", symptoms)

    prediction, probabilities = diagnose(symptoms)

    print("Posterior probabilities:")
    for disease, probability in probabilities.items():
        print(f"  {disease}: {probability:.2f}%")

    print("Most likely result:", prediction)


if __name__ == "__main__":
    # Test case 1: Fever and cough, but no sneezing
    run_test_case(
        "TEST CASE 1",
        {"fever": True, "cough": True, "sneezing": False},
    )

    # Test case 2: Sneezing and cough, but no fever
    run_test_case(
        "TEST CASE 2",
        {"fever": False, "cough": True, "sneezing": True},
    )

    # Test case 3: Sneezing only
    run_test_case(
        "TEST CASE 3",
        {"fever": False, "cough": False, "sneezing": True},
    )
