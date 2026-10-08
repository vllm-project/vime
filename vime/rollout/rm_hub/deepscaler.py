from .math_utils import extract_answer, grade_answer_mathd, grade_answer_sympy


def get_deepscaler_rule_based_reward(response, label):
    # A Python "\boxed" literal is a backspace. Put the marker back before parsing.
    response = response.replace(chr(8), "\\b")

    if "</think>" in response:
        model_solution = response.split("</think>")[-1]
    elif "<|im_end|>" in response:
        # enable_thinking=false ends the turn here. Grade the whole span so a
        # later explanation does not drop an earlier \boxed answer.
        model_solution = response.split("<|im_end|>", 1)[0]
    elif "###Response" in response:
        model_solution = response.split("###Response", 1)[1]
    else:
        return 0

    model_answer = extract_answer(model_solution)
    if model_answer is None:
        return 0
    if label == "":
        return 0

    # Convert single answer to list for uniform processing
    assert isinstance(label, (str, float, int))
    ground_truths = [label]

    # Process each ground truth
    processed_ground_truths = []
    for truth in ground_truths:
        truth = str(truth)
        if "\\boxed" in truth:
            processed_truth = extract_answer(truth)
            if processed_truth is not None:
                processed_ground_truths.append(processed_truth)
        else:
            processed_ground_truths.append(truth)

    if not processed_ground_truths:
        return 0

    # Check against all possible correct answers
    for ground_truth in processed_ground_truths:
        is_correct = grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth)
        if is_correct:
            return 1

    return 0
