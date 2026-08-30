def task_accuracy_retention(attacked_test_acc: float, baseline_test_acc: float) -> float:
    """
    Attacked model's accuracy as a fraction of the ORIGINAL non-watermarked
    baseline's accuracy (not the watermarked model's accuracy -- baseline,
    so watermarking cost and attack cost don't get conflated in this number).
    """
    return attacked_test_acc / baseline_test_acc


def watermark_retention(bit_accuracy: float) -> float:
    """
    Currently a passthrough -- bit_accuracy from wm.bit_accuracy() IS the
    retention metric for a Uchida-style watermark. Kept as a named function
    (rather than using wm.bit_accuracy() directly everywhere) so a future
    trigger-based (Adi-style) watermark, whose retention metric is verify
    success rate rather than bit accuracy, can plug into the same call site
    in run_attack.py without that script needing to change.
    """
    return bit_accuracy