"""train.py's per-step log lines, on a CPU: what a policy-gradient step keeps in phase.json."""
import pytest

train = pytest.importorskip("train")


def test_a_policy_gradient_steps_record_keeps_the_dual_clip_and_the_worst_token():
    """A pg step's record keeps ask-sd's dual-clip share and its token of largest |log r| as given, so that a step whose
    ratios ran away names its call and token in phase.json after the adapter that played it is pruned."""
    worst = {"label": {"run": "r0021", "user": 1, "i": 57}, "position": 12, "token": 4242, "log_ratio": 13.05,
             "logp": -0.05, "logp_behaviour": -13.1}
    res = {"n_examples": 64, "tokens": 22891, "skipped_long": 0, "logp": [-0.2] * 64, "loss_mean": 1.95,
           "ratio_mean": 21.285371, "clip_fraction": 0.031716, "dual_clip_fraction": 1 / 22891, "approx_kl": 20.29,
           "entropy_mean": 0.188, "log_ratio_worst": worst}
    record = train.pg_record(res)
    assert record["dual_clip_fraction"] == round(1 / 22891, 6) > 0 and record["log_ratio_worst"] == worst
    assert record["items"] == 64 and record["ratio_mean"] == 21.285371
