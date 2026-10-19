from continual_arc.aliases import TaskAliases


def test_aliases_are_stable_within_a_run_and_meaningless_outside_it(tmp_path):
    a = TaskAliases.for_run(tmp_path / "run")
    b = TaskAliases.for_run(tmp_path / "run")   # the salt is read back from the run directory
    assert a.salt == b.salt and a.alias("88a10436") == b.alias("88a10436")
    assert a.alias("88a10436").startswith("task-") and "88a10436" not in a.alias("88a10436")
    assert TaskAliases.for_run(tmp_path / "other").alias("88a10436") != a.alias("88a10436")
    assert TaskAliases.for_run(None).salt != a.salt


def test_codenames_are_distinct_within_a_run():
    names = TaskAliases(salt="s", style="names")
    got = [names.alias(f"t{i}") for i in range(200)]
    assert len(set(got)) == len(got) and all("-" in n for n in got)
