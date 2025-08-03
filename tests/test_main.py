from openpyrolysis.batch_reactor import run_batch_reactor

def test_batch_reactor_runs():
    sol = run_batch_reactor()
    assert sol.success
    assert len(sol.t) > 0
