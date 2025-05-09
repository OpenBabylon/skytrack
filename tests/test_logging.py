import skytrack as st

def test_init_idempotent():
    run1 = st.init({"project": "test", "run_name": "idemp"})
    run2 = st.init({"project": "ignored"})
    assert run1 is run2
    run1.finish()
