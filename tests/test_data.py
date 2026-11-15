from lab.data import TARGET, load_dataset, load_frame, split_data


def test_frame_shape_and_target():
    df = load_frame()
    assert df.shape == (569, 31)
    assert set(df[TARGET].unique()) == {0, 1}
    # 212 malignant cases in the original dataset; we made malignant the positive class
    assert int(df[TARGET].sum()) == 212


def test_column_names_have_no_spaces():
    assert not any(" " in c for c in load_frame().columns)


def test_split_is_stratified_and_disjoint():
    data = split_data(load_frame(), test_size=0.2, seed=0)
    assert len(data.X_test) == 114
    assert TARGET not in data.X_train.columns
    assert set(data.X_train.index).isdisjoint(data.X_test.index)
    assert abs(data.y_train.mean() - data.y_test.mean()) < 0.02


def test_split_is_reproducible():
    a, b = load_dataset(seed=7), load_dataset(seed=7)
    assert list(a.X_test.index) == list(b.X_test.index)
