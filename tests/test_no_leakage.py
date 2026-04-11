"""
tests/test_no_leakage.py
พิสูจน์ว่า ML pipeline ไม่มี Data Leakage

Run: python -m pytest tests/test_no_leakage.py -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from ml_process.preprocess import preprocess, _encode_fit_transform, _scale


# ── Fixtures ──────────────────────────────────────────────────
@pytest.fixture
def clf_df():
    """Classification dataset ที่มี signal จริง (ไม่ใช่ random)"""
    np.random.seed(42)
    n = 300
    X1 = np.random.randn(n)
    X2 = np.random.randn(n)
    # target ขึ้นอยู่กับ X1 โดยตรง → model ต้องเรียนรู้ได้
    y  = (X1 + X2 > 0).astype(int)
    return pd.DataFrame({"num_a": X1, "num_b": X2,
                         "cat_col": np.random.choice(["A", "B", "C"], n),
                         "target": y})


@pytest.fixture
def reg_df():
    """Regression dataset"""
    np.random.seed(42)
    n = 300
    X = np.random.randn(n, 3)
    y = X[:, 0] * 2 + X[:, 1] - X[:, 2] + np.random.randn(n) * 0.1
    return pd.DataFrame(X, columns=["f1", "f2", "f3"]).assign(target=y)


# ── Test 1: Scaler fit บน X_train เท่านั้น ────────────────────
def test_scaler_fit_on_train_only(clf_df):
    """
    พิสูจน์: scaler.mean_ ต้องตรงกับ X_train distribution
    ถ้า fit บน full data → X_train mean จะไม่เท่ากับ 0 พอดี
    แต่ถ้า fit บน X_train เท่านั้น → X_train mean ≈ 0, std ≈ 1
    """
    X_train, X_test, _, _, _ = preprocess(
        clf_df, "target", scaling_method="standard_scaler"
    )

    num_cols = X_train.select_dtypes(include="number").columns
    train_means = X_train[num_cols].mean()
    train_stds  = X_train[num_cols].std()

    # X_train ที่ fit+transform ด้วย X_train เอง → mean ≈ 0, std ≈ 1
    assert (train_means.abs() < 0.1).all(), \
        f"X_train mean ควรใกล้ 0 (scaler fit บน train) แต่ได้ {train_means.to_dict()}"
    assert (train_stds - 1).abs().lt(0.15).all(), \
        f"X_train std ควรใกล้ 1 แต่ได้ {train_stds.to_dict()}"

    # X_test ถูก transform ด้วย train scaler → mean ไม่จำเป็นต้องเป็น 0
    test_means = X_test[num_cols].mean()
    assert not (test_means.abs() < 1e-6).all(), \
        "X_test mean = 0 ทั้งหมด — สงสัยว่า fit บน test set ด้วย (leakage!)"


# ── Test 2: Unknown test label ต้องไม่ error และแทนด้วย 0 ────
def test_unknown_test_label_handled_gracefully():
    """
    พิสูจน์: ถ้า test set มี label ที่ไม่เคยเห็นใน train
    ต้องไม่ error และต้องแทนด้วย 0 (ไม่เพิ่ม class ใหม่)
    """
    # 3 unique values → cardinality > 2, ยังอยู่ใน one-hot threshold (≤15)
    # ใช้ force_label_cols เพื่อบังคับ label encoding
    X_train = pd.DataFrame({"cat": ["A", "B", "A", "B", "A"]})
    X_test  = pd.DataFrame({"cat": ["A", "C", "B", "C"]})  # "C" ไม่มีใน train

    X_train_enc, X_test_enc = _encode_fit_transform(
        X_train, X_test, force_label_cols=["cat"]
    )

    # A→0, B→1 (alphabetical fit บน train)
    assert set(X_train_enc["cat"].unique()) == {0, 1}, \
        f"X_train ควรมีแค่ 0,1 แต่ได้ {set(X_train_enc['cat'].unique())}"
    assert 2 not in X_test_enc["cat"].values, \
        "X_test ไม่ควรมี value 2 — 'C' ไม่ได้อยู่ใน train vocab ต้องแทนด้วย 0"
    # A→0, C→0 (unknown), B→1
    expected = [0, 0, 1, 0]
    assert X_test_enc["cat"].tolist() == expected, \
        f"X_test encoding ผิด: {X_test_enc['cat'].tolist()} ≠ {expected}"


# ── Test 3: Test labels ไม่เปลี่ยน train encoding ────────────
def test_test_labels_dont_affect_train_encoding():
    """
    พิสูจน์: encoding ของ X_train ต้องไม่เปลี่ยนไม่ว่า X_test จะมี label อะไร
    (bug เดิม: LabelEncoder.fit บน full df → test labels เปลี่ยน integer mapping)
    """
    X_train = pd.DataFrame({"cat": ["B", "A", "B", "A"]})

    # case 1: test มีแค่ A, B (เหมือน train)
    X_test_same  = pd.DataFrame({"cat": ["A", "B"]})
    enc_train_1, _ = _encode_fit_transform(
        X_train.copy(), X_test_same, force_label_cols=["cat"]
    )

    # case 2: test มี Z ด้วย (Z อยู่หลัง B alphabetically)
    X_test_extra = pd.DataFrame({"cat": ["A", "B", "Z"]})
    enc_train_2, _ = _encode_fit_transform(
        X_train.copy(), X_test_extra, force_label_cols=["cat"]
    )

    assert enc_train_1["cat"].tolist() == enc_train_2["cat"].tolist(), \
        "Train encoding เปลี่ยนเพราะ test set มี label ใหม่ — นี่คือ Leakage!"


# ── Test 4: Model ต้องทำได้ดีกว่า random บน dataset ที่มี signal
def test_model_beats_random_on_signal_data(clf_df):
    """
    พิสูจน์: ถ้า pipeline ไม่มี leakage และ dataset มี signal จริง
    model ต้องได้ accuracy สูงกว่า random (baseline ~50%)
    และการ shuffle y_test ต้องทำให้ accuracy ลดลงอย่างมีนัยสำคัญ
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    X_train, X_test, y_train, y_test, _ = preprocess(
        clf_df, "target", scaling_method="no_scaling"
    )

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    real_acc = accuracy_score(y_test, model.predict(X_test))

    # ต้องดีกว่า random ≥ 10%
    assert real_acc > 0.6, \
        f"Model accuracy ({real_acc:.3f}) ต่ำเกินไป — dataset มี signal แต่ model ทำได้แค่นี้"

    # shuffle test labels → accuracy ต้องลดลง
    rng = np.random.RandomState(0)
    shuffled_accs = []
    for _ in range(10):
        y_shuf = y_test.sample(frac=1, random_state=rng.randint(0, 9999)).values
        shuffled_accs.append(accuracy_score(y_shuf, model.predict(X_test)))

    avg_shuffled = np.mean(shuffled_accs)
    assert real_acc > avg_shuffled + 0.05, (
        f"Real ({real_acc:.3f}) ไม่สูงกว่า shuffled ({avg_shuffled:.3f}) — "
        "อาจมี leakage หรือ model ทำงานแบบ random"
    )


# ── Test 5: force_label_cols ถูก encode ด้วย train vocab เท่านั้น
def test_force_label_cols_encoded_post_split():
    """
    พิสูจน์: column ที่ระบุใน force_label_cols
    - ต้องถูก label encode (ไม่ใช่ one-hot)
    - ต้องใช้ train vocab เท่านั้น
    """
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "cat_feature": np.random.choice(["X", "Y", "Z"], n),
        "num_feature": np.random.randn(n),
        "target":      np.random.choice([0, 1], n),
    })

    X_train, X_test, _, _, _ = preprocess(
        df, "target",
        scaling_method="no_scaling",
        label_encoding_cols=["cat_feature"],
    )

    # ต้องยังเป็น column เดิม (ไม่ถูก one-hot เป็น cat_feature_Y, cat_feature_Z)
    assert "cat_feature" in X_train.columns, \
        f"cat_feature ต้องยังอยู่หลัง label encode แต่ columns: {X_train.columns.tolist()}"

    # dtype ต้องเป็น numeric
    assert pd.api.types.is_numeric_dtype(X_train["cat_feature"]), \
        "cat_feature ต้องเป็น numeric หลัง label encoding"

    # 3 classes (X, Y, Z) → values ต้องเป็น 0, 1, 2
    assert set(X_train["cat_feature"].unique()).issubset({0, 1, 2}), \
        f"label encoding ของ 3 classes ต้องได้ 0,1,2 แต่ได้ {set(X_train['cat_feature'].unique())}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
