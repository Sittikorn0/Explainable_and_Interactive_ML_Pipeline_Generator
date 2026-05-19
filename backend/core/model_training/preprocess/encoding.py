# Libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def encode_fit_transform(features_train: pd.DataFrame, features_test: pd.DataFrame,
                         encoding_decisions: dict | None = None) -> tuple:
    """
    แปลงข้อมูลตัวอักษร (Categorical) เป็นตัวเลข:
    - ป้องกัน Data Leakage โดยการสร้างกฎ (Fit) จาก features_train เท่านั้น
    - นำกฎนั้นไปประยุกต์ใช้ (Transform) กับทั้ง Train และ Test

    encoding_decisions: dict จาก Transform step {"col": "one_hot_encoding"|"label_encoding"|"ordinal_encoding"}
                        ถ้าเป็น None → fallback ใช้ cardinality อัตโนมัติ
    """
    if encoding_decisions:
        # ใช้ decisions ที่ user เลือกไว้ใน Transform step
        for column_name, method in encoding_decisions.items():
            if column_name not in features_train.columns:
                continue
            # drop_column ถูกจัดการไปแล้วใน data_transformer.py ก่อนส่งข้อมูลเข้ามา
            if method == "drop_column":
                continue

            if method == "one_hot_encoding":
                one_hot_train = pd.get_dummies(features_train[column_name], prefix=column_name, drop_first=True, dtype=int)
                features_train = pd.concat([features_train.drop(columns=[column_name]), one_hot_train], axis=1)

                # แปลง features_test ด้วย schema เดียวกับ features_train (Fit on Train)
                prefix = column_name + "_"
                for new_col_name in one_hot_train.columns:
                    category_value = new_col_name[len(prefix):]
                    features_test[new_col_name] = (features_test[column_name].astype(str) == category_value).astype(int)
                features_test = features_test.drop(columns=[column_name])

            elif method in ("label_encoding", "ordinal_encoding"):
                # ทั้งสองใช้ LabelEncoder — fit บน train เท่านั้น
                label_encoder = LabelEncoder()
                label_encoder.fit(features_train[column_name].astype(str))

                known_categories = set(label_encoder.classes_)
                features_train[column_name] = label_encoder.transform(features_train[column_name].astype(str))

                most_frequent_value = features_train[column_name].mode()
                fallback_value = int(most_frequent_value.iloc[0]) if len(most_frequent_value) > 0 else 0

                features_test[column_name] = features_test[column_name].astype(str).apply(
                    lambda val, enc=label_encoder, known=known_categories, fb=fallback_value:
                        int(enc.transform([val])[0]) if val in known else fb
                )
    else:
        # Fallback: auto cardinality-based (ใช้เมื่อไม่มี decisions จาก UI)
        categorical_columns = features_train.select_dtypes(include=["object", "category"]).columns.tolist()

        for column_name in categorical_columns:
            num_unique_values = features_train[column_name].nunique()

            if num_unique_values <= 15:
                one_hot_train = pd.get_dummies(features_train[column_name], prefix=column_name, drop_first=True, dtype=int)
                features_train = pd.concat([features_train.drop(columns=[column_name]), one_hot_train], axis=1)

                prefix = column_name + "_"
                for new_col_name in one_hot_train.columns:
                    category_value = new_col_name[len(prefix):]
                    features_test[new_col_name] = (features_test[column_name].astype(str) == category_value).astype(int)
                features_test = features_test.drop(columns=[column_name])
            else:
                label_encoder = LabelEncoder()
                label_encoder.fit(features_train[column_name].astype(str))

                known_categories = set(label_encoder.classes_)
                features_train[column_name] = label_encoder.transform(features_train[column_name].astype(str))

                most_frequent_value = features_train[column_name].mode()
                fallback_value = int(most_frequent_value.iloc[0]) if len(most_frequent_value) > 0 else 0

                features_test[column_name] = features_test[column_name].astype(str).apply(
                    lambda val, enc=label_encoder, known=known_categories, fb=fallback_value:
                        int(enc.transform([val])[0]) if val in known else fb
                )

    # จัดระเบียบคอลัมน์ของ Test Set ให้ตรงกับ Train Set หลังจาก Encode เสร็จ
    features_test = features_test.reindex(columns=features_train.columns, fill_value=0)

    return features_train, features_test