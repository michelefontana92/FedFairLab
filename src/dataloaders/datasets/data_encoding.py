import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder


def fit_scalers(df_orig, categorical_cols,
                numerical_cols, label_encoder_cols,
                categorical_categories=None):
    """Train scalers.
    
    Args:
        df_orig: Source dataframe to encode.
        categorical_cols: Categorical columns to encode.
        numerical_cols: Numerical columns to scale.
        label_encoder_cols: Columns to label-encode.
    """
    scalers = {}
    categorical_categories = categorical_categories or {}
    for cat in categorical_cols:
        declared_categories = categorical_categories.get(cat)
        encoder_kwargs = {
            'sparse_output': False,
            'handle_unknown': 'ignore',
        }
        if declared_categories is not None:
            encoder_kwargs['categories'] = [list(declared_categories)]
        scalers[cat] = OneHotEncoder(**encoder_kwargs).fit(df_orig[[cat]])
    for cat in numerical_cols:
        scalers[cat] = StandardScaler().fit(df_orig[[cat]])
    for label in label_encoder_cols:
        scalers[label] = LabelEncoder().fit(df_orig[[label]])
    return scalers


def encode_categorical(df_orig, attribute, ohe):
    """Handle encode categorical.
    
    Args:
        df_orig: Source dataframe to encode.
        attribute: Column or sensitive attribute to transform.
        ohe: One-hot encoder instance.
    """
    df = df_orig.copy()
    cols = df.columns.to_list()
    found_idx = -1
    for i, c in enumerate(cols):
        if c == attribute:
            found_idx = i
            break
    cols_before = cols[:found_idx]
    cols_after = cols[found_idx+1:]
    df_scaled = df.copy()
    p = ohe.transform(df_scaled[[attribute]])
    features = ohe.get_feature_names_out()
    cols_scaled = [c for c in cols_before]
    cols_scaled = cols_scaled + list(features)
    cols_scaled = cols_scaled + cols_after
    ohe_df = pd.DataFrame(p, columns=features, dtype=int)

    df_scaled = pd.concat([df_scaled.drop(attribute, axis=1), ohe_df], axis=1)[
        cols_scaled]

    return df_scaled


def encode_numerical(df_orig, attribute, sc):
    """Handle encode numerical.
    
    Args:
        df_orig: Source dataframe to encode.
        attribute: Column or sensitive attribute to transform.
        sc: Scaler instance.
    """
    df = df_orig.copy()
    df[attribute] = sc.transform(df[[attribute]])
    return df


def encode_label(df_orig, attribute, lb):
    """Handle encode label.
    
    Args:
        df_orig: Source dataframe to encode.
        attribute: Column or sensitive attribute to transform.
        lb: Label encoder instance.
    """
    df = df_orig.copy()
    df[attribute] = lb.transform(df[[attribute]])
    return df


def encode_dataset(df, cat_cols, num_cols, label_cols, scalers):
    """Handle encode dataset.
    
    Args:
        df: Input dataframe.
        cat_cols: Categorical columns to encode.
        num_cols: Numerical columns to scale.
        label_cols: Columns to label-encode.
        scalers: Pre-fitted encoders/scalers used for transformation.
    """
    df_enc = df.copy()
    for cat in cat_cols:
        df_enc = encode_categorical(df_enc, cat, scalers[cat])
    for num in num_cols:
        df_enc = encode_numerical(df_enc, num, scalers[num])
    for label in label_cols:
        df_enc = encode_label(df_enc, label, scalers[label])
    return df_enc


def decode_categorical(df_orig, attribute, ohe):
    """Handle decode categorical.
    
    Args:
        df_orig: Source dataframe to encode.
        attribute: Column or sensitive attribute to transform.
        ohe: One-hot encoder instance.
    """
    df = df_orig.copy()
    cols = df.columns.to_list()
    found = False
    cols_before = []
    cols_after = []
    for _, c in enumerate(cols):
        if c.startswith(attribute):
            found = True
        else:
            if not found:
                cols_before.append(c)
            else:
                cols_after.append(c)
    target_cols = [c for c in df.columns if c.startswith(attribute)]
    decoding = pd.DataFrame(ohe.inverse_transform(
        df[target_cols]), columns=[attribute])
    cols = cols_before + [attribute]+cols_after
    df_decoded = pd.concat(
        [df.drop(target_cols, axis=1), decoding], axis=1)[cols]
    return df_decoded


def decode_numerical(df_orig, attribute, sc):
    """Handle decode numerical.
    
    Args:
        df_orig: Source dataframe to encode.
        attribute: Column or sensitive attribute to transform.
        sc: Scaler instance.
    """
    df = df_orig.copy()
    df[attribute] = sc.inverse_transform(df[[attribute]])
    return df


def decode_label(df_orig, attribute, lb):
    """Handle decode label.
    
    Args:
        df_orig: Source dataframe to encode.
        attribute: Column or sensitive attribute to transform.
        lb: Label encoder instance.
    """
    df = df_orig.copy()
    df[attribute] = lb.inverse_transform(df[[attribute]])
    return df


def decode_dataset(df, cat_cols, num_cols, label_cols, scalers):
    """Handle decode dataset.
    
    Args:
        df: Input dataframe.
        cat_cols: Categorical columns to encode.
        num_cols: Numerical columns to scale.
        label_cols: Columns to label-encode.
        scalers: Pre-fitted encoders/scalers used for transformation.
    """
    df_dec = df.copy()
    for cat in cat_cols:
        df_dec = decode_categorical(df_dec, cat, scalers[cat])
    for num in num_cols:
        df_dec = decode_numerical(df_dec, num, scalers[num])
    for label in label_cols:
        df_enc = decode_label(df_dec, label, scalers[label])
    return df_enc
