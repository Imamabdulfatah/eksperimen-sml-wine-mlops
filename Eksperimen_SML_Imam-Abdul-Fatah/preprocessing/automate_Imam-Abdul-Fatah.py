"""
Automate Preprocessing - Imam Abdul Fatah
==========================================
Script otomatis untuk melakukan preprocessing pada Wine Quality Dataset.
Mengkonversi proses eksperimen dari notebook menjadi fungsi yang dapat
dijalankan secara otomatis.

Usage:
    python automate_Imam-Abdul-Fatah.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import argparse


def load_data(red_path, white_path):
    """
    Memuat dataset Wine Quality (red dan white) dan menggabungkannya.
    
    Parameters:
        red_path (str): Path ke file winequality-red.csv
        white_path (str): Path ke file winequality-white.csv
    
    Returns:
        pd.DataFrame: Dataset gabungan dengan kolom wine_type
    """
    df_red = pd.read_csv(red_path, sep=';')
    df_white = pd.read_csv(white_path, sep=';')
    
    df_red['wine_type'] = 'red'
    df_white['wine_type'] = 'white'
    
    df = pd.concat([df_red, df_white], axis=0, ignore_index=True)
    
    print(f"[INFO] Dataset loaded: {len(df)} sampel ({len(df_red)} red + {len(df_white)} white)")
    return df


def remove_duplicates(df):
    """
    Menghapus baris duplikat dari dataset.
    
    Parameters:
        df (pd.DataFrame): Dataset input
    
    Returns:
        pd.DataFrame: Dataset tanpa duplikat
    """
    before = len(df)
    df_clean = df.drop_duplicates()
    after = len(df_clean)
    print(f"[INFO] Duplikat dihapus: {before - after} baris (sisa: {after})")
    return df_clean


def handle_outliers(df, method='cap'):
    """
    Menangani outlier menggunakan IQR method.
    
    Parameters:
        df (pd.DataFrame): Dataset input
        method (str): 'cap' untuk capping, 'remove' untuk menghapus
    
    Returns:
        pd.DataFrame: Dataset setelah penanganan outlier
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop('quality', errors='ignore')
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        
        if method == 'cap':
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        elif method == 'remove':
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        if outliers > 0:
            print(f"[INFO] Outlier {col}: {outliers} values {'capped' if method == 'cap' else 'removed'}")
    
    return df


def encode_features(df):
    """
    Encoding fitur kategorikal dan konversi target.
    
    Parameters:
        df (pd.DataFrame): Dataset input
    
    Returns:
        pd.DataFrame: Dataset dengan fitur yang sudah di-encode
    """
    # Encode wine_type
    le = LabelEncoder()
    df['wine_type'] = le.fit_transform(df['wine_type'])
    print(f"[INFO] wine_type encoded: red=0, white=1")
    
    # Konversi quality ke binary
    df['quality_label'] = (df['quality'] > 5).astype(int)
    print(f"[INFO] quality -> quality_label: <=5 = 0 (low), >5 = 1 (high)")
    print(f"[INFO] Distribusi kelas: {dict(df['quality_label'].value_counts())}")
    
    return df


def scale_and_split(df, test_size=0.2, random_state=42):
    """
    Feature scaling dan train-test split.
    
    Parameters:
        df (pd.DataFrame): Dataset yang sudah di-encode
        test_size (float): Proporsi test set
        random_state (int): Random seed
    
    Returns:
        tuple: (X_train_scaled, X_test_scaled, y_train, y_test, scaler)
    """
    X = df.drop(columns=['quality', 'quality_label'])
    y = df['quality_label']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    print(f"[INFO] Train set: {len(X_train_scaled)} sampel")
    print(f"[INFO] Test set : {len(X_test_scaled)} sampel")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def save_preprocessed(X_train, X_test, y_train, y_test, df_preprocessed, output_dir):
    """
    Menyimpan dataset yang sudah dipreprocess.
    
    Parameters:
        X_train, X_test: Features train/test
        y_train, y_test: Target train/test
        df_preprocessed: Dataset lengkap yang sudah dipreprocess
        output_dir (str): Direktori output
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Gabung X dan y
    train_data = X_train.copy()
    train_data['quality_label'] = y_train.values
    
    test_data = X_test.copy()
    test_data['quality_label'] = y_test.values
    
    # Simpan
    train_data.to_csv(os.path.join(output_dir, 'train_data.csv'), index=False)
    test_data.to_csv(os.path.join(output_dir, 'test_data.csv'), index=False)
    df_preprocessed.to_csv(os.path.join(output_dir, 'winequality_preprocessed.csv'), index=False)
    
    print(f"[INFO] Dataset tersimpan di: {output_dir}")
    print(f"  - train_data.csv ({len(train_data)} baris)")
    print(f"  - test_data.csv ({len(test_data)} baris)")
    print(f"  - winequality_preprocessed.csv ({len(df_preprocessed)} baris)")


def preprocess_data(input_dir=None, output_dir=None):
    """
    Fungsi utama untuk melakukan preprocessing secara otomatis.
    Mengembalikan data yang siap dilatih.
    
    Parameters:
        input_dir (str): Direktori berisi dataset raw
        output_dir (str): Direktori output preprocessed data
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # Default paths
    if input_dir is None:
        input_dir = os.path.join(os.path.dirname(__file__), '..', 'winequality_raw')
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'winequality_preprocessing')
    
    red_path = os.path.join(input_dir, 'winequality-red.csv')
    white_path = os.path.join(input_dir, 'winequality-white.csv')
    
    print("=" * 60)
    print("  PREPROCESSING PIPELINE - Wine Quality Dataset")
    print("  Imam Abdul Fatah")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n[STEP 1] Loading data...")
    df = load_data(red_path, white_path)
    
    # Step 2: Remove duplicates
    print("\n[STEP 2] Removing duplicates...")
    df = remove_duplicates(df)
    
    # Step 3: Handle outliers
    print("\n[STEP 3] Handling outliers (IQR Capping)...")
    df = handle_outliers(df, method='cap')
    
    # Step 4: Encode features
    print("\n[STEP 4] Encoding features...")
    df = encode_features(df)
    
    # Step 5: Scale and split
    print("\n[STEP 5] Scaling features and splitting data...")
    X_train, X_test, y_train, y_test, scaler = scale_and_split(df)
    
    # Step 6: Save preprocessed data
    print("\n[STEP 6] Saving preprocessed data...")
    save_preprocessed(X_train, X_test, y_train, y_test, df, output_dir)
    
    print("\n" + "=" * 60)
    print("  PREPROCESSING SELESAI!")
    print("=" * 60)
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Wine Quality Dataset Preprocessing')
    parser.add_argument('--input-dir', type=str, default=None,
                        help='Direktori berisi dataset raw')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Direktori output preprocessed data')
    
    args = parser.parse_args()
    
    preprocess_data(
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )
