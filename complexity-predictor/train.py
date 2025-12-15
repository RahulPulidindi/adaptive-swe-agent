"""
Training script for complexity predictor
"""

import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def train_predictor(
    features_file: str,
    output_dir: str,
    use_log_transform: bool = False
):
    """
    Train complexity predictor on extracted features.
    
    Args:
        features_file: Path to CSV with features and target
        output_dir: Directory to save trained models
        use_log_transform: Whether to use log transform on target
    """
    # Load features
    df = pd.read_csv(features_file)
    print(f"Loaded {len(df)} samples")
    
    # Prepare features and target
    feature_cols = [c for c in df.columns if c not in ['instance_id', 'tokens_used', 'actual_tokens']]
    X = df[feature_cols].values
    
    # Target is tokens_used
    y = df['tokens_used'].values if 'tokens_used' in df.columns else df['actual_tokens'].values
    
    # Optional log transform
    if use_log_transform:
        y = np.log1p(y)
        print("Applied log1p transform to target")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("Training Random Forest...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    
    if use_log_transform:
        # Inverse transform for metrics
        y_test_orig = np.expm1(y_test)
        y_pred_orig = np.expm1(y_pred)
    else:
        y_test_orig = y_test
        y_pred_orig = y_pred
    
    mse = mean_squared_error(y_test_orig, y_pred_orig)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_orig, y_pred_orig)
    
    print(f"\nResults:")
    print(f"  RMSE: {rmse:.1f} tokens")
    print(f"  R²: {r2:.3f}")
    
    # Save models
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    joblib.dump(model, os.path.join(output_dir, "complexity_predictor.pkl"))
    joblib.dump(scaler, os.path.join(output_dir, "feature_scaler.pkl"))
    joblib.dump(feature_cols, os.path.join(output_dir, "feature_names.pkl"))
    joblib.dump({'use_log': use_log_transform}, os.path.join(output_dir, "transform_config.pkl"))
    
    print(f"\n✓ Saved models to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="Path to features CSV")
    parser.add_argument("--output", default="models", help="Output directory")
    parser.add_argument("--log-transform", action="store_true", help="Use log transform")
    
    args = parser.parse_args()
    
    train_predictor(
        features_file=args.features,
        output_dir=args.output,
        use_log_transform=args.log_transform
    )