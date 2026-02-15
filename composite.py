import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score

def load_and_process_data(file_path):
    """
    Loads data, aggregates replicates by median, and calculates physical coordinates.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None

    # Physical constants (Mass in kg)
    mass_map = {'iron': 0.000133, 'copper': 0.000128, 'glass': 0.000035}
    df['sphere_mass'] = df['material'].str.lower().map(mass_map)

    # Aggregate replicates using median to reduce noise
    group_cols = ['material', 'concentration', 'temp', 'rpm', 'f0_hz', 'sphere_mass']
    df_med = df.groupby(group_cols)[['viscosity', 'rho']].median().reset_index()

    # Physics Calculations
    # 1. Convert Frequency: Hz -> rad/s
    df_med['omega'] = 2 * np.pi * df_med['f0_hz']

    # 2. Calculate Abscissa (x): viscosity / (mass * omega)
    df_med['x_phys'] = df_med['viscosity'] / (df_med['sphere_mass'] * df_med['omega'])

    # 3. Calculate Ordinate (y): sqrt(1/rho^2 - 1)
    df_med = df_med[(df_med['rho'] > 0) & (df_med['rho'] <= 1)].copy()
    df_med['y_phys'] = np.sqrt((1 / df_med['rho']**2) - 1)

    return df_med

def train_soft_gated_model(df, valid_conc_range=None):
    """
    Trains a Soft-Gated Mixture Model:
    1. Physics Model: Estimates Kw using ONLY data within valid_conc_range.
    2. Complex Model: MLP trained on FULL dataset.
    3. Error Model: RF predicts deviation from Kw*x (calculated for FULL dataset).
    """
    
    feature_cols = ['concentration', 'temp', 'rpm', 'f0_hz']
    categorical_cols = ['material']
    
    X_raw = df[feature_cols + categorical_cols]
    y_target = df['y_phys'].values.reshape(-1, 1)
    x_physics = df['x_phys'].values.reshape(-1, 1)

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), feature_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ])
    X_processed = preprocessor.fit_transform(X_raw)

    # --- Step 1: Physics Model (RANSAC) on RESTRICTED Range ---
    # We must explicitly separate the calibration data to ensure mask alignment
    if valid_conc_range:
        min_c, max_c = valid_conc_range
        print(f"Restricting Physics (Kw) estimation to Concentration: {min_c}% - {max_c}%")
        physics_mask = (df['concentration'] >= min_c) & (df['concentration'] <= max_c)
        
        if physics_mask.sum() < 5:
            print("Warning: Too few points in concentration range. Using full dataset.")
            x_calib = x_physics
            y_calib = y_target
        else:
            x_calib = x_physics[physics_mask]
            y_calib = y_target[physics_mask]
            print(f"Physics calibrated on {physics_mask.sum()} points.")
    else:
        print("Using full concentration range for Physics estimation.")
        x_calib = x_physics
        y_calib = y_target

    base_linear = LinearRegression(fit_intercept=False)
    ransac = RANSACRegressor(estimator=base_linear, min_samples=0.4, residual_threshold=None, random_state=42)
    ransac.fit(x_calib, y_calib.ravel())
    
    Kw = ransac.estimator_.coef_[0]
    
    # Calculate noise sigma from the INLIERS of the calibration set
    # We index into x_calib/y_calib because ransac.inlier_mask_ corresponds to that subset
    inlier_residuals = y_calib[ransac.inlier_mask_].ravel() - Kw * x_calib[ransac.inlier_mask_].ravel()
    inlier_std = np.std(inlier_residuals)
    
    print(f"Physics Slope (Kw): {Kw:.4f}")
    print(f"Inlier Noise Std (Sigma): {inlier_std:.4f}")

    # --- Step 2: Complex Model (Full MLP) on FULL Dataset ---
    print("Training Complex MLP on full dataset...")
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', 
                       max_iter=2000, random_state=42)
    mlp.fit(X_processed, y_target.ravel())

    # --- Step 3: Error Prediction Model on FULL Dataset ---
    # We predict residuals for EVERY point based on the Kw we found in the restricted range.
    y_phys_all = Kw * x_physics.ravel()
    abs_residuals = np.abs(y_target.ravel() - y_phys_all)
    
    print("Training Error Predictor (Random Forest)...")
    error_model = RandomForestRegressor(n_estimators=100, random_state=42)
    error_model.fit(X_processed, abs_residuals)

    return {
        'Kw': Kw,
        'mlp': mlp,
        'error_model': error_model,
        'preprocessor': preprocessor,
        'sigma': inlier_std,
        'feature_cols': feature_cols,
        'valid_conc_range': valid_conc_range
    }

def predict_soft_gated(model_dict, df):
    """
    Predicts y using soft blending based on predicted error.
    """
    X_raw = df[['concentration', 'temp', 'rpm', 'f0_hz', 'material']]
    x_phys = df['x_phys'].values
    X_proc = model_dict['preprocessor'].transform(X_raw)
    
    # 1. Physics Prediction
    y_phys = model_dict['Kw'] * x_phys
    
    # 2. Complex Prediction
    y_complex = model_dict['mlp'].predict(X_proc)
    
    # 3. Error Prediction (Confidence)
    pred_error = model_dict['error_model'].predict(X_proc)
    
    # 4. Calculate Weight (Gaussian Kernel)
    sigma = model_dict['sigma'] * 3.0  # Trust region width
    weights = np.exp(- (pred_error**2) / (2 * sigma**2))
    
    # Blend
    y_final = weights * y_phys + (1 - weights) * y_complex
    
    return y_final, weights, pred_error

def plot_soft_gated_results(df, model_dict):
    y_pred, weights, pred_error = predict_soft_gated(model_dict, df)
    
    plt.figure(figsize=(14, 6))
    
    # --- Subplot 1: Physics Confidence Map ---
    plt.subplot(1, 2, 1)
    
    rf = model_dict['error_model']
    importances = rf.feature_importances_
    
    num_feats = model_dict['feature_cols']
    cat_feats = model_dict['preprocessor'].named_transformers_['cat'].get_feature_names_out()
    all_feat_names = np.concatenate([num_feats, cat_feats])
    
    num_indices = [i for i, name in enumerate(all_feat_names) if name in num_feats]
    num_importances = importances[num_indices]
    top2_idx_local = np.argsort(num_importances)[-2:] 
    top2_names = [num_feats[i] for i in top2_idx_local]
    
    x_min, x_max = df[top2_names[0]].min(), df[top2_names[0]].max()
    y_min, y_max = df[top2_names[1]].min(), df[top2_names[1]].max()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    
    mesh_df = pd.DataFrame({top2_names[0]: xx.ravel(), top2_names[1]: yy.ravel()})
    for col in num_feats:
        if col not in top2_names: mesh_df[col] = df[col].median()
    mesh_df['material'] = 'iron'
    
    mesh_X_proc = model_dict['preprocessor'].transform(mesh_df[['concentration', 'temp', 'rpm', 'f0_hz', 'material']])
    mesh_err = model_dict['error_model'].predict(mesh_X_proc)
    
    sigma = model_dict['sigma'] * 3.0
    mesh_w = np.exp(- (mesh_err**2) / (2 * sigma**2))
    Z = mesh_w.reshape(xx.shape)
    
    contour = plt.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.8)
    cbar = plt.colorbar(contour)
    cbar.set_label('Physics Confidence Weight (w)')
    
    plt.scatter(df[top2_names[0]], df[top2_names[1]], c=weights, cmap='RdYlBu', edgecolors='k', s=40)
    plt.xlabel(top2_names[0])
    plt.ylabel(top2_names[1])
    range_text = f" (Kw fit on {model_dict['valid_conc_range']})" if model_dict['valid_conc_range'] else ""
    plt.title(f'Physics Validity Map{range_text}\n(Blue = Trust Physics, Red = Trust ML); Kw={model_dict['Kw']:.4f}')

    # --- Subplot 2: Prediction ---
    plt.subplot(1, 2, 2)
    sc = plt.scatter(df['y_phys'], y_pred, c=weights, cmap='RdYlBu', alpha=0.7, edgecolors='k')
    plt.plot([0, df['y_phys'].max()], [0, df['y_phys'].max()], 'k--', label='Perfect Fit')
    
    mse = mean_squared_error(df['y_phys'], y_pred)
    r2 = r2_score(df['y_phys'], y_pred)
    
    plt.xlabel('Measured y')
    plt.ylabel('Soft-Gated Prediction')
    plt.title(f'Prediction Accuracy (Weighted)\nMSE: {mse:.3f}, R²: {r2:.3f}')
    
    cbar2 = plt.colorbar(sc)
    cbar2.set_label('Physics Weight (w)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('composite_model.png')

if __name__ == "__main__":
    file_path = 'master_per_well_lockin_matrix.csv'
    print("Loading Data...")
    df = load_and_process_data(file_path)
    if df is not None:
        # Define the valid concentration range for physics (e.g., 0 to 60%)
        # Adjust this tuple as needed: (min_conc, max_conc)
        physics_concentration_range = (20, 70) 
        
        print(f"Training Soft-Gated Model with restricted physics range: {physics_concentration_range}...")
        model_dict = train_soft_gated_model(df, valid_conc_range=physics_concentration_range)
        plot_soft_gated_results(df, model_dict)
