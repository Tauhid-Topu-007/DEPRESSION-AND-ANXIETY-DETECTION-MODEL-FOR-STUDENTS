
"""
DEPLOYMENT SCRIPT FOR DDNet MODEL
Simple prediction function for production use
"""

import joblib
import pandas as pd
import numpy as np

# Load model once at startup
MODEL = None
PREPROCESSORS = None

def load_model():
    """Load the DDNet model and preprocessors"""
    global MODEL, PREPROCESSORS
    if MODEL is None:
        MODEL = joblib.load('models/ddnet_model.joblib')
        PREPROCESSORS = joblib.load('models/preprocessors_complete.joblib')
    return MODEL, PREPROCESSORS

def predict_risk(user_data):
    """
    Predict depression risk from user data
    
    Args:
        user_data: dict with features (can be incomplete)
    
    Returns:
        dict: risk_score, risk_level, recommendation, urgency
    """
    model, preprocessors = load_model()
    
    scaler = preprocessors['scaler']
    label_encoders = preprocessors['label_encoders']
    selected_features = preprocessors['selected_features']
    categorical_cols = preprocessors['categorical_cols']
    all_features = preprocessors['all_features']
    default_values = preprocessors['default_values']
    
    # Convert to DataFrame
    if isinstance(user_data, dict):
        user_data = pd.DataFrame([user_data])
    
    # Fill missing features
    full_data = pd.DataFrame()
    for col in all_features:
        if col in user_data.columns and not pd.isna(user_data[col]).all():
            full_data[col] = user_data[col]
        elif col in default_values:
            full_data[col] = default_values[col]
        else:
            full_data[col] = 0
    
    # Auto-calculate age_group
    if 'age' in full_data.columns:
        age = full_data['age'].iloc[0]
        if age < 18: full_data['age_group'] = 'Under 18'
        elif age <= 24: full_data['age_group'] = '18-24'
        elif age <= 34: full_data['age_group'] = '25-34'
        elif age <= 44: full_data['age_group'] = '35-44'
        elif age <= 59: full_data['age_group'] = '45-59'
        else: full_data['age_group'] = '60+'
    
    # Encode categorical
    for col in categorical_cols:
        if col in full_data.columns and col in label_encoders:
            try:
                full_data[col] = label_encoders[col].transform(full_data[col].astype(str))
            except:
                full_data[col] = 0
    
    # Scale
    full_data_scaled = scaler.transform(full_data)
    full_data_scaled = pd.DataFrame(full_data_scaled, columns=full_data.columns)
    
    # Select features
    if selected_features:
        final_data = full_data_scaled[selected_features]
    else:
        final_data = full_data_scaled
    
    # Predict
    try:
        risk_score = model.predict_proba(final_data)[0, 1]
    except:
        risk_score = float(model.predict(final_data)[0])
    
    # Interpret
    if risk_score < 0.3:
        return {'risk_score': risk_score, 'risk_level': 'MINIMAL RISK', 
                'recommendation': 'Continue current lifestyle. Regular monitoring recommended.',
                'urgency': 'No immediate action required', 'high_risk': False}
    elif risk_score < 0.5:
        return {'risk_score': risk_score, 'risk_level': 'LOW RISK',
                'recommendation': 'Consider stress management techniques. Maintain healthy habits.',
                'urgency': 'Routine check-in recommended', 'high_risk': False}
    elif risk_score < 0.7:
        return {'risk_score': risk_score, 'risk_level': 'MODERATE RISK',
                'recommendation': 'Please speak with a counselor. Consider mental health assessment.',
                'urgency': 'Schedule appointment within 2 weeks', 'high_risk': True}
    elif risk_score < 0.85:
        return {'risk_score': risk_score, 'risk_level': 'HIGH RISK',
                'recommendation': 'Please consult a mental health professional promptly.',
                'urgency': 'Schedule appointment within 1 week', 'high_risk': True}
    else:
        return {'risk_score': risk_score, 'risk_level': 'VERY HIGH RISK',
                'recommendation': 'Please seek immediate mental health support.',
                'urgency': 'IMMEDIATE - Consult within 24 hours', 'high_risk': True}

# Example usage
if __name__ == "__main__":
    # Test with minimal data
    user = {'age': 22, 'gender': 'Male', 'stress_level': 'High', 'sleep_hours': 6.5}
    result = predict_risk(user)
    print(f"Risk Score: {result['risk_score']:.3f}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Recommendation: {result['recommendation']}")
