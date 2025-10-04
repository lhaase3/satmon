from http.server import BaseHTTPRequestHandler
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import export_text
import warnings
warnings.filterwarnings('ignore')

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        try:
            current_time = time.time()
            
            # Run explainable AI analysis
            explanations = self.run_explainable_ai_analysis()
            
            response_data = {
                "timestamp": current_time,
                "data_source": "explainable_ai_engine",
                "status": "success",
                "explanations": explanations
            }
            
            self.wfile.write(json.dumps(response_data, indent=2).encode('utf-8'))
            
        except Exception as e:
            error_response = {
                "error": str(e),
                "status": "error",
                "timestamp": time.time(),
                "note": "Explainable AI analysis failed"
            }
            self.wfile.write(json.dumps(error_response).encode('utf-8'))

    def run_explainable_ai_analysis(self):
        """Advanced explainable AI providing detailed decision reasoning"""
        try:
            # Load real telemetry data
            data_dir = Path(__file__).parent.parent / "data"
            demo_file = data_dir / "demo_temp.csv"
            
            if not demo_file.exists():
                raise FileNotFoundError("Demo data file not found")
            
            # Load and prepare data
            df = pd.read_csv(demo_file)
            df['ts'] = pd.to_datetime(df['ts'])
            df = df.sort_values('ts').reset_index(drop=True)
            
            # Advanced feature engineering for explainable AI
            df['rolling_mean_5'] = df['value'].rolling(window=5, min_periods=1).mean()
            df['rolling_mean_20'] = df['value'].rolling(window=20, min_periods=1).mean()
            df['rolling_std_5'] = df['value'].rolling(window=5, min_periods=1).std()
            df['rolling_std_20'] = df['value'].rolling(window=20, min_periods=1).std()
            df['value_lag1'] = df['value'].shift(1)
            df['value_lag2'] = df['value'].shift(2)
            df['rate_change'] = df['value'].diff()
            df['acceleration'] = df['rate_change'].diff()
            df['z_score'] = (df['value'] - df['rolling_mean_20']) / df['rolling_std_20']
            df['deviation_from_trend'] = df['value'] - df['rolling_mean_20']
            df['volatility'] = df['value'].rolling(window=10, min_periods=1).std()
            
            # Temperature-specific features for explanation
            df['temp_variance'] = (df['value'] - df['value'].mean()) ** 2
            df['temp_spike_indicator'] = (df['value'] > df['value'].quantile(0.95)).astype(int)
            df['temp_drop_indicator'] = (df['value'] < df['value'].quantile(0.05)).astype(int)
            df['rapid_change_indicator'] = (df['rate_change'].abs() > df['rate_change'].std() * 2).astype(int)
            
            # Remove NaN values
            df = df.dropna()
            
            if len(df) < 50:
                raise ValueError("Insufficient data for explainable AI analysis")
            
            # Feature matrix for ML models
            feature_cols = [
                'rolling_mean_5', 'rolling_mean_20', 'rolling_std_5', 'rolling_std_20',
                'value_lag1', 'value_lag2', 'rate_change', 'acceleration', 'z_score',
                'deviation_from_trend', 'volatility', 'temp_variance', 'temp_spike_indicator',
                'temp_drop_indicator', 'rapid_change_indicator'
            ]
            
            X = df[feature_cols].values
            
            # Create binary anomaly labels using multiple criteria
            anomaly_conditions = (
                (np.abs(df['z_score']) > 2.5) |  # Statistical outlier
                (df['temp_spike_indicator'] == 1) |  # Temperature spike
                (df['rapid_change_indicator'] == 1) |  # Rapid change
                (df['volatility'] > df['volatility'].quantile(0.9))  # High volatility
            )
            y = anomaly_conditions.astype(int)
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train explainable Random Forest classifier
            rf_explainer = RandomForestClassifier(
                n_estimators=50,
                max_depth=5,  # Shallow for better interpretability
                random_state=42,
                min_samples_split=10,
                min_samples_leaf=5
            )
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.3, random_state=42
            )
            
            rf_explainer.fit(X_train, y_train)
            
            # Feature importance analysis
            feature_importance = dict(zip(feature_cols, rf_explainer.feature_importances_))
            sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            # Get current prediction and explanation
            current_features = X_scaled[-1:]
            current_prediction = rf_explainer.predict(current_features)[0]
            current_probabilities = rf_explainer.predict_proba(current_features)[0]
            
            # Individual tree analysis for detailed explanation
            individual_tree = rf_explainer.estimators_[0]  # Use first tree for explanation
            decision_path = individual_tree.decision_path(current_features)
            leaf_id = individual_tree.apply(current_features)
            
            # SHAP-like explanation (simplified version)
            current_feature_values = dict(zip(feature_cols, current_features[0]))
            
            # Generate detailed explanation
            explanation_factors = []
            for feature, importance in sorted_importance[:6]:  # Top 6 features
                feature_value = current_feature_values[feature]
                
                # Determine contribution direction and magnitude
                contribution_score = importance * abs(feature_value)
                
                # Generate human-readable explanation
                explanations_map = {
                    'temp_variance': f'Temperature variance is {feature_value:.2f} ({"high" if feature_value > 0.5 else "normal"})',
                    'rate_change': f'Rate of change is {feature_value:.2f}°C/min ({"rapid" if abs(feature_value) > 1 else "stable"})',
                    'z_score': f'Statistical Z-score is {feature_value:.2f} ({"outlier" if abs(feature_value) > 2 else "normal"})',
                    'rolling_std_20': f'20-period volatility is {feature_value:.2f} ({"high" if feature_value > 1 else "low"})',
                    'deviation_from_trend': f'Deviation from trend is {feature_value:.2f}°C',
                    'temp_spike_indicator': f'Temperature spike detected' if feature_value > 0.5 else 'No temperature spike',
                    'rapid_change_indicator': f'Rapid change detected' if feature_value > 0.5 else 'Stable change rate',
                    'volatility': f'Current volatility is {feature_value:.2f}',
                    'acceleration': f'Temperature acceleration is {feature_value:.2f}°C/min²'
                }
                
                human_explanation = explanations_map.get(feature, f'{feature}: {feature_value:.2f}')
                
                explanation_factors.append({
                    'feature': feature,
                    'importance': float(importance),
                    'contribution': float(contribution_score),
                    'current_value': float(feature_value),
                    'explanation': human_explanation,
                    'impact': 'Increases anomaly likelihood' if contribution_score > 0.1 else 'Minor impact'
                })
            
            # Generate AI reasoning narrative
            confidence = float(max(current_probabilities))
            prediction_class = 'Anomaly' if current_prediction == 1 else 'Normal'
            
            # Detailed decision reasoning
            top_factor = explanation_factors[0]
            second_factor = explanation_factors[1] if len(explanation_factors) > 1 else None
            
            ai_reasoning = f"""
            The AI model predicts '{prediction_class}' with {confidence*100:.1f}% confidence.
            
            Primary factor: {top_factor['explanation']} (contributes {top_factor['importance']*100:.1f}% to decision).
            """
            
            if second_factor:
                ai_reasoning += f"""
                Secondary factor: {second_factor['explanation']} (contributes {second_factor['importance']*100:.1f}% to decision).
                """
            
            # Risk assessment based on historical patterns
            historical_similar_cases = np.sum((np.abs(X_scaled - current_features) < 0.5).all(axis=1))
            
            if current_prediction == 1:
                risk_assessment = {
                    'severity': 'Critical' if confidence > 0.8 else 'Moderate',
                    'recommended_action': 'Immediate investigation required' if confidence > 0.8 else 'Monitor closely',
                    'similar_historical_cases': int(historical_similar_cases),
                    'false_positive_likelihood': float((1 - confidence) * 100)
                }
            else:
                risk_assessment = {
                    'severity': 'Low',
                    'recommended_action': 'Continue normal operations',
                    'similar_historical_cases': int(historical_similar_cases),
                    'false_negative_likelihood': float((1 - confidence) * 100)
                }
            
            # Model interpretability metrics
            model_interpretability = {
                'model_type': 'Random Forest (Explainable)',
                'max_tree_depth': 5,
                'number_of_trees': 50,
                'feature_interactions': 'Limited to tree depth',
                'explanation_confidence': float(confidence),
                'decision_complexity': 'Low' if len(explanation_factors) <= 3 else 'Medium'
            }
            
            # Generate counterfactual explanation
            counterfactual = self.generate_counterfactual(
                current_features[0], feature_cols, rf_explainer, scaler
            )
            
            return {
                'current_prediction': {
                    'class': prediction_class,
                    'confidence': float(confidence),
                    'anomaly_probability': float(current_probabilities[1] if len(current_probabilities) > 1 else 0),
                    'normal_probability': float(current_probabilities[0])
                },
                'explanation_factors': explanation_factors,
                'ai_reasoning': ai_reasoning.strip(),
                'risk_assessment': risk_assessment,
                'model_interpretability': model_interpretability,
                'counterfactual_explanation': counterfactual,
                'feature_importance_global': [
                    {'feature': feature, 'importance': float(importance)}
                    for feature, importance in sorted_importance
                ],
                'decision_tree_rules': self.extract_decision_rules(individual_tree, feature_cols),
                'confidence_intervals': {
                    'prediction_stability': float(np.std([tree.predict_proba(current_features)[0][1] 
                                                        for tree in rf_explainer.estimators_[:10]])),
                    'explanation_consistency': 'High' if confidence > 0.8 else 'Medium'
                }
            }
            
        except Exception as e:
            raise Exception(f"Explainable AI analysis failed: {str(e)}")
    
    def generate_counterfactual(self, current_features, feature_cols, model, scaler):
        """Generate counterfactual explanation - what would need to change for different prediction"""
        try:
            current_pred = model.predict([current_features])[0]
            target_class = 1 - current_pred  # Opposite class
            
            # Simple counterfactual: find minimal changes needed
            modified_features = current_features.copy()
            changes = []
            
            # Try modifying top important features
            feature_importance = model.feature_importances_
            top_features_idx = np.argsort(feature_importance)[-3:][::-1]  # Top 3 features
            
            for idx in top_features_idx:
                # Try small modifications
                for delta in [0.5, -0.5, 1.0, -1.0]:
                    test_features = modified_features.copy()
                    test_features[idx] += delta
                    
                    if model.predict([test_features])[0] == target_class:
                        feature_name = feature_cols[idx]
                        changes.append({
                            'feature': feature_name,
                            'current_value': float(current_features[idx]),
                            'required_value': float(test_features[idx]),
                            'change_needed': float(delta),
                            'explanation': f"If {feature_name} were {test_features[idx]:.2f} instead of {current_features[idx]:.2f}"
                        })
                        break
                
                if len(changes) >= 2:  # Limit to top 2 changes
                    break
            
            return {
                'target_outcome': 'Normal' if target_class == 0 else 'Anomaly',
                'required_changes': changes,
                'explanation': f"To get '{['Normal', 'Anomaly'][target_class]}' prediction, the following changes would be needed:"
            }
            
        except:
            return {
                'target_outcome': 'Unknown',
                'required_changes': [],
                'explanation': 'Counterfactual analysis not available'
            }
    
    def extract_decision_rules(self, tree, feature_names):
        """Extract human-readable decision rules from tree"""
        try:
            tree_rules = export_text(tree, feature_names=feature_names, max_depth=3)
            # Parse and simplify the rules
            rules = []
            lines = tree_rules.split('\n')
            
            for line in lines[:10]:  # First 10 rules
                if 'if' in line.lower() or 'else' in line.lower():
                    cleaned_rule = line.strip().replace('|--- ', '').replace('|    ', '')
                    if len(cleaned_rule) > 0 and len(cleaned_rule) < 100:
                        rules.append(cleaned_rule)
            
            return {
                'decision_rules': rules[:5],  # Top 5 rules
                'explanation': 'Simplified decision tree rules used by the AI model'
            }
        except:
            return {
                'decision_rules': ['Rules extraction failed'],
                'explanation': 'Decision tree rules not available'
            }

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()