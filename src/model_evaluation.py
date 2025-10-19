"""
Model Evaluation Module
Comprehensive evaluation metrics and visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)


class ModelEvaluator:
    """Class for evaluating ML models"""
    
    def __init__(self):
        self.results = {}
        
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None, model_name='Model'):
        """
        Calculate all evaluation metrics
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        y_pred_proba : array-like
            Predicted probabilities
        model_name : str
            Name of the model
            
        Returns:
        --------
        dict
            Dictionary of metrics
        """
        metrics = {
            'Model': model_name,
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'F1-Score': f1_score(y_true, y_pred, zero_division=0)
        }
        
        if y_pred_proba is not None:
            metrics['ROC-AUC'] = roc_auc_score(y_true, y_pred_proba)
        
        self.results[model_name] = metrics
        
        return metrics
    
    def print_metrics(self, metrics):
        """
        Print metrics in formatted way
        
        Parameters:
        -----------
        metrics : dict
            Dictionary of metrics
        """
        print("\n" + "="*80)
        print(f"MODEL EVALUATION: {metrics['Model']}")
        print("="*80)
        print(f"📊 Performance Metrics:")
        print(f"   Accuracy:  {metrics['Accuracy']:.4f}")
        print(f"   Precision: {metrics['Precision']:.4f}")
        print(f"   Recall:    {metrics['Recall']:.4f}")
        print(f"   F1-Score:  {metrics['F1-Score']:.4f}")
        if 'ROC-AUC' in metrics:
            print(f"   ROC-AUC:   {metrics['ROC-AUC']:.4f}")
        print("="*80)
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name='Model', save_path=None):
        """
        Plot confusion matrix
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        model_name : str
            Name of the model
        save_path : str
            Path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                   xticklabels=['Not Clicked', 'Clicked'],
                   yticklabels=['Not Clicked', 'Clicked'])
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Confusion matrix saved to {save_path}")
        
        plt.show()
        
    def plot_roc_curve(self, y_true, y_pred_proba, model_name='Model', save_path=None):
        """
        Plot ROC curve
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred_proba : array-like
            Predicted probabilities
        model_name : str
            Name of the model
        save_path : str
            Path to save the plot
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ ROC curve saved to {save_path}")
        
        plt.show()
        
    def plot_precision_recall_curve(self, y_true, y_pred_proba, model_name='Model', save_path=None):
        """
        Plot Precision-Recall curve
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred_proba : array-like
            Predicted probabilities
        model_name : str
            Name of the model
        save_path : str
            Path to save the plot
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curve - {model_name}', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Precision-Recall curve saved to {save_path}")
        
        plt.show()
        
    def plot_feature_importance(self, model, feature_names, top_n=15, model_name='Model', save_path=None):
        """
        Plot feature importance
        
        Parameters:
        -----------
        model : trained model
            Model with feature_importances_ or coef_ attribute
        feature_names : list
            List of feature names
        top_n : int
            Number of top features to display
        model_name : str
            Name of the model
        save_path : str
            Path to save the plot
        """
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            print("⚠️  Model does not have feature importance or coefficients")
            return
        
        # Create DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_importance_df, y='feature', x='importance', palette='viridis')
        plt.title(f'Top {top_n} Feature Importances - {model_name}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Feature importance plot saved to {save_path}")
        
        plt.show()
        
        return feature_importance_df
    
    def evaluate_model(self, model, X_test, y_test, model_name='Model', 
                      feature_names=None, save_dir='visualizations/model_performance'):
        """
        Complete model evaluation with all metrics and visualizations
        
        Parameters:
        -----------
        model : trained model
            Model to evaluate
        X_test : array-like
            Test features
        y_test : array-like
            Test target
        model_name : str
            Name of the model
        feature_names : list
            List of feature names
        save_dir : str
            Directory to save visualizations
            
        Returns:
        --------
        dict
            Dictionary of metrics
        """
        print("\n" + "="*80)
        print(f"EVALUATING: {model_name}")
        print("="*80)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = None
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba, model_name)
        self.print_metrics(metrics)
        
        # Print classification report
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Not Clicked', 'Clicked']))
        
        # Create visualizations
        print("\n📊 Generating visualizations...")
        
        # Confusion Matrix
        cm_path = f"{save_dir}/confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
        self.plot_confusion_matrix(y_test, y_pred, model_name, save_path=cm_path)
        
        # ROC Curve
        if y_pred_proba is not None:
            roc_path = f"{save_dir}/roc_curve_{model_name.lower().replace(' ', '_')}.png"
            self.plot_roc_curve(y_test, y_pred_proba, model_name, save_path=roc_path)
            
            # Precision-Recall Curve
            pr_path = f"{save_dir}/precision_recall_{model_name.lower().replace(' ', '_')}.png"
            self.plot_precision_recall_curve(y_test, y_pred_proba, model_name, save_path=pr_path)
        
        # Feature Importance
        if feature_names is not None:
            fi_path = f"{save_dir}/feature_importance_{model_name.lower().replace(' ', '_')}.png"
            self.plot_feature_importance(model, feature_names, model_name=model_name, save_path=fi_path)
        
        print("\n✓ Evaluation completed")
        
        return metrics
    
    def compare_models(self, save_path='visualizations/model_performance/model_comparison.png'):
        """
        Compare all evaluated models
        
        Parameters:
        -----------
        save_path : str
            Path to save the comparison plot
            
        Returns:
        --------
        pandas.DataFrame
            Comparison table
        """
        if not self.results:
            print("⚠️  No models evaluated yet")
            return None
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(self.results).T
        
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        print(comparison_df.to_string())
        print("="*80)
        
        # Plot comparison
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        metrics_to_plot = [m for m in metrics_to_plot if m in comparison_df.columns]
        
        fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(20, 5))
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx] if len(metrics_to_plot) > 1 else axes
            comparison_df[metric].plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
            ax.set_title(metric, fontsize=12, fontweight='bold')
            ax.set_ylabel('Score', fontsize=10)
            ax.set_ylim([0, 1])
            ax.grid(axis='y', alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Model comparison plot saved to {save_path}")
        
        plt.show()
        
        return comparison_df


if __name__ == "__main__":
    print("Model Evaluation Module")
    print("Use ModelEvaluator() to evaluate machine learning models")
