import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from scipy import stats
import librosa
import librosa.display

def evaluate_predictions(y_true, y_pred, dataset_name="Test", save_dir=None):
    """Calculates accuracy and macro-F1, and optionally saves to CSV."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"[{dataset_name}] Accuracy: {acc:.4f}, Macro-F1: {f1:.4f}")
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        df = pd.DataFrame([{"Dataset": dataset_name, "Accuracy": acc, "Macro-F1": f1}])
        file_path = os.path.join(save_dir, f"{dataset_name.replace(' ', '_')}_metrics.csv")
        df.to_csv(file_path, index=False)
        
    return acc, f1

def plot_conf_matrix(y_true, y_pred, classes, title="Confusion Matrix", save_path=None):
    """Plots a heatmap confusion matrix and optionally saves the figure."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()

def run_paired_ttest(scores_model_A, scores_model_B, model_names=("Model A", "Model B"), save_path=None):
    """Runs an exploratory paired t-test on 5-fold CV scores and saves the result."""
    scores_model_A = np.array(scores_model_A)
    scores_model_B = np.array(scores_model_B)
    
    diff = scores_model_A - scores_model_B
    mean_diff = np.mean(diff)
    t_stat, p_val = stats.ttest_rel(scores_model_A, scores_model_B)
    
    res_str = f"--- Paired t-test: {model_names[0]} vs {model_names[1]} ---\n"
    res_str += f"Per-fold differences: {np.round(diff, 4)}\n"
    res_str += f"Mean difference: {mean_diff:.4f}\n"
    res_str += f"t-statistic: {t_stat:.4f}, p-value: {p_val:.4f}\n"
    if p_val < 0.05:
        res_str += "Result is statistically significant at p < 0.05\n"
    else:
        res_str += "Result is NOT statistically significant at p < 0.05\n"
    
    print(res_str)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(res_str)

def analyze_errors(X_test_raw_paths, y_test, y_pred, num_examples=5, save_dir=None):
    """Finds misclassified examples and plots their waveforms and spectrograms."""
    misclassified_idx = np.where(y_test != y_pred)[0]
    print(f"\nTotal misclassified test samples: {len(misclassified_idx)}")
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
    for i, idx in enumerate(misclassified_idx[:num_examples]):
        true_label = y_test[idx]
        pred_label = y_pred[idx]
        audio_path = X_test_raw_paths[idx]
        
        y, sr = librosa.load(audio_path, sr=None)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Waveform
        librosa.display.waveshow(y, sr=sr, ax=axes[0])
        axes[0].set_title(f"Waveform - True: {true_label}, Pred: {pred_label}")
        
        # Spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=axes[1])
        axes[1].set_title(f"Spectrogram - True: {true_label}, Pred: {pred_label}")
        
        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"error_{i+1}_true{true_label}_pred{pred_label}.png"), dpi=300)
        plt.show()