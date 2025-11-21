"""
Funciones auxiliares para entrenamiento y evaluación de modelos de Machine Learning y Deep Learning.

Este módulo contiene utilidades para:
- Entrenamiento de redes neuronales con PyTorch
- Evaluación y visualización de resultados
- Gestión de métricas en archivos JSON
"""

import numpy as np
import pandas as pd
import torch
import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, ConfusionMatrixDisplay

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt


# =============================================================================
# GESTIÓN DE MÉTRICAS (JSON)
# =============================================================================

def save_model_metrics(model_name, metrics, output_dir="./metricas"):
    """
    Guarda las métricas de un modelo en un archivo JSON.

    Args:
        model_name: Nombre del modelo (se usa para el nombre del archivo)
        metrics: Diccionario con las métricas del modelo
        output_dir: Directorio donde guardar los archivos JSON (default: ./metricas)
    """
    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)

    # Nombre del archivo: convertir a snake_case y eliminar caracteres especiales
    filename = f"{model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_metrics.json"
    filepath = os.path.join(output_dir, filename)

    # Guardar métricas
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    print(f"✓ Métricas guardadas en: {filepath}")


def load_all_metrics(metrics_dir="./metricas"):
    """
    Carga todas las métricas desde archivos JSON en un directorio.

    Args:
        metrics_dir: Directorio donde están los archivos JSON (default: ./metricas)

    Returns:
        dict: Diccionario con todos los modelos y sus métricas
            Formato: {model_name: {metric_name: value, ...}, ...}
    """
    all_metrics = {}

    # Buscar todos los archivos JSON que terminen en _metrics.json
    for filename in os.listdir(metrics_dir):
        if filename.endswith('_metrics.json'):
            filepath = os.path.join(metrics_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                model_data = json.load(f)
                # El nombre del modelo está en el JSON
                model_name = model_data.get('model_name', filename.replace('_metrics.json', ''))
                all_metrics[model_name] = model_data

    return all_metrics

def muestra_metricas(model_name, y_test, y_predic, start_time=None, end_train_time=None, end_predict_time=None):
    """
    Función simple para personalizar las métrica a imprimir para cada modelo analizado.
    model_name: string con el nombre del modelo
    y_test: recibe el vector de salida real
    y_predic: recibe el vector de salida predicho por el modelo
    """

    accuracy = accuracy_score(y_test, y_predic)
    recall = recall_score(y_test, y_predic, average='weighted') # average permite definir cómo se computan estas métricas para múltiples clases de salida
    precision = precision_score(y_test, y_predic, average='weighted')
    f1s = f1_score(y_test, y_predic, average='weighted')
    MCC = matthews_corrcoef(y_test, y_predic)

    print("#######################################")
    print("Accuracy (Exactitud): "+ "{:.2%}".format(accuracy))
    print("Recall (Recuperación): "+ "{:.2%}".format(recall))
    print("Precision (Precisión): "+ "{:.2%}".format(precision))
    print("F1-Score: "+ "{:.2%}".format(f1s))
    print("MCC (Matthews Correlation Coefficient): "+ "{:.2%}".format(MCC))    # Matthews correlation coefficient: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html

    if (start_time != None and end_train_time != None and end_predict_time != None):
        print("Tiempo de entrenamiento: {:.4f} s".format(end_train_time-start_time))
        print("Tiempo de predicción: {:.4f} s".format(end_predict_time-end_train_time))
        print("Tiempo total: {:.4f} s".format(end_predict_time-start_time))

    print("#######################################")
    model_performance = pd.DataFrame(columns=['Accuracy','Recall','Precision','F1-Score','MCC score','Entrenamiento (s)','Predicción (s)','Tiempo Total (s)'])
    model_performance.loc[model_name] = [accuracy, recall, precision, f1s, MCC, end_train_time-start_time, end_predict_time-end_train_time, end_predict_time-start_time]


def plot_confusion_matrix(y_true, y_preds, labels=None, model_name=None):
    """
    Función para graficar la Matrix de confusión
    y_true: vector de salida real
    y_predic: vector de salida predicho por el modelo
    labels: opcional, para nombrar las clases. Lista con strings
    model_name: opcional, para poner un título al gráfico
    """
    cm = confusion_matrix(y_true, y_preds)
    cmd = ConfusionMatrixDisplay(cm, display_labels=labels)
    cmd.plot()
    cmd.ax_.set(xlabel='Clase Predicha', ylabel='Clase Real')
    if model_name:
        cmd.ax_.set(title=model_name)

    # Midamos la sensibilidad y especificidad
    if model_name:
        print("Clasificador: {}".format(model_name))
    P = np.sum(cm[1, :])
    N = np.sum(cm[0, :])
    TP = cm[1, 1]
    TN = cm[0, 0]
    TPR = TP/P
    TNR = TN/N
    print("Sensibilidad: {:.4f}".format(TP/P))
    print("Especificidad: {:.4f}".format(TN/N))
    print("Exactitud balanceada: {:.4f}".format((TPR + TNR)/2))

# =============================================================================
# DEEP LEARNING - ENTRENAMIENTO
# =============================================================================

class BinaryNetwork(torch.nn.Module):
    """
    Red neuronal para clasificación binaria.

    Esta clase encapsula una secuencia de capas de PyTorch y permite opcionalmente
    aplicar sigmoid en la salida según la función de pérdida utilizada.

    Args:
        layers: torch.nn.Sequential con las capas de la red
        use_sigmoid: Si True, aplica sigmoid en forward.
                     Usar True con BCELoss, False con BCEWithLogitsLoss
    """
    def __init__(self, layers, use_sigmoid=True):
        """
        Inicializa el modelo BinaryNetwork.

        Args:
            layers: Secuencia de capas de PyTorch (torch.nn.Sequential)
            use_sigmoid: Si aplicar sigmoid en forward (default: True)
        """
        super().__init__()
        self.layers = layers
        self.use_sigmoid = use_sigmoid

    def forward(self, x):
        """
        Paso hacia adelante a través de la red.

        Args:
            x: Tensor de entrada de forma (batch_size, input_size)

        Returns:
            Tensor de salida con las predicciones
        """
        x = self.layers(x)
        if self.use_sigmoid:
            return torch.sigmoid(x)
        return x


def train_model(model, train_loader, val_loader, loss_fn, optimizer,
                epochs=450, max_patience=5, patience_alpha=0.001,
                verbose=True, print_every=1):
    """
    Entrena un modelo de red neuronal con early stopping.

    Args:
        model: Modelo de PyTorch a entrenar
        train_loader: DataLoader de entrenamiento
        val_loader: DataLoader de validación
        loss_fn: Función de pérdida (ej: BCELoss, BCEWithLogitsLoss)
        optimizer: Optimizador de PyTorch (ej: Adam)
        epochs: Número máximo de épocas (default: 450)
        max_patience: Paciencia para early stopping (default: 5)
        patience_alpha: Umbral de mejora para early stopping (default: 0.001)
        verbose: Si imprimir progreso (default: True)
        print_every: Imprimir cada N épocas (default: 1)

    Returns:
        dict: Diccionario con historial de entrenamiento
            - 'train_losses': Lista de pérdidas de entrenamiento
            - 'val_losses': Lista de pérdidas de validación
            - 'val_accuracies': Lista de accuracies de validación
            - 'best_val_loss': Mejor pérdida de validación alcanzada
            - 'best_val_acc': Mejor accuracy de validación alcanzada
            - 'epochs_trained': Número de épocas entrenadas
    """
    patience = 0
    best_val_loss = np.inf
    best_val_acc = 0
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(1, epochs + 1):
        # Entrenamiento
        model.train()
        train_loss = 0.0
        for features, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = loss_fn(outputs, targets.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * features.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validación
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for features, targets in val_loader:
                outputs = model(features)
                loss = loss_fn(outputs, targets.unsqueeze(1))
                val_loss += loss.item() * features.size(0)
                val_preds.extend(outputs.round().squeeze().tolist())
                val_targets.extend(targets.tolist())
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        # Calcular métricas de validación
        val_accuracy = accuracy_score(val_targets, val_preds)
        val_accuracies.append(val_accuracy)

        # Trackear mejor accuracy
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy

        # Print progreso
        if verbose and epoch % print_every == 0:
            print(f'Época: {epoch}/{epochs}, Train Loss: {train_loss:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')

        # Early stopping
        if val_loss < best_val_loss * (1 - patience_alpha):
            best_val_loss = val_loss
            patience = 0
        else:
            patience += 1
            if patience >= max_patience:
                if verbose:
                    print(f"Early stopping en época {epoch}")
                break

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc,
        'epochs_trained': epoch
    }


# =============================================================================
# DEEP LEARNING - EVALUACIÓN
# =============================================================================

def evaluate_model(model, test_loader, y_test, use_sigmoid=True):
    """
    Evalúa un modelo de PyTorch en el conjunto de test.

    Args:
        model: Modelo de PyTorch a evaluar
        test_loader: DataLoader de test
        y_test: Labels verdaderos de test (no usado, se mantiene por compatibilidad)
        use_sigmoid: Si aplicar sigmoid a las salidas (False para BCEWithLogitsLoss)

    Returns:
        dict: Diccionario con predicciones y probabilidades
            - 'y_pred': Predicciones binarias (0 o 1)
            - 'y_pred_proba': Probabilidades de la clase positiva
    """
    y_pred_list = []
    y_pred_proba_list = []

    model.eval()
    with torch.no_grad():
        for X, y in test_loader:
            outputs = model(X)

            if not use_sigmoid:
                # Si el modelo no tiene sigmoid, aplicarlo manualmente
                proba = torch.sigmoid(outputs)
            else:
                proba = outputs

            y_pred_proba_list.append(proba)
            y_pred_list.append(torch.round(proba))

    y_pred_tensor = torch.cat(y_pred_list).squeeze()
    y_pred_proba_tensor = torch.cat(y_pred_proba_list).squeeze()

    return {
        'y_pred': y_pred_tensor,
        'y_pred_proba': y_pred_proba_tensor
    }


# =============================================================================
# VISUALIZACIÓN - DEEP LEARNING
# =============================================================================

def plot_training_history(history, title="Training History"):
    """
    Grafica las curvas de entrenamiento (Loss y Accuracy).

    Args:
        history: Diccionario retornado por train_model()
        title: Título de la figura
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Loss
    axes[0].plot(history['train_losses'], label='Train Loss', alpha=0.8)
    axes[0].plot(history['val_losses'], label='Val Loss', alpha=0.8)
    axes[0].set_xlabel('Época', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Curvas de Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(history['val_accuracies'], label='Val Accuracy', color='green', alpha=0.8)
    axes[1].axhline(y=history['best_val_acc'], color='r', linestyle='--',
                    label=f'Best: {history["best_val_acc"]:.4f}')
    axes[1].set_xlabel('Época', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Accuracy en Validación', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


# =============================================================================
# VISUALIZACIÓN - GENERAL (ML Y DL)
# =============================================================================

def plot_evaluation_results(y_test, y_pred, y_pred_proba=None, title="Model Evaluation", labels=None):
    """
    Grafica ROC curve y matriz de confusión para cualquier modelo.

    Args:
        y_test: Labels verdaderos
        y_pred: Predicciones binarias
        y_pred_proba: Probabilidades (opcional, si no se provee solo muestra confusion matrix)
        title: Título de la figura
        labels: Labels para la matriz de confusión (default: ['No Arrestado', 'Arrestado'])

    Returns:
        np.array: Matriz de confusión
    """
    from sklearn.metrics import RocCurveDisplay, confusion_matrix, ConfusionMatrixDisplay

    if labels is None:
        labels = ['No Arrestado', 'Arrestado']

    # Si hay probabilidades, mostrar ROC + Confusion Matrix
    if y_pred_proba is not None:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # ROC Curve
        RocCurveDisplay.from_predictions(y_test, y_pred_proba, ax=axes[0])
        axes[0].set_title('Curva ROC', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=labels
        )
        disp.plot(ax=axes[1], cmap='Blues', values_format='d', colorbar=True)
        axes[1].set_title('Matriz de Confusión', fontsize=14, fontweight='bold')
        axes[1].grid(False)

    else:
        # Solo Confusion Matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=labels
        )
        disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=True)
        ax.set_title('Matriz de Confusión', fontsize=14, fontweight='bold')
        ax.grid(False)

    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

    return cm


def print_metrics(y_test, y_pred, cm=None, title="METRICS"):
    """
    Imprime métricas de evaluación de forma detallada.

    Args:
        y_test: Labels verdaderos
        y_pred: Predicciones binarias
        cm: Matriz de confusión (opcional, se calcula si no se provee)
        title: Título del reporte
    """
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

    if cm is None:
        cm = confusion_matrix(y_test, y_pred)

    print(f"{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f} ({accuracy_score(y_test, y_pred)*100:.1f}%)")
    print(f"F1-score: {f1_score(y_test, y_pred):.4f}")

    # Reporte completo
    print(f"\n{classification_report(y_test, y_pred, target_names=['No Arrestado', 'Arrestado'])}")

    # Desglose de matriz de confusión
    tn, fp, fn, tp = cm.ravel()
    print(f"\nDesglose de la Matriz de Confusión:")
    print(f"  Verdaderos Negativos (TN): {tn:,}")
    print(f"  Falsos Positivos (FP):     {fp:,}")
    print(f"  Falsos Negativos (FN):     {fn:,}")
    print(f"  Verdaderos Positivos (TP): {tp:,}")
    print(f"\nMétricas derivadas:")
    print(f"  Precisión (Precision): {tp/(tp+fp):.4f}")
    print(f"  Recall (Sensibilidad): {tp/(tp+fn):.4f}")
    print(f"  Especificidad: {tn/(tn+fp):.4f}")


def get_model_color(name):
    """
    Asigna un color a cada modelo basándose en su familia/tipo.

    Args:
        name: Nombre del modelo

    Returns:
        str: Código de color para matplotlib
            - 'coral': Redes Neuronales (Deep Learning)
            - 'forestgreen': Boosting (XGBoost, AdaBoost)
            - 'mediumseagreen': Ensembles con Bagging
            - 'lightgreen': Árboles de Decisión
            - 'steelblue': Modelos Lineales
    """
    if 'Deep Learning' in name:
        return 'coral'
    elif 'XGBoost' in name or 'AdaBoost' in name:
        return 'forestgreen'
    elif 'Random Forest' in name or 'Bagging' in name:
        return 'mediumseagreen'
    elif 'Árbol' in name or 'Tree' in name:
        return 'lightgreen'
    else:
        return 'steelblue'
