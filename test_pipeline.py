import unittest
from pipeline import evaluacion 
import numpy as np

class PruebaMLFlow(unittest.TestCase):

    def test_model(self):
        # Llamar a la función de evaluación para obtener todas las métricas
        (
            accuracy_score1, roc_auc_score1, 
            accuracy_score2, roc_auc_score2, 
            accuracy_score3, roc_auc_score3, 
            accuracy_score4, roc_auc_score4, 
            accuracy1_test, roc1_test, 
            accuracy2_test, roc2_test, 
            accuracy3_test, roc3_test, 
            accuracy4_test, roc4_test, 
            resultados_df
        ) = evaluacion() 

        # Aserción para verificar si el proceso fue exitoso
        mensaje = "Proceso Exitoso"
        self.assertEqual(mensaje, "Proceso Exitoso", "El proceso no fue exitoso")

        # Evaluar la diferencia de precisión entre entrenamiento y prueba para todos los modelos
        diferencias = [
            np.abs(accuracy_score1 - accuracy1_test),
            np.abs(accuracy_score2 - accuracy2_test),
            np.abs(accuracy_score3 - accuracy3_test),
            np.abs(accuracy_score4 - accuracy4_test)
        ]

        # Verificar si hay overfitting/underfitting
        for i, diferencia in enumerate(diferencias, start=1):
            if diferencia <= 10:
                print(f"No presenta Underfitting ni Overfitting en el modelo {i}")
            else:
                print(f"Posible Overfitting o Underfitting en el modelo {i}")
            self.assertLessEqual(diferencia, 10, f"Modelo {i}: Hay una diferencia significativa, posible Overfitting/Underfitting")

        # Imprimir los resultados de precisión y AUC
        print("\n--- Resultados de Entrenamiento ---")
        print(f"Modelo 1 (Bert sin Ingeniería) - Accuracy: {accuracy_score1:.4f}, ROC AUC: {roc_auc_score1:.4f}")
        print(f"Modelo 2 (Bert con Ingeniería) - Accuracy: {accuracy_score2:.4f}, ROC AUC: {roc_auc_score2:.4f}")
        print(f"Modelo 3 (Fast sin Ingeniería) - Accuracy: {accuracy_score3:.4f}, ROC AUC: {roc_auc_score3:.4f}")
        print(f"Modelo 4 (Fast con Ingeniería) - Accuracy: {accuracy_score4:.4f}, ROC AUC: {roc_auc_score4:.4f}")

        print("\n--- Resultados de Prueba ---")
        print(f"Modelo 1 (Bert sin Ingeniería) - Accuracy: {accuracy1_test:.4f}, ROC AUC: {roc1_test:.4f}")
        print(f"Modelo 2 (Bert con Ingeniería) - Accuracy: {accuracy2_test:.4f}, ROC AUC: {roc2_test:.4f}")
        print(f"Modelo 3 (Fast sin Ingeniería) - Accuracy: {accuracy3_test:.4f}, ROC AUC: {roc3_test:.4f}")
        print(f"Modelo 4 (Fast con Ingeniería) - Accuracy: {accuracy4_test:.4f}, ROC AUC: {roc4_test:.4f}")

        # Guardar los resultados en un archivo CSV si es necesario
        if not resultados_df.empty:
            resultados_df.to_csv("resultados_modelos.csv", index=False)
            print("\nResultados guardados en 'resultados_modelos.csv'.")

if __name__ == "__main__":
    unittest.main()
