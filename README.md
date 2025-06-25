```mermaid
flowchart LR
    subgraph extraer_datos
        A[extraccion_de_datos.py]
    end
    
    subgraph pre_web
        B[preweb.py]
    end
    
    subgraph entrenamiento
        C[entrenamiento.html]
    end
    
    A -->|X_hand_landmarks.npy| B
    A -->|y_labels.npy| B
    B -->|labels.json| C
    B -->|mode_info.json| C
    B -->|train_data.json| C
    C -->|asl-model-tfjs.json| D[probar_modelo.html]
    C -->|train_data.json| D
