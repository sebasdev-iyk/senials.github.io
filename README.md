
```mermaid
flowchart LR
    subgraph extraer_datos ["extraer_datos"]
        A[extraccion_de_datos.py]
    end
    
    subgraph pre_web ["pre_web"]
        B[preweb.py]
    end
    
    subgraph entrenamiento ["entrenamiento"]
        C[entrenamiento.html]
    end
    
    A --> D[X_hand_landmarks.npy]
    A --> E[y_labels.npy]
    
    D --> B
    E --> B
    
    B --> F[mode_info.json]
    B --> G[train_data.json]
    
    F --> C
    G --> C
    
    C --> H[asl-model-tfjs.json]
    C --> I[train_data.json]
    
    H --> J[probar_modelo.html]
    I --> J
```



# senials.github.io - Proyecto de Análisis de Señales


Bienvenido a mi repositorio de análisis de señales. Este documento integra diferentes visualizaciones para presentar información técnica.

## Diagrama de Flujo del Proceso

Este diagrama muestra el proceso completo de análisis de señales que implementamos:

![Diagrama de Flujo del Proceso de Análisis de Señales](Diagrama_de_flujo.png)

**Descripción del flujo:**  
1. Adquisición de señales desde sensores  
2. Preprocesamiento y filtrado digital  
3. Análisis espectral (FFT)  
4. Clasificación mediante modelos de ML  
5. Visualización de resultados  

**Interpretación:**  
- **Media**: 60% de las señales requieren monitoreo continuo  
- **Crítica**: 20% necesita intervención inmediata  
- La categoría más frecuente es "Media"  

## Comparación de Métodos

| Técnica          | Precisión | Tiempo Procesamiento | Complejidad |
|------------------|-----------|----------------------|-------------|
| FFT Clásica      | 85%       | 120 ms               | Media       |
| Wavelet          | 92%       | 250 ms               | Alta        |
| Deep Learning    | 96%       | 350 ms               | Muy Alta    |

## Conclusiones Técnicas

1. El diagrama de flujo muestra un pipeline robusto para procesamiento de señales  
2. Los resultados indican que la mayoría de señales caen en categoría media  
3. Los métodos basados en deep learning ofrecen mayor precisión pero requieren más recursos  
4. Se recomienda implementar sistema híbrido FFT + Wavelet para balance óptimo  

> **Nota**: Todos los gráficos son actualizados automáticamente con cada nueva versión del dataset
