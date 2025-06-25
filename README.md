# senials.github.io

Bienvenido a mi repositorio. Este documento muestra dos formas de incluir gráficos en GitHub:

## 1. Gráfico de barras usando Mermaid

Este diagrama muestra ventas mensuales usando la sintaxis nativa de GitHub:

```mermaid
bar
    title Ventas por Mes
    xAxis Ene Feb Mar Abr May Jun
    yAxis 0 100 200 300
    bar 400
    bar 300
    bar 200
    bar 100
    bar 50
    bar 300
```

**Ventajas de Mermaid:**  
- Se edita directamente como texto  
- No requiere archivos externos  
- Soporte nativo en GitHub  

## 2. Gráfico como imagen externa

Este es un gráfico generado externamente e incluido como imagen:

![Gráfico de ventas generado con Python](image.png)

**Características de la imagen:**  
- Mayor flexibilidad de diseño  
- Ideal para gráficos complejos  
- Requiere actualizar el archivo cuando cambian los datos  

## Análisis comparativo

| Característica       | Mermaid | Imagen |
|----------------------|---------|--------|
| Fácil de actualizar | ✔️      | ❌     |
| Calidad visual       | ⭐⭐     | ⭐⭐⭐⭐  |
| Soporte complejidad  | Básico  | Alto   |
| Dependencias         | Ninguna | Editor externo |

## Conclusión
Ambos métodos son válidos para mostrar información visual en GitHub. Para datos dinámicos o actualizaciones frecuentes recomiendo Mermaid, mientras que para gráficos de alta calidad con estilo personalizado es mejor usar imágenes externas.

> **Tip**: Puedes combinar ambas técnicas según tus necesidades específicas