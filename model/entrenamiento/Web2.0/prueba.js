// Variable global para la instancia de utilidades del modelo (o sus datos)
let modelInfo = null;
let trainData = null;
let currentModel = null; // Renombrado para evitar conflicto con la función createModel
let isTraining = false;

// Referencias a elementos del DOM (se obtienen una vez al inicio)
const modelInfoFile = document.getElementById('modelInfoFile');
const trainDataFile = document.getElementById('trainDataFile');
const modelInfoStatus = document.getElementById('modelInfoStatus');
const trainDataStatus = document.getElementById('trainDataStatus');
const createModelBtn = document.getElementById('createModelBtn');
const trainBtn = document.getElementById('trainBtn');
const saveBtn = document.getElementById('saveBtn');
const logContainer = document.getElementById('logContainer');
const modelStatus = document.getElementById('modelStatus');
const trainStatus = document.getElementById('trainStatus');
const saveStatus = document.getElementById('saveStatus');
const modelInfoDiv = document.getElementById('modelInfo');
const progressContainer = document.getElementById('progressContainer');
const numClassesElem = document.getElementById('numClasses');
const trainSamplesElem = document.getElementById('trainSamples');
const testSamplesElem = document.getElementById('testSamples');
const inputFeaturesElem = document.getElementById('inputFeatures');
const progressFill = document.getElementById('progressFill');
const epochInfo = document.getElementById('epochInfo');
const lossInfo = document.getElementById('lossInfo');
const accInfo = document.getElementById('accInfo');
const valLossInfo = document.getElementById('valLossInfo'); // Corregido: ya no es una asignación doble
const valAccInfo = document.getElementById('valAccInfo');


// --- Funciones de Utilidad (Adaptadas de la clase ASLModelUtils) ---

/**
 * Agrega un mensaje al contenedor de logs en la interfaz.
 * @param {string} message - El mensaje a mostrar.
 * @param {string} type - El tipo de mensaje ('info', 'success', 'error').
 */
function addLog(message, type = 'info') {
    const logEntry = document.createElement('div');
    logEntry.className = `log-entry ${type}`;

    let iconClass = '';
    switch (type) {
        case 'info':
            iconClass = 'fa-solid fa-circle-info'; // Icono para información
            break;
        case 'success':
            iconClass = 'fa-solid fa-circle-check'; // Icono para éxito
            break;
        case 'error':
            iconClass = 'fa-solid fa-circle-exclamation'; // Icono para error
            break;
        default:
            iconClass = 'fa-solid fa-circle-info';
    }

    // Usamos innerHTML para insertar el icono directamente
    logEntry.innerHTML = `<i class="${iconClass}"></i><span>${message}</span>`;
    logContainer.appendChild(logEntry);
    logContainer.scrollTop = logContainer.scrollHeight;
}

/**
 * Lee un archivo JSON desde un objeto File.
 * @param {File} file - El archivo JSON a leer.
 * @returns {Promise<Object>} Una promesa que resuelve con el contenido JSON del archivo.
 */
function readJSONFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const json = JSON.parse(e.target.result);
                resolve(json);
            } catch (error) {
                reject(new Error(`Error analizando JSON: ${error.message}`));
            }
        };
        reader.onerror = () => reject(new Error('Error leyendo archivo'));
        reader.readAsText(file);
    });
}

/**
 * Crea la arquitectura del modelo de TensorFlow.js.
 * Requiere que 'modelInfo' esté cargado.
 * @returns {tf.LayersModel} El modelo de TensorFlow.js compilado.
 */
function createASLModel() {
    if (!modelInfo) {
        addLog('¡Alto ahí! Falta la información esencial del modelo. Asegúrate de cargarla.', 'error');
        throw new Error('Model info not loaded.');
    }

    const inputShape = modelInfo.input_shape;
    const numClasses = modelInfo.num_classes;

    const model = tf.sequential({
        layers: [
            tf.layers.dense({
                inputShape: [inputShape],
                units: 256,
                activation: 'relu',
                kernelRegularizer: tf.regularizers.l2({ l2: 0.001 })
            }),
            tf.layers.batchNormalization(),
            tf.layers.dropout({ rate: 0.4 }),

            tf.layers.dense({
                units: 128,
                activation: 'relu',
                kernelRegularizer: tf.regularizers.l2({ l2: 0.001 })
            }),
            tf.layers.batchNormalization(),
            tf.layers.dropout({ rate: 0.3 }),

            tf.layers.dense({
                units: 64,
                activation: 'relu'
            }),
            tf.layers.dropout({ rate: 0.2 }),

            tf.layers.dense({
                units: numClasses,
                activation: 'softmax'
            })
        ]
    });

    // Se compila aquí para que el modelo esté listo para el entrenamiento o evaluación
    model.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    return model;
}

/**
 * Prepara los tensores de entrenamiento y prueba a partir de los datos cargados.
 * Requiere que 'modelInfo' y 'trainData' estén cargados.
 * @returns {{xTrain: tf.Tensor2D, yTrain: tf.Tensor, xTest: tf.Tensor2D, yTest: tf.Tensor}} Los tensores preparados.
 */
function prepareTrainingData() {
    if (!trainData || !modelInfo) {
        addLog('Parece que el modelo tiene hambre... ¡faltan datos de entrenamiento o su receta!.', 'error');
        throw new Error('Training data or model info not loaded.');
    }
    const xTrain = tf.tensor2d(trainData.X_train);
    const yTrain = tf.oneHot(tf.tensor1d(trainData.y_train, 'int32'), modelInfo.num_classes);
    
    const xTest = tf.tensor2d(trainData.X_test);
    const yTest = tf.oneHot(tf.tensor1d(trainData.y_test, 'int32'), modelInfo.num_classes);

    return { xTrain, yTrain, xTest, yTest };
}

/**
 * Guarda el modelo de TensorFlow.js en el navegador.
 * @param {tf.LayersModel} model - El modelo a guardar.
 * @param {string} modelName - El nombre para el archivo del modelo.
 * @returns {Promise<boolean>} Verdadero si se guarda exitosamente, falso en caso contrario.
 */
async function saveASLModel(model, modelName = 'asl-model-tfjs') {
    try {
        addLog('Guardando modelo... ¡Casi en casa!'); // Eliminado el icono literal
        await model.save(`downloads://${modelName}`);
        addLog('¡Modelo guardado exitosamente! Listo para conquistar el mundo.', 'success'); // Eliminado el icono literal
        updateControlStatus(saveStatus, 'saved'); // Actualiza el estado visual del botón de guardar
        return true;
    } catch (error) {
        addLog(`Uhm... algo salió mal al guardar el modelo. Detalles: ${error.message}`, 'error');
        updateControlStatus(saveStatus, 'error'); // Actualiza el estado visual a error
        saveBtn.disabled = false;
        return false;
    }
}

/**
 * Actualiza la interfaz de usuario con el progreso del entrenamiento.
 * @param {number} epoch - Época actual.
 * @param {number} maxEpochs - Número total de épocas.
 * @param {Object} logs - Objeto con las métricas de la época (loss, acc, val_loss, val_acc).
 */
function updateUIProgress(epoch, maxEpochs, logs) {
    const progress = (epoch / maxEpochs) * 100;
    
    progressFill.style.width = `${progress}%`;
    epochInfo.textContent = `Época: ${epoch}/${maxEpochs}`;
    lossInfo.textContent = `Loss: ${logs.loss.toFixed(4)}`;
    accInfo.textContent = `Accuracy: ${logs.acc.toFixed(4)}`;
    valLossInfo.textContent = `Val Loss: ${logs.val_loss.toFixed(4)}`;
    valAccInfo.textContent = `Val Accuracy: ${logs.val_acc.toFixed(4)}`;
}

/**
 * Actualiza el estado visual de un elemento de status de archivo (modelInfoStatus, trainDataStatus).
 * @param {HTMLElement} element - El elemento DOM a actualizar.
 * @param {string} statusType - 'pending', 'loading', 'success', 'error'.
 * @param {string} message - El texto del mensaje.
 */
function updateFileStatusUI(element, statusType, message) {
    element.className = `file-status file-status--${statusType}`;
    let iconClass = '';
    switch (statusType) {
        case 'pending':
            iconClass = 'fa-solid fa-hourglass-half';
            break;
        case 'loading':
            iconClass = 'fa-solid fa-spinner fa-spin'; // Icono de carga giratorio
            break;
        case 'success':
            iconClass = 'fa-solid fa-check-circle';
            break;
        case 'error':
            iconClass = 'fa-solid fa-times-circle';
            break;
    }
    element.innerHTML = `<i class="${iconClass}"></i> ${message}`;
}

/**
 * Actualiza el estado visual de los elementos de control (modelStatus, trainStatus, saveStatus).
 * @param {HTMLElement} element - El elemento DOM a actualizar (e.g., modelStatus, trainStatus).
 * @param {string} type - 'waiting', 'created', 'ready', 'training', 'completed', 'not_saved', 'saved', 'error'.
 */
function updateControlStatus(element, type) {
    let iconClass = '';
    let message = '';
    
    // Remover todas las clases de estado para evitar conflictos
    element.className = 'control-status';

    switch (type) {
        case 'waiting':
            iconClass = 'fa-solid fa-hourglass-start';
            message = 'Esperando datos';
            break;
        case 'created':
            iconClass = 'fa-solid fa-cubes';
            message = 'Modelo creado';
            break;
        case 'ready': // Listo para entrenar/crear, estado inicial de botón
            iconClass = 'fa-solid fa-circle-play';
            message = 'Listo para entrenar';
            break;
        case 'training':
            iconClass = 'fa-solid fa-dumbbell fa-spin'; // Icono de entrenamiento giratorio
            message = 'Entrenando...';
            break;
        case 'completed':
            iconClass = 'fa-solid fa-check-double';
            message = 'Entrenamiento completado';
            break;
        case 'not_saved':
            iconClass = 'fa-solid fa-cloud-arrow-up';
            message = 'Modelo no guardado';
            break;
        case 'saved':
            iconClass = 'fa-solid fa-check';
            message = 'Modelo guardado';
            break;
        case 'error':
            iconClass = 'fa-solid fa-triangle-exclamation';
            message = 'Error';
            break;
        default:
            iconClass = 'fa-solid fa-circle-info';
            message = 'Estado desconocido';
            break;
    }
    element.innerHTML = `<i class="${iconClass}"></i> ${message}`;
}


// --- Lógica de la Interfaz de Usuario ---

/**
 * Verifica si ambos archivos de datos han sido cargados y actualiza la UI.
 */
function checkDataReady() {
    if (modelInfo && trainData) {
        createModelBtn.disabled = false;
        // Ahora que los datos están listos, el estado del modelo puede pasar a "Listo para crear"
        updateControlStatus(modelStatus, 'ready'); 
        
        // Mostrar información del modelo
        modelInfoDiv.style.display = 'grid'; // Asegurarse de que el div sea un grid
        numClassesElem.textContent = modelInfo.num_classes;
        trainSamplesElem.textContent = trainData.X_train.length; // Usar el tamaño real de X_train
        testSamplesElem.textContent = trainData.X_test.length;   // Usar el tamaño real de X_test
        inputFeaturesElem.textContent = modelInfo.input_shape;
        
        addLog(`Datos listos: ${modelInfo.num_classes} clases, ${trainData.X_train.length} muestras de entrenamiento. ¡A construir se ha dicho!`, 'success');
    }
}

// --- Manejadores de Eventos ---

// Manejar carga de model_info.json
modelInfoFile.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (file) {
        updateFileStatusUI(modelInfoStatus, 'loading', 'Cargando...');
        try {
            addLog('Cargando model_info.json...');
            modelInfo = await readJSONFile(file);
            updateFileStatusUI(modelInfoStatus, 'success', 'model_info.json cargado');
            addLog('model_info.json cargado exitosamente. ¡La base está sentada!', 'success');
            checkDataReady();
        } catch (error) {
            addLog(`¡Houston, tenemos un problema con model_info.json! Error: ${error.message}`, 'error');
            updateFileStatusUI(modelInfoStatus, 'error', 'Error en model_info.json');
            modelInfo = null;
        }
    } else {
        updateFileStatusUI(modelInfoStatus, 'pending', 'Pendiente');
    }
});

// Manejar carga de train_data.json
trainDataFile.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (file) {
        updateFileStatusUI(trainDataStatus, 'loading', 'Cargando...');
        try {
            addLog('Cargando train_data.json...');
            trainData = await readJSONFile(file);
            updateFileStatusUI(trainDataStatus, 'success', 'train_data.json cargado');
            addLog('train_data.json cargado exitosamente. ¡Hora de alimentar al modelo!', 'success');
            checkDataReady();
        } catch (error) {
            addLog(`¡Ups! Hubo un enredo con train_data.json. Error: ${error.message}`, 'error');
            updateFileStatusUI(trainDataStatus, 'error', 'Error en train_data.json');
            trainData = null;
        }
    } else {
        updateFileStatusUI(trainDataStatus, 'pending', 'Pendiente');
    }
});

// Evento para crear el modelo
createModelBtn.addEventListener('click', () => {
    addLog('Creando modelo... ¡Dando forma a la inteligencia!');
    createModelBtn.disabled = true;
    updateControlStatus(modelStatus, 'waiting'); // Mostrar "esperando" mientras se crea

    try {
        currentModel = createASLModel(); // Usar la función adaptada
        
        addLog('¡Modelo creado exitosamente! Está listo para aprender.', 'success');
        addLog(`Arquitectura del cerebro: ${modelInfo.input_shape} → 256 → 128 → 64 → ${modelInfo.num_classes}`);
        updateControlStatus(modelStatus, 'created'); // Actualizar a "creado"
        updateControlStatus(trainStatus, 'ready'); // Habilitar el botón de entrenar
        trainBtn.disabled = false;
        
        // Mostrar resumen del modelo en la consola del navegador
        currentModel.summary();
    } catch (error) {
        addLog(`¡Rayos! Fallamos al crear el modelo. Parece que algo no encaja: ${error.message}`, 'error');
        updateControlStatus(modelStatus, 'error'); // Mostrar error en la creación
        createModelBtn.disabled = false;
    }
});

// Evento para iniciar el entrenamiento del modelo
trainBtn.addEventListener('click', async () => {
    if (isTraining) return;
    if (!currentModel) {
        addLog('¡Espera! No hay modelo para entrenar. Primero, dale una forma a este cerebro.', 'error');
        return;
    }
    if (!trainData) {
        addLog('¡Los datos de entrenamiento están desaparecidos! Sin ellos, el modelo no aprenderá.', 'error');
        return;
    }
    
    addLog('Iniciando entrenamiento... ¡La gymkhana del conocimiento ha comenzado!', 'info'); // Asegurando que el icono se añada por addLog
    trainBtn.disabled = true;
    isTraining = true;
    updateControlStatus(trainStatus, 'training'); // Mostrar estado de entrenamiento
    
    progressContainer.style.display = 'block'; // Mostrar la sección de progreso

    // Resetear la barra de progreso al inicio de un nuevo entrenamiento
    progressFill.style.width = '0%';
    epochInfo.textContent = 'Época: 0/5';
    lossInfo.textContent = 'Loss: --';
    accInfo.textContent = 'Accuracy: --';
    valLossInfo.textContent = 'Val Loss: --';
    valAccInfo.textContent = 'Val Accuracy: --';

    try {
        const data = prepareTrainingData(); // Prepara los tensores
        const epochsToTrain = 5; // Definir las épocas

        await currentModel.fit(data.xTrain, data.yTrain, {
            epochs: epochsToTrain,
            batchSize: 32,
            validationData: [data.xTest, data.yTest],
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    updateUIProgress(epoch + 1, epochsToTrain, logs); // Usar la función de UI
                    addLog(`Época ${epoch + 1}: Loss=${logs.loss.toFixed(4)}, Acc=${logs.acc.toFixed(4)}, Val_Acc=${logs.val_acc.toFixed(4)}`);
                },
                onTrainEnd: () => {
                    addLog('¡Entrenamiento completado! El modelo ha sudado la gota gorda.', 'success'); // Eliminado el icono literal
                    updateControlStatus(trainStatus, 'completed'); // Actualizar a entrenamiento completado
                    saveBtn.disabled = false;
                    isTraining = false;
                    
                    // Asegurarse de disponer los tensores al finalizar
                    tf.dispose([data.xTrain, data.yTrain, data.xTest, data.yTest]);
                    addLog('Tensores de datos dispuestos. ¡Orden y limpieza!', 'success'); // Eliminado el icono literal
                }
            },
            shuffle: true
        });
        
    } catch (error) {
        addLog(`¡Vaya! El entrenamiento tropezó. Aquí el informe: ${error.message}`, 'error');
        updateControlStatus(trainStatus, 'error'); // Mostrar error en entrenamiento
        trainBtn.disabled = false;
        isTraining = false;
    }
});

// Evento para guardar el modelo
saveBtn.addEventListener('click', async () => {
    if (!currentModel) {
        addLog('¿Intentas guardar aire? No hay ningún modelo listo para ser guardado.', 'error');
        return;
    }
    saveBtn.disabled = true;
    await saveASLModel(currentModel); // Llama a la función de guardado
});

// --- Inicialización ---
document.addEventListener('DOMContentLoaded', () => {
    addLog('¡Hola! Sistema listo. Es hora de darle vida a los datos. Selecciona los archivos JSON para comenzar.');
    // Los botones de "Crear" y "Entrenar" estarán deshabilitados hasta que se carguen los datos.
    // La sección de información del modelo y progreso se ocultan por CSS por defecto.
    updateControlStatus(modelStatus, 'waiting'); // Estado inicial del control de modelo
    updateControlStatus(trainStatus, 'ready'); // Estado inicial del control de entrenamiento
    updateControlStatus(saveStatus, 'not_saved'); // Estado inicial del control de guardado
    updateFileStatusUI(modelInfoStatus, 'pending', 'Pendiente'); // Estado inicial del archivo model_info
    updateFileStatusUI(trainDataStatus, 'pending', 'Pendiente'); // Estado inicial del archivo train_data
});
