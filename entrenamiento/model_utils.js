
class ASLModelUtils {
    constructor() {
        this.modelInfo = null;
        this.trainData = null;
    }

    async loadData() {
        try {
            console.log('🔍 Intentando cargar archivos...');
            
            // Verificar que los archivos existan antes de cargarlos
            const files = ['../data/model_info.json', '../data/train_data.json'];
            const fileChecks = await Promise.allSettled(
                files.map(file => fetch(file).then(r => ({ file, ok: r.ok, status: r.status })))
            );
            
            console.log('📂 Estado de archivos:');
            fileChecks.forEach((result, i) => {
                if (result.status === 'fulfilled') {
                    console.log(`   ${files[i]}: ${result.value.ok ? '✅ OK' : `❌ ${result.value.status}`}`);
                } else {
                    console.log(`   ${files[i]}: ❌ Error: ${result.reason.message}`);
                }
            });

            // Cargar información del modelo
            console.log('📊 Cargando model_info.json...');
            const modelInfoResponse = await fetch('../data/model_info.json');
            if (!modelInfoResponse.ok) {
                throw new Error(`Error cargando model_info.json: ${modelInfoResponse.status} ${modelInfoResponse.statusText}`);
            }
            this.modelInfo = await modelInfoResponse.json();
            console.log('✅ model_info.json cargado');

            // Cargar datos de entrenamiento
            console.log('🔢 Cargando train_data.json...');
            const trainDataResponse = await fetch('../data/train_data.json');
            if (!trainDataResponse.ok) {
                throw new Error(`Error cargando train_data.json: ${trainDataResponse.status} ${trainDataResponse.statusText}`);
            }
            this.trainData = await trainDataResponse.json();
            console.log('✅ train_data.json cargado');

            console.log('📊 Datos cargados:', {
                classes: this.modelInfo.num_classes,
                trainSamples: this.modelInfo.train_samples,
                inputShape: this.modelInfo.input_shape
            });

            return true;
        } catch (error) {
            console.error('❌ Error detallado cargando datos:', error);
            console.error('🔍 Verifica que:');
            console.error('   1. Los archivos model_info.json y train_data.json estén en la carpeta data/');
            console.error('   2. El servidor web esté sirviendo los archivos correctamente');
            console.error('   3. No hay errores de CORS');
            return false;
        }
    }

    createModel() {
        const inputShape = this.modelInfo.input_shape;
        const numClasses = this.modelInfo.num_classes;

        const model = tf.sequential({
            layers: [
                // Input layer con normalización
                tf.layers.dense({
                    inputShape: [inputShape],
                    units: 256,
                    activation: 'relu',
                    kernelRegularizer: tf.regularizers.l2({ l2: 0.001 })
                }),
                tf.layers.batchNormalization(),
                tf.layers.dropout({ rate: 0.4 }),

                // Segunda capa
                tf.layers.dense({
                    units: 128,
                    activation: 'relu',
                    kernelRegularizer: tf.regularizers.l2({ l2: 0.001 })
                }),
                tf.layers.batchNormalization(),
                tf.layers.dropout({ rate: 0.3 }),

                // Tercera capa
                tf.layers.dense({
                    units: 64,
                    activation: 'relu'
                }),
                tf.layers.dropout({ rate: 0.2 }),

                // Output layer
                tf.layers.dense({
                    units: numClasses,
                    activation: 'softmax'
                })
            ]
        });

        return model;
    }

    prepareTrainingData() {
        const xTrain = tf.tensor2d(this.trainData.X_train);
        const yTrain = tf.oneHot(tf.tensor1d(this.trainData.y_train, 'int32'), this.modelInfo.num_classes);
        
        const xTest = tf.tensor2d(this.trainData.X_test);
        const yTest = tf.oneHot(tf.tensor1d(this.trainData.y_test, 'int32'), this.modelInfo.num_classes);

        return { xTrain, yTrain, xTest, yTest };
    }

    async trainModel(model, trainData, callbacks = {}) {
        const { xTrain, yTrain, xTest, yTest } = trainData;

        // Configurar el optimizer
        const optimizer = tf.train.adam(0.001);
        
        model.compile({
            optimizer: optimizer,
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });

        console.log('🏗️ Arquitectura del modelo:');
        model.summary();

        // Configurar callbacks
        const defaultCallbacks = {
            onEpochEnd: (epoch, logs) => {
                console.log(`Época ${epoch + 1}: loss=${logs.loss.toFixed(4)}, acc=${logs.acc.toFixed(4)}, val_loss=${logs.val_loss.toFixed(4)}, val_acc=${logs.val_acc.toFixed(4)}`);
                
                if (callbacks.onProgress) {
                    callbacks.onProgress(epoch + 1, logs);
                }
            },
            onTrainEnd: () => {
                if (callbacks.onComplete) {
                    callbacks.onComplete();
                }
            }
        };

        // Entrenar el modelo
        const history = await model.fit(xTrain, yTrain, {
            epochs: 100,
            batchSize: 32,
            validationData: [xTest, yTest],
            callbacks: defaultCallbacks,
            shuffle: true
        });

        return history;
    }

    async saveModel(model, modelName = 'asl-model') {
        try {
            await model.save(`downloads://${modelName}`);
            console.log('✅ Modelo guardado exitosamente');
            return true;
        } catch (error) {
            console.error('❌ Error guardando modelo:', error);
            return false;
        }
    }

    async evaluateModel(model, testData) {
        const { xTest, yTest } = testData;
        
        const predictions = model.predict(xTest);
        const loss = tf.losses.softmaxCrossEntropy(yTest, predictions);
        
        // Calcular accuracy
        const predLabels = tf.argMax(predictions, 1);
        const trueLabels = tf.argMax(yTest, 1);
        const accuracy = tf.mean(tf.equal(predLabels, trueLabels));

        const lossValue = await loss.data();
        const accValue = await accuracy.data();

        console.log(`📈 Evaluación final: Loss=${lossValue[0].toFixed(4)}, Accuracy=${accValue[0].toFixed(4)}`);

        // Limpiar tensores
        loss.dispose();
        accuracy.dispose();
        predLabels.dispose();
        trueLabels.dispose();
        predictions.dispose();

        return {
            loss: lossValue[0],
            accuracy: accValue[0]
        };
    }

    getLabelMapping() {
        return this.modelInfo.label_mapping;
    }

    getClassNames() {
        return this.modelInfo.classes;
    }
}

// Función para mostrar progreso de entrenamiento
function createProgressDisplay() {
    const progressDiv = document.createElement('div');
    progressDiv.id = 'training-progress';
    progressDiv.innerHTML = `
        <div class="progress-container">
            <h3>Progreso de Entrenamiento</h3>
            <div class="progress-bar">
                <div class="progress-fill" id="progress-fill"></div>
            </div>
            <div class="progress-info">
                <span id="epoch-info">Época: 0/100</span>
                <span id="loss-info">Loss: --</span>
                <span id="acc-info">Accuracy: --</span>
            </div>
            <div class="progress-info">
                <span id="val-loss-info">Val Loss: --</span>
                <span id="val-acc-info">Val Accuracy: --</span>
            </div> 
        </div>
    `;
    
    document.body.appendChild(progressDiv);
    return progressDiv;
}

function updateProgress(epoch, maxEpochs, logs) {
    const progressFill = document.getElementById('progress-fill');
    const epochInfo = document.getElementById('epoch-info');
    const lossInfo = document.getElementById('loss-info');
    const accInfo = document.getElementById('acc-info');
    const valLossInfo = document.getElementById('val-loss-info');
    const valAccInfo = document.getElementById('val-acc-info');

    const progress = (epoch / maxEpochs) * 100;
    
    if (progressFill) progressFill.style.width = `${progress}%`;
    if (epochInfo) epochInfo.textContent = `Época: ${epoch}/${maxEpochs}`;
    if (lossInfo) lossInfo.textContent = `Loss: ${logs.loss.toFixed(4)}`;
    if (accInfo) accInfo.textContent = `Accuracy: ${logs.acc.toFixed(4)}`;
    if (valLossInfo) valLossInfo.textContent = `Val Loss: ${logs.val_loss.toFixed(4)}`;
    if (valAccInfo) valAccInfo.textContent = `Val Accuracy: ${logs.val_acc.toFixed(4)}`;
}