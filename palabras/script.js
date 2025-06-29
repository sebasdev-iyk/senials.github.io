const videoElement = document.getElementById('input_video');
const canvasElement = document.getElementById('output_canvas');
const canvasCtx = canvasElement.getContext('2d');
const predictionLetterElement = document.getElementById('prediction_letter');
const predictedWordElement = document.getElementById('predicted_word');
const clearWordBtn = document.getElementById('clear_word_btn');

let model;
let predictedWord = "";
const aslLabels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'];

// --- New variables for automatic letter addition ---
let stableLetter = "";      // The letter that is currently being held steadily
let letterStartTime = null; // Timestamp (Date.now()) when the stableLetter began
let lastAddedTimestamp = 0; // Timestamp when the last letter was added to the word
const STABILITY_TIME_MS = 1000; // Hold a sign for 1 second to add it
const COOLDOWN_TIME_MS = 1500;  // Wait 1.5s after adding a letter before adding another

async function loadModel() {
    try {
        model = await tf.loadLayersModel('../entrenamiento/Web2.0/asl-model-tfjs.json');
        console.log("Modelo cargado exitosamente");
    } catch (error) {
        console.error("Error al cargar el modelo:", error);
    }
}

function normalizeLandmarks(landmarks) {
    const wrist = landmarks[0];
    const relativePoints = landmarks.map(point => ({
        x: point.x - wrist.x,
        y: point.y - wrist.y,
        z: point.z - wrist.z
    }));

    let maxDistance = 0;
    for (let i = 1; i < relativePoints.length; i++) {
        const dist = Math.sqrt(
            relativePoints[i].x ** 2 +
            relativePoints[i].y ** 2 +
            relativePoints[i].z ** 2
        );
        if (dist > maxDistance) maxDistance = dist;
    }

    if (maxDistance > 0) {
        return relativePoints.map(point => ({
            x: point.x / maxDistance,
            y: point.y / maxDistance,
            z: point.z / maxDistance
        }));
    }
    return relativePoints;
}

async function onResults(results) {
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

    let detectedLetter = "";

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        canvasElement.style.borderColor = 'green';
        const landmarks = results.multiHandLandmarks[0];

        if (model) {
            const normalizedLandmarks = normalizeLandmarks(landmarks);
            const landmarkData = [];
            normalizedLandmarks.forEach(landmark => {
                landmarkData.push(landmark.x, landmark.y, landmark.z);
            });

            if (landmarkData.length === 63) {
                const inputTensor = tf.tensor2d([landmarkData]);
                const prediction = model.predict(inputTensor);
                const predictionData = await prediction.data();
                const maxIndex = predictionData.indexOf(Math.max(...predictionData));
                const confidence = predictionData[maxIndex];

                if (confidence > 0.85) { // Increased confidence for more stability
                   detectedLetter = aslLabels[maxIndex];
                }
                tf.dispose([inputTensor, prediction]);
            }
        }
    } else {
        canvasElement.style.borderColor = 'red';
    }

    predictionLetterElement.textContent = detectedLetter;

    // --- Automatic Letter Addition Logic ---
    const now = Date.now();

    if (detectedLetter !== stableLetter) {
        // The prediction has changed. Reset the stable letter and start the timer.
        stableLetter = detectedLetter;
        letterStartTime = stableLetter === "" ? null : now;
    } else if (stableLetter !== "" && letterStartTime !== null) {
        // The letter has been stable. Check if enough time has passed to add it.
        const timeHeld = now - letterStartTime;
        const timeSinceLastAdd = now - lastAddedTimestamp;

        if (timeHeld > STABILITY_TIME_MS && timeSinceLastAdd > COOLDOWN_TIME_MS) {
            predictedWord += stableLetter;
            predictedWordElement.textContent = predictedWord;

            // Reset timers to prevent re-adding the same letter on the next frame
            lastAddedTimestamp = now;
            letterStartTime = null; // This forces the user to move their hand away and back to add another letter
        }
    }
    canvasCtx.restore();
}

clearWordBtn.addEventListener('click', () => {
    predictedWord = "";
    predictedWordElement.textContent = "";
    lastAddedTimestamp = 0; // Reset cooldown
});

const hands = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
});

hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
});

hands.onResults(onResults);

const camera = new Camera(videoElement, {
    onFrame: async () => {
        await hands.send({ image: videoElement });
    },
    width: 640,
    height: 480
});
camera.start();
loadModel();
