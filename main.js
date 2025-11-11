// ImageNet class labels (top 10 common classes for demo)
const imagenetClasses = [
    [0, "tench"], [1, "goldfish"], [2, "great_white_shark"], [3, "tiger_shark"],
    [4, "hammerhead"], [5, "electric_ray"], [6, "stingray"], [7, "cock"],
    [8, "hen"], [9, "ostrich"], [10, "brambling"], [11, "goldfinch"],
    [12, "house_finch"], [13, "junco"], [14, "indigo_bunting"], [15, "robin"],
    [16, "bulbul"], [17, "jay"], [18, "magpie"], [19, "chickadee"],
    // Add more classes as needed or use full 1000 ImageNet classes
];

let modelSession = null;
let isProcessing = false;

// Initialize when page loads
window.addEventListener('DOMContentLoaded', async () => {
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const detectionsDiv = document.getElementById('detections');
    const statusDiv = document.getElementById('status');

    try {
        statusDiv.textContent = 'Loading model...';
        await initModel();
        statusDiv.textContent = 'Model loaded! Processing video frames...';

        // Wait for video to be ready
        video.addEventListener('loadeddata', () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            processVideoFrames(video, canvas, detectionsDiv, statusDiv);
        });

    } catch (error) {
        statusDiv.textContent = `Error: ${error.message}`;
        console.error('Initialization error:', error);
    }
});

async function initModel() {
    // Set up ONNX Runtime environment
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;

    const modelPath = './mobilenetv2-10.onnx';
    const options = {
        executionProviders: [
            { 
                name: 'webnn', 
                deviceType: 'npu',  // Use NPU
                powerPreference: 'default' 
            },
            'webgpu',  // Fallback to WebGPU
            'wasm'     // Final fallback to WASM
        ],
        freeDimensionOverrides: {
            "batch": 1, 
            "channels": 3, 
            "height": 224, 
            "width": 224
        }
    };

    modelSession = await ort.InferenceSession.create(modelPath, options);
    console.log('Model loaded successfully');
}

async function processVideoFrames(video, canvas, detectionsDiv, statusDiv) {
    const ctx = canvas.getContext('2d');
    let frameCount = 0;

    async function processFrame() {
        if (video.paused || video.ended) {
            return;
        }

        if (!isProcessing) {
            isProcessing = true;
            frameCount++;

            try {
                // Draw current video frame to canvas
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                
                // Get image data from canvas
                const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                
                // Convert to tensor and classify
                const tensor = imageDataToTensor(imageData, 224, 224);
                const predictions = await runModel(tensor);
                
                // Display results
                displayDetections(predictions, detectionsDiv, frameCount);
                statusDiv.textContent = `Processing frame ${frameCount}... (using NPU)`;
                
            } catch (error) {
                console.error('Frame processing error:', error);
                statusDiv.textContent = `Error processing frame: ${error.message}`;
            }
            
            isProcessing = false;
        }

        // Process next frame (adjust delay as needed for performance)
        setTimeout(() => requestAnimationFrame(processFrame), 1000); // Process 1 frame per second
    }

    processFrame();
}

function imageDataToTensor(imageData, targetWidth, targetHeight) {
    // Resize image data to target size
    const resizedData = resizeImageData(imageData, targetWidth, targetHeight);
    
    const pixelCount = targetWidth * targetHeight;
    const float32Data = new Float32Array(3 * pixelCount);
    
    // Convert RGBA to RGB and normalize to [0, 1]
    for (let i = 0; i < pixelCount; i++) {
        float32Data[pixelCount * 0 + i] = resizedData[i * 4 + 0] / 255.0; // Red
        float32Data[pixelCount * 1 + i] = resizedData[i * 4 + 1] / 255.0; // Green
        float32Data[pixelCount * 2 + i] = resizedData[i * 4 + 2] / 255.0; // Blue
    }
    
    const dimensions = [1, 3, targetHeight, targetWidth];
    return new ort.Tensor('float32', float32Data, dimensions);
}

function resizeImageData(imageData, targetWidth, targetHeight) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = targetWidth;
    canvas.height = targetHeight;
    
    // Create temporary canvas with original image
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');
    tempCanvas.width = imageData.width;
    tempCanvas.height = imageData.height;
    tempCtx.putImageData(imageData, 0, 0);
    
    // Draw resized
    ctx.drawImage(tempCanvas, 0, 0, targetWidth, targetHeight);
    
    return ctx.getImageData(0, 0, targetWidth, targetHeight).data;
}

async function runModel(preprocessedData) {
    const feeds = {};
    feeds[modelSession.inputNames[0]] = preprocessedData;
    
    const outputData = await modelSession.run(feeds);
    const output = outputData[modelSession.outputNames[0]];
    
    const outputSoftmax = softmax(Array.prototype.slice.call(output.data));
    const results = getTopK(outputSoftmax, 5);
    
    return results;
}

function softmax(resultArray) {
    const largestNumber = Math.max(...resultArray);
    const sumOfExp = resultArray
        .map(resultItem => Math.exp(resultItem - largestNumber))
        .reduce((prevNumber, currentNumber) => prevNumber + currentNumber);
    
    return resultArray.map((resultValue) => {
        return Math.exp(resultValue - largestNumber) / sumOfExp;
    });
}

function getTopK(classProbabilities, k = 5) {
    const probs = Array.prototype.slice.call(classProbabilities);
    
    const sorted = probs
        .map((prob, index) => [prob, index])
        .sort((a, b) => b[0] - a[0])
        .slice(0, k);
    
    return sorted.map(([prob, index]) => {
        const className = imagenetClasses[index] ? imagenetClasses[index][1] : `class_${index}`;
        return {
            id: index,
            name: className.replace(/_/g, ' '),
            probability: prob
        };
    });
}

function displayDetections(predictions, detectionsDiv, frameCount) {
    detectionsDiv.innerHTML = predictions.map(pred => `
        <div class="detection">
            <strong>${pred.name}</strong>: ${(pred.probability * 100).toFixed(2)}%
        </div>
    `).join('');
}
