package com.surendramaran.yolov8tflite

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStream
import java.io.InputStreamReader

class Detector(
    private val context: Context,
    private val modelPath: String,
    private val labelPath: String,
    private val detectorListener: DetectorListener,
) {

    private lateinit var interpreter: Interpreter // lateinit as it's initialized in init
    private var labels = mutableListOf<String>()

    // Variables to store model input dimensions
    private var modelInputWidth = 0
    private var modelInputHeight = 0
    private var modelInputChannel = 0 // For completion, though often 3 and implicitly handled

    // Variables for model output dimensions
    private var modelOutputClasses = 0 // e.g. 80 for COCO
    private var modelOutputElements = 0 // e.g. 8400 detections


    private val imageProcessor = ImageProcessor.Builder()
        .add(NormalizeOp(INPUT_MEAN, INPUT_STANDARD_DEVIATION))
        .add(CastOp(INPUT_IMAGE_TYPE))
        .build()

    init {
        loadLabels()
        initializeInterpreter(useGpu = true) // Default to GPU if available
    }

    private fun loadLabels() {
        try {
            val inputStream: InputStream = context.assets.open(labelPath)
            val reader = BufferedReader(InputStreamReader(inputStream))
            var line: String? = reader.readLine()
            while (line != null) { // Allow empty lines in labels.txt if they signify something (though unusual)
                labels.add(line)
                line = reader.readLine()
            }
            reader.close()
            inputStream.close()
        } catch (e: IOException) {
            Log.e("Detector", "Error loading labels from $labelPath", e)
            // Consider how to handle this error: throw, or operate without labels (class names will be "Unknown")
        }
    }

    private fun initializeInterpreter(useGpu: Boolean) {
        val options = Interpreter.Options()
        if (useGpu) {
            val compatList = CompatibilityList()
            if (compatList.isDelegateSupportedOnThisDevice) {
                val delegateOptions = compatList.bestOptionsForThisDevice
                options.addDelegate(GpuDelegate(delegateOptions))
                Log.i("Detector", "GPU delegate applied.")
            } else {
                options.setNumThreads(NUM_THREADS)
                Log.i("Detector", "GPU not supported on this device, using CPU with $NUM_THREADS threads.")
            }
        } else {
            options.setNumThreads(NUM_THREADS)
            Log.i("Detector", "Using CPU with $NUM_THREADS threads.")
        }

        try {
            val model = FileUtil.loadMappedFile(context, modelPath)
            interpreter = Interpreter(model, options)

            // Read input tensor shape
            val inputTensor = interpreter.getInputTensor(0)
            val inputShape = inputTensor?.shape() ?: throw IllegalStateException("Input tensor shape is null")
            Log.i("Detector", "Input tensor shape: ${inputShape.joinToString()}")

            // Determine model input H, W, C based on common formats
            // NHWC: [Batch, Height, Width, Channels] e.g. [1, 640, 640, 3]
            // NCHW: [Batch, Channels, Height, Width] e.g. [1, 3, 640, 640]
            if (inputShape.size == 4) {
                when (inputShape[1]) {
                    3 -> { // Likely NCHW format [1, 3, H, W]
                        modelInputChannel = inputShape[1]
                        modelInputHeight = inputShape[2]
                        modelInputWidth = inputShape[3]
                    }
                    else -> { // Likely NHWC format [1, H, W, 3]
                        modelInputHeight = inputShape[1]
                        modelInputWidth = inputShape[2]
                        modelInputChannel = inputShape[3]
                    }
                }
            } else {
                throw IllegalStateException("Unsupported input tensor shape: ${inputShape.joinToString()}. Expected 4 dimensions.")
            }
            Log.i("Detector", "Model Input - W: $modelInputWidth, H: $modelInputHeight, C: $modelInputChannel")


            // Read output tensor shape for YOLOv8-like models: [1, num_classes + 4, num_detections]
            // e.g., [1, 84, 8400] for COCO (80 classes + 4 box coords)
            val outputTensor = interpreter.getOutputTensor(0)
            val outputShape = outputTensor?.shape() ?: throw IllegalStateException("Output tensor shape is null")
            Log.i("Detector", "Output tensor shape: ${outputShape.joinToString()}")

            if (outputShape.size == 3 && outputShape[0] == 1) {
                modelOutputClasses = outputShape[1] - 4 // Assuming 4 are box coordinates
                modelOutputElements = outputShape[2]
            } else {
                throw IllegalStateException("Unsupported output tensor shape: ${outputShape.joinToString()}. Expected [1, C+4, N]")
            }
            Log.i("Detector", "Model Output - Classes: $modelOutputClasses (derived), Elements: $modelOutputElements")


        } catch (e: Exception) {
            Log.e("Detector", "Error initializing TFLite interpreter", e)
            // Propagate or handle error, app might not function without interpreter
            throw IllegalStateException("Failed to initialize TFLite interpreter: ${e.message}")
        }
    }


    fun restart(isGpu: Boolean) {
        interpreter.close()
        initializeInterpreter(useGpu = isGpu)
    }

    fun close() {
        interpreter.close()
        Log.i("Detector", "Interpreter closed.")
    }

    fun detect(frame: Bitmap) {
        if (modelInputWidth == 0 || modelInputHeight == 0) {
            Log.e("Detector", "Model input dimensions not initialized. Cannot detect.")
            return
        }

        var inferenceTime = SystemClock.uptimeMillis()

        // Ensure the bitmap is scaled to the model's expected input dimensions
        val resizedBitmap = Bitmap.createScaledBitmap(frame, modelInputWidth, modelInputHeight, false)

        val tensorImage = TensorImage(INPUT_IMAGE_TYPE)
        tensorImage.load(resizedBitmap)
        val processedImage = imageProcessor.process(tensorImage)
        val imageBuffer = processedImage.buffer

        // Output tensor: [1, 84, 8400] where 84 = 4 (box) + 80 (classes)
        val outputBufferShape = intArrayOf(1, modelOutputClasses + 4, modelOutputElements)
        val outputTensorBuffer = TensorBuffer.createFixedSize(outputBufferShape, OUTPUT_IMAGE_TYPE)

        try {
            interpreter.run(imageBuffer, outputTensorBuffer.buffer)
        } catch (e: Exception) {
            Log.e("Detector", "Error during TFLite inference", e)
            detectorListener.onEmptyDetect() // Notify listener of failure
            return
        }


        // ***** START DEBUG LOGGING (Optional: wrap with BuildConfig.DEBUG) *****
        /*
        val outputArray = outputTensorBuffer.floatArray
        var maxScoreInOutput = 0f
        var maxScoreIndex = -1
        for (i in outputArray.indices) {
            if (outputArray[i] > maxScoreInOutput) {
                maxScoreInOutput = outputArray[i]
                maxScoreIndex = i
            }
        }
        Log.d("DetectorRawOutput", "Output array size: ${outputArray.size}, Max score in raw output: $maxScoreInOutput at index $maxScoreIndex")
        // Log.d("DetectorRawOutput", "First 100 elements: ${outputArray.take(100).joinToString()}")
        */
        // ***** END DEBUG LOGGING *****


        val bestBoxes = bestBox(outputTensorBuffer.floatArray)
        inferenceTime = SystemClock.uptimeMillis() - inferenceTime

        if (bestBoxes.isNullOrEmpty()) { // Check for null or empty
            detectorListener.onEmptyDetect()
            return
        }

        detectorListener.onDetect(bestBoxes, inferenceTime)
    }

    private fun bestBox(array: FloatArray): List<BoundingBox>? {
        val boundingBoxes = mutableListOf<BoundingBox>()
        // numAttributes is modelOutputClasses + 4 (e.g., 80 classes + 4 box coords = 84)
        val numAttributes = modelOutputClasses + 4
        // numDetections is modelOutputElements (e.g., 8400)
        val numDetections = modelOutputElements

        for (d in 0 until numDetections) {
            val offset = d * numAttributes
            // Output format: [cx, cy, w, h, class_score_0, ..., class_score_N-1]
            // These are normalized coordinates (0.0 to 1.0)
            val cx = array[offset + 0]
            val cy = array[offset + 1]
            val w = array[offset + 2]
            val h = array[offset + 3]

            var maxConf = -1.0f
            var maxIdx = -1
            // Class scores start at index 4 in this segment
            for (classIdx in 0 until modelOutputClasses) {
                val score = array[offset + 4 + classIdx]
                if (score > maxConf) {
                    maxConf = score
                    maxIdx = classIdx
                }
            }

            if (maxConf > CONFIDENCE_THRESHOLD) { // Use companion object's threshold
                val x1 = cx - w / 2f
                val y1 = cy - h / 2f
                val x2 = cx + w / 2f
                val y2 = cy + h / 2f

                // Clamp normalized coordinates to [0, 1] range to be safe
                val clampedX1 = x1.coerceIn(0f, 1f)
                val clampedY1 = y1.coerceIn(0f, 1f)
                val clampedX2 = x2.coerceIn(0f, 1f)
                val clampedY2 = y2.coerceIn(0f, 1f)
                val clampedW = clampedX2 - clampedX1
                val clampedH = clampedY2 - clampedY1


                if (clampedW > 0 && clampedH > 0) { // Ensure valid box after clamping
                    val clsName = labels.getOrNull(maxIdx) ?: "Unknown_$maxIdx"
                    boundingBoxes.add(
                        BoundingBox(
                            x1 = clampedX1, y1 = clampedY1, x2 = clampedX2, y2 = clampedY2,
                            cx = (clampedX1 + clampedX2) / 2f, // Recalculate cx,cy based on clamped values
                            cy = (clampedY1 + clampedY2) / 2f,
                            w = clampedW, h = clampedH,
                            cnf = maxConf, cls = maxIdx, clsName = clsName
                        )
                    )
                }
            }
        }

        if (boundingBoxes.isEmpty()) return null
        return applyNMS(boundingBoxes)
    }

    private fun applyNMS(boxes: List<BoundingBox>): MutableList<BoundingBox> {
        val sortedBoxes = boxes.sortedByDescending { it.cnf }.toMutableList()
        val selectedBoxes = mutableListOf<BoundingBox>()

        while (sortedBoxes.isNotEmpty()) {
            val first = sortedBoxes.first()
            selectedBoxes.add(first)
            sortedBoxes.remove(first)

            val iterator = sortedBoxes.iterator()
            while (iterator.hasNext()) {
                val nextBox = iterator.next()
                val iou = calculateIoU(first, nextBox)
                if (iou >= IOU_THRESHOLD) {
                    iterator.remove()
                }
            }
        }
        return selectedBoxes
    }

    private fun calculateIoU(box1: BoundingBox, box2: BoundingBox): Float {
        val x1 = maxOf(box1.x1, box2.x1)
        val y1 = maxOf(box1.y1, box2.y1)
        val x2 = minOf(box1.x2, box2.x2)
        val y2 = minOf(box1.y2, box2.y2)

        val intersectionWidth = maxOf(0F, x2 - x1)
        val intersectionHeight = maxOf(0F, y2 - y1)
        val intersectionArea = intersectionWidth * intersectionHeight

        val box1Area = box1.w * box1.h
        val box2Area = box2.w * box2.h

        val unionArea = box1Area + box2Area - intersectionArea
        return if (unionArea > 0) intersectionArea / unionArea else 0f
    }

    interface DetectorListener {
        fun onEmptyDetect()
        fun onDetect(boundingBoxes: List<BoundingBox>, inferenceTime: Long)
    }

    companion object {
        private const val INPUT_MEAN = 0f
        private const val INPUT_STANDARD_DEVIATION = 255f
        private val INPUT_IMAGE_TYPE = DataType.FLOAT32
        private val OUTPUT_IMAGE_TYPE = DataType.FLOAT32
        private const val CONFIDENCE_THRESHOLD = 0.75F // Adjusted to 0.6F, was 0.5F locally in bestBox
        private const val IOU_THRESHOLD = 0.3F
        private const val NUM_THREADS = 4 // Number of threads for CPU inference
    }
}