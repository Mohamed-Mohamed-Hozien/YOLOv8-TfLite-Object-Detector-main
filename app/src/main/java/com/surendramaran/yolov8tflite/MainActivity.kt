package com.surendramaran.yolov8tflite

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.speech.tts.TextToSpeech // Import TextToSpeech
import android.util.Log
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.AspectRatio
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.surendramaran.yolov8tflite.Constants.LABELS_PATH
import com.surendramaran.yolov8tflite.Constants.MODEL_PATH
import com.surendramaran.yolov8tflite.databinding.ActivityMainBinding
import java.util.Locale // Import Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import androidx.core.graphics.createBitmap

class MainActivity : AppCompatActivity(), Detector.DetectorListener, TextToSpeech.OnInitListener { // Implement OnInitListener
    private lateinit var binding: ActivityMainBinding
    private val isFrontCamera = false // Remains false, front camera logic is present but not actively used by a toggle

    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var detector: Detector? = null

    private lateinit var cameraExecutor: ExecutorService

    private var tts: TextToSpeech? = null // TextToSpeech instance
    private var lastSpokenMessage: String? = null
    private var lastSpokenTime: Long = 0
    private val SPEAK_COOLDOWN_MS = 3000 // Cooldown in milliseconds (e.g., 3 seconds)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        cameraExecutor = Executors.newSingleThreadExecutor()

        // Initialize TextToSpeech
        tts = TextToSpeech(this, this)

        cameraExecutor.execute {
            detector = Detector(baseContext, MODEL_PATH, LABELS_PATH, this)
            // Log.d(TAG, "Detector initialized on ${Thread.currentThread().name}")
        }

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            // Use the modern ActivityResultLauncher for permissions
            requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
        }

        bindListeners()
    }

    private fun bindListeners() {
        binding.apply {
            isGpu.setOnCheckedChangeListener { buttonView, isChecked ->
                cameraExecutor.submit {
                    detector?.restart(isGpu = isChecked)
                }
                if (isChecked) {
                    buttonView.setBackgroundColor(ContextCompat.getColor(buttonView.context, R.color.orange))
                } else {
                    buttonView.setBackgroundColor(ContextCompat.getColor(buttonView.context, R.color.gray))
                }
            }
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            cameraProvider  = cameraProviderFuture.get()
            bindCameraUseCases()
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindCameraUseCases() {
        val cameraProvider = cameraProvider ?: throw IllegalStateException("Camera initialization failed.")

        val rotation = binding.viewFinder.display.rotation

        val cameraSelector = CameraSelector
            .Builder()
            .requireLensFacing(CameraSelector.LENS_FACING_BACK)
            .build()

        preview =  Preview.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setTargetRotation(rotation)
            .build()

        imageAnalyzer = ImageAnalysis.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setTargetRotation(binding.viewFinder.display.rotation)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()

        imageAnalyzer?.setAnalyzer(cameraExecutor) { imageProxy ->
            val bitmapBuffer =
                createBitmap(imageProxy.width, imageProxy.height, Bitmap.Config.ARGB_8888) // Specify config
            bitmapBuffer.copyPixelsFromBuffer(imageProxy.planes[0].buffer)
            // It's crucial to close the ImageProxy otherwise new images may not be produced
            imageProxy.close()

            val matrix = Matrix().apply {
                postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())

                if (isFrontCamera) { // Logic for front camera, though isFrontCamera is hardcoded to false
                    postScale(
                        -1f,
                        1f,
                        imageProxy.width.toFloat(),
                        imageProxy.height.toFloat()
                    )
                }
            }

            val rotatedBitmap = Bitmap.createBitmap(
                bitmapBuffer, 0, 0, bitmapBuffer.width, bitmapBuffer.height,
                matrix, true
            )

            detector?.detect(rotatedBitmap)
        }

        cameraProvider.unbindAll()

        try {
            camera = cameraProvider.bindToLifecycle(
                this,
                cameraSelector,
                preview,
                imageAnalyzer
            )

            preview?.setSurfaceProvider(binding.viewFinder.surfaceProvider)
        } catch(exc: Exception) {
            Log.e(TAG, "Use case binding failed", exc)
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED // Use 'this' for context
    }

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()) { permissions ->
        var allGranted = true
        for (isGranted in permissions.values) {
            if (!isGranted) {
                allGranted = false
                break
            }
        }
        if (allGranted) {
            startCamera()
        } else {
            // Handle permission denial, e.g., show a message to the user
            Log.e(TAG, "One or more permissions not granted.")
            // Optionally, inform the user that the camera feature cannot be used.
            // Toast.makeText(this, "Camera permission is required to use this feature.", Toast.LENGTH_LONG).show()
        }
    }

    // TextToSpeech OnInitListener implementation
    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            val result = tts?.setLanguage(Locale.US) // Set language, e.g., US English
            if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                Log.e("TTS", "The Language specified is not supported!")
            } else {
                Log.i("TTS", "TextToSpeech initialized successfully.")
            }
        } else {
            Log.e("TTS", "TextToSpeech initialization failed!")
        }
    }

    private fun speak(text: String) {
        if (text.isBlank()) return // Don't speak empty messages

        val currentTime = System.currentTimeMillis()
        if (text == lastSpokenMessage && (currentTime - lastSpokenTime < SPEAK_COOLDOWN_MS)) {
            // If the message is the same and cooldown hasn't passed, don't speak
            return
        }

        tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, null)
        lastSpokenMessage = text
        lastSpokenTime = currentTime
    }


    override fun onDestroy() {
        super.onDestroy() // Call super.onDestroy() first
        detector?.close()
        // It's important to shut down the executor after all pending tasks are done
        // or use shutdownNow() if immediate stop is required.
        cameraExecutor.shutdown()

        // Shutdown TTS
        if (tts != null) {
            tts!!.stop()
            tts!!.shutdown()
            Log.i("TTS", "TextToSpeech shut down.")
            tts = null // Release reference
        }
    }

    override fun onResume() {
        super.onResume()
        // Re-check permissions and start camera if needed,
        // especially if permissions were changed while app was paused.
        if (detector == null && !cameraExecutor.isShutdown) { // Re-initialize detector if it was closed
            cameraExecutor.execute {
                detector = Detector(baseContext, MODEL_PATH, LABELS_PATH, this)
            }
        }
        if (allPermissionsGranted()){
            startCamera()
        } else {
            requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
        }
    }

    override fun onPause() {
        super.onPause()
        // Consider releasing camera resources here if not using them in background
        // cameraProvider?.unbindAll() // This might be too aggressive if app is just paused briefly
    }

    companion object {
        private const val TAG = "Camera"
        // REQUEST_CODE_PERMISSIONS is no longer needed with ActivityResultContracts
        private val REQUIRED_PERMISSIONS = arrayOf( // Use arrayOf for clarity
            Manifest.permission.CAMERA
        )
    }

    override fun onEmptyDetect() {
        runOnUiThread {
            binding.overlay.clear()
            // Optionally speak "No objects detected" or similar
            // speak("No objects detected.") // Be mindful of speaking too often
        }
    }

    // OBSTACLE_CLASS_NAMES is defined but not directly used in the current obstacle detection logic below.
    // The current logic treats any non-door object meeting size/position criteria as an obstacle.
    // If specific obstacle types from this list were intended, the logic in onDetect would need adjustment.
    private val OBSTACLE_CLASS_NAMES = setOf(
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"
        // "door" is NOT an obstacle in this context
    )

    override fun onDetect(boundingBoxes: List<BoundingBox>, inferenceTime: Long) {
        runOnUiThread {
            binding.inferenceTime.text = "${inferenceTime}ms"
            binding.overlay.apply {
                setResults(boundingBoxes)
                invalidate()
            }

            var doorDetected = false
            var doorCenterX = 0f
            var doorArea = 0f

            var significantObstacleDetected = false
            var obstacleDetails = ""

            for (box in boundingBoxes) {
                if (box.clsName == "door") {
                    if (!doorDetected || (box.w * box.h > doorArea)) {
                        doorDetected = true
                        doorCenterX = box.cx
                        doorArea = box.w * box.h
                    }
                }
                // Current obstacle logic: any non-door object meeting size/centrality criteria.
                // The OBSTACLE_CLASS_NAMES set is not used here to filter by class name for obstacles.
                else if (box.w * box.h > 0.03 && box.cx > 0.15 && box.cx < 0.85) { // Example thresholds
                    // Prioritize more significant (e.g., larger) obstacles if multiple are present
                    val currentObstacleArea = box.w * box.h
                    val existingObstacleThreshold = 0.05 // Only update if new obstacle is much larger or first one
                    if (!significantObstacleDetected || currentObstacleArea > existingObstacleThreshold) {
                        significantObstacleDetected = true
                        obstacleDetails = "a ${box.clsName}"
                    }
                }
            }

            var guidanceText = ""
            if (doorDetected) {
                if (significantObstacleDetected) {
                    // Using R.string.obstacle_detected assumes it's a format string like "Obstacle detected: %1$s. "
                    guidanceText = getString(R.string.obstacle_detected, obstacleDetails) + " "
                    guidanceText += when {
                        doorCenterX < 0.35f -> "The door is to your left, beyond the obstacle."
                        doorCenterX > 0.65f -> "The door is to your right, beyond the obstacle."
                        else -> "The door is ahead, but there's an obstacle."
                    }
                } else {
                    guidanceText = when {
                        doorCenterX < 0.35f -> "Door is to your left."
                        doorCenterX > 0.65f -> "Door is to your right."
                        else -> "Door is ahead. You can go forward."
                    }
                    if (doorArea > 0.35) {
                        guidanceText += " It seems close."
                    } else if (doorArea < 0.05 && doorArea > 0) {
                        guidanceText += " It seems a bit far."
                    }
                }
            } else if (significantObstacleDetected) {
                guidanceText = "Obstacle detected, it's ${obstacleDetails} in front."
            } else {
                guidanceText = "Looking for a door."
            }
            speak(guidanceText)
        }
    }
}