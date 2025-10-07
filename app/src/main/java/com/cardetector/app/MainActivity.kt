package com.cardetector.app

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.android.material.floatingactionbutton.FloatingActionButton
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import android.graphics.*
import android.view.View
import android.widget.TextView
import java.io.File
import java.io.FileOutputStream
import java.net.URL
import kotlinx.coroutines.*

class MainActivity : AppCompatActivity(), TextToSpeech.OnInitListener {
    
    private lateinit var previewView: PreviewView
    private lateinit var overlayView: OverlayView
    private lateinit var statusText: TextView
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var tts: TextToSpeech
    
    private var interpreter: Interpreter? = null
    private var imageProcessor: ImageProcessor? = null
    private var camera: Camera? = null
    private var isProcessing = false
    private var lastWarningTime = 0L
    private val warningCooldown = 2000L // 2 seconds
    
    // YOLO parameters
    private val inputSize = 640
    private val numClasses = 80
    private val confidenceThreshold = 0.3f
    private val iouThreshold = 0.45f
    
    // Vehicle class IDs from COCO dataset
    private val vehicleClasses = mapOf(
        2 to "car",
        3 to "motorcycle",
        5 to "bus",
        7 to "truck"
    )
    
    // Proximity thresholds
    private val dangerThreshold = 0.4f
    private val warningThreshold = 0.25f
    
    companion object {
        private const val TAG = "CarDetector"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
        private const val MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.tflite"
        private const val MODEL_NAME = "yolov8n.tflite"
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        previewView = findViewById(R.id.previewView)
        overlayView = findViewById(R.id.overlayView)
        statusText = findViewById(R.id.statusText)
        
        cameraExecutor = Executors.newSingleThreadExecutor()
        tts = TextToSpeech(this, this)
        
        // Setup image processor
        imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(inputSize, inputSize, ResizeOp.ResizeMethod.BILINEAR))
            .build()
        
        // Check and download model
        checkAndDownloadModel()
        
        // Request camera permissions
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
            )
        }
        
        // Setup switch camera button
        findViewById<FloatingActionButton>(R.id.switchCamera).setOnClickListener {
            switchCamera()
        }
    }
    
    private fun checkAndDownloadModel() {
        val modelFile = File(filesDir, MODEL_NAME)
        
        if (modelFile.exists()) {
            loadModel(modelFile)
        } else {
            statusText.text = "Downloading YOLO model..."
            CoroutineScope(Dispatchers.IO).launch {
                try {
                    downloadModel(modelFile)
                    withContext(Dispatchers.Main) {
                        loadModel(modelFile)
                        statusText.text = "Model loaded. Ready!"
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        statusText.text = "Error downloading model: ${e.message}"
                        Toast.makeText(
                            this@MainActivity,
                            "Failed to download model. Check internet connection.",
                            Toast.LENGTH_LONG
                        ).show()
                    }
                }
            }
        }
    }
    
    private fun downloadModel(destFile: File) {
        URL(MODEL_URL).openStream().use { input ->
            FileOutputStream(destFile).use { output ->
                input.copyTo(output)
            }
        }
    }
    
    private fun loadModel(modelFile: File) {
        try {
            val options = Interpreter.Options().apply {
                setNumThreads(4)
                setUseNNAPI(true) // Use Android Neural Networks API for acceleration
            }
            interpreter = Interpreter(modelFile, options)
            Log.d(TAG, "Model loaded successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Error loading model", e)
            runOnUiThread {
                Toast.makeText(this, "Error loading model: ${e.message}", Toast.LENGTH_LONG).show()
            }
        }
    }
    
    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }
    
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(this, "Camera permission required", Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }
    
    private var lensFacing = CameraSelector.LENS_FACING_BACK
    
    private fun switchCamera() {
        lensFacing = if (lensFacing == CameraSelector.LENS_FACING_BACK) {
            CameraSelector.LENS_FACING_FRONT
        } else {
            CameraSelector.LENS_FACING_BACK
        }
        startCamera()
    }
    
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            
            // Preview
            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }
            
            // Image analysis
            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { imageProxy ->
                        processImage(imageProxy)
                    }
                }
            
            val cameraSelector = CameraSelector.Builder()
                .requireLensFacing(lensFacing)
                .build()
            
            try {
                cameraProvider.unbindAll()
                camera = cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalyzer
                )
            } catch (e: Exception) {
                Log.e(TAG, "Use case binding failed", e)
            }
            
        }, ContextCompat.getMainExecutor(this))
    }
    
    @androidx.camera.core.ExperimentalGetImage
    private fun processImage(imageProxy: ImageProxy) {
        if (isProcessing || interpreter == null) {
            imageProxy.close()
            return
        }
        
        isProcessing = true
        
        try {
            val image = imageProxy.image
            if (image != null) {
                val bitmap = imageProxyToBitmap(imageProxy)
                val detections = detectObjects(bitmap)
                
                runOnUiThread {
                    overlayView.setDetections(detections, bitmap.width, bitmap.height)
                    updateStatus(detections)
                    handleWarnings(detections)
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error processing image", e)
        } finally {
            isProcessing = false
            imageProxy.close()
        }
    }
    
    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap {
        val buffer = imageProxy.planes[0].buffer
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)
        
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size) ?: run {
            // Fallback: convert YUV to RGB
            val yuvImage = imageProxy.image!!
            val yuvBytes = ByteArray(
                yuvImage.width * yuvImage.height * 
                ImageFormat.getBitsPerPixel(ImageFormat.YUV_420_888) / 8
            )
            
            Bitmap.createBitmap(imageProxy.width, imageProxy.height, Bitmap.Config.ARGB_8888)
        }
    }
    
    private fun detectObjects(bitmap: Bitmap): List<Detection> {
        val tensorImage = TensorImage.fromBitmap(bitmap)
        val processedImage = imageProcessor?.process(tensorImage)
        
        // Prepare input buffer
        val inputBuffer = ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3)
        inputBuffer.order(ByteOrder.nativeOrder())
        
        processedImage?.buffer?.let {
            inputBuffer.put(it)
        }
        
        // Prepare output buffer (YOLOv8 format)
        val outputBuffer = Array(1) { Array(84) { FloatArray(8400) } }
        
        // Run inference
        interpreter?.run(inputBuffer, outputBuffer)
        
        // Parse results
        return parseYoloOutput(outputBuffer[0], bitmap.width, bitmap.height)
    }
    
    private fun parseYoloOutput(output: Array<FloatArray>, imageWidth: Int, imageHeight: Int): List<Detection> {
        val detections = mutableListOf<Detection>()
        val numPredictions = 8400
        
        for (i in 0 until numPredictions) {
            // YOLOv8 output format: [x, y, w, h, class0_conf, class1_conf, ...]
            val x = output[0][i]
            val y = output[1][i]
            val w = output[2][i]
            val h = output[3][i]
            
            // Find class with highest confidence
            var maxConf = 0f
            var maxClass = -1
            
            for (c in 0 until numClasses) {
                val conf = output[4 + c][i]
                if (conf > maxConf) {
                    maxConf = conf
                    maxClass = c
                }
            }
            
            // Filter by confidence and vehicle classes
            if (maxConf > confidenceThreshold && vehicleClasses.containsKey(maxClass)) {
                val left = ((x - w / 2) * imageWidth / inputSize).coerceIn(0f, imageWidth.toFloat())
                val top = ((y - h / 2) * imageHeight / inputSize).coerceIn(0f, imageHeight.toFloat())
                val right = ((x + w / 2) * imageWidth / inputSize).coerceIn(0f, imageWidth.toFloat())
                val bottom = ((y + h / 2) * imageHeight / inputSize).coerceIn(0f, imageHeight.toFloat())
                
                val rect = RectF(left, top, right, bottom)
                val proximity = (bottom - top) / imageHeight
                
                detections.add(
                    Detection(
                        rect,
                        maxClass,
                        vehicleClasses[maxClass] ?: "vehicle",
                        maxConf,
                        proximity
                    )
                )
            }
        }
        
        return nonMaxSuppression(detections)
    }
    
    private fun nonMaxSuppression(detections: List<Detection>): List<Detection> {
        val sortedDetections = detections.sortedByDescending { it.confidence }
        val selected = mutableListOf<Detection>()
        
        for (detection in sortedDetections) {
            var shouldSelect = true
            
            for (selectedDetection in selected) {
                if (calculateIoU(detection.rect, selectedDetection.rect) > iouThreshold) {
                    shouldSelect = false
                    break
                }
            }
            
            if (shouldSelect) {
                selected.add(detection)
            }
        }
        
        return selected
    }
    
    private fun calculateIoU(rect1: RectF, rect2: RectF): Float {
        val intersectionArea = RectF().apply {
            setIntersect(rect1, rect2)
        }.run {
            if (isEmpty) 0f else width() * height()
        }
        
        val rect1Area = rect1.width() * rect1.height()
        val rect2Area = rect2.width() * rect2.height()
        val unionArea = rect1Area + rect2Area - intersectionArea
        
        return if (unionArea > 0) intersectionArea / unionArea else 0f
    }
    
    private fun updateStatus(detections: List<Detection>) {
        val vehicleCount = detections.size
        val closestProximity = detections.maxOfOrNull { it.proximity } ?: 0f
        
        val status = when {
            closestProximity > dangerThreshold -> "⚠️ DANGER"
            closestProximity > warningThreshold -> "⚠️ WARNING"
            vehicleCount > 0 -> "✓ SAFE"
            else -> "No vehicles detected"
        }
        
        statusText.text = "Vehicles: $vehicleCount | $status"
    }
    
    private fun handleWarnings(detections: List<Detection>) {
        val closestProximity = detections.maxOfOrNull { it.proximity } ?: 0f
        val currentTime = System.currentTimeMillis()
        
        if (currentTime - lastWarningTime < warningCooldown) {
            return
        }
        
        when {
            closestProximity > dangerThreshold -> {
                speakWarning("Danger! Car very close!")
                lastWarningTime = currentTime
            }
            closestProximity > warningThreshold -> {
                speakWarning("Warning! Car approaching")
                lastWarningTime = currentTime
            }
        }
    }
    
    private fun speakWarning(text: String) {
        tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, null)
    }
    
    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            tts.language = Locale.US
            tts.setSpeechRate(1.2f)
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        tts.shutdown()
        interpreter?.close()
    }
}

data class Detection(
    val rect: RectF,
    val classId: Int,
    val label: String,
    val confidence: Float,
    val proximity: Float
)
