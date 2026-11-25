package com.example.smartpcbinspector

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import kotlin.math.min
import kotlin.math.roundToInt

data class Detection(
    val centerX: Float,
    val centerY: Float,
    val width: Float,
    val height: Float,
    val confidence: Float,
    val classId: Int,
    val angle: Float
)

class PCBDetector(context: Context) {

    private val ortEnvironment = OrtEnvironment.getEnvironment()
    private val ortSession: OrtSession

    private val inputSize = 1536
    private val numberOfBoxes = 48384
    private val numberOfClasses = 6
    private val confidenceThreshold = 0.25f

    val classNames = listOf(
        "Missing Pad",
        "Mouse Bite",
        "Open Circuit",
        "Short",
        "Spur",
        "Spurious Copper"
    )

    init {
        val modelBytes = context.assets
            .open("smartpcb_model.onnx")
            .use { it.readBytes() }

        val options = OrtSession.SessionOptions()

        ortSession = ortEnvironment.createSession(
            modelBytes,
            options
        )

        options.close()
    }

    fun detect(bitmap: Bitmap): List<Detection> {

        val scale = min(
            inputSize.toFloat() / bitmap.width,
            inputSize.toFloat() / bitmap.height
        )

        val resizedWidth = (bitmap.width * scale).roundToInt()
        val resizedHeight = (bitmap.height * scale).roundToInt()

        val paddingLeft = (inputSize - resizedWidth) / 2
        val paddingTop = (inputSize - resizedHeight) / 2

        val letterboxedBitmap = Bitmap.createBitmap(
            inputSize,
            inputSize,
            Bitmap.Config.ARGB_8888
        )

        val canvas = Canvas(letterboxedBitmap)
        canvas.drawColor(Color.rgb(114, 114, 114))

        val paint = Paint(Paint.FILTER_BITMAP_FLAG)

        val destination = Rect(
            paddingLeft,
            paddingTop,
            paddingLeft + resizedWidth,
            paddingTop + resizedHeight
        )

        canvas.drawBitmap(bitmap, null, destination, paint)

        val pixels = IntArray(inputSize * inputSize)

        letterboxedBitmap.getPixels(
            pixels,
            0,
            inputSize,
            0,
            0,
            inputSize,
            inputSize
        )

        val planeSize = inputSize * inputSize
        val inputData = FloatArray(3 * planeSize)

        for (i in pixels.indices) {

            val pixel = pixels[i]

            val red = (pixel shr 16) and 0xFF
            val green = (pixel shr 8) and 0xFF
            val blue = pixel and 0xFF

            inputData[i] = red / 255.0f
            inputData[planeSize + i] = green / 255.0f
            inputData[2 * planeSize + i] = blue / 255.0f
        }

        letterboxedBitmap.recycle()

        val inputTensor = OnnxTensor.createTensor(
            ortEnvironment,
            FloatBuffer.wrap(inputData),
            longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())
        )

        val results = try {
            ortSession.run(mapOf("images" to inputTensor))
        } finally {
            inputTensor.close()
        }

        try {
            val outputTensor = results[0] as OnnxTensor
            val output = outputTensor.floatBuffer

            val detections = mutableListOf<Detection>()

            for (boxIndex in 0 until numberOfBoxes) {

                var bestClass = -1
                var bestConfidence = 0.0f

                for (classIndex in 0 until numberOfClasses) {

                    val confidence = output.get(
                        (4 + classIndex) * numberOfBoxes + boxIndex
                    )

                    if (confidence > bestConfidence) {
                        bestConfidence = confidence
                        bestClass = classIndex
                    }
                }

                if (bestConfidence < confidenceThreshold) {
                    continue
                }

                val centerX = output.get(boxIndex)
                val centerY = output.get(numberOfBoxes + boxIndex)
                val width = output.get(2 * numberOfBoxes + boxIndex)
                val height = output.get(3 * numberOfBoxes + boxIndex)
                val angle = output.get(10 * numberOfBoxes + boxIndex)

                detections.add(
                    Detection(
                        centerX = (centerX - paddingLeft) / scale,
                        centerY = (centerY - paddingTop) / scale,
                        width = width / scale,
                        height = height / scale,
                        confidence = bestConfidence,
                        classId = bestClass,
                        angle = angle
                    )
                )
            }

            return rotatedNms(detections)

        } finally {
            results.close()
        }
    }

    private fun rotatedNms(
        detections: List<Detection>
    ): List<Detection> {

        val iouThreshold = 0.5f
        val maxDetections = 300
        val maxCandidates = 1000

        val sorted = detections
            .sortedByDescending { it.confidence }
            .take(maxCandidates)

        val selected = mutableListOf<Detection>()

        for (candidate in sorted) {

            var shouldKeep = true

            for (accepted in selected) {

                if (candidate.classId != accepted.classId) {
                    continue
                }

                val overlap = probabilisticIoU(
                    candidate,
                    accepted
                )

                if (overlap > iouThreshold) {
                    shouldKeep = false
                    break
                }
            }

            if (shouldKeep) {
                selected.add(candidate)
            }

            if (selected.size >= maxDetections) {
                break
            }
        }

        return selected
    }

    private fun probabilisticIoU(
        box1: Detection,
        box2: Detection
    ): Float {

        val eps = 1e-7f

        fun covariance(box: Detection): Triple<Float, Float, Float> {

            val a = box.width * box.width / 12f
            val b = box.height * box.height / 12f

            val cos = kotlin.math.cos(box.angle)
            val sin = kotlin.math.sin(box.angle)

            val cos2 = cos * cos
            val sin2 = sin * sin

            val covA = a * cos2 + b * sin2
            val covB = a * sin2 + b * cos2
            val covC = (a - b) * cos * sin

            return Triple(covA, covB, covC)
        }

        val (a1, b1, c1) = covariance(box1)
        val (a2, b2, c2) = covariance(box2)

        val dx = box1.centerX - box2.centerX
        val dy = box1.centerY - box2.centerY

        val sumA = a1 + a2
        val sumB = b1 + b2
        val sumC = c1 + c2

        val denominator = (
                sumA * sumB - sumC * sumC + eps
                )

        val t1 = (
                sumA * dy * dy +
                        sumB * dx * dx
                ) / denominator * 0.25f

        val t2 = (
                sumC * (box2.centerX - box1.centerX) * dy
                ) / denominator * 0.5f

        val det1 = (a1 * b1 - c1 * c1).coerceAtLeast(0f)
        val det2 = (a2 * b2 - c2 * c2).coerceAtLeast(0f)

        val t3 = 0.5f * kotlin.math.ln(
            (
                    denominator /
                            (4f * kotlin.math.sqrt(det1 * det2) + eps) + eps
                    ).coerceAtLeast(eps)
        )

        val bd = (t1 + t2 + t3).coerceIn(eps, 100f)

        val hd = kotlin.math.sqrt(
            1f - kotlin.math.exp(-bd) + eps
        )

        return (1f - hd).coerceIn(0f, 1f)
    }
    fun getModelInfo(): String {
        return """
            Model loaded successfully!

            Input:
            ${ortSession.inputInfo}

            Output:
            ${ortSession.outputInfo}
        """.trimIndent()
    }

    fun close() {
        ortSession.close()
    }
}