package com.example.smartpcbinspector

import androidx.compose.ui.graphics.nativeCanvas
import android.graphics.Paint
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Box
import androidx.activity.result.PickVisualMediaRequest
import android.os.Bundle
import android.graphics.BitmapFactory
import android.net.Uri
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.smartpcbinspector.ui.theme.SmartPCBInspectorTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            SmartPCBInspectorTheme {
                SmartPCBHomeScreen()
            }
        }
    }
}

@Composable
fun SmartPCBHomeScreen() {

    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    var selectedImageUri by remember {
        mutableStateOf<Uri?>(null)
    }

    var isDetecting by remember {
        mutableStateOf(false)
    }

    var detections by remember {
        mutableStateOf<List<Detection>>(emptyList())
    }

    var detectionError by remember {
        mutableStateOf<String?>(null)
    }

    val detector = remember(context) {
        PCBDetector(context.applicationContext)
    }

    DisposableEffect(detector) {
        onDispose {
            detector.close()
        }
    }

    val imagePicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia()
    ) { uri ->
        selectedImageUri = uri
        detections = emptyList()
        detectionError = null
    }

    val selectedBitmap = remember(selectedImageUri) {
        selectedImageUri?.let { uri ->
            try {
                context.contentResolver.openInputStream(uri)?.use {
                    BitmapFactory.decodeStream(it)
                }
            } catch (e: Exception) {
                null
            }
        }
    }

    Surface(
        modifier = Modifier.fillMaxSize(),
        color = MaterialTheme.colorScheme.background
    ) {

        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(20.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {

            Spacer(modifier = Modifier.height(25.dp))

            Text(
                text = "SMART PCB",
                fontSize = 32.sp,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.primary
            )

            Text(
                text = "INSPECTOR",
                fontSize = 24.sp,
                fontWeight = FontWeight.Medium
            )

            Spacer(modifier = Modifier.height(8.dp))

            Text(
                text = "AI-Powered PCB Defect Detection",
                fontSize = 14.sp
            )

            Spacer(modifier = Modifier.height(25.dp))

            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(16.dp)
            ) {

                Column(
                    modifier = Modifier.padding(18.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {

                    Text(
                        text = "PCB INSPECTION",
                        fontSize = 20.sp,
                        fontWeight = FontWeight.Bold
                    )

                    Spacer(modifier = Modifier.height(15.dp))

                    Button(
                        onClick = {
                            imagePicker.launch(
                                PickVisualMediaRequest(
                                    ActivityResultContracts.PickVisualMedia.ImageOnly
                                )
                            )
                        },
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text("SELECT PCB IMAGE")
                    }

                    Spacer(modifier = Modifier.height(15.dp))

                    if (selectedImageUri != null) {

                        Text(
                            text = "Selected PCB Image",
                            fontWeight = FontWeight.Bold
                        )

                        Spacer(modifier = Modifier.height(10.dp))

                        if (selectedBitmap != null) {

                            Box(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .height(220.dp)
                            ) {

                                Image(
                                    bitmap = selectedBitmap.asImageBitmap(),
                                    contentDescription = "Selected PCB Image",
                                    modifier = Modifier.matchParentSize(),
                                    contentScale = ContentScale.Fit
                                )

                                Canvas(
                                    modifier = Modifier.matchParentSize()
                                ) {

                                    val scale = minOf(
                                        size.width / selectedBitmap.width,
                                        size.height / selectedBitmap.height
                                    )

                                    val offsetX = (size.width -
                                            selectedBitmap.width * scale) / 2f

                                    val offsetY = (size.height -
                                            selectedBitmap.height * scale) / 2f

                                    detections.forEachIndexed { index, detection ->
                                        val boxColor = when (detection.classId) {
                                            0 -> androidx.compose.ui.graphics.Color.Red
                                            1 -> androidx.compose.ui.graphics.Color.Yellow
                                            2 -> androidx.compose.ui.graphics.Color.Green
                                            3 -> androidx.compose.ui.graphics.Color.Cyan
                                            4 -> androidx.compose.ui.graphics.Color.Magenta
                                            5 -> androidx.compose.ui.graphics.Color.Blue
                                            else -> androidx.compose.ui.graphics.Color.White
                                        }

                                        val halfWidth = detection.width / 2f
                                        val halfHeight = detection.height / 2f

                                        val cos = kotlin.math.cos(detection.angle)
                                        val sin = kotlin.math.sin(detection.angle)

                                        val corners = listOf(
                                            Pair(-halfWidth, -halfHeight),
                                            Pair(halfWidth, -halfHeight),
                                            Pair(halfWidth, halfHeight),
                                            Pair(-halfWidth, halfHeight)
                                        ).map { (x, y) ->

                                            val rotatedX =
                                                detection.centerX + x * cos - y * sin

                                            val rotatedY =
                                                detection.centerY + x * sin + y * cos

                                            androidx.compose.ui.geometry.Offset(
                                                offsetX + rotatedX * scale,
                                                offsetY + rotatedY * scale
                                            )
                                        }

                                        for (i in corners.indices) {

                                            drawLine(
                                                color = boxColor,
                                                start = corners[i],
                                                end = corners[(i + 1) % corners.size],
                                                strokeWidth = 3f
                                            )
                                        }
                                        val defectName = detector.classNames.getOrElse(
                                            detection.classId
                                        ) { "Unknown" }

                                        val label = "$defectName ${
                                            "%.1f".format(detection.confidence * 100)
                                        }%"

                                        val labelColor = when (detection.classId) {
                                            0 -> android.graphics.Color.RED
                                            1 -> android.graphics.Color.YELLOW
                                            2 -> android.graphics.Color.GREEN
                                            3 -> android.graphics.Color.CYAN
                                            4 -> android.graphics.Color.MAGENTA
                                            5 -> android.graphics.Color.BLUE
                                            else -> android.graphics.Color.WHITE
                                        }

                                        val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                                            color = android.graphics.Color.WHITE
                                            textSize = 18f
                                            typeface = android.graphics.Typeface.DEFAULT_BOLD
                                        }

                                        val backgroundPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                                            color = labelColor
                                        }

                                        val labelWidth = textPaint.measureText(label)

                                        val labelX = corners.minOf { it.x }.coerceIn(
                                            0f,
                                            (size.width - labelWidth - 12f).coerceAtLeast(0f)
                                        )

                                        val labelY = (
                                                corners.minOf { it.y } - 5f + (index * 24f)
                                                ).coerceIn(20f, size.height - 5f)

                                        drawContext.canvas.nativeCanvas.drawRoundRect(
                                            labelX,
                                            labelY - 30f,
                                            labelX + labelWidth + 12f,
                                            labelY + 6f,
                                            5f,
                                            5f,
                                            backgroundPaint
                                        )

                                        drawContext.canvas.nativeCanvas.drawText(
                                            label,
                                            labelX + 6f,
                                            labelY,
                                            textPaint
                                        )
                                    }
                                }
                            }

                        } else {
                            Text("Unable to load selected image")
                        }

                    } else {
                        Text("No PCB image selected")
                    }

                    Spacer(modifier = Modifier.height(15.dp))

                    OutlinedButton(
                        onClick = {
                            selectedImageUri = null
                            detections = emptyList()
                            detectionError = null
                        },
                        enabled = selectedImageUri != null && !isDetecting,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text("REMOVE IMAGE")
                    }

                    Spacer(modifier = Modifier.height(12.dp))

                    Button(
                        onClick = {

                            val bitmap = selectedBitmap

                            if (bitmap != null) {

                                scope.launch {

                                    isDetecting = true
                                    detectionError = null
                                    detections = emptyList()

                                    try {

                                        detections = withContext(Dispatchers.Default) {
                                            detector.detect(bitmap)
                                        }

                                    } catch (e: Exception) {

                                        detectionError =
                                            e.message ?: "Unknown detection error"

                                    } finally {
                                        isDetecting = false
                                    }
                                }
                            }
                        },
                        enabled = selectedBitmap != null && !isDetecting,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text(
                            if (isDetecting) {
                                "DETECTING..."
                            } else {
                                "DETECT DEFECTS"
                            }
                        )
                    }

                    if (isDetecting) {
                        Spacer(modifier = Modifier.height(12.dp))
                        CircularProgressIndicator()
                        Text("AI model is analysing the PCB...")
                    }

                    if (detectionError != null) {

                        Spacer(modifier = Modifier.height(12.dp))

                        Text(
                            text = "Detection error: $detectionError",
                            color = MaterialTheme.colorScheme.error
                        )
                    }

                    if (!isDetecting && detectionError == null &&
                        selectedImageUri != null &&
                        detections.isNotEmpty()
                    ) {

                        Spacer(modifier = Modifier.height(18.dp))

                        Text(
                            text = "DETECTION RESULTS",
                            fontSize = 18.sp,
                            fontWeight = FontWeight.Bold
                        )

                        detections.forEachIndexed { index, detection ->

                            val severityResult = SeverityClassifier().classify(
                                detection.classId
                            )

                            Spacer(modifier = Modifier.height(8.dp))

                            Text(
                                text = "${index + 1}. ${
                                    detector.classNames.getOrElse(
                                        detection.classId
                                    ) { "Unknown" }
                                } — ${
                                    "%.1f".format(detection.confidence * 100)
                                }%",
                                modifier = Modifier.fillMaxWidth(),
                                fontWeight = FontWeight.Medium
                            )

                            Text(
                                text = "Severity: ${severityResult.severity}",
                                modifier = Modifier.fillMaxWidth(),
                                fontWeight = FontWeight.Bold,
                                color = when (severityResult.severity) {
                                    DefectSeverity.CRITICAL -> androidx.compose.ui.graphics.Color.Red
                                    DefectSeverity.HIGH -> androidx.compose.ui.graphics.Color(0xFFFF5722)
                                    DefectSeverity.MEDIUM -> androidx.compose.ui.graphics.Color(0xFFFFA000)
                                    DefectSeverity.LOW -> androidx.compose.ui.graphics.Color.Green
                                }
                            )

                            Text(
                                text = "Recommendation: ${severityResult.recommendation}",
                                modifier = Modifier.fillMaxWidth()
                            )
                        }

                    } else if (
                        !isDetecting &&
                        detectionError == null &&
                        selectedImageUri != null &&
                        detections.isEmpty()
                    ) {

                        Spacer(modifier = Modifier.height(12.dp))

                        Text("No defects detected above the confidence threshold.")
                    }
                }
            }

            Spacer(modifier = Modifier.height(25.dp))

            Text(
                text = "SUPPORTED DEFECT CLASSES",
                fontSize = 17.sp,
                fontWeight = FontWeight.Bold
            )

            Spacer(modifier = Modifier.height(12.dp))

            val defects = listOf(
                "Missing Pad",
                "Mouse Bite",
                "Open Circuit",
                "Short",
                "Spur",
                "Spurious Copper"
            )

            defects.forEach { defect ->
                Text(
                    text = "• $defect",
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 4.dp)
                )
            }

            Spacer(modifier = Modifier.height(25.dp))

            Text(
                text = "SMARTPCB | ARM EDGE AI",
                fontSize = 12.sp,
                color = MaterialTheme.colorScheme.secondary
            )

            Spacer(modifier = Modifier.height(15.dp))
        }
    }
}