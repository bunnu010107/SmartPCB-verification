package com.example.smartpcbinspector

enum class DefectSeverity {
    CRITICAL,
    HIGH,
    MEDIUM,
    LOW
}

data class SeverityResult(
    val severity: DefectSeverity,
    val recommendation: String
)

class SeverityClassifier {

    fun classify(classId: Int): SeverityResult {

        return when (classId) {

            0 -> SeverityResult(
                DefectSeverity.HIGH,
                "Inspect the missing pad and repair the affected connection."
            )

            1 -> SeverityResult(
                DefectSeverity.MEDIUM,
                "Inspect the PCB edge for mouse bite damage."
            )

            2 -> SeverityResult(
                DefectSeverity.CRITICAL,
                "Inspect and repair the open circuit immediately."
            )

            3 -> SeverityResult(
                DefectSeverity.CRITICAL,
                "Inspect the short circuit immediately."
            )

            4 -> SeverityResult(
                DefectSeverity.MEDIUM,
                "Inspect and remove unwanted copper protrusions."
            )

            5 -> SeverityResult(
                DefectSeverity.HIGH,
                "Inspect and remove unintended copper deposits."
            )

            else -> SeverityResult(
                DefectSeverity.LOW,
                "Perform a manual PCB inspection."
            )
        }
    }
}