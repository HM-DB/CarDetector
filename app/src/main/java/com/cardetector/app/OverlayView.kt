package com.cardetector.app

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View

class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {
    
    private var detections: List<Detection> = emptyList()
    private var imageWidth = 0
    private var imageHeight = 0
    
    private val boxPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 5f
        isAntiAlias = true
    }
    
    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 40f
        isAntiAlias = true
        typeface = Typeface.DEFAULT_BOLD
    }
    
    private val textBackgroundPaint = Paint().apply {
        style = Paint.Style.FILL
        isAntiAlias = true
    }
    
    private val dangerThreshold = 0.4f
    private val warningThreshold = 0.25f
    
    fun setDetections(detections: List<Detection>, imageWidth: Int, imageHeight: Int) {
        this.detections = detections
        this.imageWidth = imageWidth
        this.imageHeight = imageHeight
        invalidate()
    }
    
    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        
        if (imageWidth == 0 || imageHeight == 0) return
        
        val scaleX = width.toFloat() / imageWidth
        val scaleY = height.toFloat() / imageHeight
        
        for (detection in detections) {
            // Scale the bounding box to fit the view
            val scaledRect = RectF(
                detection.rect.left * scaleX,
                detection.rect.top * scaleY,
                detection.rect.right * scaleX,
                detection.rect.bottom * scaleY
            )
            
            // Determine color based on proximity
            val color = when {
                detection.proximity > dangerThreshold -> Color.RED
                detection.proximity > warningThreshold -> Color.rgb(255, 165, 0) // Orange
                else -> Color.GREEN
            }
            
            // Draw bounding box
            boxPaint.color = color
            canvas.drawRect(scaledRect, boxPaint)
            
            // Prepare label text
            val label = "${detection.label} (${(detection.proximity * 100).toInt()}%)"
            
            // Measure text
            val textBounds = Rect()
            textPaint.getTextBounds(label, 0, label.length, textBounds)
            
            // Draw text background
            val textBackgroundRect = RectF(
                scaledRect.left,
                scaledRect.top - textBounds.height() - 20f,
                scaledRect.left + textBounds.width() + 20f,
                scaledRect.top
            )
            
            textBackgroundPaint.color = color
            canvas.drawRect(textBackgroundRect, textBackgroundPaint)
            
            // Draw text
            canvas.drawText(
                label,
                scaledRect.left + 10f,
                scaledRect.top - 10f,
                textPaint
            )
            
            // Draw proximity indicator
            val statusText = when {
                detection.proximity > dangerThreshold -> "DANGER"
                detection.proximity > warningThreshold -> "WARNING"
                else -> "SAFE"
            }
            
            canvas.drawText(
                statusText,
                scaledRect.left + 10f,
                scaledRect.bottom - 10f,
                textPaint
            )
        }
    }
}
