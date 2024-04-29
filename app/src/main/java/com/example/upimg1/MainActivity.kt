package com.example.upimg1

import android.app.AlertDialog
import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.graphics.drawable.toBitmap
import com.example.Upimg1.databinding.ActivityMainBinding
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException
import java.io.OutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private var tflite: Interpreter? = null
    private val num_classes = 2
    private val IMAGE_SIZE = 224
    private var currentImageUri: Uri? = null
    private var imageSource: ImageSource = ImageSource.NONE

    enum class ImageSource {
        GALLERY, CAMERA, NONE
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        try {
            tflite = Interpreter(loadModelFile())
        } catch (e: Exception) {
            e.printStackTrace()
        }

        binding.button.setOnClickListener {
            showImagePickerOptions()
        }

        binding.subButton.setOnClickListener {
            val bitmap = binding.image.drawable?.toBitmap() ?: return@setOnClickListener
            val byteBuffer = convertBitmapToByteBuffer(bitmap)
            val result = Array(1) { FloatArray(num_classes) }
            tflite?.run(byteBuffer, result)
            val maxIndex = result[0].withIndex().maxByOrNull { it.value }?.index ?: -1
            val confidence = result[0][maxIndex] * 100
            val resultLabel = if (maxIndex == 0) "Malignant" else "Benign"

            val targetActivity = when(imageSource) {
                ImageSource.CAMERA -> ResultActivity::class.java
                ImageSource.GALLERY -> ResultActivity::class.java
                else -> return@setOnClickListener
            }

            val intent = Intent(this@MainActivity, targetActivity).apply {
                putExtra("RESULT", "$resultLabel ($confidence%)")
                currentImageUri?.let { uri ->
                    putExtra("IMAGE_URI", uri.toString())
                }
            }
            startActivity(intent)
        }
    }

    private fun showImagePickerOptions() {
        val options = arrayOf("Gallery", "Camera")
        AlertDialog.Builder(this)
            .setTitle("Select Image From")
            .setItems(options) { _, which ->
                when (which) {
                    0 -> {
                        imageSource = ImageSource.GALLERY
                        galleryLauncher.launch("image/*")
                    }
                    1 -> {
                        imageSource = ImageSource.CAMERA
                        cameraLauncher.launch(null)
                    }
                }
            }.show()
    }

    private val galleryLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        binding.image.setImageURI(uri)
        currentImageUri = uri
    }
    private fun saveImageToInternalStorage(bitmap: Bitmap): Uri {
        // Creating a file in the internal storage
        val filename = "IMG_${System.currentTimeMillis()}.jpg"
        val file = File(getExternalFilesDir(Environment.DIRECTORY_PICTURES), filename)
        try {
            val stream: OutputStream = FileOutputStream(file)
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, stream)
            stream.flush()
            stream.close()
        } catch (e: IOException) {
            e.printStackTrace()
        }
        return Uri.fromFile(file)
    }

    private val cameraLauncher = registerForActivityResult(ActivityResultContracts.TakePicturePreview()) { bitmap ->
        bitmap?.let {
            binding.image.setImageBitmap(it)
            // Save the bitmap and update currentImageUri
            currentImageUri = saveImageToInternalStorage(it)
            imageSource = ImageSource.CAMERA
        }
    }


    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, true)
        val intValues = IntArray(resizedBitmap.width * resizedBitmap.height)
        resizedBitmap.getPixels(intValues, 0, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE)

        val byteBuffer = ByteBuffer.allocateDirect(4 * IMAGE_SIZE * IMAGE_SIZE * 3)
        byteBuffer.order(ByteOrder.nativeOrder())

        var pixel = 0
        for (i in 0 until IMAGE_SIZE) {
            for (j in 0 until IMAGE_SIZE) {
                val value = intValues[pixel++]
                byteBuffer.putFloat(((value shr 16) and 0xFF) / 255.0f)
                byteBuffer.putFloat(((value shr 8) and 0xFF) / 255.0f)
                byteBuffer.putFloat((value and 0xFF) / 255.0f)
            }
        }

        return byteBuffer
    }

    @Throws(IOException::class)
    private fun loadModelFile(): MappedByteBuffer {
        val assetFileDescriptor = assets.openFd("customm.tflite")
        val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = fileInputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        var declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
}
