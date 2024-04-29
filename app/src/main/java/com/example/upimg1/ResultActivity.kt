package com.example.upimg1

import android.net.Uri
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.example.Upimg1.R
import com.example.Upimg1.databinding.ActivityResultBinding

class ResultActivity : AppCompatActivity() {

    private lateinit var binding: ActivityResultBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityResultBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Get the result from the Intent's extras
        val result = intent.getStringExtra("RESULT") ?: "No result received"
        val parts = result.split(" ")
        val label = parts.getOrNull(0) ?: "Error"
        val percentage = parts.getOrNull(1) ?: ""

        // Set the extracted label and percentage to the resultText TextView
        binding.resultText.text = "$label\n$percentage"

        // Get the image URI passed from MainActivity
        val imageUri = intent.getStringExtra("IMAGE_URI")
        if (imageUri != null) {
            binding.image.setImageURI(Uri.parse(imageUri))
        } else {
            // Handle case where no image is provided
            // For example: Set a default image or make the ImageView invisible
           // binding.image.setImageResource(R.drawable.default_image) // Replace with your default image resource
        }
    }
}
