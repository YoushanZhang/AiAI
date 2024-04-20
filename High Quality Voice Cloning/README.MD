```python
from scipy.io.wavfile import write as write_wav



```

# High Quality Voice Cloning

This work presents two different approaches to text-to-speech (TTS) synthesis. We incorporate speaker variance information beyond duration (pitch, energy) to enrich the model’s understanding of speech variations. These features are used during training and inference, improving the one-to-many mapping problem in TTS. The second approach focuses on high-fidelity audio generation with long-term coherence.

![](./synthesized.png)

![](./Loss_total_loss.svg)

```python
scipy.io.wavfile.write("clip_slide3_0000-0010.wav", rate=sample_rate, data=audio_array)
```
