# -----------------------------------------------------
# SPEECH ENHANCEMENT USING DEEP LEARNING (AUTOENCODER)
# -----------------------------------------------------
# Author: Nehaa Prasad
# Course: Deep Learning (22ISE74A)
# -----------------------------------------------------

import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# ------------------------------
# STEP 1: LOAD CLEAN AUDIO
# ------------------------------
clean_audio, sr = librosa.load("clean_speech.wav", sr=None)
print("✅ Clean audio loaded successfully.")

# ------------------------------
# STEP 2: CREATE NOISY VERSION AUTOMATICALLY
# ------------------------------
noise = np.random.normal(0, 0.02, clean_audio.shape)  # Adjust 0.02 for noise intensity
noisy_audio = clean_audio + noise
sf.write("noisy_speech.wav", noisy_audio, sr)
print("✅ Noisy audio generated and saved as 'noisy_speech.wav'.")

# ------------------------------
# STEP 3: CONVERT TO SPECTROGRAM
# ------------------------------
clean_spec = np.abs(librosa.stft(clean_audio, n_fft=512))
noisy_spec = np.abs(librosa.stft(noisy_audio, n_fft=512))

# Transpose for model input (time steps x frequency bins)
X = noisy_spec.T
Y = clean_spec.T

print("✅ Spectrograms created. Shape:", X.shape)

# ------------------------------
# STEP 4: BUILD AUTOENCODER MODEL
# ------------------------------
input_dim = X.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(256, activation='relu')(input_layer)
encoded = Dense(128, activation='relu')(encoded)
decoded = Dense(256, activation='relu')(encoded)
output_layer = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer=Adam(0.001), loss='mse')

autoencoder.summary()

# ------------------------------
# STEP 5: TRAIN MODEL
# ------------------------------
autoencoder.fit(X, Y, epochs=25, batch_size=64, verbose=1)
print("✅ Model training complete.")

# ------------------------------
# STEP 6: DENOISE USING TRAINED MODEL
# ------------------------------
predicted_spec = autoencoder.predict(X)

# Use noisy phase for reconstruction
phase = np.angle(librosa.stft(noisy_audio, n_fft=512))
reconstructed_stft = predicted_spec.T * np.exp(1j * phase)
enhanced_audio = librosa.istft(reconstructed_stft)

# Save denoised output
sf.write("enhanced_speech.wav", enhanced_audio, sr)
print("✅ Denoised speech saved as 'enhanced_speech.wav'")

# ------------------------------
# STEP 7: VISUALIZE RESULTS
# ------------------------------
plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.title("Original Clean Speech")
plt.plot(clean_audio, color='green')

plt.subplot(3, 1, 2)
plt.title("Noisy Speech (Generated)")
plt.plot(noisy_audio, color='red')

plt.subplot(3, 1, 3)
plt.title("Enhanced Speech (After Denoising)")
plt.plot(enhanced_audio, color='blue')

plt.tight_layout()
plt.show()
