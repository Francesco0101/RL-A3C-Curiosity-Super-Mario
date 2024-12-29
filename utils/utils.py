import matplotlib.pyplot as plt
import numpy as np         
def save(images):
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    fig.suptitle("Prova environment", fontsize=16)
    for i in range(4):
        frame = images[i]  # Estrae il frame i-esimo
        # Scala i valori del frame (se necessario) per essere compatibili con cv2.imshow
        if frame.max() <= 1.0:  # Normalizzato tra 0 e 1
            frame = (frame * 255).astype(np.uint8)
        elif frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        

        axes[i].imshow(frame, cmap="gray")
        axes[i].axis("off")  # Nasconde gli assi
        axes[i].set_title(f"Frame {i + 1}")
        # Mostra il frame
    # Mostra la figura
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Per lasciare spazio al titolo principale
    plt.savefig("frame.png")