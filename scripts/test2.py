import torch
import os
from torchvision import models
import torch.nn as nn

print("🧪 TEST: Modell erstellen, vorwärts rechnen, speichern")

# === 1. Dummy-Modell (wie dein echtes Mobilenet + LSTM)
class DummyTemporalModel(nn.Module):
    def __init__(self, num_classes=3, sequence_length=10):
        super(DummyTemporalModel, self).__init__()
        self.sequence_length = sequence_length
        self.base = models.mobilenet_v2(weights='IMAGENET1K_V1').features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flat = nn.Flatten()
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.base(x)
        x = self.pool(x)
        x = self.flat(x)
        x = self.fc(x)
        x = x.view(B, T, -1)  # simulate sequence output
        return x.mean(dim=1)  # simplify for test

# === 2. Modell erzeugen
model = DummyTemporalModel(num_classes=3, sequence_length=10)
print("✅ Modell erstellt")

# === 3. Dummy-Eingabe erzeugen
dummy_input = torch.randn(2, 10, 3, 224, 224)  # (Batch=2, Seq=10)

# === 4. Forward-Test
try:
    output = model(dummy_input)
    print(f"✅ Forward erfolgreich, Output-Shape: {output.shape}")
except Exception as e:
    print(f"❌ Fehler beim Forward-Pass: {e}")
    exit(1)

# === 5. Speichern testen
save_dir = "models"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "test_model_save.pth")

try:
    torch.save(model.state_dict(), save_path)
    if os.path.exists(save_path):
        print(f"✅ Modell gespeichert unter: {os.path.abspath(save_path)}")
    else:
        print("❌ Modell wurde NICHT gespeichert.")
except Exception as e:
    print(f"❌ Fehler beim Speichern: {e}")
