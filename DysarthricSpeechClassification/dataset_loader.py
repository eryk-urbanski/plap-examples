import os
import soundfile as sf
from librosa.display import waveshow
import matplotlib.pyplot as plt

class TORGO:
    """
    Dataset class for TORGO with lazy loading and batch support.
    """

    def __init__(self, path):
        self.file_paths = []
        self.labels = []
        self.sample_rate = 16000  # 16k Hz 

        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(".wav"):
                    full_path = os.path.join(root, file)
                    try:
                        # Extract label based on top-level folder name
                        top_level_dir = os.path.normpath(full_path).split(os.sep)[-3]
                        label = 0 if "Con" in top_level_dir else 1
                        self.file_paths.append(full_path)
                        self.labels.append(label)
                    except Exception as e:
                        print(f"Error with {full_path}: {e}")

        print(f"Initialized TORGO dataset with {len(self.file_paths)} files.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        label = self.labels[idx]
        return audio_path, label

    def load_batch(self, start_idx, batch_size):
        batch_audio_paths = []
        batch_labels = []
        end_idx = min(start_idx + batch_size, len(self))
        for idx in range(start_idx, end_idx):
            audio_path, label = self[idx]
            batch_audio_paths.append(audio_path)
            batch_labels.append(label)
        return batch_audio_paths, batch_labels

    def show(self, idx):
        audio_path, _ = self[idx]
        try:
            audio, _ = sf.read(audio_path)
        except Exception as e:
            print(f"Error reading {audio_path}: {e}")
            audio = None
        plt.figure(figsize=(15, 4))
        waveshow(audio, sr=self.sample_rate)
        plt.title(self.file_paths[idx].split("\\")[-1])
        plt.show()
