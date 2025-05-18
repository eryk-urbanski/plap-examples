import os
import pandas as pd
import soundfile as sf

# root -> group -> session -> wavs
def create_csv(path) -> pd.DataFrame:
    """
    Creates a csv file from the TORGO dataset.
    """
    data = []

    print(f"Number of found groups: ", len(os.listdir(path)))
    for group in os.listdir(path):
        if group.split('_')[-1] == 'Dys':
            target = 1
        else:
            target = 0
        group_name = group.split('//')[-1]
        group_path = os.path.join(path,group)
        print(f"Number of found sessions in group{group_name}: ", len(os.listdir(group_path)))
        if not os.path.isdir(group_path):
            continue
        for session in os.listdir(group_path):
            session_name = session.split('_')[-1]
            session_path = os.path.join(group_path,session)
            mic = session.split('_')[1]
            if not os.path.isdir(session_path):
                continue
            for wav_files in os.listdir(session_path):
                if wav_files.endswith('.wav'):
                    file_path = os.path.join(session_path, wav_files)
                    try:
                        audio_data, sample_rate = sf.read(file_path)
                        length = audio_data.shape[0] / sample_rate
                        if length == 0:
                            raise ValueError("Length is 0")
                    except Exception as e:
                        print(f"Could not read {file_path}: {e}")
                        audio_data, sample_rate, length = None, None, None
                    else:
                        data.append({
                            'group': group_name,
                            'session': session_name,
                            'mic': mic,
                            'wav_file': wav_files,
                            'filepath': file_path,
                            'audio_data': audio_data,
                            'sample_rate': sample_rate,
                            'length': length,
                            'target': target
                        })
    return pd.DataFrame(data)

if __name__ == "__main__":
    path = "TORGO"
    df = create_csv(path)
    df.to_csv("torgo_data.csv", index=False)