import os
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox


def extract_audio():
    input_path = filedialog.askopenfilename(
        title="Select a video file",
        filetypes=[
            ("Video files", "*.mp4 *.mov *.mkv *.avi *.webm *.m4v"),
            ("All files", "*.*"),
        ],
    )

    if not input_path:
        return

    folder = os.path.dirname(input_path)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(folder, f"{base_name}_audio.mp3")

    try:
        command = [
            "ffmpeg",
            "-y",
            "-i", input_path,
            "-vn",
            "-acodec", "libmp3lame",
            "-q:a", "2",
            output_path,
        ]

        subprocess.run(command, check=True)

        messagebox.showinfo(
            "Done",
            f"Audio extracted successfully:\n\n{output_path}"
        )

    except FileNotFoundError:
        messagebox.showerror(
            "FFmpeg not found",
            "FFmpeg must be installed and available on PATH."
        )

    except subprocess.CalledProcessError:
        messagebox.showerror(
            "Extraction failed",
            "FFmpeg could not extract audio from this file."
        )


root = tk.Tk()
root.title("Extract Audio from Video")
root.geometry("360x160")
root.resizable(False, False)

label = tk.Label(
    root,
    text="Select a video file.\nAudio will be saved beside it as MP3.",
    pady=20,
)
label.pack()

button = tk.Button(
    root,
    text="Choose Video and Extract Audio",
    command=extract_audio,
    width=30,
    height=2,
)
button.pack()

root.mainloop()