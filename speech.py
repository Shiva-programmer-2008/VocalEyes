import subprocess

def speak(text):
    # Escape double quotes for PowerShell
    safe_text = text.replace('"', "'")

    ps_command = (
        "Add-Type -AssemblyName System.Speech; "
        "$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$speak.Speak('{safe_text}');"
    )

    subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps_command],
        check=True
    )