"""Creates a QR code image of the website link and saves it to your Downloads folder.

Run:   python make_qr.py                      (uses this computer's Wi-Fi address, port 5000)
       python make_qr.py http://192.168.1.5:5000   (or any link you want)
"""
import os
import socket
import sys
from pathlib import Path

import qrcode


def lan_ip():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


url = sys.argv[1] if len(sys.argv) > 1 else f"http://{lan_ip()}:5000"
img = qrcode.make(url)

targets = [Path(__file__).with_name("qr_code.png")]
downloads = Path.home() / "Downloads"
if downloads.exists():
    targets.append(downloads / "smart_transport_qr.png")

for t in targets:
    img.save(t)
    print("Saved:", t)
print("QR link:", url)

if os.name == "nt":  # open the image automatically on Windows
    os.startfile(targets[-1])
