import pyperclip
import pynput.keyboard
import pyautogui
import time
import winsound
import re

from pynput import mouse

# Quick and dirty script taking control of the computer to interact with page UI
# Coordinates may need to be specified for your device (change all lines labelled "SPECIFY")
# Set below line and right click to log mouse coords to console
get_coords = False

emergency_stop = False
# Emergency stop
def on_press(key):
    global emergency_stop
    try:
        if key.char == 'z':
            emergency_stop = True
    except AttributeError:
        pass
z_listener = pynput.keyboard.Listener(on_press=on_press)
z_listener.start()


coords = []

# Right click to print current mouse coordinates
while get_coords:
    def on_click(x, y, button, pressed):
        global i
        if pressed and button == mouse.Button.right:
            coords.append(int(x))  # Store click coordinates
            coords.append(int(y))
            winsound.Beep(440, 200)
            print(x, y)

    with mouse.Listener(on_click=on_click) as listener:
        listener.join()

time.sleep(5)
with open('keys.txt', 'a') as file:
    i = 1 # Index of next named key

    while True:
        # Create new
        pyautogui.leftClick(1448, 256) # SPECIFY: "Create new" button coordinates
        time.sleep(0.5)

        # Name key
        pyautogui.write(f"wai{i}")

        # Wait until selectable (match cloudflare orange logo)
        time.sleep(1)
        while pyautogui.pixelMatchesColor(1308, 789, (246, 130, 31)): # SPECIFY: Anywhere orange on the cloudflare logo
            time.sleep(1)
            pyautogui.moveTo(1004, 810, 1, pyautogui.easeOutQuad) # SPECIFY: 'Are you a robot' checkbox
            pyautogui.leftClick(1004, 810) # SPECIFY: 'Are you a robot' checkbox

        # Press submit
        pyautogui.leftClick(1308, 806) # SPECIFY: Submit button

        # Wait until ui becomes smaller
        while not pyautogui.pixelMatchesColor(927, 815, (25, 25, 25)): # SPECIFY: Bottom edge of the UI before shrinking
            time.sleep(0.1)

        # grab key
        pyautogui.hotkey('ctrl', 'c')
        print(pyperclip.paste())
        file.write(f"key{i}: {pyperclip.paste()}\n")
        file.flush()
        winsound.Beep(440, 200)
        i += 1

        # Done
        pyautogui.leftClick(1311, 723) # SPECIFY: Done button
        time.sleep(0.3)