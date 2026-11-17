# WirelessJEV field kit: start here

This kit is the whole field system, ready to install. It runs the **Field Platform**: one page on your laptop that walks you through testing the SmartCar on a real site. It has the site setup, the car's checks and radio changes, laps with waypoint marks, a live radio map of the site, the analysis, and an assistant.

What is inside:

- **The platform and all its code.** RadioMap builds the site's radio map from the car's readings.
- **WirelessJEV's d3-lite model** (in `hf-cache/`). It reads the mission in plain language and picks the AP it would use. It only advises (shadow mode); nothing is sent to the car.
- **Every Python package** (in `wheels/`), for Apple-Silicon Macs with Python 3.12 and macOS 14 or newer.

You need a Mac or Linux laptop with **Python 3.12 or newer**. A stock Mac has an older Python; the script tells you how to install it. On an Apple-Silicon Mac with Python 3.12, setup needs no internet. Any other laptop needs internet once, to download the same packages (about 0.7 GB). Windows has not been tested.

## 1. At home (once)

1. **Unzip the kit** anywhere. It is about 1.9 GB, so give it a moment. Open Terminal in the `wirelessjev-field-kit` folder; on a Mac, right-click the folder and choose New Terminal at Folder.
2. **Set up and rehearse:**

   ```bash
   ./field_platform.sh --demo
   ```

   - The first run checks the model files, installs everything, and opens the platform with a made-up site, a car driving a loop, two route laps and a scan.
   - Try every tab. On Analysis, press Lap analysis, 10-robot replay, Coverage report and WirelessJEV proposal; each should finish without an error. On Setup, the "WirelessJEV model" row should be green.
   - Press Ctrl+C in Terminal to stop.
3. **Optional, a smarter assistant:** install Ollama (ollama.com), pull a small instruct model, and set its name on Settings > Assistant. This needs internet once. Without it, the assistant uses its built-in guide, which works offline.
4. **Get the car's two files privately** from Daniel, by AirDrop or USB stick:
   - `smartcar_known_hosts`: put it in your home folder as `~/smartcar_known_hosts`;
   - `WiFi.json`: put it in your home folder as `~/WiFi.json`. Open it in a text editor and add the test network's name and password, following the entries already in it.

   They contain passwords: never put them in the kit folder, a chat or a screenshot.

## 2. On site

1. **Set up the routers in their own apps.** Follow the router checklist on the platform's Setup tab:
   - the same name and password on every router;
   - AP or bridge mode;
   - fixed and different channels;
   - not 192.168.1.x.
2. **Join the car's management hotspot** with the laptop, as the SmartCar instructions say.
3. **Start the platform:**

   ```bash
   ./field_platform.sh
   ```

   It asks for the SmartCar SSH password in Terminal; type it there, never into the page.
4. **Follow the Plan tab from top to bottom.** The Next step box always says the one thing to do now; "What next?" explains why. Steps that change the car's radio make you type the target before anything is sent.
5. **While a lap records,** press the waypoint letter (on screen or on the keyboard) as the car passes each taped marker. That is how the map knows where each reading was taken.
6. **Before leaving,** copy the `artifacts/onecar` folder inside the kit to a USB stick. The laps exist nowhere else.

## If something goes wrong

- **"Python is too old":** install Python 3.12 from python.org, then run the script again.
- **"Model files are damaged or incomplete":** the unzip did not finish. Unzip the kit again.
- **The page says the platform is not answering:** the Terminal running it was closed. Run `./field_platform.sh` again; nothing recorded is lost.
- **A step was refused:** the message says why. Common causes:
  - the SSH password was skipped (restart and type it);
  - demo mode;
  - a hold after an unknown outcome (run Car status).
- **Setup failed part-way:** delete the `.field-venv` folder and run the script again.
- **Anything else:** ask the assistant, look at the Help tab, or open the Activity tab, which records every action, and send that to Daniel.

## What is in the kit

| Item | Purpose |
|---|---|
| `field_platform.sh` | Sets up and starts the platform (`--demo` to rehearse, `--setup-only` to set up and stop) |
| `src/` | The code |
| `hf-cache/` | WirelessJEV's d3-lite model (vllm-sr/d3-lite, Apache-2.0; LICENSE and NOTICE inside) |
| `MODEL_SHA256.txt` | Checksums the script verifies after unzipping |
| `wheels/` | The Python packages for Apple-Silicon Macs with Python 3.12 |
| `requirements-*.txt` | The same packages, pinned, for other laptops |
| `docs/PLATFORM.md` | Everything the platform does, in detail |
| `docs/TUESDAY_RUNBOOK.md` | The site day as terminal commands; the same steps as the Plan tab |
| `docs/REAL_SITE_PROTOCOL.md` | The analysis that was fixed in advance |
