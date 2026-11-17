# WirelessJEV field kit: start here

This kit runs the **Field Platform**: one page on your laptop that walks you through testing the SmartCar on a real site, step by step.

- **Setup:** you fill in the site sheet and work through the router checklist.
- **The car:** you run its read-only checks, and its radio changes behind a typed confirmation.
- **Laps:** you record them while marking the taped waypoints.
- **Map:** shows how the car is mapping the site's Wi-Fi.
- **Assistant:** tells you what to do next and how to use any part.

You need:

- a Mac or Linux laptop;
- internet once, for setup;
- about 15 minutes.

Windows has not been tested; use a Mac if you can.

## 1. At home, with internet (once)

1. **Unzip the kit** anywhere, for example your Desktop. Open Terminal in the `wirelessjev-field-kit` folder. On a Mac, right-click the folder and choose New Terminal at Folder.
2. **Rehearse:**

   ```bash
   ./field_platform.sh --demo
   ```

   - **The first run** sets everything up. It needs Python 3.12 or newer; if it is missing, the script says where to get it.
   - **When it is ready,** the platform opens in your browser with a made-up site and a car driving a loop.
   - **Try every tab.** Start with the Help tab and the Map tab, and press "What next?".
   - **To stop it,** press Ctrl+C in the Terminal.
3. **Set up the assistant's local model.** On site the laptop has no internet, so this is the only way to get a smart assistant there; without it, you get the built-in guide.
   1. Install Ollama from ollama.com.
   2. In Terminal:

      ```bash
      ollama pull <model name>
      ```

      Use a small instruct model; the platform's Help tab explains how to choose one.
   3. Start the platform, go to Settings > Assistant, and type the model name.

   The banner at the top of the platform goes away once this works.
4. **Get the car's two files privately** from Daniel (not by chat or email in plain text, because they contain passwords):
   - the host key, saved as `~/smartcar_known_hosts`;
   - the Wi-Fi list, saved as `~/WiFi.json`.

   Add the test network's name and password to `WiFi.json`. Never put these files in the kit folder or in a screenshot.
5. **Read** `docs/TUESDAY_RUNBOOK.md`, sections "Bring" and "AP setup sheet". The same steps are in the platform's Help tab and its router checklist.

## 2. On site

1. **Set up the routers in their own apps.** Use the router checklist on the platform's Setup tab: the same name and password on every router, AP or bridge mode, fixed and different channels, and not 192.168.1.x.
2. **Join the car's management hotspot** with the laptop, as the SmartCar instructions say.
3. **Start the platform:**

   ```bash
   ./field_platform.sh
   ```

   It asks for the SmartCar SSH password in the Terminal; type it there, never into the page. Then the platform opens.
4. **Follow the Plan tab from top to bottom.** The Next step box always says the one thing to do now; "What next?" explains why. Read-only steps have a Run button. Steps that change the car's radio make you type the target before anything is sent.
5. **While a lap records,** press the waypoint letter (on screen, or on the keyboard) as the car passes each taped marker. That is how the map knows where each reading was taken.
6. **Before leaving,** copy the `artifacts/onecar` folder inside the kit to a USB stick. The laps exist nowhere else.

## If something goes wrong

- **The page says the platform is not answering:** the Terminal running it was closed. Run `./field_platform.sh` again; nothing recorded is lost.
- **A step was refused:** the message says why. Common causes:
  - the SSH password was skipped (restart and type it);
  - demo mode;
  - a hold after an unknown outcome (run Car status).
- **The format check failed:** the Next step box names the fix. Do not record laps until it passes.
- **Setup failed on the first run:** check the internet connection, delete the `.field-venv` folder, and run the script again.
- **Anything else:** ask the assistant, look at the Help tab, or open the Activity tab, which records every action, and send that to Daniel.

## What is in the kit

| Item | Purpose |
|---|---|
| `field_platform.sh` | Sets up and starts the platform |
| `requirements-field.txt` | The four Python packages it needs, pinned |
| `src/` | The code |
| `docs/PLATFORM.md` | Everything the platform does, in detail |
| `docs/TUESDAY_RUNBOOK.md` | The site day as terminal commands; the same steps as the Plan tab |
| `docs/REAL_SITE_PROTOCOL.md` | The analysis that was fixed in advance |
