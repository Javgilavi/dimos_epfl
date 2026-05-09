# DimOS — EPFL fork (Go2 + Bedrock Claude Sonnet 4.6)

This fork ships an agentic Unitree Go2 stack wired to **AWS Bedrock Claude Sonnet 4.6** as the LLM backend, with a few patches over upstream:

- McpClient default model = `bedrock_converse:us.anthropic.claude-sonnet-4-6`
- TTS (`speak` skill) is a no-op — avoids OpenAI TTS auth errors when using non-OpenAI keys
- `SecurityModule` removed from the spatial blueprint to free ~3.3 GiB VRAM on 8 GB cards (RTX 5070, 4070, etc.)

The original upstream README is preserved as [README2.md](./README2.md).

---

## 1. Prerequisites

- Ubuntu 22.04 or 24.04, Python 3.12, NVIDIA driver + CUDA 12.x
- AWS account with Bedrock access enabled (cross-region inference profile for Anthropic). EPFL workshop credentials work.
- For the **real robot**: a Unitree Go2, reachable from your laptop on `192.168.123.0/24` (over Ethernet to the rear LAN port, or once the robot is in STA mode on a shared WiFi). The robot's WebRTC bridge listens on `192.168.123.161`.
- For the **simulator**: nothing extra (MuJoCo is bundled).

---

## 2. Install

```bash
git clone https://github.com/Javgilavi/dimos_epfl.git ~/robohack-epfl/dimos
cd ~/robohack-epfl/dimos

uv venv --python 3.12
source .venv/bin/activate

# Core dimos extras + AWS Bedrock client
uv pip install -e '.[base,unitree,sim,agents,perception,misc,cuda]'
uv pip install langchain-aws boto3
```

One-time per-boot kernel tweaks DimOS needs for LCM (it'll prompt for sudo):
```bash
sudo sh -c 'ip link set lo multicast on; ip route add 224.0.0.0/4 dev lo; sysctl -w net.core.rmem_max=67108864 net.core.rmem_default=67108864'
```

---

## 3. Create `.env`

Create a file `~/robohack-epfl/dimos/.env` with **your** credentials. Template:

```bash
# AWS Bedrock (Claude Sonnet 4.6 in us-west-2)
AWS_DEFAULT_REGION=us-west-2
AWS_ACCESS_KEY_ID=ASIA...
AWS_SECRET_ACCESS_KEY=...
AWS_SESSION_TOKEN=...                    # only needed for STS / temporary creds

# Real robot only — IP of the Go2 WebRTC bridge.
# .161 is the AI module on the robot's internal LAN.
# After STA-mode reconfig (robot on your WiFi/hotspot), use whatever IP your DHCP gave it.
ROBOT_IP=192.168.123.161
```

**Important:**
- `.env` is loaded automatically by dimos via `python-dotenv`. No need to `source` it.
- **Never commit `.env`** — add it to `.gitignore`:
  ```bash
  grep -qxF '.env' .gitignore || echo '.env' >> .gitignore
  ```
- AWS STS session tokens (the `ASIA...` flavour) expire in 1–12 hours. When the agent suddenly stops with `ExpiredTokenException`, regenerate them and update `.env`.

### Where to get AWS credentials

If you're at an EPFL workshop, the credentials should be provided to you (region, access key, secret, session token).

If you're on your own AWS account:
1. AWS Console → IAM → create a user with `AmazonBedrockFullAccess` (or a custom policy with `bedrock:InvokeModel` on the model ARN).
2. Create access keys.
3. AWS Console → Bedrock → **Model access** → enable `Anthropic Claude Sonnet 4.6` (and any other models you want).

### Find the robot IP

- **On the robot's internal LAN (Ethernet to rear port):** `192.168.123.161` is the WebRTC bridge. `192.168.123.18` is the firmware-update UI (different module — don't confuse).
- **After STA-mode reconfig (robot joined your WiFi/hotspot):** the IP comes from your router's DHCP. Check the router's connected-clients page, or scan with `nmap -sn 192.168.43.0/24` (Android hotspot subnet) / `nmap -sn 172.20.10.0/28` (iPhone hotspot subnet).
- Quick reachability check: `ping -c 2 $ROBOT_IP` should reply <10 ms.

---

## 4. Run on the **real robot** (no simulator)

In **Terminal A**:

```bash
cd ~/robohack-epfl/dimos
source .venv/bin/activate

# Confirm robot is reachable
ping -c 2 $ROBOT_IP

# Launch the agentic stack against the real robot
dimos run unitree-go2-agentic
```

Wait for these log lines (boot ≈ 2 minutes the first time):
```
GO2Connection mode: ai
🟢 Peer Connection State: connected
🟢 Data Channel Verification: ✅ OK
Discovered tools from MCP server. n_tools=22 tools=[...]
```

When `n_tools=22` appears, the agent is ready.

---

## 5. Run in the **simulator** (no robot needed)

Same launch, just add `--simulation`:

```bash
cd ~/robohack-epfl/dimos
source .venv/bin/activate

__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia MUJOCO_GL=glfw \
    dimos --simulation run unitree-go2-agentic
```

A MuJoCo window opens with the simulated Go2. Wait for the same `Discovered tools from MCP server. n_tools=22` line.

First run downloads ~2 GB of MuJoCo Menagerie + Moondream2. Subsequent runs are much faster.

---

## 6. Chat with the robot — `humancli`

Open **Terminal B** (after you see `n_tools=22` in Terminal A):

```bash
cd ~/robohack-epfl/dimos
source .venv/bin/activate
humancli
```

Then type messages:
```
hello
explore the room
what can you do?
go to the chair
find a door
stop
```

The agent has 22 tools available: `navigate_with_text`, `begin_exploration`, `end_exploration`, `start_patrol`, `look_out_for`, `tag_location`, `follow_person`, `execute_sport_command`, `relative_move`, `wait`, `speak` (no-op TTS — text only), and more.

### Other ways to interact

- **Browser chat**: open `http://localhost:5555` once dimos is running.
- **Web command-center map**: `http://localhost:7779` — click on the costmap to send a navigation goal directly (bypasses the LLM).

---

## 7. Quick reference — common commands

```bash
# kill all dimos processes (after Ctrl-C if anything lingers)
pkill -f 'dimos|mujoco_process' 2>/dev/null ; sleep 3

# Probe MCP server tool catalog while dimos is running
curl -sS -X POST http://localhost:9990/mcp \
    -H "Content-Type: application/json" \
    -H "Accept: application/json, text/event-stream" \
    -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | head -c 400

# Verify Bedrock auth before launching
python -c "
import boto3
print(boto3.client('sts').get_caller_identity()['Arn'])
"

# Verify Bedrock model is callable
python -c "
from langchain.chat_models import init_chat_model
print(init_chat_model('bedrock_converse:us.anthropic.claude-sonnet-4-6').invoke('reply pong').content)
"
```

---

## 8. Switching the robot to your own WiFi (STA mode)

By default the Go2 broadcasts its own AP (`Go2-XXXX_5G`) and your laptop must join it to talk. To put the robot on your phone hotspot or office router instead:

1. Connect ethernet to the robot's rear LAN port.
2. Set a static IP on your wired interface: `192.168.123.99/24`, no gateway, never-default.
3. SSH into the AI module: `ssh unitree@192.168.123.161` (passwords to try: `123`, `unitree`, `Unitree0408`).
4. On the robot:
   ```bash
   sudo iw reg set CH    # or your country code
   sudo nmcli device wifi connect "YOUR_SSID" password "YOUR_PASS"
   sudo nmcli connection modify "YOUR_SSID" connection.autoconnect yes connection.autoconnect-priority 100
   ip -4 addr show wlan0 | grep inet     # write down the new IP
   ```
5. Update `ROBOT_IP=` in `.env` to the new IP.

The other path is BLE provisioning from the laptop (no SSH/passwords needed):
```bash
dimos go2tool connect-wifi --name Go2_14082 --ssid "YOUR_SSID" --password "YOUR_PASS" --country CH
```

---

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `n_tools=0` after launch | EPFL/AWS endpoint cold-starting, or skill containers not registered yet | Wait up to 3 minutes; if persistent, restart |
| Agent silent, no humancli reply | Bedrock auth expired (`ExpiredTokenException` in log) | Refresh `AWS_SESSION_TOKEN` in `.env`, restart |
| `Failed to find Rerun Viewer` on launch | venv not on PATH | Use the launch command above (it sets PATH); or `--viewer none` |
| `CUDA out of memory` on `look_out_for` | Moondream2 + other models exceed 8 GB | Already mitigated by removing SecurityModule. If still OOM, edit `unitree_go2_spatial.py` and remove `PerceiveLoopSkill` too |
| Camera frozen on real robot, restart shows blank | Stale WebRTC peer in robot firmware | Power-cycle the robot, wait 10 s, relaunch |
| `ICE: checking` lingers on connect | Previous client still held by the robot's dispatcher | Kill all dimos, wait 60 s, retry; or reboot robot |
| `Tool call started with UUID...` repeated forever (`observe`) | Async tool returns a promise the agent retries on; known dimos design quirk | Don't use `observe` for counting/Q&A; use `navigate_with_text` for spatial queries |

For the **upstream README** (full module list, capabilities deep dive, blueprint catalog), see [README2.md](./README2.md).
