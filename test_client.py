# Smoke test: send one chat message, print a sample of the frames.
import asyncio
import json

import websockets


async def main():
    async with websockets.connect("ws://127.0.0.1:8765/ws", max_size=None) as ws:
        await ws.send(json.dumps({
            "type": "chat",
            "messages": [{"role": "user", "content": "Is this correct? 12 + 5 = 1"}],
            "max_tokens": 60,
        }))
        n_read = n_think = 0
        while True:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=180))
            if msg["type"] == "frame":
                phase = msg["phase"]
                if phase == "reading":
                    n_read += 1
                    if n_read <= 3 or n_read % 5 == 0:
                        tops = ", ".join(f"{t['w']}({t['p']:.3f},d{t['d']})" for t in msg["thoughts"][:5])
                        print(f"[read {n_read:>2}] tok={msg['tok']!r:14} -> {tops}")
                else:
                    n_think += 1
                    if n_think <= 5 or n_think % 10 == 0:
                        tops = ", ".join(f"{t['w']}({t['p']:.3f})" for t in msg["thoughts"][:5])
                        print(f"[think {n_think:>2}] out={msg['out']!r:12} -> {tops}")
            elif msg["type"] == "done":
                print(f"\nDONE. reply: {msg['text']!r}")
                print(f"frames: {n_read} reading, {n_think} thinking")
                break


asyncio.run(main())
