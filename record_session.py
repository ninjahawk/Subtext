# Record a session to docs/demo_session.json for the GitHub Pages replay.
# Run with the server up: python record_session.py
import asyncio
import json
import pathlib

import websockets

PROMPTS = [
    "Is this correct? 12 + 5 = 1",
    "The currency of the country shaped like a boot is…",
]
OUT = pathlib.Path(__file__).parent / "docs" / "demo_session.json"


async def main():
    events = []
    async with websockets.connect("ws://127.0.0.1:8765/ws", max_size=None) as ws:
        events.append(json.loads(await ws.recv()))  # hello
        messages = []
        for prompt in PROMPTS:
            messages.append({"role": "user", "content": prompt})
            events.append({"type": "user", "text": prompt})
            await ws.send(json.dumps(
                {"type": "chat", "messages": messages, "max_tokens": 130}))
            while True:
                m = json.loads(await asyncio.wait_for(ws.recv(), timeout=300))
                events.append(m)
                if m["type"] == "done":
                    messages.append({"role": "assistant", "content": m["text"]})
                    print(f"{prompt!r} -> {m['text'][:70]!r}")
                    break
    OUT.write_text(json.dumps(events))
    print(f"{len(events)} events -> {OUT} "
          f"({OUT.stat().st_size / 1024:.0f} KB)")


asyncio.run(main())
