# Two-turn conversation test: verifies history handling and that the reading
# phase covers only the newest message on turn 2.
import asyncio
import json

import websockets


async def turn(ws, messages, label):
    await ws.send(json.dumps({"type": "chat", "messages": messages, "max_tokens": 50}))
    n_read = n_think = 0
    junk = []
    while True:
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=180))
        if msg["type"] == "hello":
            continue
        if msg["type"] == "frame":
            if msg["phase"] == "reading":
                n_read += 1
            else:
                n_think += 1
            for t in msg["thoughts"]:
                w = t["w"]
                if not w.replace("'", "").replace("-", "").isalpha():
                    junk.append(w)
        else:
            print(f"[{label}] read={n_read} think={n_think} junk_words={junk[:5]}")
            print(f"[{label}] reply: {msg['text'][:90]!r}")
            return msg["text"]


async def main():
    async with websockets.connect("ws://127.0.0.1:8765/ws", max_size=None) as ws:
        msgs = [{"role": "user", "content": "My name is Nate. Remember it. Say only hello."}]
        r1 = await turn(ws, msgs, "turn1")
        msgs += [{"role": "assistant", "content": r1},
                 {"role": "user", "content": "What is my name?"}]
        r2 = await turn(ws, msgs, "turn2")
        ok = "nate" in r2.lower()
        print(f"\nhistory recall: {'PASS' if ok else 'FAIL'}")


asyncio.run(main())
