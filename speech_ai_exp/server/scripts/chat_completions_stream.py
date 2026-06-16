#!/usr/bin/env python3
"""Interactive test client for the remote LLM server's chat-completions API."""

import json

import requests

URL = "http://172.16.1.7:8080/v1/chat/completions"

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer abcdefg",
}

prompt = input("You: ")
print("Assistant: ", end="", flush=True)

resp = requests.post(
    URL,
    headers=HEADERS,
    json={
        "model": "local-model",
        "stream": True,
        "messages": [{"role": "user", "content": prompt}],
    },
    stream=True,
)

for line in resp.iter_lines(decode_unicode=True):
    if not line or not line.startswith("data: "):
        continue
    data = line[6:]
    if data == "[DONE]":
        break
    chunk = json.loads(data)
    token = chunk["choices"][0]["delta"].get("content")
    if token:
        print(token, end="", flush=True)

print()
