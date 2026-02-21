import json
import urllib.request
from label_studio_sdk import Client

from .templates import LABEL_COLORS, POLYGON_XML_LAYOUT

def build_label_config(categories: list[dict]) -> str:
    tags = ""
    for i, cat in enumerate(categories):
        color = LABEL_COLORS[i % len(LABEL_COLORS)]
        tags += f'    <Label value="{cat["name"]}" background="{color}"/>\n'
    return POLYGON_XML_LAYOUT.format(label_tags=tags)

def create_project(title: str, label_config: str, ls_base: str, token: str):
    ls = Client(url=ls_base, api_key=token)
    return ls.create_project(title=title, label_config=label_config)

def api_post_json(path: str, payload: list, ls_base: str, token: str) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{ls_base.rstrip('/')}{path}",
        data=data,
        method="POST",
        headers={
            "Authorization": f"Token {token}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())