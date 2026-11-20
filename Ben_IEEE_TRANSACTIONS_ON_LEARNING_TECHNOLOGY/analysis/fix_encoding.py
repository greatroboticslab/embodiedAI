from pathlib import Path

p = Path(__file__).resolve().parent.parent / "paper.tex"
s = p.read_text(encoding="utf-8")
s = s.lstrip("﻿")
# Reverse the cp1252-as-utf8 double encoding introduced by Set-Content.
# Windows ANSI maps bytes 0x81,0x8D,0x8F,0x90,0x9D to the C1 codepoints;
# Python's cp1252 leaves them undefined, so map them back manually.
import codecs
def c1_passthrough(err):
    ch = err.object[err.start:err.end]
    return bytes(ord(c) for c in ch), err.end
codecs.register_error("c1", c1_passthrough)
b = s.encode("cp1252", errors="c1")
fixed = b.decode("utf-8")
p.write_text(fixed, encoding="utf-8", newline="\n")
print("non-ascii chars after fix:", sorted({c for c in fixed if ord(c) > 127}))
