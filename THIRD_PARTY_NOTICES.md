# Third-party material

[FeatLens](https://github.com/turhancan97/FeatLens) v0.3.2 is installed as a pip dependency at commit `07eddd718a5a53d11ca5cc21e69a21d0f7b9f2c3`. `src/attention.py` imports its cosine-similarity function and rollout helper directly; no FeatLens source is bundled here. Earlier revisions bundled the same functions, and existing figures were produced using that equivalent implementation. FeatLens is MIT-licensed; its license is reproduced below for attribution.

```text
MIT License

Copyright (c) 2026 Turhan Can Kargın

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

COCO photographs and image-containing figures retain the source-image terms listed in [images/SOURCES.md](images/SOURCES.md). The repository MIT license does not relicense those photographs or pretrained checkpoints.
