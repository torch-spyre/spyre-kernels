# Pre-flight: Consult the Spyre Knowledge Base

Shared by all kernel-authoring skills. Run this before convert / review / test.

Query the `spyre-kb` MCP server for context relevant to the kernel at hand:

1. **Check for an existing skill** — `mcp__spyre-kb__skill(name="<kernel_type>")`
   (e.g. `name="matmul"`) to see whether a wiki-defined skill already covers
   this pattern.
2. **Search for guidance** — `mcp__spyre-kb__search(query="<topic>")` with the
   kernel's domain (`"attention"`, `"layernorm"`, `"softmax"`) or an API topic
   (`"tensor descriptor"`, `"distribution loop"`, `"precision DL16 BF16"`).
3. **Read pages** — if search returns relevant pages, `mcp__spyre-kb__read(path="<path>")`
   for full content.

Use KB results to **supplement, not override** the procedure in each skill. If
the KB gives more specific or newer guidance for a pattern, prefer it and note
the discrepancy in your output.
